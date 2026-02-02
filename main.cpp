#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cuda_runtime.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>

// Match the struct in your .cu file
struct alignas(16) PointXYZRGB {
    float x, y, z;
    union {
        struct { unsigned char b, g, r, a; };
        float rgb;
    };
};

// CUDA Function Declarations
extern "C" void cuda_compute_cloud(
    const unsigned short* d_depth, const unsigned char* d_rgb, 
    PointXYZRGB* d_cloud, 
    int width, int height, 
    float fx, float fy, float cx, float cy);

extern "C" void cuda_compute_normals(
    const PointXYZRGB* d_cloud, 
    float4* d_normals, 
    int width, int height);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rgb_image> <depth_image> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  -n   Compute Surface Normals" << std::endl;
        std::cerr << "  -v   Visualize result" << std::endl;
        return -1;
    }

    // 1. Parse Arguments
    std::string rgb_path = argv[1];
    std::string depth_path = argv[2];
    bool use_normals = false;
    bool visualize = false;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--normals") use_normals = true;
        if (arg == "-v" || arg == "--viz") visualize = true;
    }

    // 2. Load & Pre-process Images
    cv::Mat rgb_img = cv::imread(rgb_path, cv::IMREAD_COLOR);
    cv::Mat depth_img = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

    if (rgb_img.empty() || depth_img.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    if (depth_img.size() != rgb_img.size()) {
        cv::resize(depth_img, depth_img, rgb_img.size(), 0, 0, cv::INTER_NEAREST);
    }

    cv::Mat rgb_conv;
    cv::cvtColor(rgb_img, rgb_conv, cv::COLOR_BGR2RGB);

    // Force continuous memory
    if (!depth_img.isContinuous()) depth_img = depth_img.clone();
    if (!rgb_conv.isContinuous()) rgb_conv = rgb_conv.clone();

    int width = rgb_conv.cols;
    int height = rgb_conv.rows;
    size_t num_pixels = width * height;

    // Intrinsics (Auto-scaled)
    float fx = 525.0f * (width / 640.0f);
    float fy = 525.0f * (height / 480.0f);
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    // 3. GPU Allocation
    unsigned short *d_depth;
    unsigned char *d_rgb;
    PointXYZRGB *d_cloud;
    float4 *d_normals = nullptr; // Initialize to nullptr

    cudaMalloc(&d_depth, num_pixels * sizeof(unsigned short));
    cudaMalloc(&d_rgb, num_pixels * 3 * sizeof(unsigned char));
    cudaMalloc(&d_cloud, num_pixels * sizeof(PointXYZRGB));

    // ONLY allocate memory for normals if the flag is set
    if (use_normals) {
        cudaMalloc(&d_normals, num_pixels * sizeof(float4));
    }

    // 4. Upload & Compute
    cudaMemcpy(d_depth, depth_img.data, num_pixels * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgb, rgb_conv.data, num_pixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Compute Points
    cuda_compute_cloud(d_depth, d_rgb, d_cloud, width, height, fx, fy, cx, cy);

    // Compute Normals (Optional)
    if (use_normals) {
        cuda_compute_normals(d_cloud, d_normals, width, height);
    }

    // 5. Download Results
    std::vector<PointXYZRGB> h_points(num_pixels);
    std::vector<float4> h_normals;

    cudaMemcpy(h_points.data(), d_cloud, num_pixels * sizeof(PointXYZRGB), cudaMemcpyDeviceToHost);

    if (use_normals) {
        h_normals.resize(num_pixels);
        cudaMemcpy(h_normals.data(), d_normals, num_pixels * sizeof(float4), cudaMemcpyDeviceToHost);
    }

    // 6. Convert to PCL & Save
    // We use PointXYZRGBNormal because it can hold both. 
    // If use_normals is false, the normal fields will just be 0.
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    cloud->width = width;
    cloud->height = height;
    cloud->is_dense = false;
    cloud->points.resize(num_pixels);

    #pragma omp parallel for
    for (size_t i = 0; i < num_pixels; ++i) {
        // Copy Point Data
        cloud->points[i].x = h_points[i].x;
        cloud->points[i].y = h_points[i].y;
        cloud->points[i].z = h_points[i].z;
        cloud->points[i].rgb = h_points[i].rgb;
        
        // Copy Normal Data (if enabled)
        if (use_normals) {
            cloud->points[i].normal_x = h_normals[i].x;
            cloud->points[i].normal_y = h_normals[i].y;
            cloud->points[i].normal_z = h_normals[i].z;
        } else {
            cloud->points[i].normal_x = 0;
            cloud->points[i].normal_y = 0;
            cloud->points[i].normal_z = 0;
        }
    }

    pcl::io::savePCDFileBinary("output.pcd", *cloud);
    std::cout << "Saved 'output.pcd' (" << (use_normals ? "With Normals" : "No Normals") << ")" << std::endl;

    // 7. Visualization
    if (visualize) {
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0.1, 0.1, 0.1);
        
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(cloud);
        viewer->addPointCloud<pcl::PointXYZRGBNormal>(cloud, rgb, "cloud");
        
        // Only draw normal lines if we actually computed them
        if (use_normals) {
            // Level=10 (every 10th point), Scale=0.05 (5cm lines)
            viewer->addPointCloudNormals<pcl::PointXYZRGBNormal>(cloud, 10, 0.05, "normals");
        }
        
        viewer->addCoordinateSystem(0.5); 
        viewer->initCameraParameters();
        viewer->setCameraPosition(0, 0, -1.0, 0, 0, 1, 0, -1, 0);

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // Cleanup
    cudaFree(d_depth);
    cudaFree(d_rgb);
    cudaFree(d_cloud);
    if (d_normals) cudaFree(d_normals); // Safety check

    return 0;
}