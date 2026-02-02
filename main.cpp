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

// Host Struct
struct alignas(16) SimplePoint {
    float x, y, z;
    union {
        struct { unsigned char b, g, r, a; };
        float rgb;
    };
};

// CUDA Wrappers
extern "C" void cuda_compute_cloud(
    const unsigned short* d_depth, const unsigned char* d_rgb, void* d_cloud, 
    int width, int height, float fx, float fy, float cx, float cy);

extern "C" void cuda_compute_normals(
    const void* d_cloud, float4* d_normals, int width, int height);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rgb_image> <depth_image> [-v]" << std::endl;
        return -1;
    }
    
    // --- 1. Load & Prep Images ---
    cv::Mat rgb_img = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat depth_img = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    bool visualize = (argc > 3 && std::string(argv[3]) == "-v");

    if (rgb_img.empty() || depth_img.empty()) return -1;
    if (depth_img.size() != rgb_img.size()) cv::resize(depth_img, depth_img, rgb_img.size(), 0, 0, cv::INTER_NEAREST);
    
    cv::Mat rgb_conv;
    cv::cvtColor(rgb_img, rgb_conv, cv::COLOR_BGR2RGB);

    if (!depth_img.isContinuous()) depth_img = depth_img.clone();
    if (!rgb_conv.isContinuous()) rgb_conv = rgb_conv.clone();

    int width = rgb_conv.cols;
    int height = rgb_conv.rows;
    size_t num_pixels = width * height;

    float fx = 525.0f * (width / 640.0f);
    float fy = 525.0f * (height / 480.0f);
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    // --- 2. GPU Allocation ---
    unsigned short *d_depth;
    unsigned char *d_rgb;
    SimplePoint *d_cloud;
    float4 *d_normals; // Buffer for Normals

    cudaMalloc(&d_depth, num_pixels * sizeof(unsigned short));
    cudaMalloc(&d_rgb, num_pixels * 3 * sizeof(unsigned char));
    cudaMalloc(&d_cloud, num_pixels * sizeof(SimplePoint));
    cudaMalloc(&d_normals, num_pixels * sizeof(float4)); // Allocate for normals

    // --- 3. Upload & Compute ---
    cudaMemcpy(d_depth, depth_img.data, num_pixels * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgb, rgb_conv.data, num_pixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Step A: Compute Points
    cuda_compute_cloud(d_depth, d_rgb, (void*)d_cloud, width, height, fx, fy, cx, cy);
    
    // Step B: Compute Normals (Using the points on GPU)
    cuda_compute_normals((void*)d_cloud, d_normals, width, height);

    // --- 4. Download ---
    std::vector<SimplePoint> h_points(num_pixels);
    std::vector<float4> h_normals(num_pixels);

    cudaMemcpy(h_points.data(), d_cloud, num_pixels * sizeof(SimplePoint), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normals.data(), d_normals, num_pixels * sizeof(float4), cudaMemcpyDeviceToHost);

    // --- 5. Convert to PCL & Save ---
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    cloud->width = width;
    cloud->height = height;
    cloud->is_dense = false;
    cloud->points.resize(num_pixels);

    #pragma omp parallel for
    for (size_t i = 0; i < num_pixels; ++i) {
        // Point
        cloud->points[i].x = h_points[i].x;
        cloud->points[i].y = h_points[i].y;
        cloud->points[i].z = h_points[i].z;
        cloud->points[i].rgb = h_points[i].rgb;
        
        // Normal
        cloud->points[i].normal_x = h_normals[i].x;
        cloud->points[i].normal_y = h_normals[i].y;
        cloud->points[i].normal_z = h_normals[i].z;
        cloud->points[i].curvature = 0;
    }

    pcl::io::savePCDFileBinary("output_with_normals.pcd", *cloud);
    std::cout << "Saved 'output_with_normals.pcd'" << std::endl;

    // --- 6. Visualize ---
    if (visualize) {
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0.1, 0.1, 0.1);
        
        // Add cloud with Normals
        // Level=10 (show every 10th normal), Scale=0.05
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(cloud);
        viewer->addPointCloud<pcl::PointXYZRGBNormal>(cloud, rgb, "cloud");
        viewer->addPointCloudNormals<pcl::PointXYZRGBNormal>(cloud, 10, 0.05, "normals");
        
        viewer->addCoordinateSystem(0.5); 
        viewer->initCameraParameters();
        viewer->setCameraPosition(0, 0, -1.0, 0, 0, 1, 0, -1, 0);

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    cudaFree(d_depth); cudaFree(d_rgb); cudaFree(d_cloud); cudaFree(d_normals);
    return 0;
}