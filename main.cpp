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

// Force 16-byte alignment on Host to match GPU
struct alignas(16) SimplePoint {
    float x, y, z;
    union {
        struct { unsigned char b, g, r, a; };
        float rgb;
    };
};

extern "C" void cuda_compute_cloud(
    const unsigned short* d_depth, 
    const unsigned char* d_rgb, 
    void* d_cloud, 
    int width, int height, 
    float fx, float fy, float cx, float cy);

int main(int argc, char** argv) {
    // 1. Argument Parsing
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rgb_image> <depth_image> [-v]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  -v   Enable visualization" << std::endl;
        return -1;
    }

    std::string rgb_path = argv[1];
    std::string depth_path = argv[2];
    bool visualize = false;

    // Check for optional flags starting from the 3rd argument
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--viz") {
            visualize = true;
        }
    }

    // 2. Load Images
    cv::Mat rgb_img = cv::imread(rgb_path, cv::IMREAD_COLOR);
    cv::Mat depth_img = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

    if (rgb_img.empty() || depth_img.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    // 3. Pre-process (Resize & Convert)
    if (depth_img.size() != rgb_img.size()) {
        cv::resize(depth_img, depth_img, rgb_img.size(), 0, 0, cv::INTER_NEAREST);
    }

    cv::Mat rgb_converted;
    cv::cvtColor(rgb_img, rgb_converted, cv::COLOR_BGR2RGB);

    // Force continuous memory
    if (!depth_img.isContinuous()) depth_img = depth_img.clone();
    if (!rgb_converted.isContinuous()) rgb_converted = rgb_converted.clone();

    int width = rgb_converted.cols;
    int height = rgb_converted.rows;
    size_t num_pixels = width * height;

    // 4. Intrinsics
    float fx = 525.0f * (width / 640.0f);
    float fy = 525.0f * (height / 480.0f);
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    // 5. GPU Allocation & Compute
    unsigned short *d_depth;
    unsigned char *d_rgb;
    SimplePoint *d_cloud;

    cudaMalloc(&d_depth, num_pixels * sizeof(unsigned short));
    cudaMalloc(&d_rgb, num_pixels * 3 * sizeof(unsigned char));
    cudaMalloc(&d_cloud, num_pixels * sizeof(SimplePoint));

    cudaMemcpy(d_depth, depth_img.data, num_pixels * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgb, rgb_converted.data, num_pixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cuda_compute_cloud(d_depth, d_rgb, (void*)d_cloud, width, height, fx, fy, cx, cy);

    // 6. Download Results
    std::vector<SimplePoint> host_buffer(num_pixels);
    cudaMemcpy(host_buffer.data(), d_cloud, num_pixels * sizeof(SimplePoint), cudaMemcpyDeviceToHost);

    // Convert to PCL Cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->width = width;
    cloud->height = height;
    cloud->is_dense = false;
    cloud->points.resize(num_pixels);

    #pragma omp parallel for
    for (size_t i = 0; i < num_pixels; ++i) {
        cloud->points[i].x = host_buffer[i].x;
        cloud->points[i].y = host_buffer[i].y;
        cloud->points[i].z = host_buffer[i].z;
        cloud->points[i].rgb = host_buffer[i].rgb;
    }

    // Save File
    pcl::io::savePCDFileBinary("output_gpu.pcd", *cloud);
    std::cout << "Saved cloud to output_gpu.pcd (" << num_pixels << " points)" << std::endl;

    // 7. Optional Visualization
    if (visualize) {
        std::cout << "Opening Viewer..." << std::endl;
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0.1, 0.1, 0.1);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
        
        viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
        
        viewer->addCoordinateSystem(0.5); 
        viewer->initCameraParameters();
        viewer->setCameraPosition(0, 0, -1.0, 0, 0, 1, 0, -1, 0);

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    cudaFree(d_depth);
    cudaFree(d_rgb);
    cudaFree(d_cloud);

    return 0;
}