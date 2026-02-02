#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// PCL Headers
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h> // To save the result

// OpenCV Headers
#include <opencv2/opencv.hpp>

// Forward declare the CUDA wrapper
extern "C" void cuda_compute_cloud(
    const unsigned short* d_depth, 
    const unsigned char* d_rgb, 
    void* d_cloud, 
    int width, int height, 
    float fx, float fy, float cx, float cy);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rgb_image> <depth_image>" << std::endl;
        return -1;
    }

    // 1. Load Images using OpenCV
    std::string rgb_path = argv[1];
    std::string depth_path = argv[2];

    // Load RGB (force 3 channels BGR)
    cv::Mat rgb_img = cv::imread(rgb_path, cv::IMREAD_COLOR);
    // Load Depth (force "unchanged" to keep 16-bit unsigned format)
    cv::Mat depth_img = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

    if (rgb_img.empty() || depth_img.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    // Check depth format (must be 16-bit)
    if (depth_img.type() != CV_16U) {
        std::cerr << "Error: Depth image must be 16-bit PNG (CV_16U)." << std::endl;
        return -1;
    }

    int width = rgb_img.cols;
    int height = rgb_img.rows;
    size_t num_pixels = width * height;

    std::cout << "Processing " << width << "x" << height << " image..." << std::endl;

    // 2. Allocate GPU Memory
    unsigned short *d_depth;
    unsigned char *d_rgb;
    void *d_cloud;

    cudaMalloc(&d_depth, num_pixels * sizeof(unsigned short));
    cudaMalloc(&d_rgb, num_pixels * 3 * sizeof(unsigned char));
    cudaMalloc(&d_cloud, num_pixels * sizeof(pcl::PointXYZRGB));

    // 3. Upload Images to GPU
    // OpenCV stores data row-by-row continuously, so we can copy directly
    cudaMemcpy(d_depth, depth_img.data, num_pixels * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgb, rgb_img.data, num_pixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 4. Run the Kernel (Intrinsics example: Kinect V1 / standard 640x480)
    // IMPORTANT: Change these to match YOUR camera!
    float fx = 525.0f;
    float fy = 525.0f;
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    cuda_compute_cloud(d_depth, d_rgb, d_cloud, width, height, fx, fy, cx, cy);

    // 5. Download Result
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->width = width;
    cloud->height = height;
    cloud->is_dense = false; // Because some points might be NaN (invalid depth)
    cloud->points.resize(num_pixels);

    cudaMemcpy(cloud->points.data(), d_cloud, num_pixels * sizeof(pcl::PointXYZRGB), cudaMemcpyDeviceToHost);

    // 6. Save to file
    pcl::io::savePCDFileBinary("output_gpu.pcd", *cloud);
    std::cout << "Saved point cloud to 'output_gpu.pcd'" << std::endl;

    // Cleanup
    cudaFree(d_depth);
    cudaFree(d_rgb);
    cudaFree(d_cloud);

    return 0;
}
