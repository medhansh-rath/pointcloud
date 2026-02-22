#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>

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

extern "C" void cuda_reproject_to_image(
    const PointXYZRGB* d_cloud,
    const float4* d_normals,
    float* d_image,
    int width, int height);

extern "C" void cuda_fill_depth_holes(
    unsigned short* d_depth,
    int width, int height,
    int max_radius);

extern "C" void cuda_fill_depth_holes_avg(
    unsigned short* d_depth,
    int width, int height,
    int max_radius);

extern "C" void cuda_fill_depth_blobs(
    unsigned short* d_depth,
    int width, int height,
    int max_iters);

extern "C" void cuda_fill_depth_jfa(
    unsigned short* d_depth,
    int width, int height);

int main(int argc, char** argv) {
    auto process_start = std::chrono::high_resolution_clock::now();
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rgb_image> <depth_image> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  -n   Compute Surface Normals" << std::endl;
        std::cerr << "  -v   Visualize result" << std::endl;
        std::cerr << "  -p   Save PCD point cloud" << std::endl;
        std::cerr << "  -t   Show timing information" << std::endl;
        std::cerr << "  -s   Save filled depth image (nearest)" << std::endl;
        std::cerr << "  -a   Save filled depth image (average)" << std::endl;
        std::cerr << "  -b   Save filled depth image (blob-based)" << std::endl;
        std::cerr << "  -j   Save filled depth image (jump flooding)" << std::endl;
        std::cerr << "  --fill-radius <r>   Set max fill radius (default 10)" << std::endl;
        std::cerr << "  --blob-iters <i>   Set max blob iterations (default 10)" << std::endl;
        return -1;
    }

    // 1. Parse Arguments
    std::string rgb_path = argv[1];
    std::string depth_path = argv[2];
    bool use_normals = false;
    bool visualize = false;
    bool save_pcd = false;
    bool show_timers = false;
    bool save_filled_depth_nearest = false;
    bool save_filled_depth_avg = false;
    bool save_filled_depth_blob = false;
    bool save_filled_depth_jfa = false;
    int fill_radius = 10;
    int blob_iters = 10;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--normals") use_normals = true;
        if (arg == "-v" || arg == "--viz") visualize = true;
        if (arg == "-p" || arg == "--save-pcd") save_pcd = true;
        if (arg == "-t" || arg == "--timers") show_timers = true;
        if (arg == "-s" || arg == "--save-depth") save_filled_depth_nearest = true;
        if (arg == "-a" || arg == "--save-depth-avg") save_filled_depth_avg = true;
        if (arg == "-b" || arg == "--save-depth-blob") save_filled_depth_blob = true;
        if (arg == "-j" || arg == "--save-depth-jfa") save_filled_depth_jfa = true;
        if (arg == "--fill-radius" && i + 1 < argc) {
            fill_radius = std::stoi(argv[++i]);
        }
        if (arg == "--blob-iters" && i + 1 < argc) {
            blob_iters = std::stoi(argv[++i]);
        }
    }

    // 2. Load & Pre-process Images
    auto t_load_start = std::chrono::high_resolution_clock::now();
    cv::Mat rgb_img = cv::imread(rgb_path, cv::IMREAD_COLOR);
    cv::Mat depth_img = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    auto t_imread_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "cv::imread time: " << std::chrono::duration<double, std::milli>(t_imread_end - t_load_start).count() << " ms" << std::endl;
    auto t_load_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "Image loading time: " << std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count() << " ms" << std::endl;

    if (rgb_img.empty() || depth_img.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    auto t_preproc_start = std::chrono::high_resolution_clock::now();
    if (depth_img.size() != rgb_img.size()) {
        cv::resize(depth_img, depth_img, rgb_img.size(), 0, 0, cv::INTER_NEAREST);
    }

    cv::Mat rgb_conv;
    cv::cvtColor(rgb_img, rgb_conv, cv::COLOR_BGR2RGB);

    // Force continuous memory
    if (!depth_img.isContinuous()) depth_img = depth_img.clone();
    if (!rgb_conv.isContinuous()) rgb_conv = rgb_conv.clone();
    auto t_preproc_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "Image preprocessing time: " << std::chrono::duration<double, std::milli>(t_preproc_end - t_preproc_start).count() << " ms" << std::endl;

    int width = rgb_conv.cols;
    int height = rgb_conv.rows;
    size_t num_pixels = width * height;

    // Intrinsics (Auto-scaled)
    float fx = 525.0f * (width / 640.0f);
    float fy = 525.0f * (height / 480.0f);
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    // 3. GPU Allocation (Async)
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    unsigned short *d_depth = nullptr;
    unsigned char *d_rgb = nullptr;
    PointXYZRGB *d_cloud = nullptr;
    float4 *d_normals = nullptr;
    float *d_image = nullptr;

    auto t_alloc_start = std::chrono::high_resolution_clock::now();
    // Allocate only what is needed (non-blocking async allocation)
    cudaMallocAsync(&d_depth, num_pixels * sizeof(unsigned short), stream);
    cudaMallocAsync(&d_rgb, num_pixels * 3 * sizeof(unsigned char), stream);
    // Only allocate d_cloud if point cloud computation or normals are needed
    bool need_cloud = true;
    if (need_cloud) cudaMallocAsync(&d_cloud, num_pixels * sizeof(PointXYZRGB), stream);
    // Only allocate d_normals and d_image if normals or reproject are needed
    if (use_normals) {
        cudaMallocAsync(&d_normals, num_pixels * sizeof(float4), stream);
        cudaMallocAsync(&d_image, num_pixels * 7 * sizeof(float), stream);
    }
    // Synchronize stream to ensure all allocations complete
    cudaStreamSynchronize(stream);
    auto t_alloc_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "GPU allocation time: " << std::chrono::duration<double, std::milli>(t_alloc_end - t_alloc_start).count() << " ms" << std::endl;

    // 4. Upload & Compute
    auto t_upload_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_depth, depth_img.data, num_pixels * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgb, rgb_conv.data, num_pixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    auto t_upload_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "Image upload time: " << std::chrono::duration<double, std::milli>(t_upload_end - t_upload_start).count() << " ms" << std::endl;

    // Fill holes in depth image (if requested)
    auto t_hole_start = std::chrono::high_resolution_clock::now();
    auto t_filesave_start = std::chrono::high_resolution_clock::now();
    if (save_filled_depth_nearest) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_holes(d_depth, width, height, fill_radius);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_nearest.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_nearest.png'" << std::endl;
        if (show_timers) std::cout << "Nearest method time: " << elapsed << " ms" << std::endl;
    }
    if (save_filled_depth_avg) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_holes_avg(d_depth, width, height, fill_radius);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_avg.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_avg.png'" << std::endl;
        if (show_timers) std::cout << "Average method time: " << elapsed << " ms" << std::endl;
    }
    if (save_filled_depth_blob) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_blobs(d_depth, width, height, blob_iters);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_blob.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_blob.png'" << std::endl;
        if (show_timers) std::cout << "Blob method time: " << elapsed << " ms" << std::endl;
    }
    if (save_filled_depth_jfa) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_jfa(d_depth, width, height);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_jfa.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_jfa.png'" << std::endl;
        if (show_timers) std::cout << "Jump Flooding method time: " << elapsed << " ms" << std::endl;
    }
    auto t_hole_end = std::chrono::high_resolution_clock::now();
    auto t_filesave_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "Hole filling time (GPU + download): " << std::chrono::duration<double, std::milli>(t_hole_end - t_hole_start).count() << " ms" << std::endl;
    if (show_timers) std::cout << "File I/O save time: " << std::chrono::duration<double, std::milli>(t_filesave_end - t_filesave_start).count() << " ms" << std::endl;

    // Compute Points
    auto t_cloud_start = std::chrono::high_resolution_clock::now();
    cuda_compute_cloud(d_depth, d_rgb, d_cloud, width, height, fx, fy, cx, cy);
    cudaDeviceSynchronize();
    auto t_cloud_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "Point cloud computation time: " << std::chrono::duration<double, std::milli>(t_cloud_end - t_cloud_start).count() << " ms" << std::endl;

    // Compute Normals (Optional)
    auto t_normals_start = std::chrono::high_resolution_clock::now();
    if (use_normals) {
        cuda_compute_normals(d_cloud, d_normals, width, height);
        cudaDeviceSynchronize();
        cuda_reproject_to_image(d_cloud, d_normals, d_image, width, height);
        cudaDeviceSynchronize();
    }
    auto t_normals_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "Normals/reproject time: " << std::chrono::duration<double, std::milli>(t_normals_end - t_normals_start).count() << " ms" << std::endl;

    // 5. Download Results
    auto t_download_start = std::chrono::high_resolution_clock::now();
    std::vector<PointXYZRGB> h_points(num_pixels);
    cudaMemcpy(h_points.data(), d_cloud, num_pixels * sizeof(PointXYZRGB), cudaMemcpyDeviceToHost);
    std::vector<float4> h_normals;
    if (use_normals) {
        h_normals.resize(num_pixels);
        cudaMemcpy(h_normals.data(), d_normals, num_pixels * sizeof(float4), cudaMemcpyDeviceToHost);
    }
    std::vector<float> h_image;
    if (use_normals) {
        h_image.resize(num_pixels * 7);
        cudaMemcpy(h_image.data(), d_image, num_pixels * 7 * sizeof(float), cudaMemcpyDeviceToHost);
    }
    auto t_download_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "Download time: " << std::chrono::duration<double, std::milli>(t_download_end - t_download_start).count() << " ms" << std::endl;

    // 6. Convert to PCL & Save (Optional)
    auto t_pcl_start = std::chrono::high_resolution_clock::now();
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

    if (save_pcd) {
        pcl::io::savePCDFileBinary("output.pcd", *cloud);
        std::cout << "Saved 'output.pcd' (" << (use_normals ? "With Normals" : "No Normals") << ")" << std::endl;
    } else {
        std::cout << "PCD save skipped (use -p to enable)" << std::endl;
    }
    auto t_pcl_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "PCL conversion & PCD save time: " << std::chrono::duration<double, std::milli>(t_pcl_end - t_pcl_start).count() << " ms" << std::endl;

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

    // Cleanup (Async)
    auto t_cleanup_start = std::chrono::high_resolution_clock::now();
    if (d_depth) cudaFreeAsync(d_depth, stream);
    if (d_rgb) cudaFreeAsync(d_rgb, stream);
    if (d_cloud) cudaFreeAsync(d_cloud, stream);
    if (d_normals) cudaFreeAsync(d_normals, stream);
    if (d_image) cudaFreeAsync(d_image, stream);
    // Synchronize stream to ensure all frees complete
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    auto t_cleanup_end = std::chrono::high_resolution_clock::now();
    if (show_timers) std::cout << "Cleanup time: " << std::chrono::duration<double, std::milli>(t_cleanup_end - t_cleanup_start).count() << " ms" << std::endl;

    auto process_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double, std::milli>(process_end - process_start).count();
    std::cout << "Total process time: " << total_elapsed << " ms" << std::endl;

    return 0;
}