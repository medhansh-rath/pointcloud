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

extern "C" void cuda_fill_depth_holes_median(
    unsigned short* d_depth,
    int width, int height,
    int max_radius);

extern "C" void cuda_fill_depth_holes_mode(
    unsigned short* d_depth,
    int width, int height,
    int max_radius);

extern "C" void cuda_fill_depth_holes_ip_basic(
    unsigned short* d_depth,
    int width, int height,
    int max_iters);

extern "C" void cuda_fill_depth_guided_filter(
    unsigned short* d_depth,
    const unsigned char* d_rgb,
    int width, int height,
    int filter_radius,
    int max_iters,
    float color_sigma);

extern "C" void cuda_fill_depth_true_guided(
    unsigned short* d_depth,
    const unsigned char* d_rgb,
    int width, int height,
    int radius,
    float eps,
    bool apply_all_pixels);

extern "C" void cuda_fill_depth_max_filter(
    unsigned short* d_depth,
    int width, int height,
    int filter_radius,
    int max_iters);

extern "C" void cuda_find_largest_interior_blob(
    const unsigned short* d_depth,
    int width, int height,
    int* max_dimension, int* blob_size, int* blob_width, int* blob_height);

int main(int argc, char** argv) {
    auto process_start = std::chrono::high_resolution_clock::now();
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rgb_image> <depth_image> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  -n   Compute Surface Normals" << std::endl;
        std::cerr << "  -v   Visualize result" << std::endl;
        std::cerr << "  -p   Save PCD point cloud" << std::endl;
        std::cerr << "  -B   Save binary point cloud (.bin)" << std::endl;
        std::cerr << "  -L   Save binary as CIELAB instead of RGB" << std::endl;
        std::cerr << "  -t   Show timing information" << std::endl;
        std::cerr << "  -s   Save filled depth image (nearest)" << std::endl;
        std::cerr << "  -a   Save filled depth image (average)" << std::endl;
        std::cerr << "  -b   Save filled depth image (blob-based)" << std::endl;
        std::cerr << "  -j   Save filled depth image (jump flooding)" << std::endl;
        std::cerr << "  -m   Save filled depth image (median)" << std::endl;
        std::cerr << "  -o   Save filled depth image (mode)" << std::endl;
        std::cerr << "  -c   Save filled depth image (IP-Basic inpainting)" << std::endl;
        std::cerr << "  -g   Save filled depth image (Guided Filter)" << std::endl;
        std::cerr << "  -G   Save filled depth image (True Guided Filter, fills zeros only)" << std::endl;
        std::cerr << "  -A   Apply true guided filter to all pixels (denoise + fill, use with -G)" << std::endl;
        std::cerr << "  -x   Save filled depth image (Maximum Filter)" << std::endl;
        std::cerr << "  -d   Save RGB+depth overlay image" << std::endl;
        std::cerr << "  --fill-radius <r>   Set max fill radius (default 10)" << std::endl;
        std::cerr << "  --blob-iters <i>   Set max blob iterations (default 10)" << std::endl;
        std::cerr << "  --guided-radius <r>   Set guided filter radius (default 2)" << std::endl;
        std::cerr << "  --guided-sigma <s>   Set guided filter color sigma (default 30.0)" << std::endl;
        std::cerr << "  --true-guided-radius <r>   Set true guided filter radius (default auto with -G, 0=auto-detect)" << std::endl;
        std::cerr << "  --true-guided-eps <e>   Set true guided filter eps (default 1e-3)" << std::endl;
        std::cerr << "  --output <file>   Set output binary file path (default: output.bin)" << std::endl;
        return -1;
    }

    // 1. Parse Arguments
    std::string rgb_path = argv[1];
    std::string depth_path = argv[2];
    bool use_normals = false;
    bool visualize = false;
    bool save_pcd = false;
    bool save_binary = false;
    bool save_binary_cielab = false;
    bool show_timers = false;
    bool save_filled_depth_nearest = false;
    bool save_filled_depth_avg = false;
    bool save_filled_depth_blob = false;
    bool save_filled_depth_jfa = false;
    bool save_filled_depth_median = false;
    bool save_filled_depth_mode = false;
    bool save_filled_depth_ip_basic = false;
    bool save_filled_depth_guided = false;
    bool save_filled_depth_true_guided = false;
    bool true_guided_all_pixels = false;
    bool save_filled_depth_max = false;
    bool save_depth_overlay = false;
    int fill_radius = 10;
    int blob_iters = 10;
    int guided_filter_radius = 2;
    float guided_color_sigma = 30.0f;
    int true_guided_radius = 0;
    float true_guided_eps = 1e-3f;
    bool auto_radius = false;
    bool true_guided_radius_set = false;
    std::string output_filename = "output.bin";

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--normals") use_normals = true;
        if (arg == "-v" || arg == "--viz") visualize = true;
        if (arg == "-p" || arg == "--save-pcd") save_pcd = true;
        if (arg == "-B" || arg == "--save-binary") save_binary = true;
        if (arg == "-L" || arg == "--save-cielab") save_binary_cielab = true;
        if (arg == "-t" || arg == "--timers") show_timers = true;
        if (arg == "-s" || arg == "--save-depth") save_filled_depth_nearest = true;
        if (arg == "-a" || arg == "--save-depth-avg") save_filled_depth_avg = true;
        if (arg == "-b" || arg == "--save-depth-blob") save_filled_depth_blob = true;
        if (arg == "-j" || arg == "--save-depth-jfa") save_filled_depth_jfa = true;
        if (arg == "-m" || arg == "--save-depth-median") save_filled_depth_median = true;
        if (arg == "-o" || arg == "--save-depth-mode") save_filled_depth_mode = true;
        if (arg == "-c" || arg == "--save-depth-ip-basic") save_filled_depth_ip_basic = true;
        if (arg == "-g" || arg == "--save-depth-guided") save_filled_depth_guided = true;
        if (arg == "-G" || arg == "--save-depth-true-guided") save_filled_depth_true_guided = true;
        if (arg == "-A" || arg == "--true-guided-all-pixels") true_guided_all_pixels = true;
        if (arg == "-x" || arg == "--save-depth-max") save_filled_depth_max = true;
        if (arg == "-d" || arg == "--save-depth-overlay") save_depth_overlay = true;
        if (arg == "--fill-radius" && i + 1 < argc) {
            fill_radius = std::stoi(argv[++i]);
        }
        if (arg == "--blob-iters" && i + 1 < argc) {
            blob_iters = std::stoi(argv[++i]);
        }
        if (arg == "--guided-radius" && i + 1 < argc) {
            guided_filter_radius = std::stoi(argv[++i]);
        }
        if (arg == "--guided-sigma" && i + 1 < argc) {
            guided_color_sigma = std::stof(argv[++i]);
        }
        if (arg == "--true-guided-radius" && i + 1 < argc) {
            true_guided_radius = std::stoi(argv[++i]);
            true_guided_radius_set = true;
            if (true_guided_radius == 0) {
                auto_radius = true;
            } else {
                auto_radius = false;
            }
        }
            if (save_filled_depth_true_guided && !true_guided_radius_set) {
                auto_radius = true;
            }

        if (arg == "--true-guided-eps" && i + 1 < argc) {
            true_guided_eps = std::stof(argv[++i]);
        }
        if (arg == "--output" && i + 1 < argc) {
            output_filename = argv[++i];
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

    // Auto-detect radius based on largest interior blob (not touching edges) - CUDA accelerated
    if (save_filled_depth_true_guided && (auto_radius || true_guided_radius == 0)) {
        // Upload depth to GPU
        unsigned short* d_depth_temp;
        size_t depth_bytes = depth_img.rows * depth_img.cols * sizeof(unsigned short);
        cudaMalloc(&d_depth_temp, depth_bytes);
        cudaMemcpy(d_depth_temp, depth_img.data, depth_bytes, cudaMemcpyHostToDevice);
        
        // Call CUDA blob detection
        int max_dimension = 0, largest_blob_size = 0, largest_blob_width = 0, largest_blob_height = 0;
        cuda_find_largest_interior_blob(d_depth_temp, depth_img.cols, depth_img.rows,
                                        &max_dimension, &largest_blob_size, 
                                        &largest_blob_width, &largest_blob_height);
        
        cudaFree(d_depth_temp);
        
        // Set radius from the shorter blob side (better for long corridor-like holes)
        if (largest_blob_size > 0) {
            int min_dimension = std::min(largest_blob_width, largest_blob_height);
            true_guided_radius = std::max(2, std::min(64, (min_dimension + 1) / 2));
            
            if (show_timers || auto_radius) {
                std::cout << "Auto-detected true-guided-radius: " << true_guided_radius 
                          << " (largest interior blob: " << largest_blob_size 
                          << " pixels, " << largest_blob_width << "x" << largest_blob_height 
                          << ", min side: " << min_dimension
                          << ", max side: " << max_dimension << ")" << std::endl;
            }
        } else {
            // Fallback: no interior blobs, use default
            true_guided_radius = 4;
            if (show_timers || auto_radius) {
                std::cout << "No interior holes found, using default radius: " << true_guided_radius << std::endl;
            }
        }
    }

    cv::Mat rgb_conv;
    cv::cvtColor(rgb_img, rgb_conv, cv::COLOR_BGR2RGB);

    // Force continuous memory
    if (!depth_img.isContinuous()) depth_img = depth_img.clone();
    if (!rgb_conv.isContinuous()) rgb_conv = rgb_conv.clone();
    if (save_depth_overlay) {
        cv::Mat depth_mask = depth_img > 0;
        double min_val = 0.0;
        double max_val = 0.0;
        if (cv::countNonZero(depth_mask) > 0) {
            cv::minMaxLoc(depth_img, &min_val, &max_val, nullptr, nullptr, depth_mask);
        }
        cv::Mat depth_norm;
        if (max_val > min_val) {
            depth_img.convertTo(depth_norm, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
        } else {
            depth_norm = cv::Mat::zeros(depth_img.size(), CV_8U);
        }
        cv::Mat depth_color;
        cv::applyColorMap(depth_norm, depth_color, cv::COLORMAP_JET);
        cv::Mat overlay;
        cv::addWeighted(rgb_img, 0.6, depth_color, 0.4, 0.0, overlay);
        cv::imwrite("overlay_depth_rgb.png", overlay);
        std::cout << "Saved depth overlay image to 'overlay_depth_rgb.png'" << std::endl;
    }
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
    if (save_filled_depth_median) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_holes_median(d_depth, width, height, fill_radius);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_median.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_median.png'" << std::endl;
        if (show_timers) std::cout << "Median method time: " << elapsed << " ms" << std::endl;
    }
    if (save_filled_depth_mode) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_holes_mode(d_depth, width, height, fill_radius);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_mode.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_mode.png'" << std::endl;
        if (show_timers) std::cout << "Mode method time: " << elapsed << " ms" << std::endl;
    }
    if (save_filled_depth_ip_basic) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_holes_ip_basic(d_depth, width, height, blob_iters);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_ip_basic.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_ip_basic.png'" << std::endl;
        if (show_timers) std::cout << "IP-Basic method time: " << elapsed << " ms" << std::endl;
    }
    if (save_filled_depth_guided) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_guided_filter(d_depth, d_rgb, width, height, guided_filter_radius, blob_iters, guided_color_sigma);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_guided.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_guided.png'" << std::endl;
        if (show_timers) std::cout << "Guided Filter method time: " << elapsed << " ms" << std::endl;
    }
    if (save_filled_depth_true_guided) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_true_guided(
            d_depth, d_rgb, width, height,
            true_guided_radius, true_guided_eps,
            true_guided_all_pixels);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_true_guided.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_true_guided.png'" << std::endl;
        if (show_timers) std::cout << "True Guided Filter method time: " << elapsed << " ms" << std::endl;
    }
    if (save_filled_depth_max) {
        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_fill_depth_max_filter(d_depth, width, height, fill_radius, blob_iters);
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        cv::Mat filled_depth_img(height, width, CV_16UC1, h_filled_depth.data());
        cv::imwrite("filled_depth_max.png", filled_depth_img);
        std::cout << "Saved filled depth image to 'filled_depth_max.png'" << std::endl;
        if (show_timers) std::cout << "Maximum Filter method time: " << elapsed << " ms" << std::endl;
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
    
    // Save binary format if requested (n x m x 7 format)
    if (save_binary) {
        // Download filled depth from GPU if not already downloaded
        std::vector<unsigned short> h_filled_depth(num_pixels);
        cudaMemcpy(h_filled_depth.data(), d_depth, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToHost);
        
        // Create n x m x 7 array: [r, g, b (or L, a, b if CIELAB), depth, nx, ny, nz]
        std::vector<float> image_data(num_pixels * 7);
        
        // If CIELAB conversion is requested, prepare LAB data
        std::vector<uint8_t> lab_data;
        if (save_binary_cielab) {
            // Create an 8-bit BGR image from the RGB data
            cv::Mat rgb_mat(height, width, CV_8UC3);
            for (size_t i = 0; i < num_pixels; ++i) {
                rgb_mat.data[i * 3 + 0] = h_points[i].r;      // R channel
                rgb_mat.data[i * 3 + 1] = h_points[i].g;      // G channel
                rgb_mat.data[i * 3 + 2] = h_points[i].b;      // B channel
            }
            
            // Convert RGB to LAB
            cv::Mat lab_mat;
            cv::cvtColor(rgb_mat, lab_mat, cv::COLOR_RGB2Lab);
            
            // Extract LAB data
            for (size_t i = 0; i < num_pixels; ++i) {
                // OpenCV LAB: L [0-255], a [0-255], b [0-255]
                image_data[i * 7 + 0] = static_cast<float>(lab_mat.data[i * 3 + 0]);  // L
                image_data[i * 7 + 1] = static_cast<float>(lab_mat.data[i * 3 + 1]);  // a
                image_data[i * 7 + 2] = static_cast<float>(lab_mat.data[i * 3 + 2]);  // b
                
                // Depth (in mm or original units)
                image_data[i * 7 + 3] = static_cast<float>(h_filled_depth[i]);
                
                // Normals (0 if not computed)
                if (use_normals && i < h_normals.size()) {
                    image_data[i * 7 + 4] = h_normals[i].x;
                    image_data[i * 7 + 5] = h_normals[i].y;
                    image_data[i * 7 + 6] = h_normals[i].z;
                } else {
                    image_data[i * 7 + 4] = 0.0f;
                    image_data[i * 7 + 5] = 0.0f;
                    image_data[i * 7 + 6] = 0.0f;
                }
            }
        } else {
            // Use RGB as-is
            for (size_t i = 0; i < num_pixels; ++i) {
                // RGB (0-255 range)
                image_data[i * 7 + 0] = static_cast<float>(h_points[i].r);
                image_data[i * 7 + 1] = static_cast<float>(h_points[i].g);
                image_data[i * 7 + 2] = static_cast<float>(h_points[i].b);
                
                // Depth (in mm or original units)
                image_data[i * 7 + 3] = static_cast<float>(h_filled_depth[i]);
                
                // Normals (0 if not computed)
                if (use_normals && i < h_normals.size()) {
                    image_data[i * 7 + 4] = h_normals[i].x;
                    image_data[i * 7 + 5] = h_normals[i].y;
                    image_data[i * 7 + 6] = h_normals[i].z;
                } else {
                    image_data[i * 7 + 4] = 0.0f;
                    image_data[i * 7 + 5] = 0.0f;
                    image_data[i * 7 + 6] = 0.0f;
                }
            }
        }
        
        std::ofstream bin_file(output_filename, std::ios::binary);
        if (bin_file.is_open()) {
            // Write raw n x m x 7 data (no header for direct numpy loading)
            bin_file.write(reinterpret_cast<const char*>(image_data.data()), 
                          image_data.size() * sizeof(float));
            
            bin_file.close();
            std::string color_format = save_binary_cielab ? "CIELAB" : "RGB";
            std::cout << "Saved '" << output_filename << "' (" << width << "x" << height << " x 7 channels: " << color_format << " + Depth + Normals)" << std::endl;
        } else {
            std::cerr << "Error: Could not open " << output_filename << " for writing" << std::endl;
        }
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