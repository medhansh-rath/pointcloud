#include <cuda_runtime.h>
#include <stdio.h>

// 1. Define the PCL Point Structure equivalent for GPU
// PCL uses 16-byte alignment for PointXYZRGB (4 floats)
struct __align__(16) PointXYZRGB
{
    float x, y, z;
    union {
        struct {
            unsigned char b, g, r, a;
        };
        float rgb;
    };
};

// 2. The Kernel
__global__ void depthToCloudKernel(
    const unsigned short* __restrict__ depth_map,
    const unsigned char* __restrict__ rgb_image,
    PointXYZRGB* __restrict__ output_cloud,
    int width, int height,
    float fx, float fy, float cx, float cy)
{
    // Calculate pixel coordinates (u, v) for this thread
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    int idx = v * width + u;

    // Read depth (in mm) and convert to meters
    unsigned short d_raw = depth_map[idx];
    
    // Check for invalid depth (0)
    if (d_raw == 0) {
        output_cloud[idx].x = __int_as_float(0x7fc00000); // NaN
        output_cloud[idx].y = __int_as_float(0x7fc00000); // NaN
        output_cloud[idx].z = __int_as_float(0x7fc00000); // NaN
        return;
    }

    float z = (float)d_raw * 0.001f; // Convert mm to meters
    float x = (u - cx) * z / fx;
    float y = (v - cy) * z / fy;

    // Read RGB
    // Assuming input is RGB packed (3 bytes). 
    // Note: If input includes Alpha (4 bytes), adjust index calculation.
    int rgb_idx = idx * 3; 
    unsigned char r = rgb_image[rgb_idx];
    unsigned char g = rgb_image[rgb_idx + 1];
    unsigned char b = rgb_image[rgb_idx + 2];

    // Store Output
    output_cloud[idx].x = x;
    output_cloud[idx].y = y;
    output_cloud[idx].z = z;
    
    // PCL packs RGB into a float/int
    // We can write bytes directly to the union components
    output_cloud[idx].r = r;
    output_cloud[idx].g = g;
    output_cloud[idx].b = b;
    output_cloud[idx].a = 255;
}

// 3. The Wrapper Function (Callable from C++)
extern "C" void cuda_compute_cloud(
    const unsigned short* d_depth, 
    const unsigned char* d_rgb, 
    PointXYZRGB* d_cloud, 
    int width, int height, 
    float fx, float fy, float cx, float cy)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    depthToCloudKernel<<<grid, block>>>(d_depth, d_rgb, d_cloud, width, height, fx, fy, cx, cy);
    
    cudaDeviceSynchronize();
}