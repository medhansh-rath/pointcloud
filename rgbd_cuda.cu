#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

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

// 2. The Kernel for Point Cloud Generation
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

// 3. The Kernel for Normal Computation
// We use a float4 to store (normal_x, normal_y, normal_z, curvature)
__global__ void computeNormalsKernel(
    const PointXYZRGB* __restrict__ cloud,
    float4* __restrict__ normals,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    int idx = v * width + u;
    
    // Get the current point
    PointXYZRGB p = cloud[idx];

    // Check if point is valid (z != NaN and z != 0)
    if (isnan(p.z) || p.z == 0) {
        normals[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // NaN normal
        return;
    }

    // Get neighbor indices with boundary checks
    int left_idx  = (u > 0)          ? idx - 1     : idx;
    int right_idx = (u < width - 1)  ? idx + 1     : idx;
    int up_idx    = (v > 0)          ? idx - width : idx;
    int down_idx  = (v < height - 1) ? idx + width : idx;

    PointXYZRGB pl = cloud[left_idx];
    PointXYZRGB pr = cloud[right_idx];
    PointXYZRGB pu = cloud[up_idx];
    PointXYZRGB pd = cloud[down_idx];

    // If any neighbor is NaN, we cannot compute a good normal
    if (isnan(pl.z) || isnan(pr.z) || isnan(pu.z) || isnan(pd.z)) {
         normals[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
         return;
    }

    // Compute vectors
    // Horizontal Vector (Right - Left)
    float3 horiz = make_float3(pr.x - pl.x, pr.y - pl.y, pr.z - pl.z);
    
    // Vertical Vector (Down - Up)
    // Note: In image space Y increases downwards, but in 3D Y often points down too.
    // We stick to (Down - Up) for the vertical difference vector.
    float3 vert = make_float3(pd.x - pu.x, pd.y - pu.y, pd.z - pu.z);

    // Cross Product: Normal = Horiz x Vert
    float3 n;
    n.x = horiz.y * vert.z - horiz.z * vert.y;
    n.y = horiz.z * vert.x - horiz.x * vert.z;
    n.z = horiz.x * vert.y - horiz.y * vert.x;

    // Normalize
    float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    
    if (norm > 1e-6f) {
        float inv_norm = 1.0f / norm;
        n.x *= inv_norm;
        n.y *= inv_norm;
        n.z *= inv_norm;
        
        // Orient towards camera (which is at 0,0,0)
        // View vector is simply -p
        // Dot product: n . (-p) > 0  =>  n . p < 0
        float dot = n.x * p.x + n.y * p.y + n.z * p.z;
        if (dot > 0) {
            n.x = -n.x;
            n.y = -n.y;
            n.z = -n.z;
        }
        
        normals[idx] = make_float4(n.x, n.y, n.z, 0.0f); // curvature = 0
    } else {
        normals[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // Invalid
    }
}

// 4. The Wrapper Functions (Callable from C++)

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

extern "C" void cuda_compute_normals(
    const PointXYZRGB* d_cloud,
    float4* d_normals,
    int width, int height)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    computeNormalsKernel<<<grid, block>>>(d_cloud, d_normals, width, height);

    cudaDeviceSynchronize();
}