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

// 4. The Kernel for Reprojecting to Image
__global__ void reprojectToImageKernel(
    const PointXYZRGB* __restrict__ cloud,
    const float4* __restrict__ normals,
    float* __restrict__ image,  // size height * width * 7: R, G, B, depth, nx, ny, nz
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    int idx = v * width + u;
    int img_idx = idx * 7;

    PointXYZRGB p = cloud[idx];
    float4 n = normals[idx];

    // Check if point is valid
    if (isnan(p.z) || p.z == 0.0f) {
        // Set to zeros or NaN
        for (int i = 0; i < 7; ++i) {
            image[img_idx + i] = 0.0f;
        }
        return;
    }

    // R, G, B (normalized to 0-1)
    image[img_idx + 0] = p.r / 255.0f;
    image[img_idx + 1] = p.g / 255.0f;
    image[img_idx + 2] = p.b / 255.0f;

    // Depth (z in meters)
    image[img_idx + 3] = p.z;

    // Normal x, y, z
    image[img_idx + 4] = n.x;
    image[img_idx + 5] = n.y;
    image[img_idx + 6] = n.z;
}

// 5. The Kernel for Depth Hole Filling (Nearest)
__global__ void fillDepthHolesKernel(
    unsigned short* depth_map,
    int width, int height,
    int max_radius)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    int idx = v * width + u;
    if (depth_map[idx] != 0) return; // Only process holes

    for (int radius = 1; radius <= max_radius; ++radius) {
        bool found = false;
        unsigned short found_depth = 0;
        for (int du = -radius; du <= radius; ++du) {
            for (int dv = -radius; dv <= radius; ++dv) {
                if (abs(du) != radius && abs(dv) != radius) continue;
                int nu = u + du;
                int nv = v + dv;
                if (nu < 0 || nu >= width || nv < 0 || nv >= height) continue;
                int nidx = nv * width + nu;
                unsigned short neighbor_depth = depth_map[nidx];
                if (neighbor_depth != 0) {
                    found_depth = neighbor_depth;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        if (found) {
            depth_map[idx] = found_depth;
            break;
        }
    }
}

extern "C" void cuda_fill_depth_holes(
    unsigned short* d_depth,
    int width, int height,
    int max_radius)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    fillDepthHolesKernel<<<grid, block>>>(d_depth, width, height, max_radius);
    cudaDeviceSynchronize();
}

// 5b. The Kernel for Depth Hole Filling (Averaging)
__global__ void fillDepthHolesAvgKernel(
    unsigned short* depth_map,
    int width, int height,
    int max_radius)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    int idx = v * width + u;
    if (depth_map[idx] != 0) return; // Only process holes

    unsigned int sum = 0;
    unsigned int count = 0;

    for (int radius = 1; radius <= max_radius; ++radius) {
        for (int du = -radius; du <= radius; ++du) {
            for (int dv = -radius; dv <= radius; ++dv) {
                int nu = u + du;
                int nv = v + dv;
                if (nu < 0 || nu >= width || nv < 0 || nv >= height) continue;
                int nidx = nv * width + nu;
                unsigned short neighbor_depth = depth_map[nidx];
                if (neighbor_depth != 0) {
                    sum += neighbor_depth;
                    count++;
                }
            }
        }
        if (count > 0) {
            depth_map[idx] = sum / count;
            break;
        }
    }
}

extern "C" void cuda_fill_depth_holes_avg(
    unsigned short* d_depth,
    int width, int height,
    int max_radius)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    fillDepthHolesAvgKernel<<<grid, block>>>(d_depth, width, height, max_radius);
    cudaDeviceSynchronize();
}

// 6. CUDA Kernel for Blob-Based Hole Filling (Fixed with two-buffer approach)
__global__ void fillDepthBlobsKernel(
    const unsigned short* __restrict__ depth_map_in,
    unsigned short* __restrict__ depth_map_out,
    int width, int height)
{
    // Each thread processes one pixel
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;

    // If pixel is already filled, just copy it
    if (depth_map_in[idx] != 0) {
        depth_map_out[idx] = depth_map_in[idx];
        return;
    }

    // Pixel is a hole (0 value)
    // Check 3x3 neighbors and fill with average of valid neighbors
    unsigned int sum = 0;
    unsigned int count = 0;
    for (int du = -1; du <= 1; ++du) {
        for (int dv = -1; dv <= 1; ++dv) {
            if (du == 0 && dv == 0) continue;
            int nu = u + du;
            int nv = v + dv;
            if (nu < 0 || nu >= width || nv < 0 || nv >= height) continue;
            int nidx = nv * width + nu;
            unsigned short neighbor_depth = depth_map_in[nidx];
            if (neighbor_depth != 0) {
                sum += neighbor_depth;
                count++;
            }
        }
    }

    // If at least one neighbor is valid, fill with average
    if (count > 0) {
        depth_map_out[idx] = sum / count;
    } else {
        // No valid neighbors yet, keep as hole
        depth_map_out[idx] = 0;
    }
}

extern "C" void cuda_fill_depth_blobs(
    unsigned short* d_depth,
    int width, int height,
    int max_iters)
{
    // Allocate temporary buffer
    unsigned short* d_temp;
    cudaMalloc(&d_temp, width * height * sizeof(unsigned short));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Iterative propagation: each iteration fills one layer of holes from the boundary
    for (int iter = 0; iter < max_iters; ++iter) {
        fillDepthBlobsKernel<<<grid, block>>>(d_depth, d_temp, width, height);
        cudaDeviceSynchronize();
        // Copy temp back to depth for next iteration
        cudaMemcpy(d_depth, d_temp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(d_temp);
}

// 6b. CUDA Kernel for Jump Flooding Algorithm (JFA) Blob Filling
__global__ void jumpFloodFillKernel(
    unsigned short* depth_map,
    unsigned short* temp_map,
    int width, int height,
    int jump)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;

    // If already filled, propagate value
    if (depth_map[idx] != 0) {
        temp_map[idx] = depth_map[idx];
        return;
    }

    unsigned short best_depth = 0;
    // Check 8 directions at distance 'jump'
    for (int du = -1; du <= 1; ++du) {
        for (int dv = -1; dv <= 1; ++dv) {
            if (du == 0 && dv == 0) continue;
            int nu = u + du * jump;
            int nv = v + dv * jump;
            if (nu < 0 || nu >= width || nv < 0 || nv >= height) continue;
            int nidx = nv * width + nu;
            unsigned short neighbor_depth = depth_map[nidx];
            if (neighbor_depth != 0) {
                best_depth = neighbor_depth;
                break;
            }
        }
        if (best_depth != 0) break;
    }
    temp_map[idx] = best_depth;
}

extern "C" void cuda_fill_depth_jfa(
    unsigned short* d_depth,
    int width, int height)
{
    // Allocate temporary buffer
    unsigned short* d_temp;
    cudaMalloc(&d_temp, width * height * sizeof(unsigned short));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    int max_jump = max(width, height) / 2;
    for (int jump = max_jump; jump >= 1; jump /= 2) {
        jumpFloodFillKernel<<<grid, block>>>(d_depth, d_temp, width, height, jump);
        cudaDeviceSynchronize();
        // Copy temp_map back to depth_map for next iteration
        cudaMemcpy(d_depth, d_temp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
    }
    cudaFree(d_temp);
}

// 6c. CUDA Kernel for Depth Hole Filling (Median)
__global__ void fillDepthHolesMedianKernel(
    unsigned short* depth_map,
    int width, int height,
    int max_radius)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    int idx = v * width + u;
    if (depth_map[idx] != 0) return; // Only process holes

    const int MAX_NEIGHBORS = 512; // Fixed size for neighbors
    unsigned short neighbors[MAX_NEIGHBORS];

    for (int radius = 1; radius <= max_radius; ++radius) {
        // Collect neighbor depths
        unsigned int count = 0;

        for (int du = -radius; du <= radius && count < MAX_NEIGHBORS; ++du) {
            for (int dv = -radius; dv <= radius && count < MAX_NEIGHBORS; ++dv) {
                int nu = u + du;
                int nv = v + dv;
                if (nu < 0 || nu >= width || nv < 0 || nv >= height) continue;
                int nidx = nv * width + nu;
                unsigned short neighbor_depth = depth_map[nidx];
                if (neighbor_depth != 0) {
                    neighbors[count++] = neighbor_depth;
                }
            }
        }

        if (count > 0) {
            // Simple bubble sort (not efficient, but works for small arrays)
            for (unsigned int i = 0; i < count - 1; ++i) {
                for (unsigned int j = i + 1; j < count; ++j) {
                    if (neighbors[i] > neighbors[j]) {
                        unsigned short tmp = neighbors[i];
                        neighbors[i] = neighbors[j];
                        neighbors[j] = tmp;
                    }
                }
            }
            // Median is middle value (or average of two middle for even count)
            depth_map[idx] = (count % 2 == 1) ? neighbors[count / 2] : (neighbors[count / 2 - 1] + neighbors[count / 2]) / 2;
            break;
        }
    }
}

extern "C" void cuda_fill_depth_holes_median(
    unsigned short* d_depth,
    int width, int height,
    int max_radius)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    fillDepthHolesMedianKernel<<<grid, block>>>(d_depth, width, height, max_radius);
    cudaDeviceSynchronize();
}

// 6d. CUDA Kernel for Depth Hole Filling (Mode)
__global__ void fillDepthHolesModeKernel(
    unsigned short* depth_map,
    int width, int height,
    int max_radius)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    int idx = v * width + u;
    if (depth_map[idx] != 0) return; // Only process holes

    const int MAX_NEIGHBORS = 512; // Fixed size for neighbors
    unsigned short neighbors[MAX_NEIGHBORS];

    for (int radius = 1; radius <= max_radius; ++radius) {
        // Collect neighbor depths
        unsigned int count = 0;

        for (int du = -radius; du <= radius && count < MAX_NEIGHBORS; ++du) {
            for (int dv = -radius; dv <= radius && count < MAX_NEIGHBORS; ++dv) {
                int nu = u + du;
                int nv = v + dv;
                if (nu < 0 || nu >= width || nv < 0 || nv >= height) continue;
                int nidx = nv * width + nu;
                unsigned short neighbor_depth = depth_map[nidx];
                if (neighbor_depth != 0) {
                    neighbors[count++] = neighbor_depth;
                }
            }
        }

        if (count > 0) {
            // Find mode (most frequent value)
            unsigned short mode = neighbors[0];
            unsigned int max_freq = 1;
            
            // Count frequencies (brute force for small arrays)
            for (unsigned int i = 0; i < count; ++i) {
                unsigned int freq = 0;
                for (unsigned int j = 0; j < count; ++j) {
                    if (neighbors[i] == neighbors[j]) freq++;
                }
                if (freq > max_freq) {
                    max_freq = freq;
                    mode = neighbors[i];
                }
            }
            depth_map[idx] = mode;
            break;
        }
    }
}

extern "C" void cuda_fill_depth_holes_mode(
    unsigned short* d_depth,
    int width, int height,
    int max_radius)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    fillDepthHolesModeKernel<<<grid, block>>>(d_depth, width, height, max_radius);
    cudaDeviceSynchronize();
}

// 6e. CUDA Kernel for IP-Basic (In-Place Basic) Inpainting Algorithm
// Uses confidence-based iterative propagation
__global__ void fillDepthIPBasicKernel(
    const unsigned short* __restrict__ depth_in,
    const float* __restrict__ confidence_in,
    unsigned short* __restrict__ depth_out,
    float* __restrict__ confidence_out,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;
    int idx = v * width + u;

    // If already filled with high confidence, keep it
    if (confidence_in[idx] > 0.9f) {
        depth_out[idx] = depth_in[idx];
        confidence_out[idx] = confidence_in[idx];
        return;
    }

    // This is a hole or low-confidence pixel
    // Collect neighbors and compute weighted average
    float weighted_sum = 0.0f;
    float total_weight = 0.0f;
    float max_neighbor_confidence = 0.0f;

    for (int du = -1; du <= 1; ++du) {
        for (int dv = -1; dv <= 1; ++dv) {
            if (du == 0 && dv == 0) continue;
            int nu = u + du;
            int nv = v + dv;
            if (nu < 0 || nu >= width || nv < 0 || nv >= height) continue;
            int nidx = nv * width + nu;
            
            float neighbor_conf = confidence_in[nidx];
            if (neighbor_conf > 0.0f) {
                // Weight by squared confidence (prioritize higher confidence)
                float weight = neighbor_conf * neighbor_conf;
                weighted_sum += depth_in[nidx] * weight;
                total_weight += weight;
                max_neighbor_confidence = fmaxf(max_neighbor_confidence, neighbor_conf);
            }
        }
    }

    if (total_weight > 0.0f) {
        // Compute new depth as weighted average
        depth_out[idx] = (unsigned short)(weighted_sum / total_weight + 0.5f);
        // New confidence is reduced from max neighbor (penalty for inpainting)
        confidence_out[idx] = max_neighbor_confidence * 0.99f;
    } else {
        // No valid neighbors yet
        depth_out[idx] = depth_in[idx];
        confidence_out[idx] = confidence_in[idx];
    }
}

extern "C" void cuda_fill_depth_holes_ip_basic(
    unsigned short* d_depth,
    int width, int height,
    int max_iters)
{
    // Allocate confidence map (1.0 = filled, 0.0 = unfilled)
    float* d_confidence;
    float* d_confidence_temp;
    cudaMalloc(&d_confidence, width * height * sizeof(float));
    cudaMalloc(&d_confidence_temp, width * height * sizeof(float));

    // Allocate temporary depth buffer
    unsigned short* d_depth_temp;
    cudaMalloc(&d_depth_temp, width * height * sizeof(unsigned short));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Initialize confidence to 0 (will build up from valid depth boundaries)
    cudaMemset(d_confidence, 0, width * height * sizeof(float));
    
    // Iteratively fill holes
    for (int iter = 0; iter < max_iters; ++iter) {
        fillDepthIPBasicKernel<<<grid, block>>>(
            d_depth, d_confidence,
            d_depth_temp, d_confidence_temp,
            width, height);
        cudaDeviceSynchronize();

        // Swap buffers
        cudaMemcpy(d_depth, d_depth_temp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_confidence, d_confidence_temp, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_confidence);
    cudaFree(d_confidence_temp);
    cudaFree(d_depth_temp);
}

// 6f. CUDA Kernel for Fast Guided Filter (RGB-guided depth inpainting)
// Uses RGB image to guide depth filling while preserving sharp edges
__global__ void fillDepthGuidedFilterKernel(
    const unsigned short* __restrict__ depth_in,
    const unsigned char* __restrict__ rgb,
    unsigned short* __restrict__ depth_out,
    int width, int height,
    int filter_radius,
    float color_sigma)  // Color difference threshold
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;
    int idx = v * width + u;

    // If already filled, copy it
    if (depth_in[idx] != 0) {
        depth_out[idx] = depth_in[idx];
        return;
    }

    // Get RGB of current pixel
    int rgb_idx = idx * 3;
    unsigned char r_center = rgb[rgb_idx];
    unsigned char g_center = rgb[rgb_idx + 1];
    unsigned char b_center = rgb[rgb_idx + 2];

    // Collect valid neighbors weighted by color similarity
    float weighted_sum = 0.0f;
    float total_weight = 0.0f;

    for (int du = -filter_radius; du <= filter_radius; ++du) {
        for (int dv = -filter_radius; dv <= filter_radius; ++dv) {
            int nu = u + du;
            int nv = v + dv;
            if (nu < 0 || nu >= width || nv < 0 || nv >= height) continue;
            int nidx = nv * width + nu;

            unsigned short neighbor_depth = depth_in[nidx];
            if (neighbor_depth == 0) continue;  // Skip unfilled pixels

            // Get RGB of neighbor
            int n_rgb_idx = nidx * 3;
            unsigned char r_n = rgb[n_rgb_idx];
            unsigned char g_n = rgb[n_rgb_idx + 1];
            unsigned char b_n = rgb[n_rgb_idx + 2];

            // Compute color distance
            float dr = (float)(r_center - r_n);
            float dg = (float)(g_center - g_n);
            float db = (float)(b_center - b_n);
            float color_dist = sqrtf(dr*dr + dg*dg + db*db);

            // Spatial distance
            float spatial_dist = sqrtf((float)(du*du + dv*dv));

            // Weight: Gaussian in both color and spatial domains
            // High weight if color is similar AND spatially close
            float color_weight = expf(-(color_dist * color_dist) / (2.0f * color_sigma * color_sigma));
            float spatial_weight = expf(-(spatial_dist * spatial_dist) / 2.0f);  // sigma=1.0 for spatial
            float weight = color_weight * spatial_weight;

            weighted_sum += neighbor_depth * weight;
            total_weight += weight;
        }
    }

    // Fill with weighted average
    if (total_weight > 0.0f) {
        depth_out[idx] = (unsigned short)(weighted_sum / total_weight + 0.5f);
    } else {
        // No valid neighbors found, keep as hole
        depth_out[idx] = 0;
    }
}

extern "C" void cuda_fill_depth_guided_filter(
    unsigned short* d_depth,
    const unsigned char* d_rgb,
    int width, int height,
    int filter_radius,
    int max_iters,
    float color_sigma)  // Color threshold (0-255 scale, typically 20-50)
{
    // Allocate temporary depth buffer
    unsigned short* d_temp;
    cudaMalloc(&d_temp, width * height * sizeof(unsigned short));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Iteratively fill holes
    for (int iter = 0; iter < max_iters; ++iter) {
        fillDepthGuidedFilterKernel<<<grid, block>>>(
            d_depth, d_rgb, d_temp, width, height, filter_radius, color_sigma);
        cudaDeviceSynchronize();
        
        // Copy temp back to depth for next iteration
        cudaMemcpy(d_depth, d_temp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_temp);
}

// 6g. CUDA Kernel for Maximum Filter (Morphological Dilation)
// Takes the maximum valid depth value in a neighborhood
__global__ void fillDepthMaxFilterKernel(
    const unsigned short* __restrict__ depth_in,
    unsigned short* __restrict__ depth_out,
    int width, int height,
    int filter_radius)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;
    int idx = v * width + u;

    // If already filled, copy it
    if (depth_in[idx] != 0) {
        depth_out[idx] = depth_in[idx];
        return;
    }

    // This is a hole, find the maximum valid depth in neighborhood
    unsigned short max_depth = 0;

    for (int du = -filter_radius; du <= filter_radius; ++du) {
        for (int dv = -filter_radius; dv <= filter_radius; ++dv) {
            int nu = u + du;
            int nv = v + dv;
            if (nu < 0 || nu >= width || nv < 0 || nv >= height) continue;
            int nidx = nv * width + nu;

            unsigned short neighbor_depth = depth_in[nidx];
            if (neighbor_depth != 0) {
                max_depth = max(max_depth, neighbor_depth);
            }
        }
    }

    depth_out[idx] = max_depth;
}

extern "C" void cuda_fill_depth_max_filter(
    unsigned short* d_depth,
    int width, int height,
    int filter_radius,
    int max_iters)
{
    // Allocate temporary depth buffer
    unsigned short* d_temp;
    cudaMalloc(&d_temp, width * height * sizeof(unsigned short));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Iteratively fill holes
    for (int iter = 0; iter < max_iters; ++iter) {
        fillDepthMaxFilterKernel<<<grid, block>>>(
            d_depth, d_temp, width, height, filter_radius);
        cudaDeviceSynchronize();
        
        // Copy temp back to depth for next iteration
        cudaMemcpy(d_depth, d_temp, width * height * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_temp);
}

// 6h. CUDA Kernels for True Guided Filter (He & Sun, 2015)
__global__ void rgbToGuidanceKernel(
    const unsigned char* __restrict__ rgb,
    float* __restrict__ guidance,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;
    int rgb_idx = idx * 3;

    float r = rgb[rgb_idx] / 255.0f;
    float g = rgb[rgb_idx + 1] / 255.0f;
    float b = rgb[rgb_idx + 2] / 255.0f;
    guidance[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
}

__global__ void depthToFloatKernel(
    const unsigned short* __restrict__ depth_in,
    float* __restrict__ depth_out,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;
    depth_out[idx] = (float)depth_in[idx];
}

__global__ void maskFromDepthKernel(
    const unsigned short* __restrict__ depth_in,
    float* __restrict__ mask,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;
    mask[idx] = (depth_in[idx] != 0) ? 1.0f : 0.0f;
}

__global__ void multiplyKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;
    out[idx] = a[idx] * b[idx];
}

__global__ void rowPrefixSumKernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int width, int height)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= height) return;
    int row = v * width;
    float sum = 0.0f;
    for (int u = 0; u < width; ++u) {
        sum += in[row + u];
        out[row + u] = sum;
    }
}

__global__ void colPrefixSumInPlaceKernel(
    float* __restrict__ inout,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= width) return;
    float sum = 0.0f;
    for (int v = 0; v < height; ++v) {
        int idx = v * width + u;
        sum += inout[idx];
        inout[idx] = sum;
    }
}

__device__ inline float readIntegral(
    const float* __restrict__ integral,
    int x, int y,
    int width, int height)
{
    if (x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
    return integral[y * width + x];
}

__global__ void boxSumFromIntegralKernel(
    const float* __restrict__ integral,
    float* __restrict__ out,
    int width, int height,
    int radius)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;

    int x0 = u - radius;
    int y0 = v - radius;
    int x1 = u + radius;
    int y1 = v + radius;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 >= width) x1 = width - 1;
    if (y1 >= height) y1 = height - 1;

    float A = readIntegral(integral, x0 - 1, y0 - 1, width, height);
    float B = readIntegral(integral, x1, y0 - 1, width, height);
    float C = readIntegral(integral, x0 - 1, y1, width, height);
    float D = readIntegral(integral, x1, y1, width, height);

    out[idx] = D - B - C + A;
}

__global__ void computeMaskedMeansKernel(
    const float* __restrict__ sum_M,
    const float* __restrict__ sum_I,
    const float* __restrict__ sum_p,
    const float* __restrict__ sum_I2,
    const float* __restrict__ sum_Ip,
    float* __restrict__ mean_I,
    float* __restrict__ mean_p,
    float* __restrict__ mean_I2,
    float* __restrict__ mean_Ip,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;

    float denom = sum_M[idx];
    if (denom > 1e-6f) {
        float inv = 1.0f / denom;
        mean_I[idx] = sum_I[idx] * inv;
        mean_p[idx] = sum_p[idx] * inv;
        mean_I2[idx] = sum_I2[idx] * inv;
        mean_Ip[idx] = sum_Ip[idx] * inv;
    } else {
        mean_I[idx] = 0.0f;
        mean_p[idx] = 0.0f;
        mean_I2[idx] = 0.0f;
        mean_Ip[idx] = 0.0f;
    }
}

__global__ void computeMeanFromSumKernel(
    const float* __restrict__ sum,
    const float* __restrict__ sum_M,
    float* __restrict__ mean,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;

    float denom = sum_M[idx];
    if (denom > 1e-6f) {
        mean[idx] = sum[idx] / denom;
    } else {
        mean[idx] = 0.0f;
    }
}

__global__ void computeABKernel(
    const float* __restrict__ mean_I,
    const float* __restrict__ mean_p,
    const float* __restrict__ mean_I2,
    const float* __restrict__ mean_Ip,
    float* __restrict__ a,
    float* __restrict__ b,
    int width, int height,
    float eps)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;

    float var_I = mean_I2[idx] - mean_I[idx] * mean_I[idx];
    float cov_Ip = mean_Ip[idx] - mean_I[idx] * mean_p[idx];
    float denom = var_I + eps;

    a[idx] = (denom > 0.0f) ? (cov_Ip / denom) : 0.0f;
    b[idx] = mean_p[idx] - a[idx] * mean_I[idx];
}

__global__ void computeQKernel(
    const float* __restrict__ guidance,
    const float* __restrict__ mean_a,
    const float* __restrict__ mean_b,
    float* __restrict__ q,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;
    q[idx] = mean_a[idx] * guidance[idx] + mean_b[idx];
}

__global__ void applyGuidedKernel(
    const unsigned short* __restrict__ depth_in,
    const float* __restrict__ q,
    unsigned short* __restrict__ depth_out,
    int width, int height)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;
    int idx = v * width + u;

    // Apply guided filter to ALL pixels (not just holes)
    // This enables denoising of existing depth values in addition to hole filling
    float val = q[idx];
    
    // If filtered result is invalid or negative, keep original (if non-zero) or zero
    if (val <= 0.0f) {
        depth_out[idx] = depth_in[idx];
        return;
    }

    // Clamp and convert to unsigned short
    float clamped = fminf(val, 65535.0f);
    depth_out[idx] = (unsigned short)(clamped + 0.5f);
}

static void boxFilterSum(
    const float* d_in,
    float* d_out,
    float* d_temp,
    int width, int height,
    int radius)
{
    dim3 block_row(256, 1);
    dim3 grid_row((height + block_row.x - 1) / block_row.x, 1);
    dim3 block_col(256, 1);
    dim3 grid_col((width + block_col.x - 1) / block_col.x, 1);
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    rowPrefixSumKernel<<<grid_row, block_row>>>(d_in, d_temp, width, height);
    cudaDeviceSynchronize();
    colPrefixSumInPlaceKernel<<<grid_col, block_col>>>(d_temp, width, height);
    cudaDeviceSynchronize();
    boxSumFromIntegralKernel<<<grid, block>>>(d_temp, d_out, width, height, radius);
    cudaDeviceSynchronize();
}

extern "C" void cuda_fill_depth_true_guided(
    unsigned short* d_depth,
    const unsigned char* d_rgb,
    int width, int height,
    int radius,
    float eps)
{
    size_t num_pixels = (size_t)width * (size_t)height;

    float* d_I = nullptr;
    float* d_p = nullptr;
    float* d_M = nullptr;
    float* d_I2 = nullptr;
    float* d_Ip = nullptr;
    float* d_I_M = nullptr;
    float* d_p_M = nullptr;
    float* d_I2_M = nullptr;
    float* d_Ip_M = nullptr;
    float* d_sum_M = nullptr;
    float* d_sum_I = nullptr;
    float* d_sum_p = nullptr;
    float* d_sum_I2 = nullptr;
    float* d_sum_Ip = nullptr;
    float* d_mean_I = nullptr;
    float* d_mean_p = nullptr;
    float* d_mean_I2 = nullptr;
    float* d_mean_Ip = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_a_M = nullptr;
    float* d_b_M = nullptr;
    float* d_sum_a = nullptr;
    float* d_sum_b = nullptr;
    float* d_mean_a = nullptr;
    float* d_mean_b = nullptr;
    float* d_q = nullptr;
    float* d_temp = nullptr;
    unsigned short* d_depth_out = nullptr;

    cudaMalloc(&d_I, num_pixels * sizeof(float));
    cudaMalloc(&d_p, num_pixels * sizeof(float));
    cudaMalloc(&d_M, num_pixels * sizeof(float));
    cudaMalloc(&d_I2, num_pixels * sizeof(float));
    cudaMalloc(&d_Ip, num_pixels * sizeof(float));
    cudaMalloc(&d_I_M, num_pixels * sizeof(float));
    cudaMalloc(&d_p_M, num_pixels * sizeof(float));
    cudaMalloc(&d_I2_M, num_pixels * sizeof(float));
    cudaMalloc(&d_Ip_M, num_pixels * sizeof(float));
    cudaMalloc(&d_sum_M, num_pixels * sizeof(float));
    cudaMalloc(&d_sum_I, num_pixels * sizeof(float));
    cudaMalloc(&d_sum_p, num_pixels * sizeof(float));
    cudaMalloc(&d_sum_I2, num_pixels * sizeof(float));
    cudaMalloc(&d_sum_Ip, num_pixels * sizeof(float));
    cudaMalloc(&d_mean_I, num_pixels * sizeof(float));
    cudaMalloc(&d_mean_p, num_pixels * sizeof(float));
    cudaMalloc(&d_mean_I2, num_pixels * sizeof(float));
    cudaMalloc(&d_mean_Ip, num_pixels * sizeof(float));
    cudaMalloc(&d_a, num_pixels * sizeof(float));
    cudaMalloc(&d_b, num_pixels * sizeof(float));
    cudaMalloc(&d_a_M, num_pixels * sizeof(float));
    cudaMalloc(&d_b_M, num_pixels * sizeof(float));
    cudaMalloc(&d_sum_a, num_pixels * sizeof(float));
    cudaMalloc(&d_sum_b, num_pixels * sizeof(float));
    cudaMalloc(&d_mean_a, num_pixels * sizeof(float));
    cudaMalloc(&d_mean_b, num_pixels * sizeof(float));
    cudaMalloc(&d_q, num_pixels * sizeof(float));
    cudaMalloc(&d_temp, num_pixels * sizeof(float));
    cudaMalloc(&d_depth_out, num_pixels * sizeof(unsigned short));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    rgbToGuidanceKernel<<<grid, block>>>(d_rgb, d_I, width, height);
    depthToFloatKernel<<<grid, block>>>(d_depth, d_p, width, height);
    maskFromDepthKernel<<<grid, block>>>(d_depth, d_M, width, height);
    multiplyKernel<<<grid, block>>>(d_I, d_I, d_I2, width, height);
    multiplyKernel<<<grid, block>>>(d_I, d_p, d_Ip, width, height);
    multiplyKernel<<<grid, block>>>(d_I, d_M, d_I_M, width, height);
    multiplyKernel<<<grid, block>>>(d_p, d_M, d_p_M, width, height);
    multiplyKernel<<<grid, block>>>(d_I2, d_M, d_I2_M, width, height);
    multiplyKernel<<<grid, block>>>(d_Ip, d_M, d_Ip_M, width, height);

    boxFilterSum(d_M, d_sum_M, d_temp, width, height, radius);
    boxFilterSum(d_I_M, d_sum_I, d_temp, width, height, radius);
    boxFilterSum(d_p_M, d_sum_p, d_temp, width, height, radius);
    boxFilterSum(d_I2_M, d_sum_I2, d_temp, width, height, radius);
    boxFilterSum(d_Ip_M, d_sum_Ip, d_temp, width, height, radius);

    computeMaskedMeansKernel<<<grid, block>>>(
        d_sum_M, d_sum_I, d_sum_p, d_sum_I2, d_sum_Ip,
        d_mean_I, d_mean_p, d_mean_I2, d_mean_Ip, width, height);

    computeABKernel<<<grid, block>>>(
        d_mean_I, d_mean_p, d_mean_I2, d_mean_Ip,
        d_a, d_b, width, height, eps);

    multiplyKernel<<<grid, block>>>(d_a, d_M, d_a_M, width, height);
    multiplyKernel<<<grid, block>>>(d_b, d_M, d_b_M, width, height);
    boxFilterSum(d_a_M, d_sum_a, d_temp, width, height, radius);
    boxFilterSum(d_b_M, d_sum_b, d_temp, width, height, radius);
    computeMeanFromSumKernel<<<grid, block>>>(d_sum_a, d_sum_M, d_mean_a, width, height);
    computeMeanFromSumKernel<<<grid, block>>>(d_sum_b, d_sum_M, d_mean_b, width, height);

    computeQKernel<<<grid, block>>>(d_I, d_mean_a, d_mean_b, d_q, width, height);
    applyGuidedKernel<<<grid, block>>>(d_depth, d_q, d_depth_out, width, height);

    cudaMemcpy(d_depth, d_depth_out, num_pixels * sizeof(unsigned short), cudaMemcpyDeviceToDevice);

    cudaFree(d_I);
    cudaFree(d_p);
    cudaFree(d_M);
    cudaFree(d_I2);
    cudaFree(d_Ip);
    cudaFree(d_I_M);
    cudaFree(d_p_M);
    cudaFree(d_I2_M);
    cudaFree(d_Ip_M);
    cudaFree(d_sum_M);
    cudaFree(d_sum_I);
    cudaFree(d_sum_p);
    cudaFree(d_sum_I2);
    cudaFree(d_sum_Ip);
    cudaFree(d_mean_I);
    cudaFree(d_mean_p);
    cudaFree(d_mean_I2);
    cudaFree(d_mean_Ip);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a_M);
    cudaFree(d_b_M);
    cudaFree(d_sum_a);
    cudaFree(d_sum_b);
    cudaFree(d_mean_a);
    cudaFree(d_mean_b);
    cudaFree(d_q);
    cudaFree(d_temp);
    cudaFree(d_depth_out);
}

// 7. The Wrapper Functions (Callable from C++)


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

extern "C" void cuda_reproject_to_image(
    const PointXYZRGB* d_cloud,
    const float4* d_normals,
    float* d_image,
    int width, int height)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    reprojectToImageKernel<<<grid, block>>>(d_cloud, d_normals, d_image, width, height);

    cudaDeviceSynchronize();
}