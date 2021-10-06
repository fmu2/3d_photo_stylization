#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"


__global__ void rasterize_kernel(
    const int b,                            // batch size (B)
    const int n,                            // number of points per sample (N)
    const int h,                            // image height (H)
    const int w,                            // image width (W)
    const float *__restrict__ xyz,          // NDC coordinates (B, N, 3)
    float *z_buffer) {                      // z-buffer (B, H, W)

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= n) return;

    xyz += (bs_idx * n + pt_idx) * 3;
    if ((xyz[0] < -1) | (xyz[0] > 1) | (xyz[1] < -1) | (xyz[1] > 1)) return;

    z_buffer += bs_idx * h * w;

    float u = (xyz[0] * 0.5 + 0.5) * w;
    float v = (xyz[1] * 0.5 + 0.5) * h;

    // indices of neighboring pixels
    int2 ul = make_int2((int) (floor(u)), (int) (floor(v)));
    int2 ur = make_int2(ul.x + 1, ul.y);
    int2 ll = make_int2(ul.x, ul.y + 1);
    int2 lr = make_int2(ul.x + 1, ul.y + 1);

    // interpolation weights
    float ul_wgt = ((float) (lr.x) - u) * ((float) (lr.y) - v);
    float ur_wgt = (u - (float) (ll.x)) * ((float) (ll.y) - v);
    float ll_wgt = ((float) (ur.x) - u) * (v - (float) (ur.y));
    float lr_wgt = (u - (float) (ul.x)) * (v - (float) (ul.y));

    // update z-buffer
    if ((ul_wgt >= ur_wgt) & (ul_wgt >= ll_wgt) & (ul_wgt >= lr_wgt)) {
        if ((ul.x >= 0) & (ul.x < w) & (ul.y >= 0) & (ul.y < h))
            atomicMin(z_buffer + ul.y * w + ul.x, xyz[2]);
    }
    else if ((ur_wgt >= ul_wgt) & (ur_wgt >= ll_wgt) & (ur_wgt >= lr_wgt)) {
        if ((ur.x >= 0) & (ur.x < w) & (ur.y >= 0) & (ur.y < h))
            atomicMin(z_buffer + ur.y * w + ur.x, xyz[2]);
    }
    else if ((ll_wgt >= ul_wgt) & (ll_wgt >= ur_wgt) & (ll_wgt >= lr_wgt)) {
        if ((ll.x >= 0) & (ll.x < w) & (ll.y >= 0) & (ll.y < h))
            atomicMin(z_buffer + ll.y * w + ll.x, xyz[2]);
    }
    else {
        if ((lr.x >= 0) & (lr.x < w) & (lr.y >= 0) & (lr.y < h))
            atomicMin(z_buffer + lr.y * w + lr.x, xyz[2]);
    }
}


void rasterize_kernel_wrapper(
    const int b, 
    const int n, 
    const int h, 
    const int w, 
    const float *xyz, 
    float *z_buffer) {

    unsigned int n_threads = opt_n_threads(n);
    dim3 blocks(DIVUP(n, n_threads), b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rasterize_kernel<<<blocks, threads, 0, stream>>>(
        b, n, h, w, xyz, z_buffer);

    CUDA_CHECK_ERRORS();
}


__global__ void refine_z_buffer_kernel(
    const int b,                            // batch size (B)
    const int h,                            // image height (H)
    const int w,                            // image width (W)
    float *z_buffer) {                      // z-buffer (B, H, W)

    int bs_idx = blockIdx.y;
    int px_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || px_idx >= h * w) return;

    int u = px_idx % w;
    int v = px_idx / w;
    z_buffer += bs_idx * h * w;

    int count = 0;
    float sum = 0.0;

    int opposing_x[] = {1, 0, 1, 1};
    int opposing_y[] = {0, 1, 1, -1};

    for (int i = 0; i < 4; i += 1) {
        int2 uv1 = make_int2(u + opposing_x[i], v + opposing_y[i]);
        int2 uv2 = make_int2(u - opposing_x[i], v - opposing_y[i]);

        if ((uv1.x < 0) | (uv1.x >= w) | (uv1.y < 0) | (uv1.y >= h)) continue;
        if ((uv2.x < 0) | (uv2.x >= w) | (uv2.y < 0) | (uv2.y >= h)) continue;

        // detect shine-throughs
        int idx1 = uv1.y * w + uv1.x;
        int idx2 = uv2.y * w + uv2.x; 
        if ((z_buffer[px_idx] > z_buffer[idx1] + 0.01) 
                & (z_buffer[px_idx] > z_buffer[idx2] + 0.01)) {
            count += 2;
            sum += z_buffer[idx1];
            sum += z_buffer[idx2];
        }

    // replace shine-through pixels with their neighbors' mean 
    if (count > 0) z_buffer[px_idx] = sum / count;
    }
}


void refine_z_buffer_kernel_wrapper(
    const int b, 
    const int h, 
    const int w, 
    float *z_buffer) {

    unsigned int n_threads = opt_n_threads(h * w);
    dim3 blocks(DIVUP(h * w, n_threads), b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    refine_z_buffer_kernel<<<blocks, threads, 0, stream>>>(
        b, h, w, z_buffer);

    CUDA_CHECK_ERRORS();
}


__global__ void splat_kernel(
    const int b,                            // batch size (B)
    const int c,                            // data dimension (C)
    const int n,                            // size of point set (N)
    const int h,                            // image height (H)
    const int w,                            // image width (W)
    const float *__restrict__ xyz,          // NDC coordinates (B, N, 3)
    const float *__restrict__ z_buffer,     // z-buffer (B, H, W)
    const float *__restrict__ data,         // point features (B, C, N)
    float *out,                             // rendered output (B, C, H, W)
    bool *is_visible) {                     // mask for visible points (B, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

    xyz += (bs_idx * n + pt_idx) * 3;
    if ((xyz[0] < -1) | (xyz[0] > 1) | (xyz[1] < -1) | (xyz[1] > 1)) return;
    is_visible += bs_idx * n + pt_idx;

    z_buffer += bs_idx * h * w;
    data += (bs_idx * c + c_idx) * n;
    out += (bs_idx * c + c_idx) * h * w;

    float u = (xyz[0] * 0.5 + 0.5) * w;
    float v = (xyz[1] * 0.5 + 0.5) * h;

    // indices of neighboring pixels
    int2 ul = make_int2((int) (floor(u)), (int) (floor(v)));
    int2 ur = make_int2(ul.x + 1, ul.y);
    int2 ll = make_int2(ul.x, ul.y + 1);
    int2 lr = make_int2(ul.x + 1, ul.y + 1);

    // interpolation weights
    float ul_wgt = ((float) (lr.x) - u) * ((float) (lr.y) - v);
    float ur_wgt = (u - (float) (ll.x)) * ((float) (ll.y) - v);
    float ll_wgt = ((float) (ur.x) - u) * (v - (float) (ur.y));
    float lr_wgt = (u - (float) (ul.x)) * (v - (float) (ul.y));

    // splat data
    if ((ul.x >= 0) & (ul.x < w) & (ul.y >= 0) & (ul.y < h)) {
        int ul_idx = ul.y * w + ul.x;
        if (xyz[2] < z_buffer[ul_idx] + 0.01) {
            atomicAdd(out + ul_idx, ul_wgt * data[pt_idx]);
            is_visible[0] = true;
        }
    }
    if ((ur.x >= 0) & (ur.x < w) & (ur.y >= 0) & (ur.y < h)) {
        int ur_idx = ur.y * w + ur.x;
        if (xyz[2] < z_buffer[ur_idx] + 0.01) {
            atomicAdd(out + ur_idx, ur_wgt * data[pt_idx]);
            is_visible[0] = true;
        }
    }
    if ((ll.x >= 0) & (ll.x < w) & (ll.y >= 0) & (ll.y < h)) {
        int ll_idx = ll.y * w + ll.x;
        if (xyz[2] < z_buffer[ll_idx] + 0.01) {
            atomicAdd(out + ll_idx, ll_wgt * data[pt_idx]);
            is_visible[0] = true;
        }
    }
    if ((lr.x >= 0) & (lr.x < w) & (lr.y >= 0) & (lr.y < h)) {
        int lr_idx = lr.y * w + lr.x;
        if (xyz[2] < z_buffer[lr_idx] + 0.01) {
            atomicAdd(out + lr_idx, lr_wgt * data[pt_idx]);
            is_visible[0] = true;
        }
    }
}


void splat_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int h, 
    const int w, 
    const float *xyz,  
    const float *z_buffer, 
    const float *data, 
    float *out,
    bool *is_visible) {

    unsigned int n_threads = opt_n_threads(n);
    dim3 blocks(DIVUP(n, n_threads), c, b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    splat_kernel<<<blocks, threads, 0, stream>>>(
        b, c, n, h, w, xyz, z_buffer, data, out, is_visible);

    CUDA_CHECK_ERRORS();
}


__global__ void splat_grad_kernel(
    const int b,                            // batch size (B)
    const int c,                            // data dimension (C)
    const int n,                            // number of points per sample (N)
    const int h,                            // image height (H)
    const int w,                            // image width (W)
    const float *__restrict__ xyz,          // NDC coordinates (B, N, 3)
    const float *__restrict__ z_buffer,     // z-buffer (B, H, W)
    const float *__restrict__ grad_out,     // output gradients (B, C, H, W)
    float *grad_data) {                     // data gradients (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

    xyz += (bs_idx * n + pt_idx) * 3;
    if ((xyz[0] < -1) | (xyz[0] > 1) | (xyz[1] < -1) | (xyz[1] > 1)) return;

    z_buffer += bs_idx * h * w;
    grad_out += (bs_idx * c + c_idx) * h * w;
    grad_data += (bs_idx * c + c_idx) * n;

    float u = (xyz[0] * 0.5 + 0.5) * w;
    float v = (xyz[1] * 0.5 + 0.5) * h;

    // indices of neighboring pixels
    int2 ul = make_int2((int) (floor(u)), (int) (floor(v)));
    int2 ur = make_int2(ul.x + 1, ul.y);
    int2 ll = make_int2(ul.x, ul.y + 1);
    int2 lr = make_int2(ul.x + 1, ul.y + 1);

    // interpolation weights
    float ul_wgt = ((float) (lr.x) - u) * ((float) (lr.y) - v);
    float ur_wgt = (u - (float) (ll.x)) * ((float) (ll.y) - v);
    float ll_wgt = ((float) (ur.x) - u) * (v - (float) (ur.y));
    float lr_wgt = (u - (float) (ul.x)) * (v - (float) (ul.y));

    // collect gradients
    if ((ul.x >= 0) & (ul.x < w) & (ul.y >= 0) & (ul.y < h)) {
        int ul_idx = ul.y * w + ul.x;
        if (xyz[2] < z_buffer[ul_idx] + 0.01)
            atomicAdd(grad_data + pt_idx, ul_wgt * grad_out[ul_idx]);
    }
    if ((ur.x >= 0) & (ur.x < w) & (ur.y >= 0) & (ur.y < h)) {
        int ur_idx = ur.y * w + ur.x;
        if (xyz[2] < z_buffer[ur_idx] + 0.01)
            atomicAdd(grad_data + pt_idx, ur_wgt * grad_out[ur_idx]);
    }
    if ((ll.x >= 0) & (ll.x < w) & (ll.y >= 0) & (ll.y < h)) {
        int ll_idx = ll.y * w + ll.x;
        if (xyz[2] < z_buffer[ll_idx] + 0.01)
            atomicAdd(grad_data + pt_idx, ll_wgt * grad_out[ll_idx]);
    }
    if ((lr.x >= 0) & (lr.x < w) & (lr.y >= 0) & (lr.y < h)) {
        int lr_idx = lr.y * w + lr.x;
        if (xyz[2] < z_buffer[lr_idx] + 0.01)
            atomicAdd(grad_data + pt_idx, lr_wgt * grad_out[lr_idx]);
    }
}


void splat_grad_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int h, 
    const int w, 
    const float *xyz, 
    const float *z_buffer, 
    const float *grad_out, 
    float *grad_data) {

    unsigned int n_threads = opt_n_threads(n);
    dim3 blocks(DIVUP(n, n_threads), c, b);
    dim3 threads(n_threads);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    splat_grad_kernel<<<blocks, threads, 0, stream>>>(
        b, c, n, h, w, xyz, z_buffer, grad_out, grad_data);

    CUDA_CHECK_ERRORS();
}