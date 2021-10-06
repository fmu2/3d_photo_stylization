#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"


__global__ void three_nn_kernel(
    const int b,                         // batch size (B)
    const int n,                         // size of parent point set (N)
    const int m,                         // size of child point set (M)
    const float *__restrict__ parent_xyz,// 3D coordinates of parent points (B, N, 3)
    const float *__restrict__ child_xyz, // 3D coordinates of child points (B, M, 3)
    float *__restrict__ dist,           // distances to three nearest child points (B, N, 3)
    int *__restrict__ idx) {             // indices of the three nearest child points (B, N, 3)
    
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= n) return;

    parent_xyz += (bs_idx * n + pt_idx) * 3;
    dist += (bs_idx * n + pt_idx) * 3;
    idx += (bs_idx * n + pt_idx) * 3;
    child_xyz += bs_idx * m * 3;

    float3 parent_pt = make_float3(
        parent_xyz[0], parent_xyz[1], parent_xyz[2]);

    float best1 = 1e6, best2 = 1e6, best3 = 1e6;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
        float3 child_pt = make_float3(
            child_xyz[k * 3], child_xyz[k * 3 + 1], child_xyz[k * 3 + 2]);
        float3 v = parent_pt - child_pt;
        float d = length(v);
        if (d < best1) {
            best3 = best2; besti3 = besti2;
            best2 = best1; besti2 = besti1;
            best1 = d; besti1 = k;
        } 
        else if (d < best2) {
            best3 = best2; besti3 = besti2;
            best2 = d; besti2 = k;
        } 
        else if (d < best3) {
            best3 = d; besti3 = k;
        }
    }
    dist[0] = best1; dist[1] = best2; dist[2] = best3;
    idx[0] = besti1; idx[1] = besti2; idx[2] = besti3;
}


void three_nn_kernel_wrapper(
    const int b, 
    const int n, 
    const int m, 
    const float *parent_xyz, 
    const float *child_xyz, 
    float *dist, 
    int *idx) {

    unsigned int n_threads = opt_n_threads(n);
    dim3 blocks(DIVUP(n, n_threads), b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    three_nn_kernel<<<blocks, threads, 0, stream>>>(
        b, n, m, parent_xyz, child_xyz, dist, idx);

    CUDA_CHECK_ERRORS();
}


__global__ void three_interpolate_kernel(
    const int b,                         // batch size (B)
    const int c,                         // feature size (C)
    const int n,                         // size of parent point set (N)
    const int m,                         // size of child point set (M)
    const float *__restrict__ points,    // child point features (B, C, M)
    const int *__restrict__ idx,         // indices of the three nearest child points (B, N, 3)
    const float *__restrict__ weight,    // interpolation weights (B, N, 3)
    float *__restrict__ out) {           // parent point features (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

    idx += (bs_idx * n + pt_idx) * 3;
    weight += (bs_idx * n + pt_idx) * 3;
    points += (bs_idx * c + c_idx) * m;
    out += (bs_idx * c + c_idx) * n;

    out[pt_idx] = weight[0] * points[idx[0]] \
                + weight[1] * points[idx[1]] \
                + weight[2] * points[idx[2]];
}


void three_interpolate_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const float *points, 
    const int *idx, 
    const float *weight, 
    float *out) {

    unsigned int n_threads = opt_n_threads(n);
    dim3 blocks(DIVUP(n, n_threads), c, b);
    dim3 threads(n_threads);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    three_interpolate_kernel<<<blocks, threads, 0, stream>>>(
        b, c, n, m, points, idx, weight, out);

    CUDA_CHECK_ERRORS();
}


__global__ void three_interpolate_grad_kernel(
    const int b,                         // batch size (B)
    const int c,                         // feature size (C)
    const int n,                         // size of parent point set (N)
    const int m,                         // size of child point set (M)
    const float *__restrict__ grad_out,  // output gradient (B, C, N)
    const int *__restrict__ idx,         // indices of the three nearest child points (B, N, 3)
    const float *__restrict__ weight,    // interpolation weights (B, N, 3)
    float *__restrict__ grad_points) {   // input gradient (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;
    
    idx += (bs_idx * n + pt_idx) * 3;
    weight += (bs_idx * n + pt_idx) * 3;
    grad_out += (bs_idx * c + c_idx) * n + pt_idx;
    grad_points += (bs_idx * c + c_idx) * m;

    atomicAdd(grad_points + idx[0], weight[0] * grad_out[0]);
    atomicAdd(grad_points + idx[1], weight[1] * grad_out[0]);
    atomicAdd(grad_points + idx[2], weight[2] * grad_out[0]);
}


void three_interpolate_grad_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const float *grad_out, 
    const int *idx, 
    const float *weight, 
    float *grad_points) {

    unsigned int n_threads = opt_n_threads(n);
    dim3 blocks(DIVUP(n, n_threads), c, b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    three_interpolate_grad_kernel<<<blocks, threads, 0, stream>>>(
        b, c, n, m, grad_out, idx, weight, grad_points);

    CUDA_CHECK_ERRORS();
}