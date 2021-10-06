#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"


__global__ void group_points_kernel(
    const int b,                         // batch size (B)
    const int c,                         // feature size (C)
    const int n,                         // size of input point set (N)
    const int m,                         // size of new point set (M)
    const int s,                         // number of samples to draw from each ball (S)
    const float *__restrict__ points,    // parent point features (B, C, N)
    const int *__restrict__ idx,         // indices of sampled points for each ball (B, M, S)
    float *__restrict__ out) {           // grouped parent point features (B, C, M, S)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ctr_idx = pt_idx / s;
    
    if (bs_idx >= b || c_idx >= c || ctr_idx >= m) return;
    
    int sample_idx = pt_idx % s;
    idx += (bs_idx * m + ctr_idx) * s + sample_idx;
    int in_idx = (bs_idx * c + c_idx) * n + idx[0];
    int out_idx = ((bs_idx * c + c_idx) * m + ctr_idx) * s + sample_idx;

    out[out_idx] = points[in_idx];
}


void group_points_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const int s, 
    const float *points, 
    const int *idx, 
    float *out) {
    
    unsigned int n_threads = opt_n_threads(m * s);
    dim3 blocks(DIVUP(m * s, n_threads), c, b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    group_points_kernel<<<blocks, threads, 0, stream>>>(
        b, c, n, m, s, points, idx, out);
    
    CUDA_CHECK_ERRORS();
}


__global__ void group_points_grad_kernel(
    const int b,                         // batch size (B)
    const int c,                         // feature size (C)
    const int n,                         // size of input point set (N)
    const int m,                         // size of new point set (M)
    const int s,                         // number of samples to draw from each ball (S)
    const float *__restrict__ grad_out,  // output gradient (B, C, M, S)
    const int *__restrict__ idx,         // indices of sampled points for each ball (B, M, S)
    float *__restrict__ grad_points) {   // input gradient (B, C, N)
    
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ctr_idx = pt_idx / s;
    
    if (bs_idx >= b || c_idx >= c || ctr_idx >= m) return;
    
    int sample_idx = pt_idx % s;
    idx += (bs_idx * m + ctr_idx) * s + sample_idx;
    grad_out += ((bs_idx * c + c_idx) * m + ctr_idx) * s + sample_idx;
    grad_points += (bs_idx * c + c_idx) * n;
    
    atomicAdd(grad_points + idx[0], grad_out[0]);
}


void group_points_grad_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const int s, 
    const float *grad_out, 
    const int *idx, 
    float *grad_points) {
    
    unsigned int n_threads = opt_n_threads(m * s);
    dim3 blocks(DIVUP(m * s, n_threads), c, b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    group_points_grad_kernel<<<blocks, threads, 0, stream>>>(
        b, c, n, m, s, grad_out, idx, grad_points);

    CUDA_CHECK_ERRORS();
}