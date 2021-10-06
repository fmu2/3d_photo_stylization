#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"


__global__ void gather_points_kernel(
    const int b,                         // batch size (B)
    const int c,                         // feature size (C)
    const int n,                         // size of parent point set (N)
    const int m,                         // size of child point set (M)
    const float *__restrict__ points,    // parent point features (B, C, N)
    const int *__restrict__ idx,         // indices of points selected as children (B, M)
    float *__restrict__ out) {           // child point features (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    out += (bs_idx * c + c_idx) * m + pt_idx;
    points += (bs_idx * c + c_idx) * n;
    idx += bs_idx * m + pt_idx;
    
    out[0] = points[idx[0]];
}

void gather_points_kernel_wrapper(
    const int b, 
    const int c, 
    const int n, 
    const int m, 
    const float *points, 
    const int *idx, 
    float *out) {

    unsigned int n_threads = opt_n_threads(m);
    dim3 blocks(DIVUP(m, n_threads), c, b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_points_kernel<<<blocks, threads, 0, stream>>>(
        b, c, n, m, points, idx, out);

    CUDA_CHECK_ERRORS();
}


__global__ void gather_points_grad_kernel(
    const int b,                         // batch size (B)
    const int c,                         // feature size (C)
    const int n,                         // size of parent point set (N)
    const int m,                         // size of child point set (M)
    const float *__restrict__ grad_out,  // output gradient (B, C, M)
    const int *__restrict__ idx,         // indices of points selected as children (B, M)
    float *__restrict__ grad_points) {   // input gradient (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    grad_out += (bs_idx * c + c_idx) * m + pt_idx;
    grad_points += (bs_idx * c + c_idx) * n;
    idx += bs_idx * m + pt_idx;

    atomicAdd(grad_points + idx[0], grad_out[0]);
}


void gather_points_grad_kernel_wrapper(
    const int b,
    const int c,
    const int n,
    const int m,
    const float *grad_out,
    const int *idx,
    float *grad_points) {

    unsigned int n_threads = opt_n_threads(m);
    dim3 blocks(DIVUP(m, n_threads), c, b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_points_grad_kernel<<<blocks, threads, 0, stream>>>(
        b, c, n, m, grad_out, idx, grad_points);

    CUDA_CHECK_ERRORS();
}


__device__ void __update(
    float *__restrict__ dists, 
    int *__restrict__ dists_i, 
    int idx1, 
    int idx2) {

    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}


template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    const int b,                         // batch size (B)
    const int n,                         // size of parent point set (N)
    const int m,                         // size of child point set (M)
    const float *__restrict__ xyz,       // 3D coordinates of parent points (B, N, 3)
    float *__restrict__ temp,            // shortest distance to the sampled point set (B, N)
    int *__restrict__ idx) {             // indices of sampled points (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    xyz += blockIdx.x * n * 3;
    temp += blockIdx.x * n;
    idx += blockIdx.x * m;

    int tid = threadIdx.x;
    int old = 0;    // index of the latest sampled point
    if (tid == 0) idx[0] = old;

    __syncthreads();
    
    for (int j = 1; j < m; ++j) {
        // latest sampled point
        float3 pt1 = make_float3(
            xyz[old * 3], xyz[old * 3 + 1], xyz[old * 3 + 2]);

        int besti = 0;
        float best = -1;
        for (int k = tid; k < n; k += block_size) {
            // candidate point
            float3 pt2 = make_float3(
                xyz[k * 3], xyz[k * 3 + 1], xyz[k * 3 + 2]);
            float3 v = pt2 - pt1;
            float d2 = min(dot(v, v), temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        
        __syncthreads();

        // identify the farthest point to all points in the sampled subset
        if (block_size >= 1024) {
            if (tid < 512) __update(dists, dists_i, tid, tid + 512);
            __syncthreads();
        }
        if (block_size >= 512) {
            if (tid < 256) __update(dists, dists_i, tid, tid + 256);
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) __update(dists, dists_i, tid, tid + 128);
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) __update(dists, dists_i, tid, tid + 64);
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) __update(dists, dists_i, tid, tid + 32);
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) __update(dists, dists_i, tid, tid + 16);
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) __update(dists, dists_i, tid, tid + 8);
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) __update(dists, dists_i, tid, tid + 4);
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) __update(dists, dists_i, tid, tid + 2);
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) __update(dists, dists_i, tid, tid + 1);
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0) idx[j] = old;
    }
}


void furthest_point_sampling_kernel_wrapper(
    const int b, 
    const int n, 
    const int m, 
    const float *xyz, 
    float *temp, 
    int *idx) {

    unsigned int n_threads = opt_n_threads(n);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (n_threads) {
        case 1024:
            furthest_point_sampling_kernel<1024><<<b, 1024, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 512:
            furthest_point_sampling_kernel<512><<<b, 512, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 256:
            furthest_point_sampling_kernel<256><<<b, 256, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 128:
            furthest_point_sampling_kernel<128><<<b, 128, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 64:
            furthest_point_sampling_kernel<64><<<b, 64, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 32:
            furthest_point_sampling_kernel<32><<<b, 32, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 16:
            furthest_point_sampling_kernel<16><<<b, 16, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 8:
            furthest_point_sampling_kernel<8><<<b, 8, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 4:
            furthest_point_sampling_kernel<4><<<b, 4, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 2:
            furthest_point_sampling_kernel<2><<<b, 2, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        case 1:
            furthest_point_sampling_kernel<1><<<b, 1, 0, stream>>>(
                b, n, m, xyz, temp, idx); break;
        default:
            furthest_point_sampling_kernel<512><<<b, 512, 0, stream>>>(
                b, n, m, xyz, temp, idx);
    }

    CUDA_CHECK_ERRORS();
}