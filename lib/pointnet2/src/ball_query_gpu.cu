#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"


__global__ void ball_query_kernel(
    const int b,                         // batch size (B)
    const int n,                         // size of parent point set (N)
    const int m,                         // size of child point set (M)
    const float r,                       // ball radius
    const int s,                         // number of samples to draw from each ball (S)
    const float *__restrict__ parent_xyz,// 3D coordinates of parent points (B, N, 3)
    const float *__restrict__ child_xyz, // 3D coordinates of child points (B, M, 3)
    int *__restrict__ idx,               // indices of sampled points for each ball (B, M, S)
    bool *__restrict__ is_valid) {       // a mask for valid indices (B, M, S)

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    child_xyz += (bs_idx * m + pt_idx) * 3;
    parent_xyz += bs_idx * n * 3;
    idx += (bs_idx * m + pt_idx) * s;
    is_valid += (bs_idx * m + pt_idx) * s;

    float3 child_pt = make_float3(child_xyz[0], child_xyz[1], child_xyz[2]);

    float r2 = r * r;
    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float3 parent_pt = make_float3(
            parent_xyz[k * 3], parent_xyz[k * 3 + 1], parent_xyz[k * 3 + 2]);
        float3 v = child_pt - parent_pt;
        float d2 = dot(v, v);
        if (d2 < r2) {
            idx[cnt] = k;
            if (cnt == 0) {
                for (int l = 1; l < s; ++l) {
                    idx[l] = k;
                }
            }
            is_valid[cnt] = true;
            ++cnt;
            if (cnt >= s) break;
        }
    }
}


void ball_query_kernel_wrapper(
    const int b, 
    const int n, 
    const int m, 
    const float r, 
    const int s, 
    const float *parent_xyz, 
    const float *child_xyz, 
    int *idx,
    bool *is_valid) {

    unsigned int n_threads = opt_n_threads(m);
    dim3 blocks(DIVUP(m, n_threads), b);
    dim3 threads(n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ball_query_kernel<<<blocks, threads, 0, stream>>>(
        b, n, m, r, s, parent_xyz, child_xyz, idx, is_valid);
    
    CUDA_CHECK_ERRORS();
}