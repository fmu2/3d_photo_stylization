#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#define TOTAL_THREADS 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return std::max(std::min(1 << pow_2, TOTAL_THREADS), 1);
}

#define CUDA_CHECK_ERRORS() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (cudaSuccess != err) { \
            fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n", \
                cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
                __FILE__); \
            exit(-1); \
        } \
    } while (0)

#endif