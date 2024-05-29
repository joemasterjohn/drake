#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

float* CudaAllocate(int64_t size)
{
    float* result;
    cudaMalloc(&result, sizeof(float) * size);
    CudaCheckErrors("Failed to allocate device occupancy grid");
    return result;
}
