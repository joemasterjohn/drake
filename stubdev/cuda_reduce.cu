#include <stdio.h>

#include <iostream>

#include "cuda_reduce.h"`

static void HandleError(cudaError_t err, const char* file, int line) {
  // CUDA error handeling from the "CUDA by example" book
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// ==========================================================================
// TODO: Using only one thread for now, needs optimization

// Device function to perform the reduction operation
__device__ void ReduceByProblemFunc(double* d_vec_in, double* d_vec_out,
                                    int equ_idx, int items_per_equation) {
  d_vec_out[equ_idx] = 0.0;

  for (int i = 0; i < items_per_equation; i++) {
    d_vec_out[equ_idx] += d_vec_in[equ_idx * items_per_equation + i];
  }
}

// Kernel function serving as a wrapper
__global__ void ReduceByProblemKernel(double* d_vec_in, double* d_vec_out,
                                      int num_problems,
                                      int items_per_equation) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_problems) {
    return;
  }

  // Use only one thread per equation for now
  if (thread_idx == 0) {
    // Call the device function
    ReduceByProblemFunc(d_vec_in, d_vec_out, equ_idx, items_per_equation);
  }
}

void reduce_by_problem(std::vector<double>& vec_in,
                       std::vector<double>& vec_out, int num_problems,
                       int items_per_equation) {
  // Allocate memory on the device
  double* d_vec_in;
  double* d_vec_out;
  size_t size_vec_in = num_problems * items_per_equation * sizeof(double);
  size_t size_vec_out = num_problems * sizeof(double);

  HANDLE_ERROR(cudaMalloc((void**)&d_vec_in, size_vec_in));
  HANDLE_ERROR(cudaMalloc((void**)&d_vec_out, size_vec_out));

  // Copy data to device
  HANDLE_ERROR(
      cudaMemcpy(d_vec_in, vec_in.data(), size_vec_in, cudaMemcpyHostToDevice));

  // Launch kernel
  int stride = (items_per_equation + 32 - 1) / 32;
  ReduceByProblemKernel<<<num_problems, 32>>>(d_vec_in, d_vec_out, num_problems,
                                              items_per_equation);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Copy data back to host
  HANDLE_ERROR(cudaMemcpy(vec_out.data(), d_vec_out, size_vec_out,
                          cudaMemcpyDeviceToHost));
}