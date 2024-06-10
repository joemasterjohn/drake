#include <stdio.h>

#include <iostream>

#include "cuda_reduce.h"

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
__global__ void reduce_by_problem_kernel(double* d_vec_in, double* d_vec_out,
                                         int num_equations,
                                         int items_per_equation) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_equations) {
    return;
  }

  if (thread_idx == 0) {
    d_vec_out[equ_idx] = 0.0;

    for (int i = 0; i < items_per_equation; i++) {
      d_vec_out[equ_idx] += d_vec_in[equ_idx * items_per_equation + i];
    }
  }
}

void reduce_by_problem(std::vector<double>& vec_in,
                       std::vector<double>& vec_out, int num_equations,
                       int items_per_equation) {
  // Allocate memory on the device
  double* d_vec_in;
  double* d_vec_out;
  size_t size_vec_in = num_equations * items_per_equation * sizeof(double);
  size_t size_vec_out = num_equations * sizeof(double);

  HANDLE_ERROR(cudaMalloc((void**)&d_vec_in, size_vec_in));
  HANDLE_ERROR(cudaMalloc((void**)&d_vec_out, size_vec_out));

  // Copy data to device
  HANDLE_ERROR(
      cudaMemcpy(d_vec_in, vec_in.data(), size_vec_in, cudaMemcpyHostToDevice));

  // Launch kernel
  int stride = (items_per_equation + 32 - 1) / 32;
  reduce_by_problem_kernel<<<num_equations, 32>>>(
      d_vec_in, d_vec_out, num_equations, items_per_equation);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Copy data back to host
  HANDLE_ERROR(cudaMemcpy(vec_out.data(), d_vec_out, size_vec_out,
                          cudaMemcpyDeviceToHost));
}