#include <stdio.h>

#include <iostream>

#include "cuda_gauss_jordan.h"

static void HandleError(cudaError_t err, const char* file, int line) {
  // CUDA error handeling from the "CUDA by example" book
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void gauss_jordan_inverse(double* M, double* inv,
                                     size_t num_equations, size_t n,
                                     size_t offset) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_equations) {
    return;
  }

  Eigen::Map<Eigen::MatrixXd> d_M(M + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::MatrixXd> d_inv(inv + equ_idx * n * n, n, n);

  size_t row = thread_idx + offset;

  if (row >= n || row == 0) {
    return;
  }

  for (size_t i = 0; i < 32; i++) {
    size_t pivot = i + offset;
    double mult = d_M(pivot, pivot) / d_M(row, pivot);
    for (size_t j = pivot; j < n; j++) {
      if (row > pivot) {
        d_M(row, j) = d_M(pivot, pivot) - mult * d_M(row, j);
        d_inv(row, j) = d_inv(pivot, pivot) - mult * d_inv(row, j);
      }
      __syncwarp();
    }
    __syncwarp();
  }

  // for (i = matrix_size − 1; i > 0; i−−) {
  //   for (y = 0; y < i; y++) {
  //     mult = matrix[i][i] / matrix[y][i];
  //     for (x = 0; x < i + 1; x++)
  //       13 matrix[y][x] = matrix[i][i] − mult ∗ matrix[y][x];
  //   }
}

double gauss_jordan_solve(std::vector<Eigen::MatrixXd>& M,
                          std::vector<Eigen::MatrixXd>& I) {
  const int num_equations = M.size();
  const int n = M[0].rows();

  // Allocate device arrays
  double *d_M, *d_inv;
  HANDLE_ERROR(
      cudaMalloc((void**)&d_M, sizeof(double) * num_equations * n * n));
  HANDLE_ERROR(
      cudaMalloc((void**)&d_inv, sizeof(double) * num_equations * n * n));

  // Copy to device
  for (int i = 0; i < num_equations; ++i) {
    HANDLE_ERROR(cudaMemcpy(d_M + i * n * n, M[i].data(),
                            sizeof(double) * n * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_inv + i * n * n, I[i].data(),
                            sizeof(double) * n * n, cudaMemcpyHostToDevice));
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  int offset = 0;
  while (offset < n) {
    gauss_jordan_inverse<<<num_equations, 32>>>(d_M, d_inv, num_equations, n,
                                                offset);
    offset += 32;

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for Gauss Jordan Inverse: " << milliseconds
            << " ms\n";

  // Copy to host
  // HANDLE_ERROR(cudaMemcpy(M.data, d_M, sizeof(double) * num_equations * n *
  // n,
  //                         cudaMemcpyDeviceToHost));  // just for testing
  for (int i = 0; i < num_equations; i++) {
    HANDLE_ERROR(cudaMemcpy(I[i].data(), d_inv + i * n * n,
                            sizeof(double) * num_equations * n * n,
                            cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(M[i].data(), d_M + i * n * n,
                            sizeof(double) * num_equations * n * n,
                            cudaMemcpyDeviceToHost));
  }

  for (int i = 0; i < num_equations; ++i) {
    std::cout << M[i] << std::endl;
    // std::cout << "inverse err: " << (M[i].inverse() - x_result_i).norm()
    //           << std::endl;
  }

  return 0;
}
