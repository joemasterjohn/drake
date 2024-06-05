#include <stdio.h>

#include <iostream>

#include "cuda_cholesky.h"

// CUDA error handeling
// =====================
static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
// =====================

// Cholesky factorization kernel
__global__ void cholesky_factorization_warp_primitive(double* M, double* L,
                                                      size_t num_equations,
                                                      size_t n,
                                                      size_t num_stride) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_equations) {
    return;
  }

  Eigen::Map<Eigen::MatrixXd> d_M(M + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::MatrixXd> d_L(L + equ_idx * n * n, n, n);

  for (int stride = 0; stride < num_stride; stride++) {
    int j = thread_idx + stride * 32;
    if (j >= n) return;

    double diagonal_val = 0.0;
    double sum = 0.0;

    for (int i = 0; i <= j; ++i) {
      sum = 0.0;

      for (int k = 0; k < i; ++k) {
        sum += d_L(j, k) * d_L(i, k);
      }

      if (i == j) {
        diagonal_val = sqrt(d_M(i, i) - sum);
        // d_L(i, i) = diagonal_val;
      }

      //__syncwarp();

      // diagonal_val = d_L(i, i);

      if (i > stride * 32) {
        __shfl_sync(0xFFFFFFFF, diagonal_val, i - stride * 32 - 1);
      }

      if (i <= stride * 32) {
        diagonal_val = d_L(i, i);
      }

      // if (i < offset) {
      //   diagonal_val = d_L(i, i);
      // }

      if (j > i) {
        diagonal_val = (d_M(j, i) - sum) / diagonal_val;
      }

      d_L(j, i) = diagonal_val;
      __syncwarp();
    }
  }
}
// Cholesky factorization kernel
__global__ void cholesky_factorization(double* M, double* L,
                                       size_t num_equations, size_t n,
                                       size_t num_stride) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_equations) {
    return;
  }

  Eigen::Map<Eigen::MatrixXd> d_M(M + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::MatrixXd> d_L(L + equ_idx * n * n, n, n);

  for (int stride = 0; stride < num_stride; stride++) {
    int j = thread_idx + stride * 32;
    if (j >= n) return;

    for (int i = 0; i <= j; ++i) {
      __syncwarp();

      if (i == j) {
        double sum = 0.0;
        for (int k = 0; k < i; ++k) {
          sum += d_L(i, k) * d_L(i, k);
        }
        d_L(i, i) = sqrt(d_M(i, i) - sum);
      }

      __syncwarp();

      if (j > i) {
        double sum = 0.0;
        for (int k = 0; k < i; ++k) {
          sum += d_L(j, k) * d_L(i, k);
        }
        d_L(j, i) = (d_M(j, i) - sum) / d_L(i, i);
      }
    }
  }
}

// Cholesky solve forward substitution kernel
__global__ void cholesky_solve_forward(double* L, double* b, double* x,
                                       double* y, size_t num_equations,
                                       size_t n, size_t num_stride) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_equations) {
    return;
  }

  Eigen::Map<Eigen::MatrixXd> d_L(L + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::VectorXd> d_b(b + equ_idx * n, n, 1);
  Eigen::Map<Eigen::VectorXd> d_x(x + equ_idx * n, n, 1);
  Eigen::Map<Eigen::VectorXd> d_y(y + equ_idx * n, n, 1);

  for (int stride = 0; stride < num_stride; stride++) {
    int i = thread_idx + stride * 32;
    if (i >= n) return;

    // Forward substitution to solve L * y = b

    double sum = 0.0;
    for (int j = 0; j <= i; ++j) {
      if (j == i) {
        d_y(i) = (d_b(i) - sum) / d_L(i, i);
      }
      __syncwarp();

      if (j < i) {
        sum += d_L(i, j) * d_y(j);
      }

      __syncwarp();
    }
  }
}

// Cholesky solve backward substitution kernel
__global__ void cholesky_solve_backward(double* L, double* b, double* x,
                                        double* y, size_t num_equations,
                                        size_t n, size_t num_stride) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_equations) {
    return;
  }

  Eigen::Map<Eigen::MatrixXd> d_L(L + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::VectorXd> d_b(b + equ_idx * n, n, 1);
  Eigen::Map<Eigen::VectorXd> d_x(x + equ_idx * n, n, 1);
  Eigen::Map<Eigen::VectorXd> d_y(y + equ_idx * n, n, 1);

  for (int stride = 0; stride < num_stride; stride++) {
    int i = n - 1 - (thread_idx + stride * 32);
    if (i < 0) return;

    double sum = 0.0;
    for (int j = n - 1; j >= i; --j) {
      if (j == i) {
        d_x(i) = (d_y(i) - sum) / d_L(i, i);
      }
      __syncwarp();

      if (j > i) {
        sum += d_L(j, i) * d_x(j);
      }

      __syncwarp();
    }
  }
}

// Main solve function - including memory allocation, copy, and kernel calls
double matrix_solve(std::vector<Eigen::MatrixXd>& M,
                    std::vector<Eigen::VectorXd>& b,
                    std::vector<Eigen::VectorXd>& x) {
  const int num_equations = M.size();
  const int n = b[0].size();

  double* x_result = new double[num_equations * n];

  // Allocate device arrays
  double *d_M, *d_b, *d_y, *d_x, *d_L;
  HANDLE_ERROR(
      cudaMalloc((void**)&d_M, sizeof(double) * num_equations * n * n));
  HANDLE_ERROR(
      cudaMalloc((void**)&d_L, sizeof(double) * num_equations * n * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(double) * num_equations * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_x, sizeof(double) * num_equations * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_y, sizeof(double) * num_equations * n));

  // Set d_L and d_x to be 0
  HANDLE_ERROR(cudaMemset(d_L, 0, sizeof(double) * num_equations * n * n));
  HANDLE_ERROR(cudaMemset(d_x, 1, sizeof(double) * num_equations * n));
  HANDLE_ERROR(cudaMemset(d_y, 0, sizeof(double) * num_equations * n));

  // Copy to device
  for (int i = 0; i < num_equations; ++i) {
    HANDLE_ERROR(cudaMemcpy(d_M + i * n * n, M[i].data(),
                            sizeof(double) * n * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b + i * n, b[i].data(), sizeof(double) * n,
                            cudaMemcpyHostToDevice));
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  int num_stride = (n + 31) / 32;

  // Matrix Cholesky factorization

  cholesky_factorization<<<num_equations, 32>>>(d_M, d_L, num_equations, n,
                                                num_stride);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Matrix Cholesky solve forward

  cholesky_solve_forward<<<num_equations, 32>>>(d_L, d_b, d_x, d_y,
                                                num_equations, n, num_stride);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Matrix Cholesky solve backward
  cholesky_solve_backward<<<num_equations, 32>>>(d_L, d_b, d_x, d_y,
                                                 num_equations, n, num_stride);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for Cholesky Solve: " << milliseconds << " ms\n";

  // Copy to host
  HANDLE_ERROR(cudaMemcpy(x_result, d_x, sizeof(double) * num_equations * n,
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_equations; ++i) {
    Eigen::Map<Eigen::VectorXd> x_result_i(x_result + i * n, n, 1);
    x[i] = x_result_i;
  }

  return 0;
}
