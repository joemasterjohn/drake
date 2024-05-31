#include <stdio.h>

#include <iostream>

#include "cuda_cholesky.h"

static void HandleError(cudaError_t err, const char* file, int line) {
  // CUDA error handeling from the "CUDA by example" book
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void cholesky_factorization(double* M, double* L,
                                       size_t num_equations, size_t n,
                                       size_t offset) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_equations) {
    return;
  }

  Eigen::Map<Eigen::MatrixXd> d_M(M + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::MatrixXd> d_L(L + equ_idx * n * n, n, n);

  int j = thread_idx + offset;
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

__global__ void cholesky_solve_forward(double* L, double* b, double* x,
                                       double* y, size_t num_equations,
                                       size_t n, size_t offset) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_equations) {
    return;
  }

  Eigen::Map<Eigen::MatrixXd> d_L(L + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::VectorXd> d_b(b + equ_idx * n, n, 1);
  Eigen::Map<Eigen::VectorXd> d_x(x + equ_idx * n, n, 1);
  Eigen::Map<Eigen::VectorXd> d_y(y + equ_idx * n, n, 1);

  int i = thread_idx + offset;
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

__global__ void cholesky_solve_backward(double* L, double* b, double* x,
                                        double* y, size_t num_equations,
                                        size_t n, size_t offset) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_equations) {
    return;
  }

  Eigen::Map<Eigen::MatrixXd> d_L(L + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::VectorXd> d_b(b + equ_idx * n, n, 1);
  Eigen::Map<Eigen::VectorXd> d_x(x + equ_idx * n, n, 1);
  Eigen::Map<Eigen::VectorXd> d_y(y + equ_idx * n, n, 1);

  int i = n - 1 - (thread_idx + offset);
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

  // Matrix Cholesky factorization
  int offset = 0;
  while (offset < n) {
    cholesky_factorization<<<num_equations, 32>>>(d_M, d_L, num_equations, n,
                                                  offset);
    offset += 32;

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  // Matrix Cholesky solve forward
  offset = 0;
  while (offset < n) {
    cholesky_solve_forward<<<num_equations, 32>>>(d_L, d_b, d_x, d_y,
                                                  num_equations, n, offset);
    offset += 32;

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  // Matrix Cholesky solve backward
  offset = 0;
  while (offset < n) {
    cholesky_solve_backward<<<num_equations, 32>>>(d_L, d_b, d_x, d_y,
                                                   num_equations, n, offset);
    offset += 32;

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

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
    std::cout << "||M*x - b||: " << (M[i] * x_result_i - b[i]).norm()
              << std::endl;
  }

  return 0;
}
