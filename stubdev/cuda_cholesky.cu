#include <stdio.h>

#include <iostream>

#include "cuda_cholesky.cuh"
#include "cuda_cholesky.h"

// =====================
// Device function to perform Cholesky factorization
__device__ void CholeskyFactorizationFunc(Eigen::Map<Eigen::MatrixXd> M,
                                          Eigen::Map<Eigen::MatrixXd> L,
                                          int equ_idx, int thread_idx, size_t n,
                                          size_t num_stride) {
                                            
  for (int stride = 0; stride < num_stride; stride++) {
    int j_up = 31 + stride * 32;
    int j = thread_idx + stride * 32;

    for (int i = 0; i <= j_up; ++i) {
      if (j < n && i <= j && i == j) {
        double sum = 0.0;
        for (int k = 0; k < i; ++k) {
          sum += L(i, k) * L(i, k);
        }
        L(i, i) = sqrt(M(i, i) - sum);
      }

      __syncwarp();

      if (j < n && i <= j && j > i) {
        double sum = 0.0;
        for (int k = 0; k < i; ++k) {
          sum += L(j, k) * L(i, k);
        }
        L(j, i) = (M(j, i) - sum) / L(i, i);
      }

      __syncwarp();
    }
  }
}

// Device function to perform forward substitution
__device__ void CholeskySolveForwardFunc(Eigen::Map<Eigen::MatrixXd> L,
                                         Eigen::Map<Eigen::MatrixXd> b,
                                         Eigen::Map<Eigen::MatrixXd> y,
                                         int equ_idx, int thread_idx, size_t n,
                                         size_t num_stride) {
  for (int stride = 0; stride < num_stride; stride++) {
    int j = thread_idx + stride * 32;
    int j_up = 31 + stride * 32;

    // Forward substitution to solve L * y = b
    double sum = 0.0;
    for (int i = 0; i <= j_up; ++i) {
      if (j < n && i <= j && i == j) {
        y(j, 0) = (b(j) - sum) / L(j, j);
      }
      __syncwarp();

      if (j < n && i <= j && i < j) {
        sum += L(j, i) * y(i, 0);
      }

      __syncwarp();
    }
  }
}

// Device function to perform backward substitution
__device__ void CholeskySolveBackwardFunc(Eigen::Map<Eigen::MatrixXd> L,
                                          Eigen::Map<Eigen::MatrixXd> y,
                                          Eigen::Map<Eigen::MatrixXd> x,
                                          int equ_idx, int thread_idx, size_t n,
                                          size_t num_stride) {
  for (int stride = 0; stride < num_stride; stride++) {
    int j = n - 1 - (thread_idx + stride * 32);
    int j_down = n - 1 - (31 + stride * 32);

    double sum = 0.0;
    for (int i = n - 1; i >= j_down; --i) {
      if (j >= 0 && i >= j && i == j) {
        x(j, 0) = (y(j, 0) - sum) / L(j, j);
      }
      __syncwarp();

      if (j >= 0 && i >= j && i > j) {
        sum += L(i, j) * x(i, 0);
      }

      __syncwarp();
    }
  }
}



// CUDA error handeling
// =====================
static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void CholeskySolveKernel(double* M, double* L, double* b, double* x,
                                    double* y, size_t num_problems, size_t n) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_problems) {
    return;
  }

  Eigen::Map<Eigen::MatrixXd> d_M(M + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::MatrixXd> d_L(L + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::MatrixXd> d_b(b + equ_idx * n, n, 1);
  Eigen::Map<Eigen::MatrixXd> d_x(x + equ_idx * n, n, 1);
  Eigen::Map<Eigen::MatrixXd> d_y(y + equ_idx * n, n, 1);

  int num_stride = (n + 31) / 32;

  CholeskyFactorizationFunc(d_M, d_L, equ_idx, thread_idx, n, num_stride);
  __syncwarp();
  CholeskySolveForwardFunc(d_L, d_b, d_y, equ_idx, thread_idx, n, num_stride);
  __syncwarp();
  CholeskySolveBackwardFunc(d_L, d_y, d_x, equ_idx, thread_idx, n, num_stride);
  __syncwarp();
}

// Main solve function - including memory allocation, copy, and kernel calls
double MatrixSolve(std::vector<Eigen::MatrixXd>& M,
                   std::vector<Eigen::MatrixXd>& b,
                   std::vector<Eigen::MatrixXd>& x) {
  const int num_problems = M.size();
  const int n = b[0].size();

  double* x_result = new double[num_problems * n];

  // Allocate device arrays
  double *d_M, *d_b, *d_y, *d_x, *d_L;
  HANDLE_ERROR(cudaMalloc((void**)&d_M, sizeof(double) * num_problems * n * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_L, sizeof(double) * num_problems * n * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(double) * num_problems * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_x, sizeof(double) * num_problems * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_y, sizeof(double) * num_problems * n));

  // Set d_L and d_x to be 0
  HANDLE_ERROR(cudaMemset(d_L, 0, sizeof(double) * num_problems * n * n));
  HANDLE_ERROR(cudaMemset(d_x, 0, sizeof(double) * num_problems * n));
  HANDLE_ERROR(cudaMemset(d_y, 0, sizeof(double) * num_problems * n));

  // Copy to device
  for (int i = 0; i < num_problems; ++i) {
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
  CholeskySolveKernel<<<num_problems, 32>>>(d_M, d_L, d_b, d_x, d_y,
                                            num_problems, n);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for Cholesky Solve: " << milliseconds << " ms\n";

  // Copy to host
  HANDLE_ERROR(cudaMemcpy(x_result, d_x, sizeof(double) * num_problems * n,
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_problems; ++i) {
    Eigen::Map<Eigen::MatrixXd> x_result_i(x_result + i * n, n, 1);
    x[i] = x_result_i;
  }

  return 0;
}
