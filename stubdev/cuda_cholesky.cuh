#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

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
      __syncwarp();

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
    }
  }
}

// Device function to perform forward substitution
__device__ void CholeskySolveForwardFunc(Eigen::Map<Eigen::MatrixXd> L,
                                         Eigen::Map<Eigen::VectorXd> b,
                                         Eigen::Map<Eigen::VectorXd> y,
                                         int equ_idx, int thread_idx, size_t n,
                                         size_t num_stride) {
  for (int stride = 0; stride < num_stride; stride++) {
    int j = thread_idx + stride * 32;
    int j_up = 31 + stride * 32;

    // Forward substitution to solve L * y = b
    double sum = 0.0;
    for (int i = 0; i <= j_up; ++i) {
      if (j < n && i <= j && i == j) {
        y(j) = (b(j) - sum) / L(j, j);
      }
      __syncwarp();

      if (j < n && i <= j && i < j) {
        sum += L(j, i) * y(i);
      }

      __syncwarp();
    }
  }
}

// Device function to perform backward substitution
__device__ void CholeskySolveBackwardFunc(Eigen::Map<Eigen::MatrixXd> L,
                                          Eigen::Map<Eigen::VectorXd> y,
                                          Eigen::Map<Eigen::VectorXd> x,
                                          int equ_idx, int thread_idx, size_t n,
                                          size_t num_stride) {
  for (int stride = 0; stride < num_stride; stride++) {
    int j = n - 1 - (thread_idx + stride * 32);
    int j_down = n - 1 - (31 + stride * 32);

    double sum = 0.0;
    for (int i = n - 1; i >= j_down; --i) {
      if (j >= 0 && i >= j && i == j) {
        x(j) = (y(j) - sum) / L(j, j);
      }
      __syncwarp();

      if (j >= 0 && i >= j && i > j) {
        sum += L(i, j) * x(i);
      }

      __syncwarp();
    }
  }
}