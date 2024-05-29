// Kernels for eigen debug test

#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

// Kernel function to perform hand-rolled version of matrix-vector
// multiplication
__global__ void matVecMultiply(double* matrix_ptr, double* vector_ptr,
                               double* result_ptr, int N) {
  Eigen::Map<Eigen::MatrixXd> matrix(matrix_ptr, N, N);
  Eigen::Map<Eigen::MatrixXd> vector(vector_ptr, N, 1);
  Eigen::Map<Eigen::MatrixXd> result(result_ptr, N, 1);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
      sum += matrix(idx, j) * vector(j, 0);
    }
    result(idx, 0) = sum;
  }
}

// Kernel function to perform matrix-vector
// multiplication using eigen function .row()
__global__ void matVecMultiplyRow(double* matrix_ptr, double* vector_ptr,
                                  double* result_ptr, int N) {
  Eigen::Map<Eigen::MatrixXd> matrix(matrix_ptr, N, N);
  Eigen::Map<Eigen::MatrixXd> vector(vector_ptr, N, 1);
  Eigen::Map<Eigen::MatrixXd> result(result_ptr, N, 1);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    double sum = 0.0;
    sum += (matrix.row(idx) * vector)(0, 0);
    result(idx, 0) = sum;
  }
}