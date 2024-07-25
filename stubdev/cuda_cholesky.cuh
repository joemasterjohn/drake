#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

// =====================
// Device function to perform Cholesky factorization
__device__ void CholeskyFactorizationFunc(Eigen::Map<Eigen::MatrixXd> M,
                                          Eigen::Map<Eigen::MatrixXd> L,
                                          int equ_idx, int thread_idx, size_t n,
                                          size_t num_stride);

// Device function to perform forward substitution
__device__ void CholeskySolveForwardFunc(Eigen::Map<Eigen::MatrixXd> L,
                                         Eigen::Map<Eigen::MatrixXd> b,
                                         Eigen::Map<Eigen::MatrixXd> y,
                                         int equ_idx, int thread_idx, size_t n,
                                         size_t num_stride);

// Device function to perform backward substitution
__device__ void CholeskySolveBackwardFunc(Eigen::Map<Eigen::MatrixXd> L,
                                          Eigen::Map<Eigen::MatrixXd> y,
                                          Eigen::Map<Eigen::MatrixXd> x,
                                          int equ_idx, int thread_idx, size_t n,
                                          size_t num_stride);