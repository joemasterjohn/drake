#pragma once

#include <vector>

#include "cuda_matmul.h"
#include <eigen3/Eigen/Dense>

__device__ void MatrixMultiply32thdFunc(double* v_A, double* v_B, double* v_C,
                                        double* sums, int thread_idx,
                                        int equ_idx, int A_row, int A_col,
                                        int B_col, int stride);

__global__ void MatrixMultiply32thdKernel(double* v_A, double* v_B, double* v_C,
                                          int A_row, int A_col, int B_col,
                                          int stride, int num_eqations);

__device__ void MatrixLinOp32thdFunc(double* v_A, double* v_B, double* v_Res,
                                     int thread_idx, int equ_idx, int row,
                                     int col, LinOpType op, int num_strides);

__global__ void MatrixLinOp32thdKernel(double* v_A, double* v_B, double* v_Res,
                                       int row, int col, LinOpType op,
                                       int num_strides, int num_eqations);
