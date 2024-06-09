#pragma once

#include <vector>

#include "cuda_matmul.h"
#include <eigen3/Eigen/Dense>

__global__ void matrixLinOp_32thd_Kernel(double* v_A, double* v_B,
                                         double* v_Res, int row, int col,
                                         LinOpType op, int num_strides,
                                         int num_eqations);

__global__ void matrixMultiply_32thd_Kernel(double* v_A, double* v_B,
                                            double* v_C, int A_row, int A_col,
                                            int B_col, int stride,
                                            int num_eqations);