#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

enum class LinOpType { ADD, SUB };

__global__ void matrixLinOp_32thd_Kernel(double* v_A, double* v_B,
                                         double* v_Res, int row, int col,
                                         LinOpType op, int num_strides,
                                         int num_eqations);

__global__ void matrixMultiply_32thd_Kernel(double* v_A, double* v_B,
                                            double* v_C, int A_row, int A_col,
                                            int B_col, int stride,
                                            int num_eqations);

// Matrix multiplication of two matrices A and B, and store the result in C
void matrixMultiply_32thd(std::vector<Eigen::MatrixXd>& v_A,
                          std::vector<Eigen::MatrixXd>& v_B,
                          std::vector<Eigen::MatrixXd>& v_C, int num_equations);

void matrixLinOp_32thd(std::vector<Eigen::MatrixXd>& v_A,
                       std::vector<Eigen::MatrixXd>& v_B,
                       std::vector<Eigen::MatrixXd>& v_C, LinOpType op,
                       int num_equations);
