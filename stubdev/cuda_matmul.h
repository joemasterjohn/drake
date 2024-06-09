#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

enum class LinOpType { ADD, SUB };

// Matrix multiplication of two matrices A and B, and store the result in C
// This function assumes that the matrices passed in are stored on CPU
void matrixMultiply_32thd_Host(std::vector<Eigen::MatrixXd>& v_A,
                               std::vector<Eigen::MatrixXd>& v_B,
                               std::vector<Eigen::MatrixXd>& v_C,
                               int num_equations);

// Matrix addition or subtraction of two matrices A and B, and store the result
// in C This function assumes that the matrices passed in are stored on CPU
void matrixLinOp_32thd_Host(std::vector<Eigen::MatrixXd>& v_A,
                            std::vector<Eigen::MatrixXd>& v_B,
                            std::vector<Eigen::MatrixXd>& v_C, LinOpType op,
                            int num_equations);

// Matrix multiplication of two matrices A and B, and store the result in C
// This function assumes that the matrices passed in are stored on GPU
void matrixMultiply_32thd_Device(std::vector<Eigen::MatrixXd>& d_v_A,
                                 std::vector<Eigen::MatrixXd>& d_v_B,
                                 std::vector<Eigen::MatrixXd>& d_v_C, int A_row,
                                 int A_col, int B_col, int num_equations);

// Matrix addition or subtraction of two matrices A and B, and store the result
// in C This function assumes that the matrices passed in are stored on GPU
void matrixLinOp_32thd_Device(std::vector<Eigen::MatrixXd>& d_v_A,
                              std::vector<Eigen::MatrixXd>& d_v_B,
                              std::vector<Eigen::MatrixXd>& d_v_Res, int row,
                              int col, LinOpType op, int num_eqations);
