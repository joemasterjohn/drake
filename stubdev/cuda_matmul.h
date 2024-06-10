#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

enum class LinOpType { ADD, SUB };

// Matrix multiplication of two matrices A and B, and store the result in C
// This function assumes that the matrices passed in are stored on CPU
void MatrixMultiply32thdHost(std::vector<Eigen::MatrixXd>& v_A,
                             std::vector<Eigen::MatrixXd>& v_B,
                             std::vector<Eigen::MatrixXd>& v_C,
                             int num_equations);

// Matrix addition or subtraction of two matrices A and B, and store the result
// in C This function assumes that the matrices passed in are stored on CPU
void MatrixLinOp32thdHost(std::vector<Eigen::MatrixXd>& v_A,
                          std::vector<Eigen::MatrixXd>& v_B,
                          std::vector<Eigen::MatrixXd>& v_C, LinOpType op,
                          int num_equations);
