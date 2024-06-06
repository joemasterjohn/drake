#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

// Matrix multiplication of two matrices A and B, and store the result in C
void matrixMultiply_32thd(std::vector<Eigen::MatrixXd>& v_A,
                          std::vector<Eigen::MatrixXd>& v_B,
                          std::vector<Eigen::MatrixXd>& v_C, int num_equations);

enum class LinOpType { ADD, SUB };

void matrixLinOp_32hd(std::vector<Eigen::MatrixXd>& v_A,
                      std::vector<Eigen::MatrixXd>& v_B,
                      std::vector<Eigen::MatrixXd>& v_C, LinOpType op,
                      int num_equations);

// // Evaluate the gradient of the primal cost l_p
// void evalute_l_p_grad(std::vector<Eigen::MatrixXd>& v_A,
//                       std::vector<Eigen::MatrixXd>& v_v,
//                       std::vector<Eigen::MatrixXd>& v_v_m,
//                       std::vector<Eigen::MatrixXd>& v_J,
//                       std::vector<Eigen::MatrixXd>& v_gamma,
//                       std::vector<Eigen::MatrixXd>& v_grad_l_p,  //
//                       result int num_equations);
