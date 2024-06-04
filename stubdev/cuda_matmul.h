#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

void matrixMultiply_32thd(std::vector<Eigen::MatrixXd>& v_A,
                          std::vector<Eigen::MatrixXd>& v_B,
                          std::vector<Eigen::MatrixXd>& v_C, int num_equations);