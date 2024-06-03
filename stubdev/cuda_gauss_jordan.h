#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

double gauss_jordan_solve(std::vector<Eigen::MatrixXd>& M,
                          std::vector<Eigen::MatrixXd>& I);