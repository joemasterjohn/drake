#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

double matrix_solve(std::vector<Eigen::MatrixXd>& M,
                    std::vector<Eigen::VectorXd>& b,
                    std::vector<Eigen::VectorXd>& x);