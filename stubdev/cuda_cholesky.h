#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

double MatrixSolve(std::vector<Eigen::MatrixXd>& M,
                   std::vector<Eigen::MatrixXd>& b,
                   std::vector<Eigen::MatrixXd>& x);
