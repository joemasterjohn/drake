// Cuda cholesky factorization and solve nvcc library under
// stubdev folder.

#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

// CPU Entry Point
double MatrixSolve(std::vector<Eigen::MatrixXd>& M,
                   std::vector<Eigen::MatrixXd>& b,
                   std::vector<Eigen::MatrixXd>& x);
