#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

double dot(const std::vector<Eigen::Vector3d> & v1, const std::vector<Eigen::Vector3d> & v2);
double dot_vector_x(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2);
double matrix_solve(const std::vector<Eigen::MatrixXd>& M, const std::vector<Eigen::VectorXd>& b, const std::vector<Eigen::VectorXd>& x);
