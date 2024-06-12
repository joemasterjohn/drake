#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

void reduce_by_problem(std::vector<double>& vec_int,
                       std::vector<double>& vec_out, int num_problems,
                       int items_per_equation);