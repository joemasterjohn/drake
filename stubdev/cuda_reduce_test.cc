#include "cuda_reduce.h"

#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, Reduce) {
  const int items_per_equation = 423;
  int num_equations = 300;

  std::vector<double> vec_int;
  std::vector<double> vec_out;
  for (int i = 0; i < num_equations; ++i) {
    for (int j = 0; j < items_per_equation; ++j) {
      // generate random numbers between 0 and 1
      vec_int.push_back(static_cast<double>(rand()) / RAND_MAX);
    }
  }

  // resize vec_out
  vec_out.resize(num_equations);

  reduce_by_problem(vec_int, vec_out, num_equations, items_per_equation);

  for (int i = 0; i < num_equations; ++i) {
    double sum = 0;
    for (int j = 0; j < items_per_equation; ++j) {
      sum += vec_int[i * items_per_equation + j];
    }
    EXPECT_NEAR(sum, vec_out[i], 1e-10);
  }
}

}  // namespace
}  // namespace drake
