#include "stubdev/cuda_cholesky.h"

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, Cholesky) {
  const int N = 256;
  std::vector<Eigen::MatrixXd> M;
  std::vector<Eigen::VectorXd> b;
  std::vector<Eigen::VectorXd> x;
  for (int i = 0; i < 100; ++i) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    M.push_back(A.transpose() * A);
    x.push_back(Eigen::VectorXd::Random(N));
    b.push_back(M[i] * x[i]);
  }

  matrix_solve(M, b, x);
}

}  // namespace
}  // namespace drake
