#include "stubdev/cuda_gauss_jordan.h"

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, Cholesky) {
  const int N = 16;
  std::vector<Eigen::MatrixXd> M;
  std::vector<Eigen::MatrixXd> I;
  for (int i = 0; i < 1; ++i) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    M.push_back(A.transpose() * A);

    Eigen::MatrixXd identity_matrix = Eigen::MatrixXd::Identity(N, N);
    I.push_back(identity_matrix);
  }

  gauss_jordan_solve(M, I);
}

}  // namespace
}  // namespace drake
