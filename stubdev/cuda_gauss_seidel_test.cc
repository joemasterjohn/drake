#include "stubdev/cuda_gauss_seidel.h"

#include <Eigen/Dense>

#include <vector>
#include <iostream>

#include <gtest/gtest.h>


namespace drake {
namespace {

GTEST_TEST(KernelTest, GaussSeidel) {

  std::vector<Eigen::Vector3d> v1(100, Eigen::Vector3d{ 1.0, 1.0, 1.0 });
  std::vector<Eigen::Vector3d> v2(100, Eigen::Vector3d{ 1.0, 1.0, 1.0 });

  const double d = dot(v1, v2);

  std::cout << "result: " << d << std::endl;

  Eigen::VectorXd v3 = Eigen::VectorXd::Ones(300);
  Eigen::VectorXd v4 = Eigen::VectorXd::Ones(300);
  
  const double m = dot_vector_x(v3, v4);

  std::cout << "result: " << m << std::endl;

  const int N = 30;
  std::vector<Eigen::MatrixXd> M;
  std::vector<Eigen::VectorXd> b;
  std::vector<Eigen::VectorXd> x;
  for(int i = 0; i < 100; ++i) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N/2,N);
    M.push_back(A.transpose()*A);
    x.push_back(Eigen::VectorXd::Random(N));
    b.push_back(M[i]*x[i]);
  }

  matrix_solve(M, b, x);

}

}  // namespace
}  // namespace drake
