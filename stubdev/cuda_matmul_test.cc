#include "stubdev/cuda_matmul.h"

#include <chrono>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, CudaMatmul) {
  int num_equations = 1000;

  std::vector<Eigen::MatrixXd> v_A;
  std::vector<Eigen::MatrixXd> v_B;
  std::vector<Eigen::MatrixXd> v_C_1;  // for CPU validation results
  std::vector<Eigen::MatrixXd> v_C_2;  // for CPU validation results

  for (int i = 0; i < num_equations; i++) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(128, 128);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(128, 128);
    Eigen::MatrixXd C_1(128, 128);  // matrix multiplication res
    Eigen::MatrixXd C_2(128, 128);  // validation res

    v_A.push_back(A);
    v_B.push_back(B);
    v_C_1.push_back(C_1);
    v_C_2.push_back(C_2);
  }

  // Time the Eigen matrix multiplication for C_t3 in milliseconds
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_equations; i++) {
    v_C_1[i] = v_A[i] * v_B[i];
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  std::cout << "Elapsed time for Eigen mat mul CPU: " << elapsed.count()
            << " ms\n";

  // Perform matrix multiplication
  matrixMultiply_32thd(v_A, v_B, v_C_2, num_equations);

  for (int i = 0; i < num_equations; i++) {
    EXPECT_LT((v_C_1[i] - v_C_2[i]).norm(), 1e-10);
  }
}

}  // namespace
}  // namespace drake
