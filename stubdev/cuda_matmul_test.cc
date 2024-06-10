#include "stubdev/cuda_matmul.h"

#include <chrono>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, CudaMatmul_Host) {
  int num_equations = 1000;
  int num_checks = 10;
  int N = 120;

  std::cout << "Started testing host wrapper" << std::endl;

  for (int iter = 0; iter < num_checks; iter++) {
    std::vector<Eigen::MatrixXd> v_A;
    std::vector<Eigen::MatrixXd> v_B;

    std::vector<Eigen::MatrixXd> v_C_1;  // Matrix Multiplication A*B = C
    std::vector<Eigen::MatrixXd> v_C_2;  // for CPU validation results

    std::vector<Eigen::MatrixXd> v_Res_add_1;  // A + B = Res_add
    std::vector<Eigen::MatrixXd> v_Res_add_2;  // for CPU validation results

    std::vector<Eigen::MatrixXd> v_Res_sub_1;  // A - B = Res_sub
    std::vector<Eigen::MatrixXd> v_Res_sub_2;  // for CPU validation results

    for (int i = 0; i < num_equations; i++) {
      Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
      Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
      Eigen::MatrixXd C_1(N, N);
      Eigen::MatrixXd C_2(N, N);
      Eigen::MatrixXd Res_add_1(N, N);
      Eigen::MatrixXd Res_add_2(N, N);
      Eigen::MatrixXd Res_sub_1(N, N);
      Eigen::MatrixXd Res_sub_2(N, N);

      v_A.push_back(A);
      v_B.push_back(B);
      v_C_1.push_back(C_1);
      v_C_2.push_back(C_2);
      v_Res_add_1.push_back(Res_add_1);
      v_Res_add_2.push_back(Res_add_2);
      v_Res_sub_1.push_back(Res_sub_1);
      v_Res_sub_2.push_back(Res_sub_2);
    }

    // test 1 - matrix multiplication

    // Perform matrix multiplication
    MatrixMultiply32thdHost(v_A, v_B, v_C_1, num_equations);

    // Time the Eigen matrix multiplication for C_t3 in milliseconds
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_equations; i++) {
      v_C_2[i] = v_A[i] * v_B[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Elapsed time for Eigen mat mul CPU: " << elapsed.count()
              << " ms\n";

    for (int i = 0; i < num_equations; i++) {
      EXPECT_LT((v_C_1[i] - v_C_2[i]).norm(), 1e-10);
    }

    // test 2 - matrix addition

    MatrixLinOp32thdHost(v_A, v_B, v_Res_add_1, LinOpType::ADD,
                           num_equations);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_equations; i++) {
      v_Res_add_2[i] = v_A[i] + v_B[i];
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Elapsed time for Eigen mat add CPU: " << elapsed.count()
              << " ms\n";

    for (int i = 0; i < num_equations; i++) {
      EXPECT_LT((v_Res_add_1[i] - v_Res_add_2[i]).norm(), 1e-10);
    }

    // test 3 - matrix subtraction
    MatrixLinOp32thdHost(v_A, v_B, v_Res_sub_1, LinOpType::SUB,
                           num_equations);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_equations; i++) {
      v_Res_sub_2[i] = v_A[i] - v_B[i];
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Elapsed time for Eigen mat sub CPU: " << elapsed.count()
              << " ms\n";

    for (int i = 0; i < num_equations; i++) {
      EXPECT_LT((v_Res_sub_1[i] - v_Res_sub_2[i]).norm(), 1e-10);
    }
  }
}

}  // namespace
}  // namespace drake
