// Cuda eigen debug test, used for testing the performance of .row() eigen
// function

// This test checks runtime for hand-rolled matrix vector multiplication with
// multiplicatoin using .row() function

#include "stubdev/cuda_eigen_debug.h"

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, Cholesky) {
  EigenRowTest();
  double sum = 0.f;
  for (int i = 0; i < 1000; i++) {
    sum += EigenRowTest();
  }
  std::cout << "Eigen Row avg_time:" << sum / 1000 << "ms" << std::endl;

  EigenTest();
  sum = 0.f;
  for (int i = 0; i < 1000; i++) {
    sum += EigenTest();
  }
  std::cout << "Eigen No Row avg_time:" << sum / 1000 << "ms" << std::endl;
}

}  // namespace
}  // namespace drake
