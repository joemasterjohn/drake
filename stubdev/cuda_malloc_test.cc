#undef PRINTOUT

#include "cuda_malloc.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, MallocTest) {
  malloc_test_wrap();
}

// ===================================================
// END OF ACTUAL SAP SOLVER FUNCTION CALLS
// ===================================================

}  // namespace
}  // namespace drake
