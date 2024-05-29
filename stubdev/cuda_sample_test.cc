#include "stubdev/cuda_sample.h"

#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(EstTest, SmokeTest) {
  EXPECT_TRUE(CudaAllocate(1024));
}

}  // namespace
}  // namespace drake
