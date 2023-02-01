#include "drake/multibody/fem/mooney_rivlin_model.h"

#include <gtest/gtest.h>

#include "drake/multibody/fem/test/constitutive_model_test_utilities.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace test {

constexpr int kNumLocations = 1;

GTEST_TEST(MooneyRivlinModelTest, UndeformedState) {
  TestUndeformedState<MooneyRivlinModel<double, kNumLocations>>();
  TestUndeformedState<MooneyRivlinModel<AutoDiffXd, kNumLocations>>();
}

GTEST_TEST(MooneyRivlinModelTest, PIsDerivativeOfPsi) {
  TestPIsDerivativeOfPsi<MooneyRivlinModel<AutoDiffXd, kNumLocations>>();
}

GTEST_TEST(MooneyRivlinModelTest, dPdFIsDerivativeOfP) {
  TestdPdFIsDerivativeOfP<MooneyRivlinModel<AutoDiffXd, kNumLocations>>();
}

}  // namespace test
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
