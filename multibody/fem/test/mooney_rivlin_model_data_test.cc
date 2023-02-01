#include "drake/multibody/fem/mooney_rivlin_model_data.h"

#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/fem/matrix_utilities.h"
#include "drake/multibody/fem/test/constitutive_model_test_utilities.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace {

using Eigen::Matrix3d;
using test::MakeDeformationGradients;
constexpr int kNumLocations = 1;
constexpr double kTolerance = 1e-12;

GTEST_TEST(MooneyRivlinModelDataTest, TestdIdFIsDerivativeOfInvariants) {
  MooneyRivlinModelData<AutoDiffXd, kNumLocations> data;
  const std::array<Matrix3<AutoDiffXd>, kNumLocations> deformation_gradients =
      MakeDeformationGradients<kNumLocations>();
  data.UpdateData(deformation_gradients);

  std::array<AutoDiffXd, kNumLocations> I1 = data.I1();
  std::array<AutoDiffXd, kNumLocations> I2 = data.I2();
  std::array<AutoDiffXd, kNumLocations> I3 = data.I3();
  std::array<Matrix3<AutoDiffXd>, kNumLocations> dI1_dF = data.dI1dF();
  std::array<Matrix3<AutoDiffXd>, kNumLocations> dI2_dF = data.dI2dF();
  std::array<Matrix3<AutoDiffXd>, kNumLocations> dI3_dF = data.dI3dF();

  for (int i = 0; i < kNumLocations; ++i) {
    EXPECT_TRUE(CompareMatrices(
        Eigen::Map<const Matrix3d>(I1[i].derivatives().data(), 3, 3), dI1_dF[i],
        CalcConditionNumberOfInvertibleMatrix<AutoDiffXd>(dI1_dF[i]) *
            kTolerance));
    EXPECT_TRUE(CompareMatrices(
        Eigen::Map<const Matrix3d>(I2[i].derivatives().data(), 3, 3), dI2_dF[i],
        CalcConditionNumberOfInvertibleMatrix<AutoDiffXd>(dI2_dF[i]) *
            kTolerance));
    EXPECT_TRUE(CompareMatrices(
        Eigen::Map<const Matrix3d>(I3[i].derivatives().data(), 3, 3), dI3_dF[i],
        CalcConditionNumberOfInvertibleMatrix<AutoDiffXd>(dI3_dF[i]) *
            kTolerance));
  }
}

GTEST_TEST(MooneyRivlinModelDataTest, Testd2IdF2IsDerivativeOfdIdF) {
  MooneyRivlinModelData<AutoDiffXd, kNumLocations> data;
  const std::array<Matrix3<AutoDiffXd>, kNumLocations> deformation_gradients =
      MakeDeformationGradients<kNumLocations>();
  data.UpdateData(deformation_gradients);

  std::array<Matrix3<AutoDiffXd>, kNumLocations> dI1_dF = data.dI1dF();
  std::array<Matrix3<AutoDiffXd>, kNumLocations> dI2_dF = data.dI2dF();
  std::array<Matrix3<AutoDiffXd>, kNumLocations> dI3_dF = data.dI3dF();
  std::array<Eigen::Matrix<AutoDiffXd, 9, 9>, kNumLocations> d2I1_dF2 =
      data.d2I1dF2();
  std::array<Eigen::Matrix<AutoDiffXd, 9, 9>, kNumLocations> d2I2_dF2 =
      data.d2I2dF2();
  std::array<Eigen::Matrix<AutoDiffXd, 9, 9>, kNumLocations> d2I3_dF2 =
      data.d2I3dF2();

  for (int q = 0; q < kNumLocations; ++q) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Matrix3d d2I1_dFijdF;
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            d2I1_dFijdF(k, l) = d2I1_dF2[q](3 * j + i, 3 * l + k).value();
          }
        }
        EXPECT_TRUE(CompareMatrices(
            Eigen::Map<const Matrix3d>(dI1_dF[q](i, j).derivatives().data(), 3,
                                       3),
            d2I1_dFijdF,
            CalcConditionNumberOfInvertibleMatrix<AutoDiffXd>(dI1_dF[q]) *
                kTolerance));
      }
    }
  }

  for (int q = 0; q < kNumLocations; ++q) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Matrix3d d2I2_dFijdF;
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            d2I2_dFijdF(k, l) = d2I2_dF2[q](3 * j + i, 3 * l + k).value();
          }
        }
        EXPECT_TRUE(CompareMatrices(
            Eigen::Map<const Matrix3d>(dI2_dF[q](i, j).derivatives().data(), 3,
                                       3),
            d2I2_dFijdF,
            CalcConditionNumberOfInvertibleMatrix<AutoDiffXd>(dI2_dF[q]) *
                kTolerance));
      }
    }
  }

  for (int q = 0; q < kNumLocations; ++q) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Matrix3d d2I3_dFijdF;
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            d2I3_dFijdF(k, l) = d2I3_dF2[q](3 * j + i, 3 * l + k).value();
          }
        }
        EXPECT_TRUE(CompareMatrices(
            Eigen::Map<const Matrix3d>(dI3_dF[q](i, j).derivatives().data(), 3,
                                       3),
            d2I3_dFijdF,
            CalcConditionNumberOfInvertibleMatrix<AutoDiffXd>(dI3_dF[q]) *
                kTolerance));
      }
    }
  }
}

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
