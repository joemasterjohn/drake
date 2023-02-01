#pragma once

#include <array>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/deformation_gradient_data.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* Data supporting calculations in MooneyRivlinModel.
 See DeformationGradientData for more about constitutive model data.
 @tparam_nonsymbolic_scalar
 @tparam num_locations Number of locations at which the deformation gradient
 dependent quantities are evaluated. */
template <typename T, int num_locations>
class MooneyRivlinModelData
    : public DeformationGradientData<MooneyRivlinModelData<T, num_locations>> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MooneyRivlinModelData);

  /* Constructs a MooneyRivlinModelData with no deformation. */
  MooneyRivlinModelData();

  /* I1 = tr(F^T * F). aka the trace of the cauchy green tensor. aka the sum of
   square of singular values of F. */
  const std::array<T, num_locations>& I1() const { return I1_; }
  /* I2 = (F^T * F).squaredNorm(). aka the squared frobenius norm of the cauchy
   green tensor. */
  const std::array<T, num_locations>& I2() const { return I2_; }
  /* I3 = J = det(F). */
  const std::array<T, num_locations>& I3() const { return I3_; }

  const std::array<Matrix3<T>, num_locations>& dI1dF() const { return dI1_dF_; }
  const std::array<Matrix3<T>, num_locations>& dI2dF() const { return dI2_dF_; }
  const std::array<Matrix3<T>, num_locations>& dI3dF() const { return dI3_dF_; }

  const std::array<Eigen::Matrix<T, 9, 9>, num_locations>& d2I1dF2() const {
    return d2I1_dF2_;
  }
  const std::array<Eigen::Matrix<T, 9, 9>, num_locations>& d2I2dF2() const {
    return d2I2_dF2_;
  }
  const std::array<Eigen::Matrix<T, 9, 9>, num_locations>& d2I3dF2() const {
    return d2I3_dF2_;
  }

 private:
  friend DeformationGradientData<MooneyRivlinModelData<T, num_locations>>;

  /* Shadows DeformationGradientData::UpdateFromDeformationGradient() as
   required by the CRTP base class. */
  void UpdateFromDeformationGradient();

  std::array<T, num_locations> I1_;
  std::array<T, num_locations> I2_;
  std::array<T, num_locations> I3_;

  std::array<Matrix3<T>, num_locations> dI1_dF_;
  std::array<Matrix3<T>, num_locations> dI2_dF_;
  std::array<Matrix3<T>, num_locations> dI3_dF_;

  std::array<Eigen::Matrix<T, 9, 9>, num_locations> d2I1_dF2_;
  std::array<Eigen::Matrix<T, 9, 9>, num_locations> d2I2_dF2_;
  std::array<Eigen::Matrix<T, 9, 9>, num_locations> d2I3_dF2_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
