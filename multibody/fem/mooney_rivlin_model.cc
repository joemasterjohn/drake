#include "drake/multibody/fem/mooney_rivlin_model.h"

#include <array>
#include <utility>

#include "drake/common/autodiff.h"
#include "drake/multibody/fem/matrix_utilities.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T, int num_locations>
MooneyRivlinModel<T, num_locations>::MooneyRivlinModel(const T& mu0,
                                                       const T& mu1,
                                                       const T& lambda)
    : mu0_(mu0),
      mu1_(mu1),
      lambda_(lambda),
      alpha_((2.0 * mu0 + 4.0 * mu1) / lambda + 1.0) {}

template <typename T, int num_locations>
void MooneyRivlinModel<T, num_locations>::CalcElasticEnergyDensityImpl(
    const Data& data, std::array<T, num_locations>* Psi) const {
  for (int i = 0; i < num_locations; ++i) {
    const T& I1 = data.I1()[i];
    const T& I2 = data.I2()[i];
    const T& I3 = data.I3()[i];
    (*Psi)[i] = mu0_ * (I1 - 3.0) + 0.5 * mu1_ * (I1 * I1 - I2 - 3.0) +
                0.5 * lambda_ * (I3 - alpha_) * (I3 - alpha_) - 1.5 * mu1_ -
                0.5 * lambda_ * (1.0 - alpha_) * (1.0 - alpha_);
  }
}

template <typename T, int num_locations>
void MooneyRivlinModel<T, num_locations>::CalcFirstPiolaStressImpl(
    const Data& data, std::array<Matrix3<T>, num_locations>* P) const {
  for (int i = 0; i < num_locations; ++i) {
    const Matrix3<T>& dI1_dF = data.dI1dF()[i];
    const Matrix3<T>& dI2_dF = data.dI2dF()[i];
    const Matrix3<T>& dI3_dF = data.dI3dF()[i];

    const T& I1 = data.I1()[i];
    const T& I3 = data.I3()[i];

    const T dPsi_dI1 = mu0_ + mu1_ * I1;
    const T dPsi_dI2 = -0.5 * mu1_;
    const T dPsi_dI3 = lambda_ * (I3 - alpha_);

    (*P)[i] = dPsi_dI1 * dI1_dF + dPsi_dI2 * dI2_dF + dPsi_dI3 * dI3_dF;
  }
}

template <typename T, int num_locations>
void MooneyRivlinModel<T, num_locations>::CalcFirstPiolaStressDerivativeImpl(
    const Data& data,
    std::array<Eigen::Matrix<T, 9, 9>, num_locations>* dPdF) const {
  for (int q = 0; q < num_locations; ++q) {
    const T& I1 = data.I1()[q];
    const T& I3 = data.I3()[q];

    const T dPsi_dI1 = mu0_ + mu1_ * I1;
    const T dPsi_dI2 = -0.5 * mu1_;
    const T dPsi_dI3 = lambda_ * (I3 - alpha_);

    const T& d2Psi_dI1dI1 = mu1_;
    const T& d2Psi_dI3dI3 = lambda_;

    const Matrix3<T>& dI1_dF = data.dI1dF()[q];
    const Matrix3<T>& dI3_dF = data.dI3dF()[q];

    const Eigen::Matrix<T, 9, 9>& d2I1_dF2 = data.d2I1dF2()[q];
    const Eigen::Matrix<T, 9, 9>& d2I2_dF2 = data.d2I2dF2()[q];
    const Eigen::Matrix<T, 9, 9>& d2I3_dF2 = data.d2I3dF2()[q];

    (*dPdF)[q] =
        dPsi_dI1 * d2I1_dF2 + dPsi_dI2 * d2I2_dF2 + dPsi_dI3 * d2I3_dF2;
    /* Add in d2Psi_dI1dI1 term and d2Psi_dI3dI3 term. */
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) {
            (*dPdF)[q](3 * j + i, 3 * l + k) +=
                d2Psi_dI1dI1 * dI1_dF(i, j) * dI1_dF(k, l) +
                d2Psi_dI3dI3 * dI3_dF(i, j) * dI3_dF(k, l);
          }
        }
      }
    }
  }
}

template class MooneyRivlinModel<double, 1>;
template class MooneyRivlinModel<AutoDiffXd, 1>;

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
