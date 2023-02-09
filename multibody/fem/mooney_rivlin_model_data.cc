#include "drake/multibody/fem/mooney_rivlin_model_data.h"

#include "drake/common/autodiff.h"
#include "drake/multibody/fem/matrix_utilities.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T, int num_locations>
MooneyRivlinModelData<T, num_locations>::MooneyRivlinModelData() {
  std::fill(I1_.begin(), I1_.end(), 3.0);
  std::fill(I2_.begin(), I2_.end(), 3.0);
  std::fill(I3_.begin(), I3_.end(), 1.0);

  std::fill(dI1_dF_.begin(), dI1_dF_.end(), 2.0 * Matrix3<T>::Identity());
  std::fill(dI2_dF_.begin(), dI2_dF_.end(), Matrix3<T>::Zero());
  std::fill(dI3_dF_.begin(), dI3_dF_.end(), Matrix3<T>::Identity());

  std::fill(d2I1_dF2_.begin(), d2I1_dF2_.end(),
            2.0 * Eigen::Matrix<T, 9, 9>::Identity());
  // TODO(xuchenhan-tri): properly initialize d2I2_dF2_ and d2I3_dF2.
  std::fill(d2I2_dF2_.begin(), d2I2_dF2_.end(), Eigen::Matrix<T, 9, 9>::Zero());
  std::fill(d2I3_dF2_.begin(), d2I3_dF2_.end(), Eigen::Matrix<T, 9, 9>::Zero());
}

template <typename T, int num_locations>
void MooneyRivlinModelData<T, num_locations>::UpdateFromDeformationGradient() {
  const std::array<Matrix3<T>, num_locations>& F = this->deformation_gradient();
  for (int i = 0; i < num_locations; ++i) {
    const Matrix3<T> FTF = F[i].transpose() * F[i];
    const Matrix3<T> FFT = F[i] * F[i].transpose();
    strain_[i] = 0.5 * (FTF - Matrix3<T>::Identity());

    I1_[i] = F[i].squaredNorm();
    I2_[i] = FTF.squaredNorm();
    I3_[i] = F[i].determinant();

    dI1_dF_[i] = 2.0 * F[i];
    dI2_dF_[i] = 4.0 * F[i] * FTF;
    /* Derivative of the determinant is the cofactor matrix. */
    Matrix3<T>& C = dI3_dF_[i];
    internal::CalcCofactorMatrix<T>(F[i], &C);

    d2I1_dF2_[i] = 2.0 * Eigen::Matrix<T, 9, 9>::Identity();
    // TODO(xuchenhan-tri): KroneckerProduct with identity matrix can be
    // implemented more efficiently than for generic matrices.
    d2I2_dF2_[i] = 4.0 * (KroneckerProduct<T, 3>(Matrix3<T>::Identity(), FFT) +
                          KroneckerProduct<T, 3>(FTF, Matrix3<T>::Identity()));
    /* Add in the D matrix. */
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        d2I2_dF2_[i].template block<3, 3>(3 * a, 3 * b) +=
            4.0 * F[i].col(b) * F[i].col(a).transpose();
      }
    }
    d2I3_dF2_[i].setZero();
    internal::AddScaledCofactorMatrixDerivative<T>(F[i], 1.0, &d2I3_dF2_[i]);
  }
}

template class MooneyRivlinModelData<double, 1>;
template class MooneyRivlinModelData<AutoDiffXd, 1>;

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
