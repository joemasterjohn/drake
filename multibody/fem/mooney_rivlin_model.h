#pragma once

#include <array>

#include "drake/common/unused.h"
#include "drake/multibody/fem/constitutive_model.h"
#include "drake/multibody/fem/mooney_rivlin_model_data.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* Traits for MooneyRivlinModel. */
template <typename T, int num_locations>
struct MooneyRivlinModelTraits {
  using Scalar = T;
  using Data = MooneyRivlinModelData<T, num_locations>;
};

/* Implements the Mooney-Rivlin hyperelastic constitutive model. The original
 Mooney-Rivlin model is incompressible and the compressible Mooney-Rivlin model
 is not rest stable (i.e non-zero stress at rest configuration). Hence, we adopt
 the stable Mooney-Rivlin model in [insert reference].
 @tparam_nonsymbolic_scalar
 @tparam num_locations Number of locations at which the constitutive
 relationship is evaluated. We currently only provide one instantiation of this
 template with `num_locations = 1`, but more instantiations can easily be added
 when needed. */

template <typename T, int num_locations>
class MooneyRivlinModel final
    : public ConstitutiveModel<MooneyRivlinModel<T, num_locations>,
                               MooneyRivlinModelTraits<T, num_locations>> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MooneyRivlinModel)

  using Traits = MooneyRivlinModelTraits<T, num_locations>;
  using Data = typename Traits::Data;

  /* Constructs a MooneyRivlinModel constitutive model with prescribed
   parameters.
   @param kappa is the bulk modulus.
   @pre All paremters are positive. */
  MooneyRivlinModel(const T& mu0, const T& mu1, const T& lambda);

  // TODO(xuchenhan-tri): This is a hack to facilitate testing. It creates an
  // arbitrary MooneyRivlinModel with valid parameters.
  MooneyRivlinModel(const T& E, const T& nu) : MooneyRivlinModel(293, 177, 1410) {
    unused(E, nu);
  }

 private:
  friend ConstitutiveModel<MooneyRivlinModel<T, num_locations>,
                           MooneyRivlinModelTraits<T, num_locations>>;

  /* Shadows ConstitutiveModel::CalcElasticEnergyDensityImpl() as required by
   the CRTP base class. */
  void CalcElasticEnergyDensityImpl(const Data& data,
                                    std::array<T, num_locations>* Psi) const;

  /* Shadows ConstitutiveModel::CalcFirstPiolaStressImpl() as required by the
   CRTP base class. */
  void CalcFirstPiolaStressImpl(const Data& data,
                                std::array<Matrix3<T>, num_locations>* P) const;

  /* Shadows ConstitutiveModel::CalcFirstPiolaStressDerivativeImpl() as required
   by the CRTP base class. */
  void CalcFirstPiolaStressDerivativeImpl(
      const Data& data,
      std::array<Eigen::Matrix<T, 9, 9>, num_locations>* dPdF) const;

  T mu0_;
  T mu1_;
  T lambda_;
  T alpha_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
