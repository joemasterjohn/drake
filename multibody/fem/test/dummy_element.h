#pragma once

#include <array>
#include <utility>
#include <vector>

#include "drake/multibody/fem/damping_model.h"
#include "drake/multibody/fem/fem_element.h"
#include "drake/multibody/fem/linear_constitutive_model.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* Forward declaration needed for defining the Traits below. */
class DummyElement;

struct DummyData {
  using T = double;
  static constexpr int num_dofs = 12;
  static constexpr int num_nodes = 4;
  double value{0};
  Vector<T, num_dofs> element_q;
  Vector<T, num_dofs> element_v;
  Vector<T, num_dofs> element_a;
  Vector<T, num_dofs> inverse_dynamics_force;
  Eigen::Matrix<T, num_dofs, num_dofs> mass_matrix;
  Eigen::Matrix<T, num_dofs, num_dofs> stiffness_matrix;
  Vector<T, num_nodes> strain_measure;
  Vector<T, num_nodes> lumped_mass;
};

/* The traits for the DummyElement. In this case, all of the traits are unique
 values so we can detect that each value is used in the expected context. */
template <>
struct FemElementTraits<DummyElement> {
  using T = double;
  static constexpr int num_quadrature_points = 1;
  static constexpr int num_natural_dimension = 2;
  static constexpr int num_nodes = 4;
  /* See `DoComputeData` below on how this dummy data is updated. */
  using Data = DummyData;
  static constexpr int num_dofs = Data::num_dofs;
  using ConstitutiveModel = LinearConstitutiveModel<T, num_quadrature_points>;
};

/* A simple FemElement implementation. The calculation methods are implemented
 as returning a fixed value (which can independently be accessed by calling
 the corresponding dummy method -- e.g., CalcInverseDynamics() should return
 the value in inverse_dynamics_force(). */
class DummyElement final : public FemElement<DummyElement> {
 public:
  using Base = FemElement<DummyElement>;
  using Traits = FemElementTraits<DummyElement>;
  using ConstitutiveModel = typename Traits::ConstitutiveModel;
  using T = typename Base::T;
  static constexpr int kNumDofs = Traits::num_dofs;

  DummyElement(const std::array<FemNodeIndex, Traits::num_nodes>& node_indices,
               ConstitutiveModel constitutive_model,
               DampingModel<T> damping_model)
      : Base(node_indices, std::move(constitutive_model),
             std::move(damping_model)) {}

  static Vector<T, kNumDofs> inverse_dynamics_force() {
    return Vector<T, kNumDofs>::Constant(1.23456);
  }

  static Eigen::Matrix<T, kNumDofs, kNumDofs> stiffness_matrix() {
    Eigen::Matrix<T, kNumDofs, kNumDofs> A;
    for (int i = 0; i < kNumDofs; ++i) {
      for (int j = 0; j < kNumDofs; ++j) {
        A(i, j) = 2.7 * i + 3.1 * j;
      }
    }
    /* A + A^T is guaranteed PSD. Adding the identity matrix to it makes it SPD.
     */
    return (A + A.transpose()) +
           Eigen::Matrix<T, kNumDofs, kNumDofs>::Identity();
  }

  static Eigen::Matrix<T, kNumDofs, kNumDofs> mass_matrix() {
    return 0.5 * stiffness_matrix();
  }

  /* Dummy element provides zero gravity force so that we have complete control
   over what the residual is. */
  void AddScaledGravityForce(const Data&, const T&, const Vector3<T>&,
                             EigenPtr<Vector<T, num_dofs>>) const {}

 private:
  /* Friend the base class so that the interface in the CRTP base class can
   access the private implementations of this class. */
  friend Base;

  /* Implements FemElement::ComputeData(). Returns the sum of the last entries
   in each state. */
  typename Traits::Data DoComputeData(const FemState<T>& state) const {
    const int state_dofs = state.num_dofs();
    const auto& q = state.GetPositions();
    const auto& v = state.GetVelocities();
    const auto& a = state.GetAccelerations();
    typename Traits::Data data;
    data.value = q(state_dofs - 1);
    data.value += v(state_dofs - 1);
    data.value += a(state_dofs - 1);
    data.element_q = this->ExtractElementDofs(q);
    data.element_v = this->ExtractElementDofs(v);
    data.element_a = this->ExtractElementDofs(a);
    if (data.element_v.norm() == 0.0 && data.element_a.norm() == 0.0) {
      data.inverse_dynamics_force = inverse_dynamics_force();
    } else {
        data.inverse_dynamics_force.setZero();
    }
    data.mass_matrix = mass_matrix();
    data.stiffness_matrix = stiffness_matrix();
    return data;
  }
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
