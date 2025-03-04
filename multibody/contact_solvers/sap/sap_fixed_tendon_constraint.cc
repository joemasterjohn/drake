#include "drake/multibody/contact_solvers/sap/sap_fixed_tendon_constraint.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <typename T>
SapFixedTendonConstraint<T>::Parameters::Parameters(const T& lower_limit,
                                                    const T& upper_limit,
                                                    const T& stiffness,
                                                    const T& damping,
                                                    double beta)
    : lower_limit_(lower_limit),
      upper_limit_(upper_limit),
      stiffness_(stiffness),
      damping_(damping),
      beta_(beta) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  DRAKE_DEMAND(lower_limit < kInf);
  DRAKE_DEMAND(upper_limit > -kInf);
  DRAKE_DEMAND(lower_limit <= upper_limit);
  DRAKE_DEMAND(stiffness > 0);
  DRAKE_DEMAND(damping >= 0);
}

template <typename T>
SapFixedTendonConstraint<T>::Kinematics::Kinematics(
    int clique0, int clique1, int clique0_nv, int clique1_nv,
    Eigen::SparseVector<T> q0, Eigen::SparseVector<T> q1,
    Eigen::SparseVector<T> a0, Eigen::SparseVector<T> a1)
    : clique0_(clique0),
      clique1_(clique1),
      clique0_nv_(clique0_nv),
      clique1_nv_(clique1_nv),
      q0_(std::move(q0)),
      q1_(std::move(q1)),
      a0_(std::move(a0)),
      a1_(std::move(a1)) {
  DRAKE_DEMAND(clique0_ >= 0);
  DRAKE_DEMAND(clique1_ >= 0);
  DRAKE_DEMAND(clique0_ != clique1_);
  DRAKE_DEMAND(clique0_nv_ >= 0);
  DRAKE_DEMAND(clique1_nv_ >= 0);
  DRAKE_DEMAND(q0_.size() == clique0_nv_);
  DRAKE_DEMAND(q1_.size() == clique1_nv_);
  DRAKE_DEMAND(a0_.size() == clique0_nv_);
  DRAKE_DEMAND(a1_.size() == clique1_nv_);
}

template <typename T>
SapFixedTendonConstraint<T>::Kinematics::Kinematics(int clique0, int clique0_nv,
                                                    Eigen::SparseVector<T> q0,
                                                    Eigen::SparseVector<T> a0)
    : clique0_(clique0),
      clique0_nv_(clique0_nv),
      q0_(std::move(q0)),
      a0_(std::move(a0)) {
  DRAKE_DEMAND(clique0_ >= 0);
  DRAKE_DEMAND(clique0_ != clique1_);
  DRAKE_DEMAND(clique0_nv_ >= 0);
  DRAKE_DEMAND(q0_.size() == clique0_nv_);
  DRAKE_DEMAND(a0_.size() == clique0_nv_);
}

template <typename T>
SapFixedTendonConstraint<T>::SapFixedTendonConstraint(Parameters parameters,
                                                      Kinematics kinematics)
    : SapConstraint<T>(CalcConstraintJacobian(parameters, kinematics), {}),
      g_(CalcConstraintFunction(parameters, kinematics)),
      parameters_(std::move(parameters)) {}

template <typename T>
SapFixedTendonConstraint<T>::SapFixedTendonConstraint(
    VectorX<T> g, SapConstraintJacobian<T> J, Parameters parameters)
    : SapConstraint<T>(std::move(J), {}),
      g_(std::move(g)),
      parameters_(std::move(parameters)) {
  DRAKE_THROW_UNLESS(g_.size() <= 2);
  DRAKE_THROW_UNLESS(g_.size() == this->jacobian().rows());
}

template <typename T>
VectorX<T> SapFixedTendonConstraint<T>::CalcConstraintFunction(
    const Parameters& parameters, const Kinematics& kinematics) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const T& ll = parameters.lower_limit();
  const T& ul = parameters.upper_limit();

  const int nk = ll > -kInf && ul < kInf ? 2 : 1;
  VectorX<T> g0(nk);

  int i = 0;
  if (ll > -kInf) {
    g0(i) = kinematics.a0_.dot(kinematics.q0_) + kinematics.offset_ - ll;
    if (kinematics.clique_1_ >= 0) {
      g0(i) += kinematics.a1_.dot(kinematics.q1_);
    }
    ++i;
  }
  if (ul < kInf) {
    g0(i) = ul - kinematics.a0_.dot(kinematics.q0_) - kinematics.offset_;
    if (kinematics.clique_1_ >= 0) {
      g0(i) -= kinematics.a1_.dot(kinematics.q1_);
    }
  }

  return g0;
}

template <typename T>
SapConstraintJacobian<T> SapFixedTendonConstraint<T>::CalcConstraintJacobian(
    const Parameter& parameters, const Kinematics& kinematics) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const T& ll = parameters.lower_limit();
  const T& ul = parameters.upper_limit();

  const int nk = ll > -kInf && ul < kInf ? 2 : 1;
  MatrixX<T> J0 = MatrixX<T>::Zero(nk, kinematics.clique0_nv_);

  int i = 0;
  if (ll > -kInf) J0.row(i++) += kinematics.a0_;
  if (ul < kInf) J0.row(i) -= kinematics.a0_;

  if (kinematics.clique1_ >= 0) {
    MatrixX<T> J1 = MatrixX<T>::Zero(nk, kinematics.clique1_nv_);

    i = 0;
    if (ll > -kInf) J1.row(i++) += kinematics.a1_;
    if (ul < kInf) J1.row(i) -= kinematics.a1_;

    return SapConstraintJacobian<T>(kinematics.clique0_, std::move(J0),
                                    kinematics.clique1_, std::move(J1));
  } else {
    return SapConstraintJacobian<T>(kinematics.clique0_, std::move(J0));
  }
}

template <typename T>
std::unique_ptr<AbstractValue> SapFixedTendonConstraint<T>::DoMakeData(
    const T& dt,
    const Eigen::Ref<const VectorX<T>>& delassus_estimation) const {
  // Estimate regularization based on near-rigid regime threshold.
  // Rigid approximation constant: Rₙ = β²/(4π²)⋅wᵢ when the contact frequency
  // ωₙ is below the limit ωₙ⋅δt ≤ 2π. That is, the period is Tₙ = β⋅δt. See
  // [Castro et al., 2021] for details.
  const double beta_factor =
      parameters_.beta() * parameters_.beta() / (4.0 * M_PI * M_PI);

  T k_eff = parameters_.stiffness();
  T taud_eff = parameters_.damping() / k_eff;

  // "Effective regularization" [Castro et al., 2021] for this constraint.
  const T R_eff = 1.0 / (dt * k_eff * (dt + taud_eff));

  // "Near-rigid" regularization, [Castro et al., 2021].
  VectorX<T> R = (beta_factor * delassus_estimation).cwiseMax(R_eff);

  // Make data.
  SapFixedTendonConstraintData<T> data;
  typename SapFixedTendonConstraintData<T>::InvariantData& p =
      data.invariant_data;
  p.dt = time_step;
  p.R_inv = R.cwiseInverse();
  p.v_hat = -g_ / (dt + taud);

  return AbstractValue::Make(data);
}

template <typename T>
void SapFixedTendonConstraint<T>::DoCalcData(
    const Eigen::Ref<const VectorX<T>>& v, AbstractValue* abstract_data) const {
  auto& data =
      abstract_data->get_mutable_value<SapFixedTendonConstraintData<T>>();

  const T& dt = data.dt;
  const VectorX<T>& v_hat = data.vhat;
  const VectorX<T>& R_inv = data.R_inv;

  // This constraint is formulated such that the cost, impulse, and hessian
  // are all zero when the constraint is not active.
  data.v_ = v;
  data.hessian_.setZero();
  data.gamma_.setZero();
  data.cost_.setZero();

  for (int i = 0; i < num_constraint_equations(); ++i) {
    const T v_tilde = std::min(v_hat(i), -g_(i) / dt);
    // Constraint is active when v < ṽ
    if (v(i) < v_tilde) {
      const T dv = v_hat(i) - v(i);
      const T dv_tilde = v_hat(i) - v_tilde;
      data.hessian()(i) = R_inv(i);
      data.gamma()(i) = R_inv(i) * dv;
      data.cost() += 0.5 * R_inv(i) * (dv * dv - dv_tilde * dv_tilde);
    }
  }
}

template <typename T>
T SapFixedTendonConstraint<T>::DoCalcCost(
    const AbstractValue& abstract_data) const {
  const auto& data = abstract_data.get_value<SapFixedTendonConstraintData<T>>();
  return data.cost();
}

template <typename T>
void SapFixedTendonConstraint<T>::DoCalcImpulse(
    const AbstractValue& abstract_data, EigenPtr<VectorX<T>> gamma) const {
  const auto& data = abstract_data.get_value<SapFixedTendonConstraintData<T>>();
  *gamma = data.gamma();
}

template <typename T>
void SapFixedTendonConstraint<T>::DoCalcCostHessian(
    const AbstractValue& abstract_data, MatrixX<T>* G) const {
  const auto& data = abstract_data.get_value<SapFixedTendonConstraintData<T>>();
  *G = data.hessian().asDiagonal();
}

template <typename T>
void SapFixedTendonConstraint<T>::DoAccumulateGeneralizedImpulses(
    int c, const Eigen::Ref<const VectorX<T>>& gamma,
    EigenPtr<VectorX<T>> tau) const {
  // For this constraint the generalized impulses are simply τ = Jᵀ⋅γ.
  if (c == 0) {
    *tau += first_clique_jacobian().transpose() * gamma;
  } else if (c == 1) {
    *tau += second_clique_jacobian().transpose() * gamma;
  } else {
    DRAKE_UNREACHABLE();
  }
}

template <typename T>
std::unique_ptr<SapConstraint<double>> SapFixedTendonConstraint<T>::DoToDouble()
    const {
  const typename SapFixedTendonConstraint<T>::Parameters& p = parameters_;

  SapFixedTendonConstraint<double>::Parameters p_to_double(
      math::DiscardGradient(p.lower_limit()),
      math::DiscardGradient(p.upper_limit()),
      math::DiscardGradient(p.stiffness()), math::DiscardGradient(p.damping()),
      p.beta());

  return std::make_unique<SapFixedTendonConstraint<double>>(
      math::DiscardGradient(constraint_function()), this->jacobian().ToDouble(),
      std::move(p_to_double));
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::
        SapFixedTendonConstraint);
