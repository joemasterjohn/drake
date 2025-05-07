#include "drake/multibody/contact_solvers/sap/sap_hunt_crossley_constraint.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/common/unused.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

template <typename T>
using InvariantData = typename SapHuntCrossleyConstraintData<T>::InvariantData;

template <typename T>
SapHuntCrossleyConstraint<T>::SapHuntCrossleyConstraint(
    ContactConfiguration<T> configuration, SapConstraintJacobian<T> J,
    Parameters params)
    : SapConstraint<T>(std::move(J),
                       {configuration.objectA, configuration.objectB}),
      parameters_(std::move(params)),
      configuration_(std::move(configuration)) {
  DRAKE_DEMAND(parameters().friction >= 0.0);
  DRAKE_DEMAND(parameters().dissipation >= 0.0);
  DRAKE_DEMAND(parameters().sigma >= 0.0);
  DRAKE_DEMAND(parameters().stiction_tolerance > 0.0);
  DRAKE_DEMAND(this->jacobian().rows() == 3);
  if (parameters().speculative) {
    using std::abs;
    const auto& speculative = *parameters().speculative;
    DRAKE_DEMAND(speculative.volume_factor >= 0.0);
    DRAKE_DEMAND(abs(speculative.cos_theta) <= 1.0);
    DRAKE_DEMAND(speculative.distance0 >= 0.0);
    DRAKE_DEMAND(speculative.toc >= 0.0);
    DRAKE_DEMAND(speculative.kappa >= 0.0);
  } else {
    DRAKE_DEMAND(parameters().stiffness >= 0.0);
  }
}

template <typename T>
std::unique_ptr<AbstractValue> SapHuntCrossleyConstraint<T>::MakeConstraintData(
    const T& time_step,
    const Eigen::Ref<const VectorX<T>>& delassus_estimation) const {
  DRAKE_DEMAND(!is_speculative());
  using std::max;

  const T& mu = parameters().friction;
  const T& d = parameters().dissipation;
  const double stiction_tolerance = parameters().stiction_tolerance;

  // Similar to SAP's sigma parameter.
  const double sigma = parameters().sigma;

  // Estimate a w_rms guaranteed to be larger than zero.
  const T w_rms = delassus_estimation.norm() / sqrt(3.0);

  // Compute tangent and normal scaling from Delassus estimation, bounded with
  // w_rms to avoid zero values.
  const T wt =
      max(0.5 * (delassus_estimation(0) + delassus_estimation(1)), w_rms);
  const T Rt = sigma * wt;  // SAP's regularization.

  SapHuntCrossleyConstraintData<T> data;
  typename SapHuntCrossleyConstraintData<T>::InvariantData& p =
      data.invariant_data;
  p.dt = time_step;
  const T& fe0 = configuration_.fe;
  const T& vn0 = configuration_.vn;
  const T damping = max(0.0, 1.0 - d * vn0);
  const T ne0 = max(0.0, time_step * fe0);
  p.n0 = ne0 * damping;
  const T sap_stiction_tolerance = mu * Rt * p.n0;
  p.epsilon_soft = max(stiction_tolerance, sap_stiction_tolerance);

  return AbstractValue::Make(data);
}

template <typename T>
std::unique_ptr<AbstractValue>
SapHuntCrossleyConstraint<T>::MakeSpeculativeConstraintData(
    const T& time_step,
    const Eigen::Ref<const VectorX<T>>& delassus_estimation) const {
  DRAKE_DEMAND(is_speculative());
  using std::abs;
  unused(delassus_estimation);

  const Parameters& params = parameters();
  const SpeculativeParameters& s = *params.speculative;
  const T& dt = time_step;                    // Time step, seconds.
  const T& abs_cos_theta = abs(s.cos_theta);  // = |n̂⋅ẑ|.
  const T& z0 = -s.distance0;                 // Penetration along ẑ, meters.
  const T& toc = s.toc;                       // Time of contact, seconds.
  const T& kappa = s.kappa;                   // Pressure gradient, in N/m³.

  SapHuntCrossleyConstraintData<T> data;
  typename SapHuntCrossleyConstraintData<T>::InvariantData& p =
      data.invariant_data;
  p.dt = dt;
  // N.B. Compared to non-speculative constraint, fe0 = 0 always. Therefore the
  // SAP regularization estimate is zero.
  p.epsilon_soft = params.stiction_tolerance;

  if (toc < dt) {
    // Method 1:
    // TODO(amcastro-tri): Consider using Method 2 below always.
    p.a = (dt - toc) * abs_cos_theta;
    p.b = 0.0;
  } else {
    // Method 2:
    // Sometimes we'd allow toc > dt by a safety factor.
    p.a = dt * abs_cos_theta;
    p.b = z0;
  }
  // p.c = dt * kappa * s.volume_factor;
  p.c = dt * kappa * s.effective_radius * std::numbers::pi;
  p.d = params.dissipation;

  return AbstractValue::Make(data);
}

template <typename T>
std::unique_ptr<AbstractValue> SapHuntCrossleyConstraint<T>::DoMakeData(
    const T& time_step,
    const Eigen::Ref<const VectorX<T>>& delassus_estimation) const {
  if (is_speculative()) {
    return MakeSpeculativeConstraintData(time_step, delassus_estimation);
  }
  return MakeConstraintData(time_step, delassus_estimation);
}

template <typename T>
T SapHuntCrossleyConstraint<T>::CalcHuntCrossleyAntiderivative(
    const T& dt, const T& vn) const {
  DRAKE_DEMAND(!is_speculative());

  using std::min;

  // The discrete impulse is modeled as:
  //   n(v; fₑ₀) = (fₑ₀ - δt⋅k⋅v)₊⋅(1 - d⋅v)₊.
  // We see that n(v; fe0) = 0 for v ≥ v̂, with v̂ = min(vx, vd) and:
  //  vx = x₀/δt
  //  vd = 1/d
  // Then for v < v̂, n(v; fₑ₀) is positive and we can verify that:
  //   N⁺(v; fe₀) = δt⋅[v⋅(fₑ₀ + 1/2⋅Δf)-d⋅v²/2⋅(fₑ₀ + 2/3⋅Δf)],
  // with Δf = -δt⋅k⋅v, is its antiderivative.
  // Since n(v; fₑ₀) = 0 for v ≥ v̂, then N(v; fₑ₀) must be constant for v ≥ v̂.
  // Therefore we define it as:
  //   N(v; fₑ₀) = N⁺(min(vn, v̂); fₑ₀)

  // Parameters:
  const T& k = parameters().stiffness;
  const T& d = parameters().dissipation;
  const T& fe0 = configuration_.fe;

  // We define the "dissipation" velocity vd at which the dissipation term
  // vanishes using a small tolerance so that vd goes to a very large number in
  // the limit to d = 0.
  const T vd = 1.0 / (d + 1.0e-20);

  // Similarly, we define vx as the velocity at which the elastic term goes
  // to zero. Using a small tolerance so that it goes to a very large
  // number in the limit to k = 0 (e.g. from discrete hydroelastic).
  const T vx = fe0 / dt / (k + 1.0e-20);

  // With the tolerances above in vd and vx, we can define a v̂ that goes to a
  // large number in the limit to either d = 0 or k = 0.
  const T v_hat = min(vx, vd);

  // Clamp vn to v̂.
  const T vn_clamped = min(vn, v_hat);

  // From the derivation above, N(v; fₑ₀) = N⁺(vn_clamped; fₑ₀).
  const T& v = vn_clamped;  // Alias to shorten notation.
  const T df = -dt * k * v;
  const T N = dt * (v * (fe0 + 1.0 / 2.0 * df) -
                    d * v * v / 2.0 * (fe0 + 2.0 / 3.0 * df));

  return N;
}

template <typename T>
T SapHuntCrossleyConstraint<T>::CalcSpeculativeHuntCrossleyAntiderivative(
    const InvariantData<T>& data, const T& vn) const {
  DRAKE_DEMAND(is_speculative());
  using std::min;

  const T& a = data.a;
  const T& b = data.b;
  const T& c = data.c;
  const T& d = data.d;

  // We define these velocities which the force goes to zero using a small
  // tolerance so that they go to a very large number in the limit to d = 0, or
  // a = 0.
  const T v_dis = 1.0 / (d + 1.0e-20);  // Dissipation goes to zero.
  const T v_vol = b / (a + 1.0e-20);    // Volume goes to zero.
  const T v_hat = min(v_dis, v_vol);

  // Clamp vn to v̂.
  const T vn_clamped = min(vn, v_hat);

  auto N_plus = [&a, &b, &c, &d](const T& v) {
    const T z = b - a * v;
    // const T z4 = z * z * z * z;
    const T z3 = z*z*z;
    const T a2 = a * a;
    // const T N = c / a2 * z4 * ((d * b - a) / 4.0 - d * z / 5.0);
    const T N = c / a2 * z3 * ((d * b - a) / 3.0 - d * z / 4.0);
    return N;
  };

  return N_plus(vn_clamped);
}

template <typename T>
T SapHuntCrossleyConstraint<T>::CalcHuntCrossleyImpulse(const T& dt,
                                                        const T& vn) const {
  DRAKE_DEMAND(!is_speculative());

  // Parameters:
  const T& k = parameters().stiffness;
  const T& d = parameters().dissipation;
  const T& fe0 = configuration_.fe;

  // Penetration and rate:
  const T xdot = -vn;
  const T fe = fe0 + dt * k * xdot;
  if (fe <= 0.0) return 0.0;
  const T damping = 1.0 + d * xdot;
  if (damping <= 0.0) return 0.0;
  const T gamma = dt * fe * damping;

  return gamma;
}

template <typename T>
T SapHuntCrossleyConstraint<T>::CalcSpeculativeHuntCrossleyImpulse(
    const InvariantData<T>& data, const T& vn) const {
  DRAKE_DEMAND(is_speculative());

  const T z = data.b - data.a * vn;
  if (z <= 0.0) return 0.0;

  const T damping = 1.0 - data.d * vn;
  if (damping <= 0.0) return 0.0;

  const T volume = z * z;
  const T gamma = data.c * volume * damping;

  return gamma;
}

template <typename T>
T SapHuntCrossleyConstraint<T>::CalcSpeculativeHuntCrossleyImpulseGradient(
    const InvariantData<T>& data, const T& vn) const {
  DRAKE_DEMAND(is_speculative());
  const T& a = data.a;
  const T& b = data.b;
  const T& c = data.c;
  const T& d = data.d;

  const T z = b - a * vn;
  if (z <= 0.0) return 0.0;

  const T damping = 1.0 - d * vn;
  if (damping <= 0.0) return 0.0;

  // const T z2 = z * z;
  // const T dn_dvn = -c * z2 * (3.0 * a * damping + d * z);
  const T dn_dvn = -c * z * (2.0 * a * damping + d * z);
  return dn_dvn;
}

template <typename T>
T SapHuntCrossleyConstraint<T>::CalcHuntCrossleyImpulseGradient(
    const T& dt, const T& vn) const {
  DRAKE_DEMAND(!is_speculative());

  // Parameters:
  const T& k = parameters().stiffness;
  const T& d = parameters().dissipation;
  const T& fe0 = configuration_.fe;

  const T xdot = -vn;                // Penetration rate.
  const T fe = fe0 + dt * k * xdot;  // Elastic force.

  // Quick exits.
  if (fe <= 0.0) return 0.0;
  const T damping = 1.0 + d * xdot;
  if (damping <= 0.0) return 0.0;

  // dn/dv = -δt⋅[k⋅δt⋅(1+d⋅ẋ) + d⋅(fe₀+δt⋅k⋅ẋ)]
  const T dn_dvn = -dt * (k * dt * damping + d * fe);

  return dn_dvn;
}

template <typename T>
void SapHuntCrossleyConstraint<T>::DoCalcData(
    const Eigen::Ref<const VectorX<T>>& vc,
    AbstractValue* abstract_data) const {
  auto& data =
      abstract_data->get_mutable_value<SapHuntCrossleyConstraintData<T>>();

  // Parameters:
  const T& mu = parameters().friction;
  const T& dt = data.invariant_data.dt;
  const T& epsilon_soft = data.invariant_data.epsilon_soft;

  // Computations dependent on vc.
  data.vc = vc;
  data.vn = vc[2];
  data.vt = vc.template head<2>();
  data.vt_soft = SoftNorm(data.vt, epsilon_soft);
  data.t_soft = data.vt / (data.vt_soft + epsilon_soft);
  switch (parameters().model) {
    case SapHuntCrossleyApproximation::kSimilar:
      data.z = data.vn - mu * data.vt_soft;
      break;
    case SapHuntCrossleyApproximation::kLagged:
      // This effectively evaluates n and N at z = vn for the lagged model.
      data.z = data.vn;
      break;
  }
  if (is_speculative()) {
    data.nz = CalcSpeculativeHuntCrossleyImpulse(data.invariant_data, data.z);
    data.Nz =
        CalcSpeculativeHuntCrossleyAntiderivative(data.invariant_data, data.z);
  } else {
    data.nz = CalcHuntCrossleyImpulse(dt, data.z);
    data.Nz = CalcHuntCrossleyAntiderivative(dt, data.z);
  }
}

template <typename T>
T SapHuntCrossleyConstraint<T>::DoCalcCost(
    const AbstractValue& abstract_data) const {
  const auto& data =
      abstract_data.get_value<SapHuntCrossleyConstraintData<T>>();
  switch (parameters().model) {
    case SapHuntCrossleyApproximation::kSimilar:
      return -data.Nz;
    case SapHuntCrossleyApproximation::kLagged:
      const T& mu = parameters().friction;
      const T& n0 = data.invariant_data.n0;
      const T& N = data.Nz;
      const T& vt_soft = data.vt_soft;
      return -N + mu * vt_soft * n0;
  }
  DRAKE_UNREACHABLE();
}

template <typename T>
void SapHuntCrossleyConstraint<T>::DoCalcImpulse(
    const AbstractValue& abstract_data, EigenPtr<VectorX<T>> gamma) const {
  const auto& data =
      abstract_data.get_value<SapHuntCrossleyConstraintData<T>>();
  const T& mu = parameters().friction;
  const T& n0 = data.invariant_data.n0;
  const T& n = data.nz;
  const Vector2<T>& t_soft = data.t_soft;
  // Value of n(vn) used in the friction model.
  const T n_friction =
      parameters().model == SapHuntCrossleyApproximation::kSimilar ? n : n0;
  const Vector2<T> gt = -mu * n_friction * t_soft;
  *gamma << gt, n;
}

template <typename T>
void SapHuntCrossleyConstraint<T>::DoCalcCostHessian(
    const AbstractValue& abstract_data, MatrixX<T>* G) const {
  const auto& data =
      abstract_data.get_value<SapHuntCrossleyConstraintData<T>>();

  // Const data.
  const T& mu = parameters().friction;
  const T& dt = data.invariant_data.dt;
  const T& epsilon_soft = data.invariant_data.epsilon_soft;
  const T& n0 = data.invariant_data.n0;

  // Tangential velocity and its soft norm.
  const T vt_soft = data.vt_soft;
  const Vector2<T>& t_soft = data.t_soft;

  // n(z) and its derivative n'(z).
  const T& n = data.nz;
  const T np = is_speculative() ? CalcSpeculativeHuntCrossleyImpulseGradient(
                                      data.invariant_data, data.z)
                                : CalcHuntCrossleyImpulseGradient(dt, data.z);

  // Projection matrices.
  const Matrix2<T> P = t_soft * t_soft.transpose();
  const Matrix2<T> Pperp = Matrix2<T>::Identity() - P;

  Matrix2<T> Gt;
  Vector2<T> Gtn;
  switch (parameters().model) {
    case SapHuntCrossleyApproximation::kSimilar:
      Gt = -mu * mu * np * P + mu * n * Pperp / (vt_soft + epsilon_soft);
      Gtn = mu * np * t_soft;
      break;
    case SapHuntCrossleyApproximation::kLagged:
      // N.B. This term is zero for speculative constraints.
      // TODO(amcastro-tri): Consider to ignore friction for speculative
      // constraints so that they are always simple 1-equation constraints.
      Gt = mu * n0 * Pperp / (vt_soft + epsilon_soft);
      Gtn.setZero();
      break;
  }

  G->template topLeftCorner<2, 2>() = Gt;
  G->template topRightCorner<2, 1>() = Gtn;
  G->template bottomLeftCorner<1, 2>() = Gtn.transpose();
  (*G)(2, 2) = -np;
}

template <typename T>
void SapHuntCrossleyConstraint<T>::DoAccumulateSpatialImpulses(
    int i, const Eigen::Ref<const VectorX<T>>& gamma,
    SpatialForce<T>* F) const {
  const math::RotationMatrix<T>& R_WC = configuration_.R_WC;
  const Vector3<T> f_Bc_W = R_WC * gamma;
  const SpatialForce<T> F_Bc_W(Vector3<T>::Zero(), f_Bc_W);

  if (i == 0) {
    // Object A.
    const Vector3<T> p_CAp_W = -configuration_.p_ApC_W;
    // N.B. F_Ap = F_Ac.Shift(p_CAp) = -F_Bc.Shift(p_CAp)
    *F -= F_Bc_W.Shift(p_CAp_W);
  } else {
    // Object B.
    const Vector3<T> p_CBq_W = -configuration_.p_BqC_W;
    // N.B. F_Bq = F_Bc.Shift(p_CBq)
    *F += F_Bc_W.Shift(p_CBq_W);
    return;
  }
}

template <typename T>
std::unique_ptr<SapConstraint<double>>
SapHuntCrossleyConstraint<T>::DoToDouble() const {
  SapConstraintJacobian<double> J = this->jacobian().ToDouble();
  const auto& p = parameters();

  std::optional<SapHuntCrossleyConstraint<double>::SpeculativeParameters>
      s_double;
  if (p.speculative) {
    const auto& s = *p.speculative;
    s_double = SapHuntCrossleyConstraint<double>::SpeculativeParameters{
        ExtractDoubleOrThrow(s.kappa), ExtractDoubleOrThrow(s.volume_factor),
        ExtractDoubleOrThrow(s.cos_theta), ExtractDoubleOrThrow(s.distance0),
        ExtractDoubleOrThrow(s.toc)};
  }

  SapHuntCrossleyConstraint<double>::Parameters parameters{
      p.model,
      ExtractDoubleOrThrow(p.friction),
      ExtractDoubleOrThrow(p.stiffness),
      ExtractDoubleOrThrow(p.dissipation),
      p.stiction_tolerance,
      p.sigma,
      s_double};
  ContactConfiguration<double> configuration = configuration_.ToDouble();
  return std::make_unique<SapHuntCrossleyConstraint<double>>(
      std::move(configuration), std::move(J), std::move(parameters));
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::contact_solvers::internal::
        SapHuntCrossleyConstraint);
