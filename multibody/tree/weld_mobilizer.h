#pragma once

#include <limits>
#include <memory>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/tree/frame.h"
#include "drake/multibody/tree/mobilizer_impl.h"
#include "drake/multibody/tree/multibody_tree_topology.h"
#include "drake/systems/framework/context.h"

namespace drake {
namespace multibody {
namespace internal {

// This mobilizer fixes the relative pose `X_FM` of an outboard frame M in an
// inboard frame F as if "welding" them together at this fixed relative pose.
// Therefore, this mobilizer has no associated state with it.
//
// @tparam_default_scalar
template <typename T>
class WeldMobilizer final : public MobilizerImpl<T, 0, 0> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(WeldMobilizer)

  // Constructor for a %WeldMobilizer between the `inboard_frame_F` and
  // `outboard_frame_M`.
  // @param[in] X_FM Pose of `outboard_frame_M` in the `inboard_frame_F`.
  WeldMobilizer(const Frame<T>& inboard_frame_F,
                const Frame<T>& outboard_frame_M,
                const math::RigidTransform<double>& X_FM) :
      MobilizerBase(inboard_frame_F, outboard_frame_M), X_FM_(X_FM) {}

  // @retval X_FM The pose of the outboard frame M in the inboard frame F.
  const math::RigidTransform<double>& get_X_FM() const { return X_FM_; }

  // Sets the default value of X_FM, the pose of the outboard frame M in the
  // inboard frame F.
  void set_X_FM(const math::RigidTransform<double>& X_FM) { X_FM_ = X_FM; }

  // Gets the value of X_FM, the pose of the outboard frame M in the
  // inboard frame F, stored in `context`.
  math::RigidTransform<T> get_X_FM(const systems::Context<T>& context) const {
    const systems::BasicVector<T>& X_FM_parameter =
        context.get_numeric_parameter(X_FM_parameter_index_);
    return math::RigidTransform<T>(Eigen::Map<const Eigen::Matrix<T, 3, 4>>(
        X_FM_parameter.get_value().data()));
  }

  // Sets the value of X_FM, the pose of the outboard frame M in the
  // inboard frame F, in `context`.
  void set_X_FM(systems::Context<T>* context,
                const math::RigidTransform<T>& X_FM) const {
    systems::BasicVector<T>& X_FM_parameter =
        context->get_mutable_numeric_parameter(X_FM_parameter_index_);
    X_FM_parameter.set_value(
        Eigen::Map<const VectorX<T>>(X_FM.GetAsMatrix34().data(), 12, 1));
  }

  // Computes the across-mobilizer transform `X_FM`, which for this mobilizer
  // is independent of the state stored in `context`.
  math::RigidTransform<T> CalcAcrossMobilizerTransform(
      const systems::Context<T>& context) const final;

  // Computes the across-mobilizer velocity `V_FM` which for this mobilizer is
  // always zero since the outboard frame M is fixed to the inboard frame F.
  SpatialVelocity<T> CalcAcrossMobilizerSpatialVelocity(
      const systems::Context<T>& context,
      const Eigen::Ref<const VectorX<T>>& v) const final;

  // Computes the across-mobilizer acceleration `A_FM` which for this mobilizer
  // is always zero since the outboard frame M is fixed to the inboard frame F.
  SpatialAcceleration<T> CalcAcrossMobilizerSpatialAcceleration(
      const systems::Context<T>& context,
      const Eigen::Ref<const VectorX<T>>& vdot) const final;

  // Since this mobilizer has no generalized velocities associated with it,
  // this override is a no-op.
  void ProjectSpatialForce(
      const systems::Context<T>& context,
      const SpatialForce<T>& F_Mo_F,
      Eigen::Ref<VectorX<T>> tau) const final;

  // This override is a no-op since this mobilizer has no generalized
  // velocities associated with it.
  void MapVelocityToQDot(
      const systems::Context<T>& context,
      const Eigen::Ref<const VectorX<T>>& v,
      EigenPtr<VectorX<T>> qdot) const final;

  // This override is a no-op since this mobilizer has no generalized
  // velocities associated with it.
  void MapQDotToVelocity(
      const systems::Context<T>& context,
      const Eigen::Ref<const VectorX<T>>& qdot,
      EigenPtr<VectorX<T>> v) const final;

 protected:
  void DoCalcNMatrix(const systems::Context<T>& context,
                     EigenPtr<MatrixX<T>> N) const final;

  void DoCalcNplusMatrix(
      const systems::Context<T>& context,
      EigenPtr<MatrixX<T>> Nplus) const final;

  std::unique_ptr<Mobilizer<double>> DoCloneToScalar(
      const MultibodyTree<double>& tree_clone) const final;

  std::unique_ptr<Mobilizer<AutoDiffXd>> DoCloneToScalar(
      const MultibodyTree<AutoDiffXd>& tree_clone) const final;

  std::unique_ptr<Mobilizer<symbolic::Expression>> DoCloneToScalar(
      const MultibodyTree<symbolic::Expression>& tree_clone) const final;

  // Implementation for MultibodyElement::DoDeclareParameters().
  // WeldMobilizer declares a single parameter for X_FM_.
  void DoDeclareParameters(
      internal::MultibodyTreeSystem<T>* tree_system) override {
    // Declare parent classes' parameters
    MobilizerImpl<T, 0, 0>::DoDeclareParameters(tree_system);

    X_FM_parameter_index_ = this->DeclareNumericParameter(
        tree_system,
        systems::BasicVector<T>(Eigen::Map<const VectorX<T>>(
            X_FM_.template cast<T>().GetAsMatrix34().data(), 12, 1)));
  }

 private:
  typedef MobilizerImpl<T, 0, 0> MobilizerBase;
  // Bring the handy number of position and velocities MobilizerImpl enums into
  // this class' scope. This is useful when writing mathematical expressions
  // with fixed-sized vectors since we can do things like Vector<T, nq>.
  // Operations with fixed-sized quantities can be optimized at compile time
  // and therefore they are highly preferred compared to the very slow dynamic
  // sized quantities.
  using MobilizerBase::kNq;
  using MobilizerBase::kNv;

  // Helper method to make a clone templated on ToScalar.
  template <typename ToScalar>
  std::unique_ptr<Mobilizer<ToScalar>> TemplatedDoCloneToScalar(
      const MultibodyTree<ToScalar>& tree_clone) const;

  // Pose of the outboard frame M in the inboard frame F.
  math::RigidTransform<double> X_FM_;

  // System parameter index the value of X_FM_ stored in the context.
  systems::NumericParameterIndex X_FM_parameter_index_;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::internal::WeldMobilizer)
