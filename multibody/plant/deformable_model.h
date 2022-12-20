#pragma once

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/identifier.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/fem/fem_model.h"
#include "drake/multibody/plant/deformable_indexes.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/physical_model.h"

namespace drake {
namespace multibody {

/** DeformableModel implements the interface in PhysicalModel and provides the
 functionalities to specify deformable bodies. Unlike rigid bodies, the shape of
 deformable bodies can change in a simulation. Each deformable body is modeled
 as a volumetric mesh with persisting topology, changing vertex positions, and
 an approximated signed distance field. A finite element model is built for each
 registered deformable body that is used to evaluate the dynamics of the body.
 @experimental
 @tparam_double_only */
template <typename T>
class DeformableModel final : public multibody::internal::PhysicalModel<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeformableModel)

  /** Constructs a DeformableModel to be owned by the given MultibodyPlant.
   @pre plant != nullptr.
   @pre Finalize() has not been called on `plant`. */
  explicit DeformableModel(MultibodyPlant<T>* plant) : plant_(plant) {
    DRAKE_DEMAND(plant_ != nullptr);
    DRAKE_DEMAND(!plant_->is_finalized());
  }

  /** Returns the number of deformable bodies registered with this
   DeformableModel. */
  int num_bodies() const { return reference_positions_.size(); }

  // TODO(xuchenhan-tri): Document the minimal requirement on the geometry
  //  instance. For example, it must have a friction proximity property to be
  //  simulated with an MbP that involves contact.
  // TODO(xuchenhan-tri): Consider allowing registering deformable bodies with
  //  non-world frames.
  /** Registers a deformable body in `this` DeformableModel with the given
   GeometryInstance. The body is represented in the world frame and simulated
   with FEM with linear elements and a first order quadrature rule that
   integrates linear functions exactly. See FemModel for details. Returns a
   unique identifier for the added geometry.
   @param[in] geometry_instance  The geometry to be registered with the model.
   @param[in] config             The physical properties of deformable body.
   @param[in] resolution_hint    The parameter that guides the level of mesh
                                 refinement of the deformable geometry. It has
                                 length units (in meters) and roughly
                                 corresponds to a typical edge length in the
                                 resulting mesh for a primitive shape.
   @pre resolution_hint > 0.
   @throws std::exception if Finalize() has been called on the multibody plant
   owning this deformable model. */
  DeformableBodyId RegisterDeformableBody(
      std::unique_ptr<geometry::GeometryInstance> geometry_instance,
      const fem::DeformableBodyConfig<T>& config, double resolution_hint);

  // TODO(xuchenhan-tri): Consider pulling PosedHalfSpace out of internal
  // namespace and use it here.
  /** Sets wall boundary conditions for the body with the given `id`. All
   vertices of the mesh of the deformable body whose reference positions are
   inside the prescribed open half space are put under zero displacement
   boundary conditions. The open half space is defined by a plane with outward
   normal n_W. A vertex V is considered to be subject to the boundary condition
   if n̂ ⋅ p_QV < 0 where Q is a point on the plane and n̂ is normalized n_W.
   @param[in] id    The body to be put under boundary condition.
   @param[in] p_WQ  The position of a point Q on the plane in the world frame.
   @param[in] n_W   Outward normal to the half space expressed in the world
                    frame.
   @pre n_W.norm() > 1e-10.
   @warning Be aware of round-off errors in floating computations when placing a
   vertex very close to the plane defining the half space.
   @throws std::exception if Finalize() has been called on the multibody plant
   owning this deformable model or if no deformable body with the given `id` has
   been registered in this model. */
  void SetWallBoundaryCondition(DeformableBodyId id, const Vector3<T>& p_WQ,
                                const Vector3<T>& n_W);

  /** Returns the discrete state index of the deformable body identified by the
   given `id`.
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet.
   or if no deformable body with the given `id` has been registered in this
   model. */
  systems::DiscreteStateIndex GetDiscreteStateIndex(DeformableBodyId id) const;

  /** Returns the FemModel for the body with `id`.
   @throws exception if no deformable body with `id` is registered with `this`
   %DeformableModel. */
  const fem::FemModel<T>& GetFemModel(DeformableBodyId id) const;

  // TODO(xuchenhan-tri): The use of T over double is not well-reasoned.
  //  Consider whether T is really necessary when we support autodiff in
  //  deformable simulations.
  /** Returns the reference positions of the vertices of the deformable body
   identified by the given `id`.
   The reference positions are represented as a VectorX with 3N values where N
   is the number of vertices. The x-, y-, and z-positions (measured and
   expressed in the world frame) of the j-th vertex are 3j, 3j + 1, and 3j + 2
   in the VectorX.
   @throws std::exception if no deformable body with the given `id` has been
   registered in this model. */
  const VectorX<T>& GetReferencePositions(DeformableBodyId id) const;

  /** Returns the vertex position of v-th vertex in the deformable body with the
   given `id` in the world frame in the given `context`.
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet
   or if no body with the given `id` has been registered.
   @pre v is non-negative and smaller than the number of vertices
   in body with the given `id`. */
  Vector3<T> GetVertexPosition(const systems::Context<T>& context,
                               DeformableBodyId id, int v) const;

  /** Returns the DeformableBodyId of the body with the given body index.
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet
   or if index is larger than or equal to the total number of registered
   deformable bodies. */
  DeformableBodyId GetBodyId(DeformableBodyIndex index) const;

  /** (Internal) Returns the DeformableBodyIndex of the body with the given id.
   This function is for internal bookkeeping use only. Most users should use
   DeformableBodyId instead.
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet
   or if no body with the given `id` has been registered. */
  DeformableBodyIndex GetBodyIndex(DeformableBodyId id) const;

  /** Returns the GeometryId of the geometry associated with the body with the
   given `id`.
   @throws std::exception if no body with the given `id` has been registered. */
  geometry::GeometryId GetGeometryId(DeformableBodyId id) const;

  /** Returns the DeformableBodyId associated with the given `geometry_id`.
   @throws std::exception if the given `geometry_id` does not correspond to a
   deformable body registered with this model. */
  DeformableBodyId GetBodyId(geometry::GeometryId geometry_id) const;

  /** Returns the output port of the vertex positions for all registered
   deformable bodies.
   @throws std::exception if MultibodyPlant::Finalize() has not been called yet.
  */
  const systems::OutputPort<T>& vertex_positions_port() const {
    this->ThrowIfSystemResourcesNotDeclared(__func__);
    return plant_->get_output_port(vertex_positions_port_index_);
  }

  /** Returns the weld constraints registered for the body the given `id`.
   @throws std::exception if the given `id` does not correspond to a deformable
   body registered with this model. */
  const std::vector<internal::DeformableRigidWeldConstraintSpecs>&
  weld_constraint_specs(DeformableBodyId id) const {
    return weld_constraint_specs_.at(id);
  }

  bool has_constraint(DeformableBodyId id) const {
    return weld_constraint_specs_.count(id) > 0 &&
           weld_constraint_specs_.at(id).size() > 0;
  }

  /** Defines a distance constraint between a vertex P of a deformable body A
   and a point Q rigidly affixed to a rigid body B.

   This constraint can be compliant, modeling a spring with zero rest length
   and given `stiffness` and `damping` parameters between points P
   and Q. For d = ‖p_PQ‖, a compliant weld constraint models a spring with force
   along p_PQ given by:

      f = −stiffness ⋅ d − damping ⋅ ḋ

   @param[in] body_A_id    Id of the deformable body
   @param[in] vertex_index The index of the deformable body under constraint
   @param[in] body_B       Body to which point Q is rigidly attached.
   @param[in] p_BQ         Position of point Q in body B's frame.
   @param[in] stiffness    For modeling a spring with zero rest length, the
                           stiffness parameter in N/m. Optional, with its
                           default value being infinite.
   @param[in] damping      For modeling a spring with zero rest length, the
                           damping parameter in N⋅s/m. Optional, with its
                           default value being zero for a non-dissipative
                           constraint.
   @returns the index to the newly added constraint.
   @pre `body_B` is registered with the same multibody plant owning this
   deformable model.
   @pre `vertex_index` is non-negative and smaller than the number of vertices
   in body A.
   @throws std::exception if no deformable body with the given `body_A_id`
           has been registered.
   @throws std::exception if `stiffness` is not positive or zero.
   @throws std::exception if `damping` is not positive or zero.
   @throws std::exception if Finalize() has been called on the multibody plant
   owning this deformable model. */
  void AddWeldConstraint(
      DeformableBodyId body_A_id, int vertex_index, const Body<T>& body_B,
      const Vector3<double>& p_BQ,
      double stiffness = std::numeric_limits<double>::infinity(),
      double damping = 0.0);

  /* Welds vertices of deformable body A to rigid body B. The vertices that are
   welded are those that are close to z=0 in its body frame. */
  void Weld(DeformableBodyId body_A_id, const Body<T>& body_B,
            const math::RigidTransform<T>& X_WA,
            const math::RigidTransform<T>& X_WB) {
    const VectorX<T>& reference_positions = GetReferencePositions(body_A_id);
    const int num_verts = reference_positions.size() / 3;
    const double tol = 5e-4;
    for (int i = 0; i < num_verts; ++i) {
      const Vector3<T>& p_WV = reference_positions.template segment<3>(3 * i);
      const Vector3<T> p_AV = X_WA.inverse() * p_WV;
      if (std::abs(p_AV(2)) < tol) {
        const Vector3<T> p_BV = X_WB.inverse() * p_WV;
        AddWeldConstraint(body_A_id, i, body_B, p_BV);
      }
    }
  }

 private:
  internal::PhysicalModelPointerVariant<T> DoToPhysicalModelPointerVariant()
      const final {
    return internal::PhysicalModelPointerVariant<T>(this);
  }

  // TODO(xuchenhan-tri): Implement CloneToDouble() and CloneToAutoDiffXd()
  // and the corresponding is_cloneable methods.

  void DoDeclareSystemResources(MultibodyPlant<T>* plant) final;

  /* Builds a FEM model for the body with `id` with linear tetrahedral elements
   and a single quadrature point. The reference positions as well as the
   connectivity of the elements are given by `mesh`, and physical properties
   such as the material model of the body are given by `config`.
   @throws exception if an FEM model corresponding to `id` already exists. */
  void BuildLinearVolumetricModel(DeformableBodyId id,
                                  const geometry::VolumeMesh<double>& mesh,
                                  const fem::DeformableBodyConfig<T>& config);

  template <template <class, int> class Model>
  void BuildLinearVolumetricModelHelper(
      DeformableBodyId id, const geometry::VolumeMesh<double>& mesh,
      const fem::DeformableBodyConfig<T>& config);

  /* Copies the vertex positions of all deformable bodies to the output port
   value which is guaranteed to be of type GeometryConfigurationVector. */
  void CopyVertexPositions(const systems::Context<T>& context,
                           AbstractValue* output) const;

  /* Helper to throw a useful message if a deformable body with the given `id`
   doesn't exist. */
  void ThrowUnlessRegistered(const char* source_method,
                             DeformableBodyId id) const;

  /* The MultibodyPlant that owns `this` DeformableModel. */
  MultibodyPlant<T>* plant_{nullptr};
  /* The positions of each vertex of deformable body at reference configuration.
   */
  std::unordered_map<DeformableBodyId, VectorX<T>> reference_positions_;
  /* The discrete state indexes for all deformable bodies. */
  std::unordered_map<DeformableBodyId, systems::DiscreteStateIndex>
      discrete_state_indexes_;
  std::unordered_map<DeformableBodyId, geometry::GeometryId>
      body_id_to_geometry_id_;
  std::unordered_map<geometry::GeometryId, DeformableBodyId>
      geometry_id_to_body_id_;
  std::unordered_map<DeformableBodyId, std::unique_ptr<fem::FemModel<T>>>
      fem_models_;
  std::vector<DeformableBodyId> body_ids_;
  std::unordered_map<DeformableBodyId, DeformableBodyIndex> body_id_to_index_;
  systems::OutputPortIndex vertex_positions_port_index_;
  /* Vector of weld constraints specifications. */
  std::unordered_map<DeformableBodyId,
                     std::vector<internal::DeformableRigidWeldConstraintSpecs>>
      weld_constraint_specs_;
};

}  // namespace multibody
}  // namespace drake
