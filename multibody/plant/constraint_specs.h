#pragma once

/// @file
/// This files contains simple structs used to store constraint specifications
/// defined by the user through MultibodyPlant API calls. These specifications
/// are later on used by our discrete solvers to build a model.

#include <limits>

#include "drake/multibody/plant/deformable_indexes.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"

namespace drake {
namespace multibody {
namespace internal {

// Struct to store coupler constraint parameters.
// Coupler constraints are modeled as a holonomic constraint of the form q₀ =
// ρ⋅q₁ + Δq, where q₀ and q₁ are the positions of two one-DOF joints, ρ the
// gear ratio and Δq a fixed offset. Per equation above, ρ has units of q₀/q₁
// and Δq has units of q₀.
struct CouplerConstraintSpecs {
  // First joint with position q₀.
  JointIndex joint0_index;
  // Second joint with position q₁.
  JointIndex joint1_index;
  // Gear ratio ρ.
  double gear_ratio{1.0};
  // Offset Δq.
  double offset{0.0};
};

// Struct to store the specification for a weld constraint between a vertex P on
// a deformable body A and a rigid body B. Such a weld constraint is modeled as
// a holonomic constraint. Weld constraints can be "soft" which imposes the the
// condition:
//   d(q) + c/k⋅ḋ(q) + 1/k⋅f = 0
// where k a stiffness parameter in N/m and c a damping parameter in N⋅s/m. We
// use d(q) to denote the Euclidean distance between two points P and Q, where Q
// is rigidly affixed to the rigid body B, as a function of the configuration of
// the model q. This constraint reduces to d(q) = 0 in the limit to infinite
// stiffness and it behaves as a linear spring damper (with zero rest length)
// for finite values of stiffness and damping.
//
// @pre d₀ > 0, k >= 0, c >= 0. @see IsValid().
struct DeformableRigidWeldConstraintSpecs {
  // Returns `true` iff `this` specification is valid to define a valid weld
  // constraint between a point on a deformable body and a rigid body. The
  // specification is considered to be valid iff stiffness >= 0 and
  // damping >= 0.
  bool IsValid() { return stiffness >= 0.0 && damping >= 0.0; }

  DeformableBodyId body_A;  // Index of body A.
  // TODO(xuchenhan-tri): consider allowing points on the body not at the vertex
  // to be constrained.
  int vertex_index{-1};  // Index of point P in the mesh of body A.
  BodyIndex body_B;      // Index of body B.
  Vector3<double> p_BQ;  // Position of point Q in body frame B.
  double stiffness{
      std::numeric_limits<double>::infinity()};  // Constraint stiffness
                                                 // k in N/m.
  double damping{0.0};  // Constraint damping c in N⋅s/m.
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake
