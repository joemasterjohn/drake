#pragma once

#include <array>
#include <utility>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/plant/jacobian_matrix.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"

namespace drake {
namespace multibody {
namespace internal {

// Struct to store kinematics information for each SAP weld constraint pair. For
// each weld constraint, this struct stores the current displacement, Jacobian
// w.r.t. velocities for each participating tree, and the tree indexes.
template <typename T>
struct WeldConstraintData {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(WeldConstraintData);

  // Struct to store the block contribution from a given tree to the Jacobian
  // for a contact constraint.
  struct JacobianTreeBlock {
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(JacobianTreeBlock);

    JacobianTreeBlock() = default;

    JacobianTreeBlock(TreeIndex tree_in, JacobianBlock<T> J_in)
        : tree(tree_in), J(std::move(J_in)) {}

    // Index of the tree for this block.
    TreeIndex tree;

    // J.cols() must equal the number of generalized velocities for
    // the corresponding tree.
    JacobianBlock<T> J;
  };

  // Constructs a WeldConstraintData.
  // @param[in] p_PQ_W_in position vector from P to Q expressed in the world
  // frame.
  // @param[in] jacobian_in The jacobian of the constraint with respect to tree
  // degrees of freedom, expressed in the world frame, i.e., -Jv_WAp_W and
  // Jv_WBq_W.
  WeldConstraintData(Vector3<T> p_PQ_W_in,
                     std::array<JacobianTreeBlock, 2> jacobian_in)
      : p_PQ_W(std::move(p_PQ_W_in)), jacobian(std::move(jacobian_in)) {}

  // Displacement as a measure of violation of weld constraint.
  Vector3<T> p_PQ_W{};

  // Jacobian for trees under constraint stored as individual blocks for each
  // of the trees participating in the contact.
  std::array<JacobianTreeBlock, 2> jacobian;
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    struct ::drake::multibody::internal::WeldConstraintData)
