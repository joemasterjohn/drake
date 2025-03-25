#pragma once

#include <array>
#include <memory>
#include <stack>
#include <utility>
#include <variant>
#include <vector>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/aabb.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/utilities.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

class DynamicBvNode {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(DynamicBvNode);

  static constexpr int kMaxElementPerLeaf = 1;

  /* A leaf node can store as many as kMaxElementPerLeaf elements.
   The actual number of stored element indices is `num_index`. */
  struct LeafData {
    int num_index;
    std::array<int, kMaxElementPerLeaf> indices;
  };

  /* Constructor for leaf nodes consisting of multiple elements.
   @param bv    The bounding volume encompassing the elements.
   @param data  The indices of the mesh elements contained in the leaf. */
  DynamicBvNode(Aabb bv, LeafData data)
      : bv_(std::move(bv)), child_(std::move(data)) {}

  /* Constructor for branch/internal nodes.
   @param bv The bounding volume encompassing the elements in child branches.
   @param left Unique pointer to the left child branch.
   @param right Unique pointer to the right child branch.
   @pre Both children must be distinct and not null.   */
  DynamicBvNode(Aabb bv, std::unique_ptr<DynamicBvNode> left,
                std::unique_ptr<DynamicBvNode> right)
      : bv_(std::move(bv)),
        child_(NodeChildren(std::move(left), std::move(right))) {}

  /* Returns the bounding volume.  */
  const Aabb& bv() const { return bv_; }

  void setBV(Aabb bv) { bv_ = std::move(bv); }

  /* Returns the number of element indices.
   @pre is_leaf() returns true. */
  int num_element_indices() const {
    return std::get<LeafData>(child_).num_index;
  }

  /* Returns the i-th element index in the leaf data.
   @pre is_leaf() returns true.
   @pre `i` is less than LeafData::num_index, and i >= 0. */
  int element_index(int i) const {
    DRAKE_ASSERT(0 <= i && i < std::get<LeafData>(child_).num_index);
    return std::get<LeafData>(child_).indices[i];
  }

  /* Returns the left child branch.
   @pre is_leaf() returns false.  */
  const DynamicBvNode& left() const {
    return *(std::get<NodeChildren>(child_).left);
  }

  /* Returns the right child branch.
   @pre is_leaf() returns false.  */
  const DynamicBvNode& right() const {
    return *(std::get<NodeChildren>(child_).right);
  }

  /* Returns whether this is a leaf node as opposed to a branch node.  */
  bool is_leaf() const { return std::holds_alternative<LeafData>(child_); }

 private:
  friend class DynamicBvh;

  /* Provide disciplined access to DynamicBvhUpdater to a mutable child node. */
  DynamicBvNode& left() { return *(std::get<NodeChildren>(child_).left); }

  /* Provide disciplined access to DynamicBvhUpdater to a mutable child node. */
  DynamicBvNode& right() { return *(std::get<NodeChildren>(child_).right); }

  /* Provide disciplined access to DynamicBvhUpdater to a mutable bounding
   * volume. */
  Aabb& bv() { return bv_; }

  struct NodeChildren {
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(NodeChildren);

    NodeChildren(std::unique_ptr<DynamicBvNode> left_in,
                 std::unique_ptr<DynamicBvNode> right_in)
        : left(std::move(left_in)), right(std::move(right_in)) {
      DRAKE_DEMAND(left != nullptr);
      DRAKE_DEMAND(right != nullptr);
      DRAKE_DEMAND(left != right);
    }

    copyable_unique_ptr<DynamicBvNode> left;
    copyable_unique_ptr<DynamicBvNode> right;
  };

  Aabb bv_;

  // If this is a leaf node then the child refers to indices into the mesh's
  // elements (i.e., triangles or tetrahedra) bounded by the node's bounding
  // volume. Otherwise, it refers to child nodes further down the tree.
  std::variant<LeafData, NodeChildren> child_;
};

/* Resulting instruction from performing the bounding volume tree traversal
 (BVTT) callback on two potentially colliding pairs. Note that this is not the
 mathematical result but information on how the traversal should proceed.  */
enum class DynamicBvttCallbackResult { Continue, Terminate };

/* Bounding volume tree traversal (BVTT) callback. Returns a
 DynamicBvttCallbackResult for further action, e.g. deciding whether to exit
 early. The parameters are an index into the elements of the *first* mesh
 followed by the index into the elements of the *second* mesh. */
using DynamicBvttCallback = std::function<DynamicBvttCallbackResult(int, int)>;

/* Function to calculate the Aabb from the leaf element at the given index. */
using AabbCalculator = std::function<Aabb(int)>;

/* TODO(joemasterjohn): consider different build strategies as options. */

class DynamicBvh {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(DynamicBvh);

  DynamicBvh();

  /* Given a set of `num_leaves` elements, constructs a BVH of the elements
   * using the provided AabbCalculator and the build strategy provided.*/
  DynamicBvh(int num_leaves, AabbCalculator leafCalculator);

  const DynamicBvNode& root_node() const { return *root_node_; }

  int num_leaves() const { return num_leaves_; }

  /* Refit the bounding volumes of the already calculated tree using the given
   * leafCalculator. Topology of the tree is unchanged. */
  void Refit(AabbCalculator leafCalculator);

  /* Complately Build the tree given the provided AabbCalculator and build
   * strategy. */
  void Build(int num_leaves, AabbCalculator leafCalculator);

  /* Perform a query of this %DynamicBvh's mesh elements (measured and expressed
   in Frame A) against the given %DynamicBvh's mesh elements (measured and
   expressed in Frame A). The callback is invoked on every pair of elements that
   cannot conclusively be shown to be separated via bounding-volume comparisons
   (the unculled pairs).

   @param bvh_B           The bounding volume hierarchy to collide with.
   @param callback        The callback to invoke on each unculled pair.*/
  void Collide(const DynamicBvh& bvh_B, DynamicBvttCallback callback) const;

  void SelfCollide(DynamicBvttCallback callback) const;

  /* Wrapper around `Collide` with a callback that accumulates each pair of
   collision candidates and returns them all.
   @return Vector of element index pairs whose elements are candidates for
   collision. */
  std::vector<std::pair<int, int>> GetCollisionCandidates(
      const DynamicBvh& bvh_B) const;

 private:
  static void RefitRecursive(DynamicBvNode* node,
                             AabbCalculator leafCalculator);

  static Aabb CalcAabb(std::vector<std::pair<int, Aabb>>::iterator start,
                       std::vector<std::pair<int, Aabb>>::iterator end);

  static std::unique_ptr<DynamicBvNode> BuildRecursive(
      std::vector<std::pair<int, Aabb>>::iterator start,
      std::vector<std::pair<int, Aabb>>::iterator end);

  DynamicBvNode& mutable_root_node() { return *root_node_; }
  copyable_unique_ptr<DynamicBvNode> root_node_;
  int num_leaves_{};
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
