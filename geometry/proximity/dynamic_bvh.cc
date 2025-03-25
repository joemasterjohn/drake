#include "drake/geometry/proximity/dynamic_bvh.h"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "drake/geometry/utilities.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;

DynamicBvh::DynamicBvh() : num_leaves_(0) {}

DynamicBvh::DynamicBvh(int num_leaves, AabbCalculator leafCalculator) {
  Build(num_leaves, leafCalculator);
}

void DynamicBvh::Refit(AabbCalculator leafCalculator) {
  RefitRecursive(&mutable_root_node(), leafCalculator);
}

void DynamicBvh::RefitRecursive(DynamicBvNode* node,
                                AabbCalculator leafCalculator) {
  if (node->is_leaf()) {
    /* Compute the BV that encompases all of the leaf element BVs. */
    const int num_elements = node->num_element_indices();
    Aabb bv = leafCalculator(node->element_index(0));
    for (int i = 1; i < num_elements; ++i) {
      bv = Aabb(bv, leafCalculator(node->element_index(i)));
    }
    node->setBV(bv);
  } else {
    RefitRecursive(&node->left(), leafCalculator);
    RefitRecursive(&node->right(), leafCalculator);
    node->setBV(Aabb(node->left().bv(), node->right().bv()));
  }
}

Aabb DynamicBvh::CalcAabb(std::vector<std::pair<int, Aabb>>::iterator start,
                          std::vector<std::pair<int, Aabb>>::iterator end) {
  Vector3d min_corner = start->second.center() - start->second.half_width();
  Vector3d max_corner = start->second.center() + start->second.half_width();
  std::vector<std::pair<int, Aabb>>::iterator current = start + 1;
  while (current != end) {
    min_corner = min_corner.cwiseMin(current->second.center() -
                                     current->second.half_width());
    max_corner = max_corner.cwiseMax(current->second.center() +
                                     current->second.half_width());
    ++current;
  }
  const Vector3d center = (max_corner + min_corner) / 2;
  const Vector3d half_width = max_corner - center;
  return Aabb(center, half_width);
}

std::unique_ptr<DynamicBvNode> DynamicBvh::BuildRecursive(
    std::vector<std::pair<int, Aabb>>::iterator start,
    std::vector<std::pair<int, Aabb>>::iterator end) {
  Aabb bv = CalcAabb(start, end);

  const int num_elements = end - start;
  if (num_elements <= DynamicBvNode::kMaxElementPerLeaf) {
    typename DynamicBvNode::LeafData data{num_elements, {}};
    for (int i = 0; i < num_elements; ++i) {
      data.indices[i] = (start + i)->first;
    }
    // Store element indices in this leaf node.
    return std::make_unique<DynamicBvNode>(bv, data);
  } else {
    int axis{};
    bv.half_width().maxCoeff(&axis);
    std::sort(
        start, end,
        [axis](const std::pair<int, Aabb>& a, const std::pair<int, Aabb>& b) {
          return a.second.center()(axis) < b.second.center()(axis);
        });

    const typename std::vector<std::pair<int, Aabb>>::iterator mid =
        start + num_elements / 2;
    return std::make_unique<DynamicBvNode>(bv, BuildRecursive(start, mid),
                                           BuildRecursive(mid, end));
  }
}

void DynamicBvh::Build(int num_leaves, AabbCalculator leafCalculator) {
  DRAKE_DEMAND(num_leaves > 0);
  num_leaves_ = num_leaves;
  // Simple top-down build.
  std::vector<std::pair<int, Aabb>> leaf_aabbs;
  for (int i = 0; i < num_leaves_; ++i) {
    leaf_aabbs.emplace_back(i, leafCalculator(i));
  }

  root_node_ = BuildRecursive(leaf_aabbs.begin(), leaf_aabbs.end());
}

void DynamicBvh::Collide(const DynamicBvh& bvh_B,
                         DynamicBvttCallback callback) const {
  // Check for the case of self collision.
  if (&bvh_B == this) {
    this->SelfCollide(callback);
    return;
  }

  using NodePair = std::pair<const DynamicBvNode&, const DynamicBvNode&>;
  std::stack<NodePair> node_pairs;
  node_pairs.emplace(root_node(), bvh_B.root_node());

  while (!node_pairs.empty()) {
    const auto& [node_a, node_b] = node_pairs.top();
    node_pairs.pop();

    // Check if the bounding volumes overlap.
    if (!Aabb::HasOverlap(node_a.bv(), node_b.bv())) {
      continue;
    }

    // Run the callback on the pair if they are both leaf nodes, otherwise
    // check each branch.
    if (node_a.is_leaf() && node_b.is_leaf()) {
      const int num_a_elements = node_a.num_element_indices();
      const int num_b_elements = node_b.num_element_indices();
      for (int a = 0; a < num_a_elements; ++a) {
        for (int b = 0; b < num_b_elements; ++b) {
          const DynamicBvttCallbackResult result =
              callback(node_a.element_index(a), node_b.element_index(b));
          if (result == DynamicBvttCallbackResult::Terminate) return;
        }
      }
    } else if (node_b.is_leaf()) {
      node_pairs.emplace(node_a.left(), node_b);
      node_pairs.emplace(node_a.right(), node_b);
    } else if (node_a.is_leaf()) {
      node_pairs.emplace(node_a, node_b.left());
      node_pairs.emplace(node_a, node_b.right());
    } else {
      node_pairs.emplace(node_a.left(), node_b.left());
      node_pairs.emplace(node_a.right(), node_b.left());
      node_pairs.emplace(node_a.left(), node_b.right());
      node_pairs.emplace(node_a.right(), node_b.right());
    }
  }
}

void DynamicBvh::SelfCollide(DynamicBvttCallback callback) const {
  using NodePair = std::pair<const DynamicBvNode&, const DynamicBvNode&>;
  std::stack<NodePair> node_pairs;
  node_pairs.emplace(root_node(), root_node());

  while (!node_pairs.empty()) {
    const auto& [node_a, node_b] = node_pairs.top();
    node_pairs.pop();

    // Same node case. No need to check overlap. Simplified branch checking.
    if (&node_a == &node_b) {
      if (node_a.is_leaf()) {
        // Callback on all **unique** pairs.
        const int num_elements = node_a.num_element_indices();
        for (int i = 0; i < num_elements; ++i) {
          for (int j = i + 1; j < num_elements; ++j) {
            const DynamicBvttCallbackResult result =
                callback(node_a.element_index(i), node_a.element_index(j));
            if (result == DynamicBvttCallbackResult::Terminate) return;
          }
        }
      } else {
        node_pairs.emplace(node_a.left(), node_a.left());
        node_pairs.emplace(node_a.right(), node_a.right());
        node_pairs.emplace(node_a.left(), node_a.right());
      }
      continue;
    }

    // Check if the bounding volumes overlap.
    if (!Aabb::HasOverlap(node_a.bv(), node_b.bv())) {
      continue;
    }

    // Run the callback on the pair if they are both leaf nodes, otherwise
    // check each branch.
    if (node_a.is_leaf() && node_b.is_leaf()) {
      const int num_a_elements = node_a.num_element_indices();
      const int num_b_elements = node_b.num_element_indices();
      for (int a = 0; a < num_a_elements; ++a) {
        for (int b = 0; b < num_b_elements; ++b) {
          const DynamicBvttCallbackResult result =
              callback(node_a.element_index(a), node_b.element_index(b));
          if (result == DynamicBvttCallbackResult::Terminate) return;
        }
      }
    } else if (node_b.is_leaf()) {
      node_pairs.emplace(node_a.left(), node_b);
      node_pairs.emplace(node_a.right(), node_b);
    } else if (node_a.is_leaf()) {
      node_pairs.emplace(node_a, node_b.left());
      node_pairs.emplace(node_a, node_b.right());
    } else {
      node_pairs.emplace(node_a.left(), node_b.left());
      node_pairs.emplace(node_a.right(), node_b.left());
      node_pairs.emplace(node_a.left(), node_b.right());
      node_pairs.emplace(node_a.right(), node_b.right());
    }
  }
}

std::vector<std::pair<int, int>> DynamicBvh::GetCollisionCandidates(
    const DynamicBvh& bvh_B) const {
  std::vector<std::pair<int, int>> result;
  DynamicBvttCallback callback = [&result](int a,
                                           int b) -> DynamicBvttCallbackResult {
    result.emplace_back(a, b);
    return DynamicBvttCallbackResult::Continue;
  };
  Collide(bvh_B, callback);
  return result;
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
