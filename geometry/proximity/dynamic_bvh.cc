#include "drake/geometry/proximity/bvh.h"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "drake/geometry/utilities.h"

namespace drake {
namespace geometry {
namespace internal {

DynamicBvh::DynamicBvh(int num_leaves, AabbCalculator leafCalculator)
    : num_leaves_(num_leaves) {
  DRAKE_DEMAND(num_leaves > 0);
  Rebuild(leafCalculator);
}

void DynamicBvh::Refit(AabbCalculator leafCalculator) {
  /* Update the BVs of each node in a DFS manner. */
  std::stack<const DynamicBvhNode&> nodes;
  nodes.emplate(root_node());
  while (!nodes.empty()) {
    const DynamicBvhNode& node = nodes.top();
    nodes.pop();
    if (node.is_leaf()) {
      /* Compute the BV that encompases all of the leaf element BVs. */
      const int num_elements = node.num_elements_indices();
      DRAKE_DEMAND(num_elements > 0);
      Aabb bv = leafCalculator(node.element(0));
      for (int i = 1; i < num_elements; ++i) {
        bv = Aabb(bv, leafCalculator(node.element(i)));
      }
      node.setBV(bv);
    } else {
      /* Set BV to the BV encompasing the children's BVs.
         DFS order ensures the children's BVs have been refitted prior to this
         line. */
      node.setBV(Aabb(node.left().bv(), node.right.bv()));
    }
  }
}

void DynamicBvh::Rebuild(AabbCalculator leafCalculator) {
  /* */
}

void DynamicBvh::Collide(const DynamicBvh& bvh_B, BvttCallback callback) const {
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
          const BvttCallbackResult result =
              callback(node_a.element_index(a), node_b.element_index(b));
          if (result == BvttCallbackResult::Terminate) return;
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

void DynamicBvh::SelfCollide(BvttCallback callback) const {
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
            const BvttCallbackResult result =
                callback(node_a.element_index(i), node_a.element_index(j));
            if (result == BvttCallbackResult::Terminate) return;
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
          const BvttCallbackResult result =
              callback(node_a.element_index(a), node_b.element_index(b));
          if (result == BvttCallbackResult::Terminate) return;
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
  BvttCallback callback = [&result](int a, int b) -> BvttCallbackResult {
    result.emplace_back(a, b);
    return BvttCallbackResult::Continue;
  };
  Collide(bvh_B, callback);
  return result;
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
