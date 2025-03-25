#include "drake/geometry/proximity/dynamic_bvh.h"

#include <utility>

#include <gtest/gtest.h>

#include "drake/common/fmt_eigen.h"
#include "drake/geometry/proximity/aabb.h"
#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;
using math::RigidTransformd;
using math::RollPitchYawd;

namespace {

// Line of boxes that do not overlap, each with half_width of space in between.
AabbCalculator MakeNonOverlappingBoxes(double half_width, double x_offset = 0.0,
                                       double y_offset = 0.0) {
  return [=](int i) -> Aabb {
    return Aabb(
        Vector3d(x_offset + half_width + 3 * i * half_width, y_offset, 0),
        Vector3d(half_width, 1, 1));
  };
}

// Line of boxes where each neighbor overlaps by an amount of 0.5*half_width
AabbCalculator MakeOverlappingNeighborBoxes(double half_width,
                                            double x_offset = 0.0,
                                            double y_offset = 0.0) {
  return [=](int i) -> Aabb {
    return Aabb(
        Vector3d(x_offset + half_width + 1.5 * i * half_width, y_offset, 0),
        Vector3d(half_width, 1, 1));
  };
}

GTEST_TEST(DynamicBoundingVolumeHierarchyTest, TwoNonOverlapping) {
  constexpr int num_leaves = 2;
  constexpr double half_width = 1;
  AabbCalculator calculator = MakeNonOverlappingBoxes(half_width);
  DynamicBvh bvh(num_leaves, calculator);

  EXPECT_TRUE(bvh.root_node().left().is_leaf());
  EXPECT_TRUE(bvh.root_node().right().is_leaf());
  EXPECT_TRUE(bvh.root_node().left().bv().Equal(calculator(0)));
  EXPECT_TRUE(bvh.root_node().right().bv().Equal(calculator(1)));
  EXPECT_TRUE(bvh.root_node().bv().Equal(Aabb(calculator(0), calculator(1))));
  EXPECT_EQ(bvh.root_node().bv().center().x(), 2.5 * half_width);
  EXPECT_EQ(bvh.root_node().bv().half_width().x(), 2.5 * half_width);
  auto candidates = bvh.GetCollisionCandidates(bvh);
  EXPECT_EQ(ssize(candidates), 0);
}

GTEST_TEST(DynamicBoundingVolumeHierarchyTest, TwoOverlapping) {
  constexpr int num_leaves = 2;
  constexpr double half_width = 1;
  AabbCalculator calculator = MakeOverlappingNeighborBoxes(half_width);
  DynamicBvh bvh(num_leaves, calculator);

  EXPECT_TRUE(bvh.root_node().left().is_leaf());
  EXPECT_TRUE(bvh.root_node().right().is_leaf());
  EXPECT_TRUE(bvh.root_node().left().bv().Equal(calculator(0)));
  EXPECT_TRUE(bvh.root_node().right().bv().Equal(calculator(1)));
  EXPECT_TRUE(bvh.root_node().bv().Equal(Aabb(calculator(0), calculator(1))));
  EXPECT_EQ(bvh.root_node().bv().center().x(), 0.5 * 3.5 * half_width);
  EXPECT_EQ(bvh.root_node().bv().half_width().x(), 0.5 * 3.5 * half_width);
  auto candidates = bvh.GetCollisionCandidates(bvh);
  EXPECT_EQ(ssize(candidates), 1);
  EXPECT_EQ(candidates[0].first, 0);
  EXPECT_EQ(candidates[0].second, 1);
}

GTEST_TEST(DynamicBoundingVolumeHierarchyTest, TwoRefit) {
  constexpr int num_leaves = 2;
  constexpr double half_width = 1;
  AabbCalculator non_overlapping_calculator =
      MakeNonOverlappingBoxes(half_width);
  DynamicBvh bvh(num_leaves, non_overlapping_calculator);

  EXPECT_TRUE(bvh.root_node().left().is_leaf());
  EXPECT_TRUE(bvh.root_node().right().is_leaf());
  EXPECT_TRUE(bvh.root_node().left().bv().Equal(non_overlapping_calculator(0)));
  EXPECT_TRUE(
      bvh.root_node().right().bv().Equal(non_overlapping_calculator(1)));
  EXPECT_TRUE(bvh.root_node().bv().Equal(
      Aabb(non_overlapping_calculator(0), non_overlapping_calculator(1))));
  auto candidates = bvh.GetCollisionCandidates(bvh);
  EXPECT_EQ(ssize(candidates), 0);

  AabbCalculator overlapping_calculator =
      MakeOverlappingNeighborBoxes(half_width);
  bvh.Refit(overlapping_calculator);

  EXPECT_TRUE(bvh.root_node().left().is_leaf());
  EXPECT_TRUE(bvh.root_node().right().is_leaf());
  EXPECT_TRUE(bvh.root_node().left().bv().Equal(overlapping_calculator(0)));
  EXPECT_TRUE(bvh.root_node().right().bv().Equal(overlapping_calculator(1)));
  EXPECT_TRUE(bvh.root_node().bv().Equal(
      Aabb(overlapping_calculator(0), overlapping_calculator(1))));
  candidates = bvh.GetCollisionCandidates(bvh);
  EXPECT_EQ(ssize(candidates), 1);
  EXPECT_EQ(candidates[0].first, 0);
  EXPECT_EQ(candidates[0].second, 1);
}

GTEST_TEST(DynamicBoundingVolumeHierarchyTest, TenNonOverlapping) {
  constexpr int num_leaves = 10;
  constexpr double half_width = 1;
  AabbCalculator calculator = MakeNonOverlappingBoxes(half_width);
  DynamicBvh bvh(num_leaves, calculator);

  EXPECT_EQ(bvh.root_node().bv().center().x(),
            0.5 * (2 + 3 * (num_leaves - 1)) * half_width);
  EXPECT_EQ(bvh.root_node().bv().half_width().x(),
            0.5 * (2 + 3 * (num_leaves - 1)) * half_width);
  auto candidates = bvh.GetCollisionCandidates(bvh);
  EXPECT_EQ(ssize(candidates), 0);
}

GTEST_TEST(DynamicBoundingVolumeHierarchyTest, TenOverlapping) {
  constexpr int num_leaves = 10;
  constexpr double half_width = 1;
  AabbCalculator calculator = MakeOverlappingNeighborBoxes(half_width);
  DynamicBvh bvh(num_leaves, calculator);

  EXPECT_EQ(bvh.root_node().bv().center().x(),
            0.5 * (2 + 1.5 * (num_leaves - 1)) * half_width);
  EXPECT_EQ(bvh.root_node().bv().half_width().x(),
            0.5 * (2 + 1.5 * (num_leaves - 1)) * half_width);
  auto candidates = bvh.GetCollisionCandidates(bvh);
  EXPECT_EQ(ssize(candidates), num_leaves - 1);
  // All candidates should be neighbors
  for (int i = 0; i < ssize(candidates); ++i) {
    EXPECT_EQ(abs(candidates[i].first - candidates[i].second), 1);
  }
}

GTEST_TEST(DynamicBoundingVolumeHierarchyTest, TenRefit) {
  constexpr int num_leaves = 10;
  constexpr double half_width = 1;
  AabbCalculator non_overlapping_calculator =
      MakeNonOverlappingBoxes(half_width);
  DynamicBvh bvh(num_leaves, non_overlapping_calculator);

  EXPECT_EQ(bvh.root_node().bv().center().x(),
            0.5 * (2 + 3 * (num_leaves - 1)) * half_width);
  EXPECT_EQ(bvh.root_node().bv().half_width().x(),
            0.5 * (2 + 3 * (num_leaves - 1)) * half_width);
  auto candidates = bvh.GetCollisionCandidates(bvh);
  EXPECT_EQ(ssize(candidates), 0);

  AabbCalculator overlapping_calculator =
      MakeOverlappingNeighborBoxes(half_width);
  bvh.Refit(overlapping_calculator);

  EXPECT_EQ(bvh.root_node().bv().center().x(),
            0.5 * (2 + 1.5 * (num_leaves - 1)) * half_width);
  EXPECT_EQ(bvh.root_node().bv().half_width().x(),
            0.5 * (2 + 1.5 * (num_leaves - 1)) * half_width);
  candidates = bvh.GetCollisionCandidates(bvh);
  EXPECT_EQ(ssize(candidates), num_leaves - 1);
  // All candidates should be neighbors
  for (int i = 0; i < ssize(candidates); ++i) {
    EXPECT_EQ(abs(candidates[i].first - candidates[i].second), 1);
  }
}

GTEST_TEST(DynamicBoundingVolumeHierarchyTest, TenOffsetY) {
  constexpr int num_leaves = 10;
  constexpr double half_width = 1;
  constexpr double y_offset = 0.5;
  AabbCalculator non_overlapping_calculator =
      MakeNonOverlappingBoxes(half_width);
  AabbCalculator non_overlapping_calculator_offset =
      MakeNonOverlappingBoxes(half_width, 0.0, y_offset);

  DynamicBvh bvhA(num_leaves, non_overlapping_calculator);
  DynamicBvh bvhB(num_leaves, non_overlapping_calculator_offset);

  auto candidates = bvhA.GetCollisionCandidates(bvhB);
  EXPECT_EQ(ssize(candidates), num_leaves);
  // Candidate leaves should have matching indexes.
  for (int i = 0; i < ssize(candidates); ++i) {
    EXPECT_EQ(candidates[i].first, candidates[i].second);
  }
}

GTEST_TEST(DynamicBoundingVolumeHierarchyTest, TenOffsetYAndX) {
  constexpr int num_leaves = 10;
  constexpr double half_width = 1;
  constexpr double x_offset = 3 * half_width;
  constexpr double y_offset = 0.5;
  AabbCalculator non_overlapping_calculator =
      MakeNonOverlappingBoxes(half_width);
  AabbCalculator non_overlapping_calculator_offset =
      MakeNonOverlappingBoxes(half_width, x_offset, y_offset);

  DynamicBvh bvhA(num_leaves, non_overlapping_calculator);
  DynamicBvh bvhB(num_leaves, non_overlapping_calculator_offset);

  auto candidates = bvhA.GetCollisionCandidates(bvhB);
  EXPECT_EQ(ssize(candidates), num_leaves - 1);
  // Candidate leaves should have matching indexes.
  for (int i = 0; i < ssize(candidates); ++i) {
    EXPECT_EQ(abs(candidates[i].first - candidates[i].second), 1);
  }
}

// Compute Aabb for the bounding box of a tetrahedral element.
AabbCalculator MeshCalculator(const VolumeMesh<double>& mesh) {
  return [&](int e) -> Aabb {
    VolumeElement element = mesh.element(e);
    const std::set<int> verts = {element.vertex(0), element.vertex(1),
                                 element.vertex(2), element.vertex(3)};
    return Aabb::Maker<VolumeMesh<double>>(mesh, verts).Compute();
  };
}

// Compute the Aabb for a tetrahedral element undergoing the translation t.
AabbCalculator TranslatedMeshCalculator(const VolumeMesh<double>& mesh,
                                        const Vector3d& t) {
  return [&](int e) -> Aabb {
    VolumeElement element = mesh.element(e);
    const std::set<int> verts = {element.vertex(0), element.vertex(1),
                                 element.vertex(2), element.vertex(3)};
    Aabb aabb = Aabb::Maker<VolumeMesh<double>>(mesh, verts).Compute();
    Vector3d min_corner = aabb.center() - aabb.half_width();
    Vector3d max_corner = aabb.center() + aabb.half_width();
    min_corner = min_corner.cwiseMin(min_corner + t);
    max_corner = max_corner.cwiseMax(max_corner + t);
    const Vector3d center = (max_corner + min_corner) / 2;
    const Vector3d half_width = max_corner - center;
    return Aabb(center, half_width);
  };
}

std::vector<std::pair<int, int>> BruteForceOverlap(
    const VolumeMesh<double>& meshA, const VolumeMesh<double>& meshB) {
  std::vector<Aabb> bvA;
  std::vector<Aabb> bvB;
  AabbCalculator calculatorA = MeshCalculator(meshA);
  AabbCalculator calculatorB = MeshCalculator(meshB);

  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < meshA.num_elements(); ++i) {
    bvA.emplace_back(calculatorA(i));
  }
  for (int i = 0; i < meshB.num_elements(); ++i) {
    bvB.emplace_back(calculatorB(i));
  }

  for (int i = 0; i < meshA.num_elements(); ++i) {
    for (int j = 0; j < meshB.num_elements(); ++j) {
      if (Aabb::HasOverlap(bvA[i], bvB[j])) {
        pairs.emplace_back(i, j);
      }
    }
  }
  return pairs;
}

bool ValidTree(const DynamicBvNode& root) {
  if (root.is_leaf()) return true;
  return Aabb::Contains(root.bv(), root.left().bv()) &&
         Aabb::Contains(root.bv(), root.right().bv());
}

// Create a DynamicBVH of tetrahedral meshes and intersect.
// Check that the candidates generated are the same as those generated by the
// brute force check.
GTEST_TEST(DynamicBoundingVolumeHierarchyTest, BVHSameCandidates) {
  const VolumeMesh<double> meshA = MakeSphereVolumeMesh<double>(
      Sphere(2.), 1., TessellationStrategy::kDenseInteriorVertices);
  VolumeMesh<double> meshB = MakeSphereVolumeMesh<double>(
      Sphere(1.), 0.5, TessellationStrategy::kDenseInteriorVertices);

  DynamicBvh bvhA(meshA.num_elements(), MeshCalculator(meshA));
  DynamicBvh bvhB(meshB.num_elements(), MeshCalculator(meshB));

  ASSERT_TRUE(ValidTree(bvhA.root_node()));
  ASSERT_TRUE(ValidTree(bvhB.root_node()));

  auto lexicographic = [](const std::pair<int, int>& a,
                          const std::pair<int, int>& b) {
    return (a.first == b.first) ? a.second < b.second : a.first < b.first;
  };

  std::vector<std::pair<int, int>> bvh_candidates =
      bvhA.GetCollisionCandidates(bvhB);
  std::vector<std::pair<int, int>> brute_force_candidates =
      BruteForceOverlap(meshA, meshB);

  // Sanity check, we actually have collision candidates.
  ASSERT_GT(ssize(bvh_candidates), 10);
  ASSERT_GT(ssize(brute_force_candidates), 10);

  std::sort(bvh_candidates.begin(), bvh_candidates.end(), lexicographic);
  std::sort(brute_force_candidates.begin(), brute_force_candidates.end(),
            lexicographic);

  EXPECT_TRUE(std::includes(bvh_candidates.begin(), bvh_candidates.end(),
                            brute_force_candidates.begin(),
                            brute_force_candidates.end()));

  // Test that refitting after an arbitrary transformation still produces the
  // expected candidates.
  RigidTransformd X_AB(RollPitchYawd(0.5 * M_PI, 0.2 * M_PI, 0.35 * M_PI),
                       Vector3d(0.2, 0.5, 0.4));
  meshB.TransformVertices(X_AB);
  bvhB.Refit(MeshCalculator(meshB));

  std::vector<std::pair<int, int>> transformed_bvh_candidates =
      bvhA.GetCollisionCandidates(bvhB);
  std::vector<std::pair<int, int>> transformed_brute_force_candidates =
      BruteForceOverlap(meshA, meshB);

  // Sanity check
  ASSERT_GT(ssize(transformed_bvh_candidates), 10);
  ASSERT_GT(ssize(transformed_brute_force_candidates), 10);

  std::sort(transformed_bvh_candidates.begin(),
            transformed_bvh_candidates.end(), lexicographic);
  std::sort(transformed_brute_force_candidates.begin(),
            transformed_brute_force_candidates.end(), lexicographic);

  EXPECT_TRUE(std::includes(transformed_bvh_candidates.begin(),
                            transformed_bvh_candidates.end(),
                            transformed_brute_force_candidates.begin(),
                            transformed_brute_force_candidates.end()));

  // Sanity check we have actually different candidates this time.
  std::vector<std::pair<int, int>> intersection;
  std::set_intersection(bvh_candidates.begin(), bvh_candidates.end(),
                        transformed_bvh_candidates.begin(),
                        transformed_bvh_candidates.end(),
                        std::back_inserter(intersection));
  ASSERT_LT(ssize(intersection), ssize(bvh_candidates));
}

// Check that the candidates generated by an "inflated" bvh contain all of the
// candidates generated by brute force intersecting the meshes while sampling a
// translated motion of one of the meshes.
GTEST_TEST(DynamicBoundingVolumeHierarchyTest, MovingLeaves) {
  const VolumeMesh<double> meshA = MakeSphereVolumeMesh<double>(
      Sphere(2.), 1, TessellationStrategy::kDenseInteriorVertices);
  VolumeMesh<double> meshB = MakeSphereVolumeMesh<double>(
      Sphere(1.), 0.5, TessellationStrategy::kDenseInteriorVertices);
  RigidTransformd X_AB(RollPitchYawd(0.5 * M_PI, 0.2 * M_PI, 0.35 * M_PI),
                       Vector3d(2.9, 0, 0));
  meshB.TransformVertices(X_AB);

  const Vector3d translation(-5.8, 0, 0);

  DynamicBvh bvhA(meshA.num_elements(), MeshCalculator(meshA));
  DynamicBvh bvhB(meshB.num_elements(),
                  TranslatedMeshCalculator(meshB, translation));

  ASSERT_TRUE(ValidTree(bvhA.root_node()));
  ASSERT_TRUE(ValidTree(bvhB.root_node()));

  auto lexicographic = [](const std::pair<int, int>& a,
                          const std::pair<int, int>& b) {
    return (a.first == b.first) ? a.second < b.second : a.first < b.first;
  };

  std::vector<std::pair<int, int>> bvh_candidates =
      bvhA.GetCollisionCandidates(bvhB);
  std::sort(bvh_candidates.begin(), bvh_candidates.end(), lexicographic);

  // Sanity check, we actually have collision candidates.
  ASSERT_GT(ssize(bvh_candidates), 10);

  // Transform the mesh by small increments collecting all collision candidates
  // along the way.
  constexpr double N = 10;
  std::vector<std::pair<int, int>> all_brute_force_candidates =
      BruteForceOverlap(meshA, meshB);
  for (int i = 1; i <= N; ++i) {
    meshB.TransformVertices(RigidTransformd((i / N) * translation));
    std::vector<std::pair<int, int>> candidates =
        BruteForceOverlap(meshA, meshB);
    all_brute_force_candidates.insert(all_brute_force_candidates.end(),
                                      candidates.begin(), candidates.end());
  }

  // Sort and remove duplicates.
  std::sort(all_brute_force_candidates.begin(),
            all_brute_force_candidates.end(), lexicographic);
  auto unique_end = std::unique(all_brute_force_candidates.begin(),
                                all_brute_force_candidates.end());
  all_brute_force_candidates.erase(unique_end,
                                   all_brute_force_candidates.end());

  // Sanity check, we actually have collision candidates.
  ASSERT_GT(ssize(all_brute_force_candidates), 10);

  // Check that the candidates generated by the "inflated" BVH are the same as
  // brute force with sampling.
  EXPECT_TRUE(std::includes(bvh_candidates.begin(), bvh_candidates.end(),
                            all_brute_force_candidates.begin(),
                            all_brute_force_candidates.end()));
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
