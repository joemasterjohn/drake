#include "drake/geometry/proximity/ccd.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/default_scalars.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;
using math::RigidTransform;
using math::RollPitchYaw;
using math::RotationMatrix;
namespace {

const std::array<std::array<int, 3>, 4> triangles = {
    {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}}};
const std::array<std::array<int, 2>, 6> edges = {
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

// Constructs a cannonical configuration of two tetraheda where the closest
// points between the two correspond to vertex 0 of the first tet and face {0,
// 1, 2} of the second tet.
template <typename T>
std::pair<std::array<Vector3<T>, 4>, std::array<Vector3<T>, 4>>
CreateVertexFaceTetPair(const RigidTransform<T> X_WG, const T theta,
                        const T desired_distance, int closest_vertex,
                        std::array<int, 3> closest_face) {
  const Vector3<T> p_RAp(0, 0, 0);
  const Vector3<T> p_RAq(-1, 1, 0);
  const Vector3<T> p_RAr(-1, -0.5, -0.5);
  const Vector3<T> p_RAs(-1, -0.5, 0.5);
  const Vector3<T> p_GBp(desired_distance, 0.5, 0.5);
  const Vector3<T> p_GBq(desired_distance, 0.5, -0.5);
  const Vector3<T> p_GBr(desired_distance, -1, 0);
  const Vector3<T> p_GBs(desired_distance + 1, 0, 0);

  // Rotate A around the x axis in frame G. This doesn't change the closet
  // points, but it rotates the normals adjacent to vertex p_RAp.
  const RigidTransform<T> X_GR(RotationMatrix<T>::MakeXRotation(theta));
  const RigidTransform<T> X_WR = X_WG * X_GR;
  std::array<Vector3<T>, 4> tet_A = {X_WR * p_RAp, X_WR * p_RAq, X_WR * p_RAr,
                                     X_WR * p_RAs};
  std::swap(
      tet_A[0],
      tet_A[closest_vertex]);  // Put the closest vertex at the desired index.
  std::array<Vector3<T>, 4> tet_B;
  int furthest_vertex =
      6 - (closest_face[0] + closest_face[1] + closest_face[2]);
  tet_B[closest_face[0]] = X_WG * p_GBp;
  tet_B[closest_face[1]] = X_WG * p_GBq;
  tet_B[closest_face[2]] = X_WG * p_GBr;
  tet_B[furthest_vertex] = X_WG * p_GBs;

  return std::make_pair(tet_A, tet_B);
}

// Constructs a cannonical configuration of two tetraheda where the closest
// points between the two correspond to points on edge {0,1} of the first tet
// and edge {0, 1} of the second tet.
template <typename T>
std::pair<std::array<Vector3<T>, 4>, std::array<Vector3<T>, 4>>
CreateEdgeEdgeTetPair(const RigidTransform<T> X_WG, const T theta,
                      const T desired_distance, int edge_A, int edge_B) {
  const Vector3<T> p_RAp(0, 0, -1);
  const Vector3<T> p_RAq(0, 0, 1);
  const Vector3<T> p_RAr(-1, -1, 0);
  const Vector3<T> p_RAs(-1, 1, 0);
  const Vector3<T> p_GBp(desired_distance, -1, 0);
  const Vector3<T> p_GBq(desired_distance, 1, 0);
  const Vector3<T> p_GBr(desired_distance + 1, 0, -1);
  const Vector3<T> p_GBs(desired_distance + 1, 0, 1);

  const auto e_A = edges[edge_A];
  const auto opposite_e_A = edges[5 - edge_A];
  const auto e_B = edges[edge_B];
  const auto opposite_e_B = edges[5 - edge_B];

  // Rotate A around the x axis in frame G. This doesn't change the closet
  // points, but it rotates the normals adjacent edge e_A.
  const RigidTransform<T> X_GR(RotationMatrix<T>::MakeXRotation(theta));
  const RigidTransform<T> X_WR = X_WG * X_GR;

  std::array<Vector3<T>, 4> tet_A;
  // Placed the closest edge at e_A and the not closest at opposite_e_A.
  tet_A[e_A[0]] = X_WR * p_RAp;
  tet_A[e_A[1]] = X_WR * p_RAq;
  tet_A[opposite_e_A[0]] = X_WR * p_RAr;
  tet_A[opposite_e_A[1]] = X_WR * p_RAs;

  // Placed the closest edge at e_B and the not closest at opposite_e_B.
  std::array<Vector3<T>, 4> tet_B;
  tet_B[e_B[0]] = X_WG * p_GBp;
  tet_B[e_B[1]] = X_WG * p_GBq;
  tet_B[opposite_e_B[0]] = X_WG * p_GBr;
  tet_B[opposite_e_B[1]] = X_WG * p_GBs;

  return std::make_pair(tet_A, tet_B);
}

template <typename T>
class TetTetDistanceTests : public ::testing::Test {};

using DefaultScalars = ::testing::Types<double, AutoDiffXd>;
TYPED_TEST_SUITE(TetTetDistanceTests, DefaultScalars);

TYPED_TEST(TetTetDistanceTests, Basic) {
  using T = TypeParam;
  const T kEps{1e3 * std::numeric_limits<double>::epsilon()};

  for (int iterations = 0; iterations < 20; ++iterations) {
    // Generate random transforms and distances
    const T desired_distance = 0.3 + 0.1 * Vector1<T>::Random()(0);
    Vector3<T> rpy = Vector3<T>::Random();
    RigidTransform<T> X_WA(RollPitchYaw<T>(rpy), Vector3<T>::Random());
    const T theta = M_PI * Vector1<T>::Random()(0);

    for (int i = 0; i < 4; ++i) {
      for (const std::array<int, 3> t : triangles) {
        const auto [tet_A, tet_B] =
            CreateVertexFaceTetPair(X_WA, theta, desired_distance, i, t);
        ClosestPointResult<T> result =
            ClosestPointTetrahedronToTetrahedron(tet_A, tet_B);
        EXPECT_LE((result.squared_dist - desired_distance * desired_distance),
                  kEps);
        EXPECT_EQ(result.closest_A.p, tet_A[i]);
        EXPECT_EQ(result.closest_A.type, ClosestPointType::Vertex);
        EXPECT_EQ(result.closest_A.indices[0], i);
        EXPECT_EQ(result.closest_B.type, ClosestPointType::Face);
        EXPECT_EQ(result.closest_B.indices, t);

        // Swap arguments A and B and see that we get the mirror result.
        result = ClosestPointTetrahedronToTetrahedron(tet_B, tet_A);
        EXPECT_LE((result.squared_dist - desired_distance * desired_distance),
                  kEps);
        EXPECT_EQ(result.closest_B.p, tet_A[i]);
        EXPECT_EQ(result.closest_B.type, ClosestPointType::Vertex);
        EXPECT_EQ(result.closest_B.indices[0], i);
        EXPECT_EQ(result.closest_A.type, ClosestPointType::Face);
        EXPECT_EQ(result.closest_A.indices, t);
      }
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        const auto e_A = edges[i];
        const auto e_B = edges[j];

        const auto [tet_A, tet_B] =
            CreateEdgeEdgeTetPair(X_WA, theta, desired_distance, i, j);
        ClosestPointResult<T> result =
            ClosestPointTetrahedronToTetrahedron(tet_A, tet_B);
        EXPECT_LE((result.squared_dist - desired_distance * desired_distance),
                  kEps);
        // Tets are constructed such that the witness points are at the
        // midpoints of the edges.
        EXPECT_LE(
            (result.closest_A.p - (tet_A[e_A[0]] + tet_A[e_A[1]]) / 2).norm(),
            kEps);
        EXPECT_EQ(result.closest_A.type, ClosestPointType::Edge);
        EXPECT_EQ(result.closest_A.indices[0], e_A[0]);
        EXPECT_EQ(result.closest_A.indices[1], e_A[1]);
        EXPECT_LE(
            (result.closest_B.p - (tet_B[e_B[0]] + tet_B[e_B[1]]) / 2).norm(),
            kEps);
        EXPECT_EQ(result.closest_B.type, ClosestPointType::Edge);
        EXPECT_EQ(result.closest_B.indices[0], e_B[0]);
        EXPECT_EQ(result.closest_B.indices[1], e_B[1]);

        // // Swap arguments A and B and see that we get the mirror result.
        result = ClosestPointTetrahedronToTetrahedron(tet_B, tet_A);
        EXPECT_LE((result.squared_dist - desired_distance * desired_distance),
                  kEps);
        // Tets are constructed such that the witness points are at the
        // midpoints of the edges.
        EXPECT_LE(
            (result.closest_A.p - (tet_B[e_B[0]] + tet_B[e_B[1]]) / 2).norm(),
            kEps);
        EXPECT_EQ(result.closest_A.type, ClosestPointType::Edge);
        EXPECT_EQ(result.closest_A.indices[0], e_B[0]);
        EXPECT_EQ(result.closest_A.indices[1], e_B[1]);
        EXPECT_LE(
            (result.closest_B.p - (tet_A[e_A[0]] + tet_A[e_A[1]]) / 2).norm(),
            kEps);
        EXPECT_EQ(result.closest_B.type, ClosestPointType::Edge);
        EXPECT_EQ(result.closest_B.indices[0], e_A[0]);
        EXPECT_EQ(result.closest_B.indices[1], e_A[1]);
      }
    }
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
