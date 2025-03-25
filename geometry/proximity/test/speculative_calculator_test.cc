#include "drake/geometry/proximity/speculative_calculator.h"

#include <limits>
#include <tuple>

#include <gtest/gtest.h>

#include "drake/common/default_scalars.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity/ccd.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;
using hydroelastic::SoftGeometry;
using hydroelastic::SoftMesh;
using math::RigidTransform;
using math::RollPitchYaw;
using math::RotationMatrix;
using multibody::SpatialVelocity;

namespace {

// Triangles with CCW winding facing inwards.
const std::array<std::array<int, 3>, 4> triangles = {
    {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}}};
// Edges where e[i] and it's neighbor e[6-i] are an even permutation of
// {0,1,2,3} (i.e. positive volume).
const std::array<std::array<int, 2>, 6> edges = {
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {3, 1}, {2, 3}}};

template <typename T>
using Tetrahedron = std::array<Vector3<T>, 4>;

// Compute signed volume of {A,B,C,D}.
template <typename T>
T Volume(const Vector3<T>& A, const Vector3<T>& B, const Vector3<T>& C,
         const Vector3<T>& D) {
  return (D - A).dot((B - A).cross(C - A)) / 6.0;
}

// Calculate the volume of overlap of two tetrahedra. The tetrahedra are
// expected to overlap in simple way where the entire overlap volume is a single
// tetrahedron. These configurations are expected from small perturbations of a
// Vertex-Face or Edge-Edge kissing configuration.
template <typename T>
T CalcVolumeOfOverlap(const Tetrahedron<T>& tetA, const Tetrahedron<T>& tetB) {
  Tetrahedron<T> overlap;
  int num_intersections = 0;
  const auto sign = [](const T& v) {
    return v == T(0) ? 0 : (v < T(0) ? -1 : 1);
  };

  // Go through triangles of A and intersect with edges of B.
  for (const auto& t : triangles) {
    const Vector3<T>& a0 = tetA[t[0]];
    const Vector3<T>& a1 = tetA[t[1]];
    const Vector3<T>& a2 = tetA[t[2]];
    const Vector3<T> n = (a1 - a0).cross(a2 - a0).normalized();

    for (const auto& e : edges) {
      const Vector3<T>& b0 = tetB[e[0]];
      const Vector3<T>& b1 = tetB[e[1]];

      const T h0 = (b0 - a0).dot(n);
      const T h1 = (b1 - a0).dot(n);
      if (sign(h0) != sign(h1)) {
        const T wb0 = h1 / (h1 - h0);
        const T wb1 = T(1.0) - wb0;
        const Vector3<T> intersection = wb0 * b0 + wb1 * b1;
        const bool inside =
            (sign(n.dot((a1 - a0).cross(intersection - a0))) +
             sign(n.dot((a2 - a1).cross(intersection - a1))) +
             sign(n.dot((a0 - a2).cross(intersection - a2)))) == 3;

        if (inside) {
          // fmt::print("intersection: A {} {} {}  B {} {}\n", t[0], t[1], t[2],
          //            e[0], e[1]);
          DRAKE_ASSERT(num_intersections < 4);
          overlap[num_intersections] = intersection;
          num_intersections++;
        }
      }
    }
  }
  // Go through triangles of B and intersect with edges of A.
  for (const auto& t : triangles) {
    const Vector3<T>& b0 = tetB[t[0]];
    const Vector3<T>& b1 = tetB[t[1]];
    const Vector3<T>& b2 = tetB[t[2]];
    const Vector3<T> n = (b1 - b0).cross(b2 - b0).normalized();

    for (const auto& e : edges) {
      const Vector3<T>& a0 = tetA[e[0]];
      const Vector3<T>& a1 = tetA[e[1]];

      const T h0 = (a0 - b0).dot(n);
      const T h1 = (a1 - b0).dot(n);
      if (sign(h0) != sign(h1)) {
        const T wa0 = h1 / (h1 - h0);
        const T wa1 = T(1.0) - wa0;
        const Vector3<T> intersection = wa0 * a0 + wa1 * a1;
        const bool inside =
            (sign(n.dot((b1 - b0).cross(intersection - b0))) +
             sign(n.dot((b2 - b1).cross(intersection - b1))) +
             sign(n.dot((b0 - b2).cross(intersection - b2)))) == 3;

        if (inside) {
          // fmt::print("intersection: A {} {}  B {} {} {}\n", e[0], e[1], t[0],
          //            t[1], t[2]);
          DRAKE_ASSERT(num_intersections < 4);
          overlap[num_intersections] = intersection;
          num_intersections++;
        }
      }
    }
  }

  // Check for vertices inside of the other tet.
  for (int i = 0; i < 4; ++i) {
    const Vector3<T>& a = tetA[i];
    const Vector3<T>& b = tetB[i];
    int sA = 0;
    int sB = 0;

    for (const auto& t : triangles) {
      const Vector3<T>& a0 = tetA[t[0]];
      const Vector3<T>& a1 = tetA[t[1]];
      const Vector3<T>& a2 = tetA[t[2]];
      sB += sign(Volume(b, a0, a1, a2));
      const Vector3<T>& b0 = tetB[t[0]];
      const Vector3<T>& b1 = tetB[t[1]];
      const Vector3<T>& b2 = tetB[t[2]];
      sA += sign(Volume(a, b0, b1, b2));
    }

    if (std::abs(sA) == 4) {
      // fmt::print("intersection: A {}\n", i);
      DRAKE_ASSERT(num_intersections < 4);
      overlap[num_intersections] = a;
      num_intersections++;
    }
    if (std::abs(sB) == 4) {
      // fmt::print("intersection: B {}\n", i);
      DRAKE_ASSERT(num_intersections < 4);
      overlap[num_intersections] = b;
      num_intersections++;
    }
  }

  // fmt::print("num_intersections: {}\n\n", num_intersections);
  DRAKE_ASSERT(num_intersections == 0 || num_intersections == 4);

  if (num_intersections == 0) return T(0.0);

  const Vector3<T>& A = overlap[0];
  const Vector3<T>& B = overlap[1];
  const Vector3<T>& C = overlap[2];
  const Vector3<T>& D = overlap[3];

  // There is no guarantee that we computed the intersections in an order that
  // gives positive volume, so just take the absolute of the signed volume.
  const T signed_volume = Volume(A, B, C, D);
  return signed_volume >= 0 ? signed_volume : -signed_volume;
}

// Constructs a cannonical configuration of two tetraheda where the closest
// points between the two correspond to vertex 0 of the first tet and face {0,
// 1, 2} of the second tet.
template <typename T>
std::tuple<Tetrahedron<T>, Tetrahedron<T>, std::array<T, 4>, std::array<T, 4>>
CreateVertexFaceTetPair(const RigidTransform<T> X_WG, const T theta,
                        const T desired_distance, int closest_vertex,
                        std::array<int, 3> closest_face) {
  const Vector3<T> p_RAp(0, 0, 0);
  const Vector3<T> p_RAq(-1, 1, 0);
  const Vector3<T> p_RAr(-1, -0.5, -0.5);
  const Vector3<T> p_RAs(-1, -0.5, 0.5);
  const Vector3<T> p_GBp(desired_distance, 0.5, -0.5);
  const Vector3<T> p_GBq(desired_distance, 0.5, 0.5);
  const Vector3<T> p_GBr(desired_distance, -1, 0);
  const Vector3<T> p_GBs(desired_distance + 1, 0, 0);

  DRAKE_ASSERT(Volume(p_RAp, p_RAq, p_RAr, p_RAs) > 0);
  DRAKE_ASSERT(Volume(p_GBp, p_GBq, p_GBr, p_GBs) > 0);

  // Rotate A around the x axis in frame G. This doesn't change the closet
  // points, but it rotates the normals adjacent to vertex p_RAp.
  const RigidTransform<T> X_GR(RotationMatrix<T>::MakeXRotation(theta));
  const RigidTransform<T> X_WR = X_WG * X_GR;
  Tetrahedron<T> tet_A = {X_WR * p_RAp, X_WR * p_RAq, X_WR * p_RAr,
                          X_WR * p_RAs};
  // 0 pressure at the closest features, 1 elsewhere.
  std::array<T, 4> pressure_A = {T(0), T(1), T(1), T(1)};
  std::swap(
      tet_A[0],
      tet_A[closest_vertex]);  // Put the closest vertex at the desired index.
  std::swap(pressure_A[0], pressure_A[closest_vertex]);
  // Swap the other vertices (if neeeded) to keep positive volume.
  switch (closest_vertex) {
    case 0:
      break;
    case 1:
      std::swap(tet_A[2], tet_A[3]);
      std::swap(pressure_A[2], pressure_A[3]);
      break;
    case 2:
      std::swap(tet_A[1], tet_A[3]);
      std::swap(pressure_A[1], pressure_A[3]);
      break;
    case 3:
      std::swap(tet_A[1], tet_A[2]);
      std::swap(pressure_A[1], pressure_A[2]);
      break;
    default:
      DRAKE_ASSERT(false);
      break;
  }

  Tetrahedron<T> tet_B;
  std::array<T, 4> pressure_B;
  int furthest_vertex =
      6 - (closest_face[0] + closest_face[1] + closest_face[2]);
  tet_B[closest_face[0]] = X_WG * p_GBp;
  tet_B[closest_face[1]] = X_WG * p_GBq;
  tet_B[closest_face[2]] = X_WG * p_GBr;
  tet_B[furthest_vertex] = X_WG * p_GBs;
  // 0 pressure at the closest features, 1 elsewhere.
  pressure_B[closest_face[0]] = T(0);
  pressure_B[closest_face[1]] = T(0);
  pressure_B[closest_face[2]] = T(0);
  pressure_B[furthest_vertex] = T(1);

  DRAKE_ASSERT(Volume(tet_A[0], tet_A[1], tet_A[2], tet_A[3]) > 0);
  DRAKE_ASSERT(Volume(tet_B[0], tet_B[1], tet_B[2], tet_B[3]) > 0);

  return std::make_tuple(tet_A, tet_B, pressure_A, pressure_B);
}

// Constructs a cannonical configuration of two tetraheda where the closest
// points between the two correspond to points on edge {0,1} of the first tet
// and edge {0, 1} of the second tet.
template <typename T>
std::tuple<Tetrahedron<T>, Tetrahedron<T>, std::array<T, 4>, std::array<T, 4>>
CreateEdgeEdgeTetPair(const RigidTransform<T> X_WG, const T theta,
                      const T desired_distance, int edge_A, int edge_B) {
  const Vector3<T> p_RAp(0, 0, -1);
  const Vector3<T> p_RAq(0, 0, 1);
  const Vector3<T> p_RAr(-1, 1, 0);
  const Vector3<T> p_RAs(-1, -1, 0);
  const Vector3<T> p_GBp(desired_distance, -1, 0);
  const Vector3<T> p_GBq(desired_distance, 1, 0);
  const Vector3<T> p_GBr(desired_distance + 1, 0, 1);
  const Vector3<T> p_GBs(desired_distance + 1, 0, -1);

  DRAKE_ASSERT(Volume(p_RAp, p_RAq, p_RAr, p_RAs) > 0);
  DRAKE_ASSERT(Volume(p_GBp, p_GBq, p_GBr, p_GBs) > 0);

  const auto e_A = edges[edge_A];
  const auto opposite_e_A = edges[5 - edge_A];
  const auto e_B = edges[edge_B];
  const auto opposite_e_B = edges[5 - edge_B];

  // Rotate A around the x axis in frame G. This doesn't change the closet
  // points, but it rotates the normals adjacent edge e_A.
  const RigidTransform<T> X_GR(RotationMatrix<T>::MakeXRotation(theta));
  const RigidTransform<T> X_WR = X_WG * X_GR;

  Tetrahedron<T> tet_A;
  std::array<T, 4> pressure_A;
  // Placed the closest edge at e_A and the not closest at opposite_e_A.
  tet_A[e_A[0]] = X_WR * p_RAp;
  tet_A[e_A[1]] = X_WR * p_RAq;
  tet_A[opposite_e_A[0]] = X_WR * p_RAr;
  tet_A[opposite_e_A[1]] = X_WR * p_RAs;
  // 0 pressure at the closest features, 1 elsewhere.
  pressure_A[e_A[0]] = T(0.0);
  pressure_A[e_A[1]] = T(0.0);
  pressure_A[opposite_e_A[0]] = T(1.0);
  pressure_A[opposite_e_A[1]] = T(1.0);

  // Placed the closest edge at e_B and the not closest at opposite_e_B.
  Tetrahedron<T> tet_B;
  std::array<T, 4> pressure_B;
  tet_B[e_B[0]] = X_WG * p_GBp;
  tet_B[e_B[1]] = X_WG * p_GBq;
  tet_B[opposite_e_B[0]] = X_WG * p_GBr;
  tet_B[opposite_e_B[1]] = X_WG * p_GBs;
  // 0 pressure at the closest features, 1 elsewhere.
  pressure_B[e_B[0]] = T(0.0);
  pressure_B[e_B[1]] = T(0.0);
  pressure_B[opposite_e_B[0]] = T(1.0);
  pressure_B[opposite_e_B[1]] = T(1.0);

  DRAKE_ASSERT(Volume(tet_A[0], tet_A[1], tet_A[2], tet_A[3]) > 0);
  DRAKE_ASSERT(Volume(tet_B[0], tet_B[1], tet_B[2], tet_B[3]) > 0);

  return std::make_tuple(tet_A, tet_B, pressure_A, pressure_B);
}

template <typename T>
class TetTetDistanceTests : public ::testing::Test {};

// TODO(joemasterjohn): Test on AutoDiffXd after debugging.
// using DefaultScalars = ::testing::Types<double, AutoDiffXd>;
using DefaultScalars = ::testing::Types<double>;
TYPED_TEST_SUITE(TetTetDistanceTests, DefaultScalars);

template <typename T>
SoftGeometry MakeSoftGeometry(Tetrahedron<T> tet,
                              const std::array<T, 4>& pressure) {
  std::vector<VolumeElement> elements = {VolumeElement(0, 1, 2, 3)};
  std::vector<Vector3<double>> vertices = {
      ExtractDoubleOrThrow(tet[0]), ExtractDoubleOrThrow(tet[1]),
      ExtractDoubleOrThrow(tet[2]), ExtractDoubleOrThrow(tet[3])};
  std::vector<double> pressures = {
      ExtractDoubleOrThrow(pressure[0]), ExtractDoubleOrThrow(pressure[1]),
      ExtractDoubleOrThrow(pressure[2]), ExtractDoubleOrThrow(pressure[3])};
  std::unique_ptr<VolumeMesh<double>> mesh =
      std::make_unique<VolumeMesh<double>>(std::move(elements),
                                           std::move(vertices));
  std::unique_ptr<VolumeMeshFieldLinear<double, double>> field =
      std::make_unique<VolumeMeshFieldLinear<double, double>>(
          std::move(pressures), mesh.get());
  return SoftGeometry(SoftMesh(std::move(mesh), std::move(field)));
}

template <typename T>
std::vector<SpeculativeContactSurface<T>> TestExpectedSurface(
    GeometryId id_A, GeometryId id_B, const Tetrahedron<T>& tet_A,
    const Tetrahedron<T>& tet_B, const std::array<T, 4>& pressure_A,
    const std::array<T, 4>& pressure_B, const RigidTransform<T>& X_WA,
    const RigidTransform<T>& X_WB, const SpatialVelocity<T> V_WA,
    const SpatialVelocity<T> V_WB, const double dt,
    const int expected_num_candidates, const int expected_num_surfaces) {
  // Soft geometries.
  SoftGeometry soft_A = MakeSoftGeometry(tet_A, pressure_A);
  SoftGeometry soft_B = MakeSoftGeometry(tet_B, pressure_B);
  // Bounding spheres.
  const std::vector<PosedSphere<double>> s_A = {MinimumBoundingSphere(
      ExtractDoubleOrThrow(tet_A[0]), ExtractDoubleOrThrow(tet_A[1]),
      ExtractDoubleOrThrow(tet_A[2]), ExtractDoubleOrThrow(tet_A[3]))};
  const std::vector<PosedSphere<double>> s_B = {MinimumBoundingSphere(
      ExtractDoubleOrThrow(tet_B[0]), ExtractDoubleOrThrow(tet_B[1]),
      ExtractDoubleOrThrow(tet_B[2]), ExtractDoubleOrThrow(tet_B[3]))};

  // Refit the BVHs to zero velocity.
  soft_A.mutable_soft_mesh().mutable_mesh_dynamic_bvh().Refit(
      hydroelastic::MovingBoundingSphereAabbCalculator(s_A, X_WA, V_WA, dt));
  soft_B.mutable_soft_mesh().mutable_mesh_dynamic_bvh().Refit(
      hydroelastic::MovingBoundingSphereAabbCalculator(s_B, X_WB, V_WB, dt));

  std::vector<std::pair<int, int>> candidates =
      soft_A.soft_mesh().mesh_dynamic_bvh().GetCollisionCandidates(
          soft_B.soft_mesh().mesh_dynamic_bvh());
  EXPECT_EQ(ssize(candidates), expected_num_candidates);

  std::vector<SpeculativeContactSurface<T>> speculative_surfaces;
  ComputeSpeculativeContactSurfaceByClosestPoints(id_A, id_B, soft_A, soft_B,
                                                  X_WA, X_WB, V_WA, V_WB, dt,
                                                  &speculative_surfaces);

  EXPECT_EQ(ssize(speculative_surfaces), expected_num_surfaces);

  return speculative_surfaces;
}

TYPED_TEST(TetTetDistanceTests, Basic) {
  using T = TypeParam;
  const double dt = 0.1;

  // Arbitrary Ids.
  GeometryId id_A = GeometryId::get_new_id();
  GeometryId id_B = GeometryId::get_new_id();

  // Meshes are already specified in world.
  RigidTransform<T> X_WA = RigidTransform<T>::Identity();
  RigidTransform<T> X_WB = RigidTransform<T>::Identity();

  for (int iterations = 0; iterations < 20; ++iterations) {
    // Generate arbitrary random transforms and distances
    Vector3<T> rpy = Vector3<T>::Random();
    RigidTransform<T> X_WG(RollPitchYaw<T>(rpy), Vector3<T>::Random());
    // Small rotation to perturb while maintaining hand crafted closest
    // features.
    const T theta = -0.1 + 0.2 * M_PI * Vector1<T>::Random()(0);
    // Initially use zero velocity.
    SpatialVelocity<T> V_WA(Vector3<T>::Zero(), Vector3<T>::Zero());
    SpatialVelocity<T> V_WB(Vector3<T>::Zero(), Vector3<T>::Zero());

    // Distance such that the bounding boxes do not overlap.
    T desired_distance = 3;
    for (int i = 0; i < 4; ++i) {
      for (const std::array<int, 3> t : triangles) {
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateVertexFaceTetPair(X_WG, theta, T(desired_distance), i, t);
        ASSERT_FALSE(Intersects(tet_A, tet_B));
        TestExpectedSurface(
            id_A, id_B, tet_A, tet_B, pressure_A, pressure_B, X_WA, X_WB, V_WA,
            V_WB, dt, 0 /* expected candidates */, 0 /* expected surfaces */);
      }
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateEdgeEdgeTetPair(X_WG, theta, T(desired_distance), i, j);
        ASSERT_FALSE(Intersects(tet_A, tet_B));
        TestExpectedSurface(
            id_A, id_B, tet_A, tet_B, pressure_A, pressure_B, X_WA, X_WB, V_WA,
            V_WB, dt, 0 /* expected candidates */, 0 /* expected surfaces */);
      }
    }

    // Tests the situation where the Aabbs overlap but the tetrahedra do not, no
    // contact surface is created because the relative velocity is 0.
    desired_distance = 0.1;
    for (int i = 0; i < 4; ++i) {
      for (const std::array<int, 3> t : triangles) {
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateVertexFaceTetPair(X_WG, theta, T(desired_distance), i, t);
        ASSERT_FALSE(Intersects(tet_A, tet_B));
        TestExpectedSurface(
            id_A, id_B, tet_A, tet_B, pressure_A, pressure_B, X_WA, X_WB, V_WA,
            V_WB, dt, 1 /* expected candidates */, 0 /* expected surfaces */);
      }
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateEdgeEdgeTetPair(X_WG, theta, T(desired_distance), i, j);
        ASSERT_FALSE(Intersects(tet_A, tet_B));
        TestExpectedSurface(
            id_A, id_B, tet_A, tet_B, pressure_A, pressure_B, X_WA, X_WB, V_WA,
            V_WB, dt, 1 /* expected candidates */, 0 /* expected surfaces */);
      }
    }

    // Tests the situation where the tetrahedra are disjoint, their static
    // bounding volumes do not overlap, but they have sufficient velocity such
    // that the inflated bounding volumes overlap. We already verified that this
    // distance with 0 velocity produces no candidates, now we expect one
    // candidate.
    // Distance such that the static bounding boxes do not overlap.
    desired_distance = 3;
    // Velocity such that the inflated bounding boxes do not overlap.
    V_WA = SpatialVelocity<T>(
        Vector3<T>::Zero(),
        X_WG.rotation() * Vector3<T>(2 * desired_distance / dt, 0, 0));
    for (int i = 0; i < 4; ++i) {
      for (const std::array<int, 3> t : triangles) {
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateVertexFaceTetPair(X_WG, theta, T(desired_distance), i, t);
        ASSERT_FALSE(Intersects(tet_A, tet_B));
        TestExpectedSurface(
            id_A, id_B, tet_A, tet_B, pressure_A, pressure_B, X_WA, X_WB, V_WA,
            V_WB, dt, 1 /* expected candidates */, 1 /* expected surfaces */);
      }
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateEdgeEdgeTetPair(X_WG, theta, T(desired_distance), i, j);
        ASSERT_FALSE(Intersects(tet_A, tet_B));
        TestExpectedSurface(
            id_A, id_B, tet_A, tet_B, pressure_A, pressure_B, X_WA, X_WB, V_WA,
            V_WB, dt, 1 /* expected candidates */, 1 /* expected surfaces */);
      }
    }

    // Tests the situation where the Aabbs overlap and the tetrahedra overlap,
    // no contact surface is created because the tetrahedra overlap.
    desired_distance = -0.2;
    // Make sure there is non-zero relative velocity between the tetrahedra so
    // the query doesn't exit because of 0 relative velocity.
    V_WA = SpatialVelocity<T>(Vector3<T>::Zero(), Vector3<T>(1.0, 0, 0));
    for (int i = 0; i < 4; ++i) {
      for (const std::array<int, 3> t : triangles) {
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateVertexFaceTetPair(X_WG, theta, T(desired_distance), i, t);
        ASSERT_TRUE(Intersects(tet_A, tet_B));
        TestExpectedSurface(
            id_A, id_B, tet_A, tet_B, pressure_A, pressure_B, X_WA, X_WB, V_WA,
            V_WB, dt, 1 /* expected candidates */, 0 /* expected surfaces */);
      }
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateEdgeEdgeTetPair(X_WG, theta, T(desired_distance), i, j);
        ASSERT_TRUE(Intersects(tet_A, tet_B));
        TestExpectedSurface(
            id_A, id_B, tet_A, tet_B, pressure_A, pressure_B, X_WA, X_WB, V_WA,
            V_WB, dt, 1 /* expected candidates */, 0 /* expected surfaces */);
      }
    }

    // Test various properties of the speculative contact surface.
    const T kEps{std::numeric_limits<double>::epsilon()};
    desired_distance = 0.1;
    const T dx_dt = 2 * desired_distance / dt;
    V_WA = SpatialVelocity<T>(Vector3<T>::Zero(),
                              X_WG.rotation() * Vector3<T>(dx_dt, 0, 0));
    for (int i = 0; i < 4; ++i) {
      for (const std::array<int, 3> t : triangles) {
        desired_distance = 0.1;
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateVertexFaceTetPair(X_WG, theta, T(desired_distance), i, t);
        ASSERT_FALSE(Intersects(tet_A, tet_B));
        std::vector<SpeculativeContactSurface<T>> surfaces =
            TestExpectedSurface(id_A, id_B, tet_A, tet_B, pressure_A,
                                pressure_B, X_WA, X_WB, V_WA, V_WB, dt,
                                1 /* expected candidates */,
                                1 /* expected surfaces */);
        ASSERT_EQ(ssize(surfaces), 1);

        const SpeculativeContactSurface<T>& surface = surfaces[0];

        EXPECT_EQ(surface.id_A(), id_A);
        EXPECT_EQ(surface.id_B(), id_B);
        EXPECT_EQ(surface.num_contact_points(), 1);

        // In 1D we have a distance of desired_distance and a relative velocity
        // of dx_dt. Solve for t.
        EXPECT_NEAR(surface.time_of_contact()[0], desired_distance / dx_dt,
                    kEps);
        // Only geometry A has velocity, thus the contact point on the segments
        // between P and Q will be Q.
        EXPECT_TRUE(CompareMatrices(surface.p_WC()[0],
                                    surface.closest_points()[0].closest_B.p,
                                    kEps, MatrixCompareType::relative));
        // Volume normal zhat should point from B into A, thus from Q to P.
        const Vector3<T> expected_zhat_BA_W =
            (surface.closest_points()[0].closest_A.p -
             surface.closest_points()[0].closest_B.p)
                .normalized();
        EXPECT_TRUE(CompareMatrices(surface.zhat_BA_W()[0], expected_zhat_BA_W,
                                    1e1 * kEps, MatrixCompareType::relative));

        // Test that the local polynomial approximation of the overlap volume
        // matches the volume computed explicitly on the transformed tetrahedra.
        for (int delta = 1; delta < 8; ++delta) {
          desired_distance = -0.05 * delta;
          const auto [tet_A_overlap, tet_B_overlap, pressure_A_overlap,
                      pressure_B_overlap] =
              CreateVertexFaceTetPair(X_WG, theta, T(desired_distance), i, t);
          EXPECT_NEAR(
              CalcVolumeOfOverlap(tet_A_overlap, tet_B_overlap),
              std::pow(-desired_distance, 3) * surfaces[0].coefficient()[0],
              kEps);
        }

        // TODO(joemasterjohn): Clean up these tests to not make duplicates of
        // soft_A and soft_B. Also make the test smaller by consolidating
        // duplicated code.
        SoftGeometry soft_A = MakeSoftGeometry(tet_A, pressure_A);
        SoftGeometry soft_B = MakeSoftGeometry(tet_B, pressure_B);
        const Vector3<T> expected_grad_eA_W =
            X_WA.rotation() * soft_A.pressure_field().EvaluateGradient(0);
        const Vector3<T> expected_grad_eB_W =
            X_WB.rotation() * soft_B.pressure_field().EvaluateGradient(0);
        const Vector3<T> expected_nhat_BA_W =
            (expected_grad_eA_W - expected_grad_eB_W).normalized();
        EXPECT_EQ(surface.grad_eA_W()[0], expected_grad_eA_W);
        EXPECT_EQ(surface.grad_eB_W()[0], expected_grad_eB_W);
        EXPECT_EQ(surface.nhat_BA_W()[0], expected_nhat_BA_W);
      }
    }

    // Test swapping the place of A and B.
    for (int i = 0; i < 4; ++i) {
      for (const std::array<int, 3> t : triangles) {
        desired_distance = 0.1;
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateVertexFaceTetPair(X_WG, theta, T(desired_distance), i, t);
        ASSERT_FALSE(Intersects(tet_A, tet_B));
        // Swap the order to the query.
        std::vector<SpeculativeContactSurface<T>> surfaces =
            TestExpectedSurface(id_B, id_A, tet_B, tet_A, pressure_B,
                                pressure_A, X_WB, X_WA, V_WB, V_WA, dt,
                                1 /* expected candidates */,
                                1 /* expected surfaces */);
        ASSERT_EQ(ssize(surfaces), 1);

        const SpeculativeContactSurface<T>& surface = surfaces[0];

        EXPECT_EQ(surface.id_A(), id_B);
        EXPECT_EQ(surface.id_B(), id_A);
        EXPECT_EQ(surface.num_contact_points(), 1);

        // In 1D we have a distance of desired_distance and a relative velocity
        // of dx_dt. Solve for t.
        EXPECT_NEAR(surface.time_of_contact()[0], desired_distance / dx_dt,
                    kEps);
        // Only geometry B has velocity, thus the contact point on the segments
        // between P and Q will be P.
        EXPECT_TRUE(CompareMatrices(surface.p_WC()[0],
                                    surface.closest_points()[0].closest_A.p,
                                    kEps, MatrixCompareType::relative));
        // Volume normal zhat should point from B into A, thus from Q to P.
        const Vector3<T> expected_zhat_BA_W =
            (surface.closest_points()[0].closest_A.p -
             surface.closest_points()[0].closest_B.p)
                .normalized();
        EXPECT_TRUE(CompareMatrices(surface.zhat_BA_W()[0], expected_zhat_BA_W,
                                    1e1 * kEps, MatrixCompareType::relative));

        // Test that the local polynomial approximation of the overlap volume
        // matches the volume computed explicitly on the transformed tetrahedra.
        for (int delta = 1; delta < 8; ++delta) {
          desired_distance = -0.05 * delta;
          const auto [tet_A_overlap, tet_B_overlap, pressure_A_overlap,
                      pressure_B_overlap] =
              CreateVertexFaceTetPair(X_WG, theta, T(desired_distance), i, t);
          EXPECT_NEAR(
              CalcVolumeOfOverlap(tet_A_overlap, tet_B_overlap),
              std::pow(-desired_distance, 3) * surfaces[0].coefficient()[0],
              kEps);
        }

        SoftGeometry soft_A = MakeSoftGeometry(tet_B, pressure_B);
        SoftGeometry soft_B = MakeSoftGeometry(tet_A, pressure_A);
        const Vector3<T> expected_grad_eA_W =
            X_WA.rotation() * soft_A.pressure_field().EvaluateGradient(0);
        const Vector3<T> expected_grad_eB_W =
            X_WB.rotation() * soft_B.pressure_field().EvaluateGradient(0);
        const Vector3<T> expected_nhat_BA_W =
            (expected_grad_eA_W - expected_grad_eB_W).normalized();
        EXPECT_EQ(surface.grad_eA_W()[0], expected_grad_eA_W);
        EXPECT_EQ(surface.grad_eB_W()[0], expected_grad_eB_W);
        EXPECT_EQ(surface.nhat_BA_W()[0], expected_nhat_BA_W);
      }
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        desired_distance = 0.1;
        const auto [tet_A, tet_B, pressure_A, pressure_B] =
            CreateEdgeEdgeTetPair(X_WG, theta, T(desired_distance), i, j);
        ASSERT_FALSE(Intersects(tet_A, tet_B));
        std::vector<SpeculativeContactSurface<T>> surfaces =
            TestExpectedSurface(id_A, id_B, tet_A, tet_B, pressure_A,
                                pressure_B, X_WA, X_WB, V_WA, V_WB, dt,
                                1 /* expected candidates */,
                                1 /* expected surfaces */);
        ASSERT_EQ(ssize(surfaces), 1);

        const SpeculativeContactSurface<T>& surface = surfaces[0];

        EXPECT_EQ(surface.id_A(), id_A);
        EXPECT_EQ(surface.id_B(), id_B);
        EXPECT_EQ(surface.num_contact_points(), 1);

        // In 1D we have a distance of desired_distance and a relative velocity
        // of dx_dt. Solve for t.
        EXPECT_NEAR(surface.time_of_contact()[0], desired_distance / dx_dt,
                    kEps);
        // Only geometry A has velocity, thus the contact point on the segments
        // between P and Q will be Q.
        EXPECT_TRUE(CompareMatrices(surface.p_WC()[0],
                                    surface.closest_points()[0].closest_B.p,
                                    kEps, MatrixCompareType::relative));
        // Volume normal zhat should point from B into A, thus from Q to P.
        const Vector3<T> expected_zhat_BA_W =
            (surface.closest_points()[0].closest_A.p -
             surface.closest_points()[0].closest_B.p)
                .normalized();
        EXPECT_TRUE(CompareMatrices(surface.zhat_BA_W()[0], expected_zhat_BA_W,
                                    1e2 * kEps, MatrixCompareType::relative));

        // Test that the local polynomial approximation of the overlap volume
        // matches the volume computed explicitly on the transformed tetrahedra.
        for (int delta = 1; delta < 8; ++delta) {
          desired_distance = -0.05 * delta;
          const auto [tet_A_overlap, tet_B_overlap, pressure_A_overlap,
                      pressure_B_overlap] =
              CreateEdgeEdgeTetPair(X_WG, theta, T(desired_distance), i, j);
          EXPECT_NEAR(
              CalcVolumeOfOverlap(tet_A_overlap, tet_B_overlap),
              std::pow(-desired_distance, 3) * surfaces[0].coefficient()[0],
              kEps);
        }

        SoftGeometry soft_A = MakeSoftGeometry(tet_A, pressure_A);
        SoftGeometry soft_B = MakeSoftGeometry(tet_B, pressure_B);
        const Vector3<T> expected_grad_eA_W =
            X_WA.rotation() * soft_A.pressure_field().EvaluateGradient(0);
        const Vector3<T> expected_grad_eB_W =
            X_WB.rotation() * soft_B.pressure_field().EvaluateGradient(0);
        const Vector3<T> expected_nhat_BA_W =
            (expected_grad_eA_W - expected_grad_eB_W).normalized();
        EXPECT_EQ(surface.grad_eA_W()[0], expected_grad_eA_W);
        EXPECT_EQ(surface.grad_eB_W()[0], expected_grad_eB_W);
        EXPECT_EQ(surface.nhat_BA_W()[0], expected_nhat_BA_W);
      }
    }
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
