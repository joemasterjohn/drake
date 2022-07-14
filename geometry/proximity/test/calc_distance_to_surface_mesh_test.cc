#include "drake/geometry/proximity/calc_distance_to_surface_mesh.h"

#include <gtest/gtest.h>

#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;
using std::vector;

GTEST_TEST(CalcDistanceToSurfaceMeshTest, SingleTriangle) {
  const Vector3d p_TV0(0, 0, 0);
  const Vector3d p_TV1(1, 0, 0);
  const Vector3d p_TV2(2, 2, 0);
  const math::RollPitchYaw<double> rpy(1, 2, 3);
  const Vector3d position(4, 5, 6);
  const math::RigidTransform<double> X_WT(rpy, position);

  vector<Vector3d> vertices = {X_WT * p_TV0, X_WT * p_TV1, X_WT * p_TV2};
  vector<SurfaceTriangle> triangles = {{0, 1, 2}};
  TriangleSurfaceMesh<double> mesh_W(std::move(triangles), std::move(vertices));

  constexpr double kEps = 1e-14;
  {
    // The point is on the interior of the triangle.
    const Vector3d p_TQ(0.5, 0.25, 0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(0.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The point is on the edge of the triangle.
    const Vector3d p_TQ(0.5, 0.5, 0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(0.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The point is on a vertex of the triangle.
    const Vector3d p_TQ(2.0, 2.0, 0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(0.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The point is in the plane of the triangle, and the shortest distance is
    // achieved on an edge.
    const Vector3d p_TQ(0.5, -1.0, 0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(1.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The point is in the plane of the triangle, and the shortest distance is
    // achieved on a vertex.
    const Vector3d p_TQ(-3.0, -4.0, 0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(5.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The projection of the point is on the interior of the triangle.
    const Vector3d p_TQ(0.5, 0.25, 1.0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(1.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The projection of the point is on the edge of the triangle.
    const Vector3d p_TQ(0.5, 0.5, 1.0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(1.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The projection of the point is on a vertex of the triangle.
    const Vector3d p_TQ(2.0, 2.0, 1.0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(1.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The projection of the point is not in the triangle, and the shortest
    // distance is achieved on an edge.
    const Vector3d p_TQ(0.5, -1.0, 1.0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(std::sqrt(2.0), CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The projection of the point is not in the triangle, and the shortest
    // distance is achieved on a vertex.
    const Vector3d p_TQ(-3.0, -4.0, 1.0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(std::sqrt(26), CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
}

GTEST_TEST(CalcDistanceToSurfaceMeshTest, FullMesh) {
  const Vector3d p_TV0(1, 0, 0);
  const Vector3d p_TV1(0, 1, 0);
  const Vector3d p_TV2(0, 0, 1);
  const Vector3d p_TV3(-1, 0, 0);
  const Vector3d p_TV4(0, -1, 0);
  const Vector3d p_TV5(0, 0, -1);

  const math::RollPitchYaw<double> rpy(1, 2, 3);
  const Vector3d position(4, 5, 6);
  const math::RigidTransform<double> X_WT(rpy, position);

  vector<Vector3d> vertices = {X_WT * p_TV0, X_WT * p_TV1, X_WT * p_TV2,
                               X_WT * p_TV3, X_WT * p_TV4, X_WT * p_TV5};
  vector<SurfaceTriangle> triangles = {
      {0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {0, 2, 4},
      {0, 5, 1}, {1, 5, 3}, {3, 5, 4}, {0, 4, 5},
  };
  // The surface mesh of an octahedron.
  TriangleSurfaceMesh<double> mesh_W(std::move(triangles), std::move(vertices));

  constexpr double kEps = 1e-14;
  {
    // The point is on a triangle.
    const Vector3d p_TQ(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(0.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The point is on a edge.
    const Vector3d p_TQ(0.5, 0.5, 0.0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(0.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The point is at a vertex.
    const Vector3d p_TQ(1.0, 0.0, 0.0);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(0.0, CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
  }
  {
    // The point is "inside" the mesh
    const Vector3d p_TQ(1.0 / 6, 1.0 / 6, 1.0 / 6);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(1.0 / sqrt(12.0), CalcDistanceToSurfaceMesh(p_WQ, mesh_W),
                kEps);
  }
  {
    // The point is "outside" the mesh
    const Vector3d p_TQ(0.5, 0.5, 0.5);
    const Vector3d p_WQ = X_WT * p_TQ;
    EXPECT_NEAR(1.0 / sqrt(12.0), CalcDistanceToSurfaceMesh(p_WQ, mesh_W),
                kEps);
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
