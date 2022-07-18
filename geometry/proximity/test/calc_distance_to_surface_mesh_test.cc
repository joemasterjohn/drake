#include "drake/geometry/proximity/calc_distance_to_surface_mesh.h"

#include <gtest/gtest.h>

#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;
using std::vector;

double CalcDistanceToLineAB(const Vector3d& p_TQ) {
  return std::sqrt(p_TQ(1) * p_TQ(1) + p_TQ(2) * p_TQ(2));
}

double CalcDistanceToLineAC(const Vector3d& p_TQ) {
  return std::sqrt(p_TQ(0) * p_TQ(0) + p_TQ(2) * p_TQ(2));
}

double CalcDistanceToLineBC(const Vector3d& p_TQ) {
  // p_TP is the projection of p_TQ on the line BC.
  const Vector3d p_TP((1.0 + p_TQ(0) - p_TQ(1)) / 2.0,
                      (1.0 - p_TQ(0) + p_TQ(1)) / 2.0, 0.0);
  return (p_TQ - p_TP).norm();
}

double CalcDistanceToA(const Vector3d& p_TQ) { return p_TQ.norm(); }

double CalcDistanceToB(const Vector3d& p_TQ) {
  return (p_TQ - Vector3d(1, 0, 0)).norm();
}

double CalcDistanceToC(const Vector3d& p_TQ) {
  return (p_TQ - Vector3d(0, 1, 0)).norm();
}

double CalcDistanceToPlaneABC(const Vector3d& p_TQ) {
  return std::abs(p_TQ(2));
}

/* We test the distance calculation from various points to a triangle ABC that
 has its three vertices at (0, 0, 0), (1, 0, 0), and (0, 1, 0) when posed in its
 reference configuration.

                  \  4  |
                    \   |
                      \ | y-axis
                      C •
                        | \
                        |   \
                        |     \
                        |       \
                        |         \
                3       |    0      \     5
                        |             \
                        |               \
        --------------- • --------------- • ------- x-axis
                      A |                 B \
                2       |    1                \  6
                        |                       \
 We sample points from regions 0 through 6 (with negative, positive, and zero z
 values) and verify that the calculated distances match analytic results. */
GTEST_TEST(CalcDistanceToSurfaceMeshTest, SingleTriangle) {
  const Vector3d p_TA(0, 0, 0);
  const Vector3d p_TB(1, 0, 0);
  const Vector3d p_TC(0, 1, 0);
  const math::RollPitchYaw<double> rpy_WT(1, 2, 3);
  const Vector3d p_WTo(4, 5, 6);
  const math::RigidTransform<double> X_WT(rpy_WT, p_WTo);

  vector<Vector3d> vertices = {X_WT * p_TA, X_WT * p_TB, X_WT * p_TC};
  vector<SurfaceTriangle> triangles = {{0, 1, 2}};
  TriangleSurfaceMesh<double> mesh_W(std::move(triangles), std::move(vertices));

  constexpr double kEps = 1e-14;
  constexpr double kZScale = 0.5;
  // Region 0.
  {
    for (int sign_z = -1; sign_z <= 1; ++sign_z) {
      const Vector3d p_TQ(0.25, 0.25, sign_z * kZScale);
      const Vector3d p_WQ = X_WT * p_TQ;
      EXPECT_NEAR(CalcDistanceToPlaneABC(p_TQ),
                  CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
    }
  }
  // Region 1.
  {
    for (int sign_z = -1; sign_z <= 1; ++sign_z) {
      const Vector3d p_TQ(0.25, -0.25, sign_z * kZScale);
      const Vector3d p_WQ = X_WT * p_TQ;
      EXPECT_NEAR(CalcDistanceToLineAB(p_TQ),
                  CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
    }
  }
  // Region 2.
  {
    for (int sign_z = -1; sign_z <= 1; ++sign_z) {
      const Vector3d p_TQ(-0.25, -0.25, sign_z * kZScale);
      const Vector3d p_WQ = X_WT * p_TQ;
      EXPECT_NEAR(CalcDistanceToA(p_TQ),
                  CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
    }
  }
  // Region 3.
  {
    for (int sign_z = -1; sign_z <= 1; ++sign_z) {
      const Vector3d p_TQ(-0.25, 0.0, sign_z * kZScale);
      const Vector3d p_WQ = X_WT * p_TQ;
      EXPECT_NEAR(CalcDistanceToLineAC(p_TQ),
                  CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
    }
  }
  // Region 4.
  {
    for (int sign_z = -1; sign_z <= 1; ++sign_z) {
      const Vector3d p_TQ(-0.25, 1.5, sign_z * kZScale);
      const Vector3d p_WQ = X_WT * p_TQ;
      EXPECT_NEAR(CalcDistanceToC(p_TQ),
                  CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
    }
  }
  // Region 5.
  {
    for (int sign_z = -1; sign_z <= 1; ++sign_z) {
      const Vector3d p_TQ(1.0, 2.0, sign_z * kZScale);
      const Vector3d p_WQ = X_WT * p_TQ;
      EXPECT_NEAR(CalcDistanceToLineBC(p_TQ),
                  CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
    }
  }
  // Region 6.
  {
    for (int sign_z = -1; sign_z <= 1; ++sign_z) {
      const Vector3d p_TQ(1.2, -0.3, sign_z * kZScale);
      const Vector3d p_WQ = X_WT * p_TQ;
      EXPECT_NEAR(CalcDistanceToB(p_TQ),
                  CalcDistanceToSurfaceMesh(p_WQ, mesh_W), kEps);
    }
  }
}

/* Tests that given a mesh with two triangles, the distance to the mesh is
  indeed taking the minimum of the distance to each triangles.

                        |  y-axis
                        |
                        |
                      C •
                      / | \
                    /   |   \
                  /     |     \
                /       |       \
              /         |         \
            /           |           \
          /             |             \
        /               |               \
  ---- • -------------- • --------------- • ------- x-axis
     D                A |                 B
                        |
                        |
*/
GTEST_TEST(CalcDistanceToSurfaceMeshTest, MultipleTriangles) {
  const Vector3d p_TA(0, 0, 0);
  const Vector3d p_TB(1, 0, 0);
  const Vector3d p_TC(0, 1, 0);
  const Vector3d p_TD(-1, 0, 0);

  const math::RollPitchYaw<double> rpy_WT(1, 2, 3);
  const Vector3d p_WTo(4, 5, 6);
  const math::RigidTransform<double> X_WT(rpy_WT, p_WTo);

  vector<Vector3d> vertices = {X_WT * p_TA, X_WT * p_TB, X_WT * p_TC,
                               X_WT * p_TD};
  vector<SurfaceTriangle> triangles = {{0, 1, 2}, {0, 2, 3}};
  TriangleSurfaceMesh<double> mesh_W(std::move(triangles), std::move(vertices));

  constexpr double kEps = 1e-14;
  // Q0 is on the equidistance surface of the two triangles.
  const Vector3d p_TQ0(0, 0.5, 1.0);
  const Vector3d p_WQ0 = X_WT * p_TQ0;
  const double d_Q0 = CalcDistanceToSurfaceMesh(p_WQ0, mesh_W);
  EXPECT_NEAR(CalcDistanceToPlaneABC(p_TQ0), d_Q0, kEps);
  // Q1 is slightly to the left of Q0.
  const Vector3d p_TQ1(-1e-3, 0.5, 1.0);
  const Vector3d p_WQ1 = X_WT * p_TQ1;
  const double d_Q1 = CalcDistanceToSurfaceMesh(p_WQ1, mesh_W);
  EXPECT_NEAR(CalcDistanceToPlaneABC(p_TQ1), d_Q1, kEps);
  EXPECT_LE(d_Q1, CalcDistanceToLineAC(p_TQ1));

  // Q2 is slightly to the right of Q0.
  const Vector3d p_TQ2(1e-3, 0.5, 1.0);
  const Vector3d p_WQ2 = X_WT * p_TQ2;
  const double d_Q2 = CalcDistanceToSurfaceMesh(p_WQ2, mesh_W);
  EXPECT_NEAR(CalcDistanceToPlaneABC(p_TQ2), d_Q2, kEps);
  EXPECT_LE(d_Q2, CalcDistanceToLineAC(p_TQ1));
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
