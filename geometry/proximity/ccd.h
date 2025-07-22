#pragma once

#include <array>

#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {

// Indicates whether the closest point is on a vertex, edge or face of the
// element.
enum class ClosestPointType { Vertex, Edge, Face };

template <typename T>
struct ClosestPoint {
  Vector3<T> p;                // Value of the closest point.
  ClosestPointType type;       // Type point.
  std::array<int, 3> indices;  // Indices of the element this point is on.
                               // If type == Vertex only indices[0] is valid.
                               // If type == Edge only indices[0,1] are valid.
                               // If type == Face all indices are valid.
};

template <typename T>
struct ClosestPointResult {
  ClosestPoint<T> closest_A;
  ClosestPoint<T> closest_B;
  T squared_dist;
};

// Distance between point and triangle.
template <typename T>
void ClosestPointPointToTriangle(const Vector3<T>& p, const Vector3<T>& a,
                                 const Vector3<T>& b, const Vector3<T>& c,
                                 ClosestPoint<T>* c1, ClosestPoint<T>* c2);

// Distance between two edge segments.
template <typename T>
void ClosestPointEdgeToEdge(const Vector3<T>& p1, const Vector3<T>& q1,
                            const Vector3<T>& p2, const Vector3<T>& q2,
                            ClosestPoint<T>* c1, ClosestPoint<T>* c2);

// Distance between two non-intersecting triangles.
template <typename T>
ClosestPointResult<T> ClosestPointTriangleToTriangle(
    const std::array<Vector3<T>, 3> p_A, const std::array<Vector3<T>, 3> p_B);

// Distance between two non-intersedting tetrahedra.
template <typename T>
ClosestPointResult<T> ClosestPointTetrahedronToTetrahedron(
    const std::array<Vector3<T>, 4> p_A, const std::array<Vector3<T>, 4> p_B);

// Returns true if the tetrahedra intersect, false otherwise. Uses separating
// axis test to determine intersection.
template <typename T>
bool Intersects(const std::array<Vector3<T>, 4> p_A,
                const std::array<Vector3<T>, 4> p_B);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
