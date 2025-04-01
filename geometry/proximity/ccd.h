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
                                 Vector3<T>* closest);

// Distance between two edge segments.
template <typename T>
void ClosestPointEdgeToEdge(const Vector3<T>& p1, const Vector3<T>& q1,
                            const Vector3<T>& p2, const Vector3<T>& q2,
                            Vector3<T>* c1, Vector3<T>* c2);

// Distance between two non-intersedting tetrahedra.
template <typename T>
ClosestPointResult<T> ClosestPointTetrahedronToTetrahedron(
    const std::array<Vector3<T>, 4> p_A, const std::array<Vector3<T>, 4> p_B);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
