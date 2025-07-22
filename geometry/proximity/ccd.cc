#include "drake/geometry/proximity/ccd.h"

#include <limits>

#include "drake/common/default_scalars.h"

namespace drake {
namespace geometry {
namespace internal {

constexpr double kEps = std::numeric_limits<double>::epsilon();

template <typename T>
void ClosestPointPointToTriangle(const Vector3<T>& p, const Vector3<T>& a,
                                 const Vector3<T>& b, const Vector3<T>& c,
                                 ClosestPoint<T>* c1, ClosestPoint<T>* c2) {
  // Adapted from:
  //   Ericson, Christer. Real-time collision detection. Crc Press, 2004.
  //   Section 5.1.5
  (*c1).p = p;
  (*c1).type = ClosestPointType::Vertex;
  (*c1).indices = {0};

  // Check if p in vertex region outside a.
  const Vector3<T> ab = b - a;
  const Vector3<T> ac = c - a;
  const Vector3<T> ap = p - a;
  const T d1 = ab.dot(ap);
  const T d2 = ac.dot(ap);
  if (d1 <= 0 && d2 <= 0) {
    (*c2).p = a;
    (*c2).type = ClosestPointType::Vertex;
    (*c2).indices = {0};
    return;
  }

  // Check if p in vertex region outside b.
  const Vector3<T> bp = p - b;
  const T d3 = ab.dot(bp);
  const T d4 = ac.dot(bp);
  if (d3 >= 0 && d4 <= d3) {
    (*c2).p = b;
    (*c2).type = ClosestPointType::Vertex;
    (*c2).indices = {1};
    return;
  }

  // Check if p in edge region of ac, if so return projection of p onto ab.
  const T vc = d1 * d4 - d2 * d3;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const T v = d1 / (d1 - d3);
    (*c2).p = a + v * ab;
    (*c2).type = ClosestPointType::Edge;
    (*c2).indices = {0, 1};
    return;
  }

  // Check if p in vertex region outside c.
  const Vector3<T> cp = p - c;
  const T d5 = ab.dot(cp);
  const T d6 = ac.dot(cp);
  if (d6 >= 0 && d5 <= d6) {
    (*c2).p = c;
    (*c2).type = ClosestPointType::Vertex;
    (*c2).indices = {2};
    return;
  }

  // Check if p in edge region of ac, if so return projection of p onto ac.
  const T vb = d2 * d5 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const T w = d2 / (d2 - d6);
    (*c2).p = a + w * ac;
    (*c2).type = ClosestPointType::Edge;
    (*c2).indices = {0, 2};
    return;
  }

  // Check if p in edge region of bc, if so return projection of p onto bc.
  const T va = d3 * d6 - d4 * d5;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const T w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    (*c2).p = b + w * (c - b);
    (*c2).type = ClosestPointType::Edge;
    (*c2).indices = {1, 2};
    return;
  }

  // p inside face region. Compute closest through its barycentric coordinates
  // (u,v,w).
  const T denom = 1 / (va + vb + vc);
  const T v = vb * denom;
  const T w = vc * denom;
  (*c2).p = a + v * ab + w * ac;
  (*c2).type = ClosestPointType::Face;
  (*c2).indices = {0, 1, 2};
  return;
}

template <typename T>
void ClosestPointEdgeToEdge(const Vector3<T>& p1, const Vector3<T>& q1,
                            const Vector3<T>& p2, const Vector3<T>& q2,
                            ClosestPoint<T>* c1, ClosestPoint<T>* c2) {
  // Adapted from:
  //   Ericson, Christer. Real-time collision detection. Crc Press, 2004.
  //   Section 5.1.9

  T s, t;
  const Vector3<T> d1 = q1 - p1;  // Direction vector of segment s1.
  const Vector3<T> d2 = q2 - p2;  // Direction vector of segment s2.
  const Vector3<T> r = p1 - p2;
  const T a = d1.dot(d1);  // Squared length of s1.
  const T e = d2.dot(d2);  // Squared length of s2.
  const T f = d2.dot(r);

  const T zero(0.0);
  const T one(1.0);

  // Check if either or both segments degenerate into points.
  if (a < kEps && e < kEps) {
    s = t = 0;
  } else if (a < kEps) {
    // First segment degenerates into a point.
    s = 0;
    t = f / e;  // s = 0 => t = (b*s + f) / e = f / e.
    t = std::clamp(t, zero, one);
  } else {
    const T c = d1.dot(r);
    if (e < kEps) {
      // Second segment degenerates into a point.
      t = 0;
      s = std::clamp(-c / a, zero,
                     one);  // t = 0 => s = (b*t - c) / a = -c / a.
    } else {
      // The general non-degenerate case starts here.
      const T b = d1.dot(d2);
      const T denom = a * e - b * b;  // Always non-negative.

      // If segments not parallel, compute closest point on L1 to L2 and
      // std::clamp to segment s1. Else pick arbitrary s (here 0).
      if (denom != 0) {
        s = std::clamp((b * f - c * e) / denom, zero, one);
      } else {
        s = 0;
      }

      // Compute point on L2 closest to s1(s) using
      // t = dot((p1 + d1*s) - p2, d2) / dot(d2, d2) = (b*s + f) / e.
      t = (b * s + f) / e;

      // If t in [0, 1] done. Else std::clamp t, recompute s for the new value
      // of t using s = dot((p2 + d2*t) - p1, d1) / dot(d1, d1) = (t*b - c) / a
      // and std::clamp s to [0, 1].
      if (t < 0) {
        t = 0;
        s = std::clamp(-c / a, zero, one);
      } else if (t > 1) {
        t = 1;
        s = std::clamp((b - c) / a, zero, one);
      }
    }
  }

  (*c1).p = p1 + s * d1;
  (*c2).p = p2 + t * d2;

  if (s == zero) {
    (*c1).type = ClosestPointType::Vertex;
    (*c1).indices = {0};
  } else if (s == one) {
    (*c1).type = ClosestPointType::Vertex;
    (*c1).indices = {1};
  } else {
    (*c1).type = ClosestPointType::Edge;
    (*c1).indices = {0, 1};
  }

  if (t == zero) {
    (*c2).type = ClosestPointType::Vertex;
    (*c2).indices = {0};
  } else if (t == one) {
    (*c2).type = ClosestPointType::Vertex;
    (*c2).indices = {1};
  } else {
    (*c2).type = ClosestPointType::Edge;
    (*c2).indices = {0, 1};
  }
}

static const std::array<std::array<int, 2>, 6> tri_edges = {
    {{0, 1}, {0, 2}, {1, 2}}};

// Closest point between two non-intersecting triangles A and B.
template <typename T>
ClosestPointResult<T> ClosestPointTriangleToTriangle(
    const std::array<Vector3<T>, 3> p_A, const std::array<Vector3<T>, 3> p_B) {
  ClosestPointResult<T> current_result;
  current_result.squared_dist = std::numeric_limits<double>::infinity();

  ClosestPoint<T> current_A, current_B;
  T squared_dist;

  // Go through all of the vertex of A to B cases.
  for (int i = 0; i < 3; ++i) {
    ClosestPointPointToTriangle(p_A[i], p_B[0], p_B[1], p_B[2], &current_A,
                                &current_B);
    squared_dist = (p_A[i] - current_B.p).dot(p_A[i] - current_B.p);
    if (squared_dist < current_result.squared_dist) {
      current_result.squared_dist = squared_dist;
      current_result.closest_A.p = p_A[i];
      current_result.closest_A.type = ClosestPointType::Vertex;
      current_result.closest_A.indices[0] = i;
      current_result.closest_B.p = current_B.p;
      current_result.closest_B.type = current_B.type;
      current_result.closest_B.indices = current_B.indices;
    }
  }

  // Go through all of the vertex of B to A cases.
  for (int i = 0; i < 3; ++i) {
    ClosestPointPointToTriangle(p_B[i], p_A[0], p_A[1], p_A[2], &current_B,
                                &current_A);
    squared_dist = (p_B[i] - current_A.p).dot(p_B[i] - current_A.p);
    if (squared_dist < current_result.squared_dist) {
      current_result.squared_dist = squared_dist;
      current_result.closest_B.p = p_B[i];
      current_result.closest_B.type = ClosestPointType::Vertex;
      current_result.closest_B.indices[0] = i;
      current_result.closest_A.p = current_A.p;
      current_result.closest_A.type = current_A.type;
      current_result.closest_A.indices = current_A.indices;
    }
  }

  // Go through all of the edge to edge cases.
  for (const std::array<int, 2> e_A : tri_edges) {
    for (const std::array<int, 2> e_B : tri_edges) {
      ClosestPointEdgeToEdge(p_A[e_A[0]], p_A[e_A[1]], p_B[e_B[0]], p_B[e_B[1]],
                             &current_A, &current_B);
      squared_dist = (current_A - current_B).dot(current_A - current_B);
      if (squared_dist < current_result.squared_dist) {
        current_result.squared_dist = squared_dist;
        current_result.closest_A.p = current_A.p;
        current_result.closest_A.type = current_A.type;
        switch (current_A.type) {
          case ClosestPointType::Vertex:
            current_result.closest_A.indices = {e_A[current_A.indices[0]]};
            break;
          case ClosestPointType::Edge:
            current_result.closest_A.indices = {e_A[current_A.indices[0]],
                                                e_A[current_A.indices[1]]};
            break;
          default:
            DRAKE_UNREACHABLE();
            break;
        }
        current_result.closest_B.p = current_B.p;
        current_result.closest_B.type = current_B.type;
        switch (current_B.type) {
          case ClosestPointType::Vertex:
            current_result.closest_B.indices = {e_B[current_B.indices[0]]};
            break;
          case ClosestPointType::Edge:
            current_result.closest_B.indices = {e_B[current_B.indices[0]],
                                                e_B[current_B.indices[1]]};
            break;
          default:
            DRAKE_UNREACHABLE();
            break;
        }
      }
    }
  }

  return current_result;
}

// // Triangles with CCW winding facing inwards.
// static const std::array<std::array<int, 3>, 4> triangles = {
//     {{0, 1, 2}, {0, 3, 1}, {0, 2, 3}, {1, 3, 2}}};
// // Edges where e[i] and it's neighbor e[6-i] are an even permutation of
// // {0,1,2,3} (i.e. positive volume).
// static const std::array<std::array<int, 2>, 6> edges = {
//     {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {3, 1}, {2, 3}}};

const std::array<std::array<int, 3>, 4> triangles = {
    {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}}};
const std::array<std::array<int, 2>, 6> edges = {
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};


// Closest point between two non-intersecting triangles A and B.
template <typename T>
ClosestPointResult<T> ClosestPointTetrahedronToTetrahedron(
    const std::array<Vector3<T>, 4> p_A, const std::array<Vector3<T>, 4> p_B) {
  ClosestPointResult<T> current_result;
  current_result.squared_dist = std::numeric_limits<double>::infinity();

  ClosestPoint<T> current_A, current_B;
  T squared_dist;

  // Go through all of the vertex of A to face of B cases.
  for (int i = 0; i < 4; ++i) {
    for (const std::array<int, 3> t : triangles) {
      ClosestPointPointToTriangle(p_A[i], p_B[t[0]], p_B[t[1]], p_B[t[2]],
                                  &current_A, &current_B);
      squared_dist = (p_A[i] - current_B.p).dot(p_A[i] - current_B.p);
      if (squared_dist < current_result.squared_dist) {
        current_result.squared_dist = squared_dist;
        current_result.closest_A.p = p_A[i];
        current_result.closest_A.type = ClosestPointType::Vertex;
        current_result.closest_A.indices[0] = i;
        current_result.closest_B.p = current_B.p;
        current_result.closest_B.type = current_B.type;
        switch (current_B.type) {
          case ClosestPointType::Vertex:
            current_result.closest_B.indices = {t[current_B.indices[0]]};
            break;
          case ClosestPointType::Edge:
            current_result.closest_B.indices = {t[current_B.indices[0]],
                                                t[current_B.indices[1]]};
            break;
          default:
            current_result.closest_B.indices = t;
            break;
        }
      }
    }
  }

  // Go through all of the vertex of B to face of A cases.
  for (int i = 0; i < 4; ++i) {
    for (const std::array<int, 3> t : triangles) {
      ClosestPointPointToTriangle(p_B[i], p_A[t[0]], p_A[t[1]], p_A[t[2]],
                                  &current_B, &current_A);
      squared_dist = (p_B[i] - current_A.p).dot(p_B[i] - current_A.p);
      if (squared_dist < current_result.squared_dist) {
        current_result.squared_dist = squared_dist;
        current_result.closest_B.p = p_B[i];
        current_result.closest_B.type = ClosestPointType::Vertex;
        current_result.closest_B.indices[0] = i;
        current_result.closest_A.p = current_A.p;
        current_result.closest_A.type = current_A.type;
        switch (current_A.type) {
          case ClosestPointType::Vertex:
            current_result.closest_A.indices = {t[current_A.indices[0]]};
            break;
          case ClosestPointType::Edge:
            current_result.closest_A.indices = {t[current_A.indices[0]],
                                                t[current_A.indices[1]]};
            break;
          default:
            current_result.closest_A.indices = t;
            break;
        }
      }
    }
  }

  // Go through all of the edge to edge cases.
  for (const std::array<int, 2> e_A : edges) {
    for (const std::array<int, 2> e_B : edges) {
      ClosestPointEdgeToEdge(p_A[e_A[0]], p_A[e_A[1]], p_B[e_B[0]], p_B[e_B[1]],
                             &current_A, &current_B);
      squared_dist = (current_A.p - current_B.p).dot(current_A.p - current_B.p);
      if (squared_dist < current_result.squared_dist) {
        current_result.squared_dist = squared_dist;
        current_result.closest_A.p = current_A.p;
        current_result.closest_A.type = current_A.type;
        switch (current_A.type) {
          case ClosestPointType::Vertex:
            current_result.closest_A.indices = {e_A[current_A.indices[0]]};
            break;
          case ClosestPointType::Edge:
            current_result.closest_A.indices = {e_A[current_A.indices[0]],
                                                e_A[current_A.indices[1]]};
            break;
          default:
            DRAKE_UNREACHABLE();
            break;
        }
        current_result.closest_B.p = current_B.p;
        current_result.closest_B.type = current_B.type;
        switch (current_B.type) {
          case ClosestPointType::Vertex:
            current_result.closest_B.indices = {e_B[current_B.indices[0]]};
            break;
          case ClosestPointType::Edge:
            current_result.closest_B.indices = {e_B[current_B.indices[0]],
                                                e_B[current_B.indices[1]]};
            break;
          default:
            DRAKE_UNREACHABLE();
            break;
        }
      }
    }
  }

  return current_result;
}

template <typename T>
bool Intersects(const std::array<Vector3<T>, 4> p_A,
                const std::array<Vector3<T>, 4> p_B) {
  const auto project = [](const std::array<Vector3<T>, 4> tet,
                          const Vector3<T> n) {
    T lower = n.dot(tet[0]);
    T upper = lower;
    for (int i = 1; i < 4; ++i) {
      const T v = n.dot(tet[i]);
      if (v < lower) {
        lower = v;
      }
      if (v > upper) {
        upper = v;
      }
    }
    return std::make_pair(lower, upper);
  };

  const auto disjoint = [](const std::pair<T, T>& bounds_A,
                           const std::pair<T, T>& bounds_B) {
    return (bounds_B.first > bounds_A.second) ||
           (bounds_A.first > bounds_B.second);
  };

  // Check all of the face normals.
  for (const auto& t : triangles) {
    const Vector3<T> n_A = (p_A[t[1]] - p_A[t[0]]).cross(p_A[t[2]] - p_A[t[0]]);
    if (disjoint(project(p_A, n_A), project(p_B, n_A))) {
      return false;
    }
    const Vector3<T> n_B = (p_B[t[1]] - p_B[t[0]]).cross(p_B[t[2]] - p_B[t[0]]);
    if (disjoint(project(p_A, n_B), project(p_B, n_B))) {
      return false;
    }
  }

  // Check the edge/edge normals.
  for (const auto& e_A : edges) {
    for (const auto& e_B : edges) {
      const Vector3<T> n =
          (p_A[e_A[1]] - p_A[e_A[0]]).cross(p_B[e_B[1]] - p_B[e_B[0]]);
      if (disjoint(project(p_A, n), project(p_B, n))) {
        return false;
      }
    }
  }

  return true;
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&ClosestPointPointToTriangle<T>, &ClosestPointEdgeToEdge<T>,
     &ClosestPointTetrahedronToTetrahedron<T>, &Intersects<T>));

}  // namespace internal
}  // namespace geometry
}  // namespace drake
