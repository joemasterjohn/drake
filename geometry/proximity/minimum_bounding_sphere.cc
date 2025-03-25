#include "drake/geometry/proximity/minimum_bounding_sphere.h"

#include <limits>

#include "drake/common/default_scalars.h"
#include "drake/math/linear_solve.h"

namespace drake {
namespace geometry {
namespace internal {

constexpr double kEps = std::numeric_limits<double>::epsilon();

template <typename T>
bool IsInside(const PosedSphere<T>& s, const Vector3<T>& p) {
  return (p - s.p_FSo).norm() - s.radius <= kEps;
}

template <typename T>
bool IsOn(const PosedSphere<T>& s, const Vector3<T>& p) {
  return abs((p - s.p_FSo).norm() - s.radius) <= 2e3 * kEps;
}

template <typename T>
Vector2<T> CircumsphereBarycentric(const Vector3<T>& p, const Vector3<T>& q,
                                   const Vector3<T>& r) {
  // This implementation has been adapted from:
  // https://realtimecollisiondetection.net/blog/?p=20

  // The circumcenter of pqr is given in Barycentric coordinates as:
  //
  //   C = P + s⋅(Q - P) + t⋅(R - P);
  //
  // With the equidistant constraints:
  //
  //  (C - Q)⋅(C - Q) = (C - P)⋅(C - P)
  //  (C - R)⋅(C - R) = (C - P)⋅(C - P)
  //
  // Which, after a bit of algebraic rearranging, leads to the system:
  //
  //   [(P - Q)⋅(P - Q)  (P - Q)⋅(P - R)] ⋅[s]  = [0.5⋅(P - Q)⋅(P - Q)]
  //   [(P - R)⋅(P - Q)  (P - R)⋅(P - R)] ⋅[t]  = [0.5⋅(P - R)⋅(P - R)]
  //
  const Vector3<T> qp = p - q;
  const Vector3<T> rp = p - r;
  const T qp2 = qp.dot(qp);
  const T rp2 = rp.dot(rp);
  const T qprp = qp.dot(rp);
  // clang-format off
  const Matrix2<T> M =
      (Matrix2<T>() << qp2, qprp,
                       qprp, rp2).finished();
  // clang-format on
  const Vector2<T> b(0.5 * qp2, 0.5 * rp2);
  math::LinearSolver<Eigen::LLT, Matrix2<T>> M_llt(M);
  return M_llt.Solve(b);
}

template <typename T>
Vector3<T> CircumsphereBarycentric(const Vector3<T>& p, const Vector3<T>& q,
                                   const Vector3<T>& r, const Vector3<T>& s) {
  // clang-format off
  //
  // The circumcenter of pqr is given in Barycentric coordinates as:
  //
  //   C = P + s⋅(Q - P) + t⋅(R - P) + u⋅(S - P)
  //
  // With the equidistant constraints:
  //
  //  (C - Q)⋅(C - Q) = (C - P)⋅(C - P)
  //  (C - R)⋅(C - R) = (C - P)⋅(C - P)
  //  (C - S)⋅(C - S) = (C - P)⋅(C - P)
  //
  // Which, after a bit of algebraic rearranging, leads to the system:
  //
  //  [(P - Q)⋅(P - Q) (P - Q)⋅(P - R) (P - Q)⋅(P - S)]⋅[s] = [0.5⋅(P - Q)⋅(P - Q)]  // NOLINT(*)
  //  [(P - R)⋅(P - Q) (P - R)⋅(P - R) (P - R)⋅(P - S)]⋅[t] = [0.5⋅(P - R)⋅(P - R)]  // NOLINT(*)
  //  [(P - S)⋅(P - Q) (P - S)⋅(P - R) (P - S)⋅(P - S)]⋅[u] = [0.5⋅(P - S)⋅(P - S)]  // NOLINT(*)
  //
  // clang-format on
  const Vector3<T> qp = p - q;
  const Vector3<T> rp = p - r;
  const Vector3<T> sp = p - s;
  const T qp2 = qp.dot(qp);
  const T rp2 = rp.dot(rp);
  const T sp2 = sp.dot(sp);
  const T qprp = qp.dot(rp);
  const T qpsp = qp.dot(sp);
  const T rpsp = rp.dot(sp);
  // clang-format off
  const Matrix3<T> M =
      (Matrix3<T>() << qp2, qprp, qpsp,
                       qprp, rp2, rpsp,
                       qpsp, rpsp, sp2)
          .finished();
  // clang-format on
  const Vector3<T> b(0.5 * qp2, 0.5 * rp2, 0.5 * sp2);
  math::LinearSolver<Eigen::LLT, Matrix3<T>> M_llt(M);
  return M_llt.Solve(b);
}

template <typename T>
PosedSphere<T> Circumsphere(const Vector3<T>& p) {
  return {p, 0.0};
}

template <typename T>
PosedSphere<T> Circumsphere(const Vector3<T>& p, const Vector3<T>& q) {
  return {0.5 * (p + q), 0.5 * (p - q).norm() + kEps};
}

template <typename T>
PosedSphere<T> Circumsphere(const Vector3<T>& p, const Vector3<T>& q,
                            const Vector3<T>& r) {
  const Vector2<T> st = CircumsphereBarycentric(p, q, r);
  const Vector3<T> p_FSo = (1 - st.sum()) * p + st(0) * q + st(1) * r;
  // Due to numerical imprecision, we must choose the largest radius such that
  // each of p,q,r are inside the circumsphere.
  const T radius = std::max((p - p_FSo).norm(),
                            std::max((q - p_FSo).norm(), (r - p_FSo).norm()));
  return {p_FSo, radius};
}

template <typename T>
PosedSphere<T> Circumsphere(const Vector3<T>& p, const Vector3<T>& q,
                            const Vector3<T>& r, const Vector3<T>& s) {
  const Vector3<T> stu = CircumsphereBarycentric(p, q, r, s);
  const Vector3<T> p_FSo =
      (1 - stu.sum()) * p + stu(0) * q + stu(1) * r + stu(2) * s;
  // Due to numerical imprecision, we must choose the largest radius such that
  // each of p,q,r,s  are inside the circumsphere.
  const T radius =
      std::max((p - p_FSo).norm(),
               std::max((q - p_FSo).norm(),
                        std::max((r - p_FSo).norm(), (s - p_FSo).norm())));
  return {p_FSo, radius};
}

template <typename T>
PosedSphere<T> MinimumBoundingSphere(const Vector3<T>& p) {
  return Circumsphere(p);
}

template <typename T>
PosedSphere<T> MinimumBoundingSphere(const Vector3<T>& p, const Vector3<T>& q) {
  return Circumsphere(p, q);
}

template <typename T>
PosedSphere<T> MinimumBoundingSphere(const Vector3<T>& p, const Vector3<T>& q,
                                     const Vector3<T>& r) {
  const Vector2<T> st = CircumsphereBarycentric(p, q, r);
  // This implementation has been adapted from:
  // https://realtimecollisiondetection.net/blog/?p=20

  //
  if (st(0) <= 0) {
    return MinimumBoundingSphere(p, r);
  } else if (st(1) <= 0) {
    return MinimumBoundingSphere(p, q);
  } else if (st.sum() >= 1) {
    return MinimumBoundingSphere(q, r);
  } else {
    const Vector3<T> p_FSo = (1 - st.sum()) * p + st(0) * q + st(1) * r;
    // Due to numerical imprecision, we must choose the largest radius such that
    // each of p,q,r are inside the circumsphere.
    const T radius = std::max((p - p_FSo).norm(),
                              std::max((q - p_FSo).norm(), (r - p_FSo).norm()));
    return {p_FSo, radius};
  }
}

template <typename T>
PosedSphere<T> MinimumBoundingSphere(const Vector3<T>& p, const Vector3<T>& q,
                                     const Vector3<T>& r, const Vector3<T>& s) {
  // Specialized unrolled version of Wezlz's algorithm for 4 points.

  // If p is inside of the MBS of (q, r, s), then that is the MBS.
  const PosedSphere<T> sphere_qrs = MinimumBoundingSphere(q, r, s);
  if (IsInside(sphere_qrs, p)) {
    return sphere_qrs;
  }

  // p must lie on the minimum bounding sphere of (p, q, r, s).
  // Recurse on (q, r, s), given p on the sphere.

  // Compute MBS of (r, s) given p on sphere.
  PosedSphere<T> min_p_rs;  // MBS of (r, s) given p on sphere.
  const PosedSphere<T> sphere_ps = MinimumBoundingSphere(p, s);
  if (IsInside(sphere_ps, r)) {
    min_p_rs = sphere_ps;
  } else {
    // r must lie on the MBS with p.
    const PosedSphere<T> sphere_pr = MinimumBoundingSphere(p, r);
    if (IsInside(sphere_pr, s)) {
      min_p_rs = sphere_pr;
    } else {
      // s must lie on the MBS with p and r.
      min_p_rs = MinimumBoundingSphere(p, r, s);
    }
  }

  // If q is inside the MBS of (r, s) given p on sphere, then that is the MBS.
  if (IsInside(min_p_rs, q)) {
    return min_p_rs;
  }

  // p and q must lie on the MBS of (p, q, r, s).
  // Recurse on (r, s) given p and q on the sphere.
  const PosedSphere<T> sphere_pqs = MinimumBoundingSphere(p, q, s);
  if (IsInside(sphere_pqs, r)) {
    return sphere_pqs;
  }
  // r must lie on the sphere with p and q.
  const PosedSphere<T> sphere_pqr = MinimumBoundingSphere(p, q, r);
  if (IsInside(sphere_pqr, s)) {
    return sphere_pqr;
  }

  // All points lie on the minimum bounding sphere, return the circumsphere.
  return Circumsphere(p, q, r, s);
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS((
    &IsInside<T>,  // BR
    &IsOn<T>,      // BR
    static_cast<Vector2<T> (*)(const Vector3<T>&, const Vector3<T>&,
                               const Vector3<T>&)>(
        &CircumsphereBarycentric<T>),  // BR
    static_cast<Vector3<T> (*)(const Vector3<T>&, const Vector3<T>&,
                               const Vector3<T>&, const Vector3<T>&)>(
        &CircumsphereBarycentric<T>),                                      // BR
    static_cast<PosedSphere<T> (*)(const Vector3<T>&)>(&Circumsphere<T>),  // BR
    static_cast<PosedSphere<T> (*)(const Vector3<T>&, const Vector3<T>&)>(
        &Circumsphere<T>),  // BR
    static_cast<PosedSphere<T> (*)(const Vector3<T>&, const Vector3<T>&,
                                   const Vector3<T>&)>(&Circumsphere<T>),  // BR
    static_cast<PosedSphere<T> (*)(const Vector3<T>&, const Vector3<T>&,
                                   const Vector3<T>&, const Vector3<T>&)>(
        &Circumsphere<T>),  // BR
    static_cast<PosedSphere<T> (*)(const Vector3<T>&)>(
        &MinimumBoundingSphere<T>),  // BR
    static_cast<PosedSphere<T> (*)(const Vector3<T>&, const Vector3<T>&)>(
        &MinimumBoundingSphere<T>),  // BR
    static_cast<PosedSphere<T> (*)(const Vector3<T>&, const Vector3<T>&,
                                   const Vector3<T>&)>(
        &MinimumBoundingSphere<T>),  // BR
    static_cast<PosedSphere<T> (*)(const Vector3<T>&, const Vector3<T>&,
                                   const Vector3<T>&, const Vector3<T>&)>(
        &MinimumBoundingSphere<T>)));

}  // namespace internal
}  // namespace geometry
}  // namespace drake
