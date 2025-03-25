#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {

template <typename T>
struct PosedSphere {
  Vector3<T> p_FSo;  // Position vector from frame F's origin to So (the center
                     // of the sphere).
  T radius{};        // Radius of the sphere.
};

template <typename T>
bool IsInside(const PosedSphere<T>& s, const Vector3<T>& p);

template <typename T>
bool IsOn(const PosedSphere<T>& s, const Vector3<T>& p);

template <typename T>
Vector2<T> CircumsphereBarycentric(const Vector3<T>& p, const Vector3<T>& q,
                                   const Vector3<T>& r);

template <typename T>
Vector3<T> CircumsphereBarycentric(const Vector3<T>& p, const Vector3<T>& q,
                                   const Vector3<T>& r, const Vector3<T>& s);

template <typename T>
PosedSphere<T> Circumsphere(const Vector3<T>& p);

template <typename T>
PosedSphere<T> Circumsphere(const Vector3<T>& p, const Vector3<T>& q);

template <typename T>
PosedSphere<T> Circumsphere(const Vector3<T>& p, const Vector3<T>& q,
                            const Vector3<T>& r);

template <typename T>
PosedSphere<T> Circumsphere(const Vector3<T>& p, const Vector3<T>& q,
                            const Vector3<T>& r, const Vector3<T>& s);

template <typename T>
PosedSphere<T> MinimumBoundingSphere(const Vector3<T>& p);

template <typename T>
PosedSphere<T> MinimumBoundingSphere(const Vector3<T>& p, const Vector3<T>& q);

template <typename T>
PosedSphere<T> MinimumBoundingSphere(const Vector3<T>& p, const Vector3<T>& q,
                                     const Vector3<T>& r);

template <typename T>
PosedSphere<T> MinimumBoundingSphere(const Vector3<T>& p, const Vector3<T>& q,
                                     const Vector3<T>& r, const Vector3<T>& s);

using PosedSphered = PosedSphere<double>;

}  // namespace internal
}  // namespace geometry
}  // namespace drake
