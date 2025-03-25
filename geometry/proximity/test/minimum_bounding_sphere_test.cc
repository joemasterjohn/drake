#include "drake/geometry/proximity/minimum_bounding_sphere.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/default_scalars.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

template <typename T>
class MinimumBoundingSphereTests : public ::testing::Test {};

using DefaultScalars = ::testing::Types<double, AutoDiffXd>;
TYPED_TEST_SUITE(MinimumBoundingSphereTests, DefaultScalars);

TYPED_TEST(MinimumBoundingSphereTests, Circumsphere) {
  using T = TypeParam;

  for (int i = 0; i < 100; ++i) {
    Vector3<T> p = Vector3<T>::Random();
    Vector3<T> q = Vector3<T>::Random();
    Vector3<T> r = Vector3<T>::Random();
    Vector3<T> s = Vector3<T>::Random();

    // Single point.
    {
      const PosedSphere<T> s_p = Circumsphere(p);

      EXPECT_TRUE(IsInside(s_p, p));
      EXPECT_TRUE(IsOn(s_p, p));
      EXPECT_FALSE(IsInside(s_p, q));
      EXPECT_FALSE(IsInside(s_p, r));
      EXPECT_FALSE(IsInside(s_p, s));
    }

    // Two points.
    {
      const PosedSphere<T> s_pq = Circumsphere(p, q);
      EXPECT_TRUE(IsInside(s_pq, p));
      EXPECT_TRUE(IsInside(s_pq, q));
      EXPECT_TRUE(IsOn(s_pq, p));
      EXPECT_TRUE(IsOn(s_pq, q));


      const Vector3<T> mid_pq = (p + q) / 2;
      const Vector3<T> p_outside = p + (p - s_pq.p_FSo);
      const Vector3<T> q_outside = q + (q - s_pq.p_FSo);
      EXPECT_TRUE(IsInside(s_pq, mid_pq));
      EXPECT_FALSE(IsInside(s_pq, p_outside));
      EXPECT_FALSE(IsInside(s_pq, q_outside));
    }

    // Three points.
    {
      const PosedSphere<T> s_pqr = Circumsphere(p, q, r);
      EXPECT_TRUE(IsInside(s_pqr, p));
      EXPECT_TRUE(IsInside(s_pqr, q));
      EXPECT_TRUE(IsInside(s_pqr, r));
      EXPECT_TRUE(IsOn(s_pqr, p));
      EXPECT_TRUE(IsOn(s_pqr, q));
      EXPECT_TRUE(IsOn(s_pqr, r));

      const Vector3<T> mid_pq = (p + q) / 2;
      const Vector3<T> mid_pr = (p + r) / 2;
      const Vector3<T> mid_qr = (q + r) / 2;
      const Vector3<T> p_outside = p + (p - s_pqr.p_FSo);
      const Vector3<T> q_outside = q + (q - s_pqr.p_FSo);
      const Vector3<T> r_outside = r + (r - s_pqr.p_FSo);
      EXPECT_TRUE(IsInside(s_pqr, mid_pq));
      EXPECT_TRUE(IsInside(s_pqr, mid_pr));
      EXPECT_TRUE(IsInside(s_pqr, mid_qr));
      EXPECT_FALSE(IsInside(s_pqr, p_outside));
      EXPECT_FALSE(IsInside(s_pqr, q_outside));
      EXPECT_FALSE(IsInside(s_pqr, r_outside));
    }

    // Four points.
    {
      const PosedSphere<T> s_pqrs = Circumsphere(p, q, r, s);
      EXPECT_TRUE(IsInside(s_pqrs, p));
      EXPECT_TRUE(IsInside(s_pqrs, q));
      EXPECT_TRUE(IsInside(s_pqrs, r));
      EXPECT_TRUE(IsInside(s_pqrs, s));
      EXPECT_TRUE(IsOn(s_pqrs, p));
      EXPECT_TRUE(IsOn(s_pqrs, q));
      EXPECT_TRUE(IsOn(s_pqrs, r));
      EXPECT_TRUE(IsOn(s_pqrs, s));

      const Vector3<T> mid_pq = (p + q) / 2;
      const Vector3<T> mid_qr = (q + r) / 2;
      const Vector3<T> mid_rs = (r + s) / 2;
      const Vector3<T> mid_sp = (s + p) / 2;
      const Vector3<T> p_outside = p + (p - s_pqrs.p_FSo);
      const Vector3<T> q_outside = q + (q - s_pqrs.p_FSo);
      const Vector3<T> r_outside = r + (r - s_pqrs.p_FSo);
      const Vector3<T> s_outside = s + (s - s_pqrs.p_FSo);
      EXPECT_TRUE(IsInside(s_pqrs, mid_pq));
      EXPECT_TRUE(IsInside(s_pqrs, mid_qr));
      EXPECT_TRUE(IsInside(s_pqrs, mid_rs));
      EXPECT_TRUE(IsInside(s_pqrs, mid_sp));
      EXPECT_FALSE(IsInside(s_pqrs, p_outside));
      EXPECT_FALSE(IsInside(s_pqrs, q_outside));
      EXPECT_FALSE(IsInside(s_pqrs, r_outside));
      EXPECT_FALSE(IsInside(s_pqrs, s_outside));
    }
  }
}

TYPED_TEST(MinimumBoundingSphereTests, MinimumBoundingSphere) {
  using T = TypeParam;
  for (int i = 0; i < 100; ++i) {
    Vector3<T> p = Vector3<T>::Random();
    Vector3<T> q = Vector3<T>::Random();
    Vector3<T> r = Vector3<T>::Random();
    Vector3<T> s = Vector3<T>::Random();
    // Single point.
    {
      const PosedSphere<T> s_min_p = MinimumBoundingSphere(p);

      EXPECT_TRUE(IsInside(s_min_p, p));
      EXPECT_FALSE(IsInside(s_min_p, q));
      EXPECT_FALSE(IsInside(s_min_p, r));
      EXPECT_FALSE(IsInside(s_min_p, s));
    }

    // Two points.
    {
      const PosedSphere<T> s_min_pq = MinimumBoundingSphere(p, q);
      EXPECT_TRUE(IsInside(s_min_pq, p));
      EXPECT_TRUE(IsInside(s_min_pq, q));

      const Vector3<T> mid_pq = (p + q) / 2;
      const Vector3<T> p_outside = p + (p - s_min_pq.p_FSo);
      const Vector3<T> q_outside = q + (q - s_min_pq.p_FSo);
      EXPECT_TRUE(IsInside(s_min_pq, mid_pq));
      EXPECT_FALSE(IsInside(s_min_pq, p_outside));
      EXPECT_FALSE(IsInside(s_min_pq, q_outside));
    }

    // Three points.
    {
      const PosedSphere<T> s_min_pqr = MinimumBoundingSphere(p, q, r);
      EXPECT_TRUE(IsInside(s_min_pqr, p));
      EXPECT_TRUE(IsInside(s_min_pqr, q));
      EXPECT_TRUE(IsInside(s_min_pqr, r));

      const Vector3<T> mid_pq = (p + q) / 2;
      const Vector3<T> mid_pr = (p + r) / 2;
      const Vector3<T> mid_qr = (q + r) / 2;
      const Vector3<T> p_outside =
          p + 1.1 * s_min_pqr.radius * (p - s_min_pqr.p_FSo).normalized();
      const Vector3<T> q_outside =
          q + 1.1 * s_min_pqr.radius * (q - s_min_pqr.p_FSo).normalized();
      const Vector3<T> r_outside =
          r + 1.1 * s_min_pqr.radius * (r - s_min_pqr.p_FSo).normalized();
      EXPECT_TRUE(IsInside(s_min_pqr, mid_pq));
      EXPECT_TRUE(IsInside(s_min_pqr, mid_pr));
      EXPECT_TRUE(IsInside(s_min_pqr, mid_qr));
      EXPECT_FALSE(IsInside(s_min_pqr, p_outside));
      EXPECT_FALSE(IsInside(s_min_pqr, q_outside));
      EXPECT_FALSE(IsInside(s_min_pqr, r_outside));

      // Brute force check all possible bounding sphere's and assert that the
      // computed MBS is indeed minimal.
      const PosedSphere<T> s_pq = Circumsphere(p, q);
      if (IsInside(s_pq, r)) {
        EXPECT_LE(s_min_pqr.radius, s_pq.radius);
      }
      const PosedSphere<T> s_qr = Circumsphere(q, r);
      if (IsInside(s_qr, p)) {
        EXPECT_LE(s_min_pqr.radius, s_qr.radius);
      }
      const PosedSphere<T> s_pr = Circumsphere(p, r);
      if (IsInside(s_pr, q)) {
        EXPECT_LE(s_min_pqr.radius, s_pr.radius);
      }
      const PosedSphere<T> s_pqr = Circumsphere(p, q, r);
      EXPECT_LE(s_min_pqr.radius, s_pqr.radius);
    }

    // Four points.
    {
      const PosedSphere<T> s_min_pqrs = MinimumBoundingSphere(p, q, r, s);
      EXPECT_TRUE(IsInside(s_min_pqrs, p));
      EXPECT_TRUE(IsInside(s_min_pqrs, q));
      EXPECT_TRUE(IsInside(s_min_pqrs, r));
      EXPECT_TRUE(IsInside(s_min_pqrs, s));

      const Vector3<T> mid_pq = (p + q) / 2;
      const Vector3<T> mid_qr = (q + r) / 2;
      const Vector3<T> mid_rs = (r + s) / 2;
      const Vector3<T> mid_sp = (s + p) / 2;
      const Vector3<T> p_outside =
          p + 1.1 * s_min_pqrs.radius * (p - s_min_pqrs.p_FSo).normalized();
      const Vector3<T> q_outside =
          q + 1.1 * s_min_pqrs.radius * (q - s_min_pqrs.p_FSo).normalized();
      const Vector3<T> r_outside =
          r + 1.1 * s_min_pqrs.radius * (r - s_min_pqrs.p_FSo).normalized();
      const Vector3<T> s_outside =
          s + 1.1 * s_min_pqrs.radius * (s - s_min_pqrs.p_FSo).normalized();
      EXPECT_TRUE(IsInside(s_min_pqrs, mid_pq));
      EXPECT_TRUE(IsInside(s_min_pqrs, mid_qr));
      EXPECT_TRUE(IsInside(s_min_pqrs, mid_rs));
      EXPECT_TRUE(IsInside(s_min_pqrs, mid_sp));
      EXPECT_FALSE(IsInside(s_min_pqrs, p_outside));
      EXPECT_FALSE(IsInside(s_min_pqrs, q_outside));
      EXPECT_FALSE(IsInside(s_min_pqrs, r_outside));
      EXPECT_FALSE(IsInside(s_min_pqrs, s_outside));

      // Brute force check all possible bounding sphere's and assert that the
      // computed MBS is indeed minimal.
      const PosedSphere<T> s_pq = Circumsphere(p, q);
      if (IsInside(s_pq, r) && IsInside(s_pq, s)) {
        EXPECT_LE(s_min_pqrs.radius, s_pq.radius);
      }
      const PosedSphere<T> s_pr = Circumsphere(p, r);
      if (IsInside(s_pr, q) && IsInside(s_pr, s)) {
        EXPECT_LE(s_min_pqrs.radius, s_pr.radius);
      }
      const PosedSphere<T> s_ps = Circumsphere(p, s);
      if (IsInside(s_ps, q) && IsInside(s_ps, r)) {
        EXPECT_LE(s_min_pqrs.radius, s_ps.radius);
      }
      const PosedSphere<T> s_qr = Circumsphere(q, r);
      if (IsInside(s_qr, p) && IsInside(s_qr, s)) {
        EXPECT_LE(s_min_pqrs.radius, s_qr.radius);
      }
      const PosedSphere<T> s_qs = Circumsphere(q, s);
      if (IsInside(s_qs, p) && IsInside(s_qs, r)) {
        EXPECT_LE(s_min_pqrs.radius, s_qs.radius);
      }
      const PosedSphere<T> s_rs = Circumsphere(r, s);
      if (IsInside(s_rs, p) && IsInside(s_rs, q)) {
        EXPECT_LE(s_min_pqrs.radius, s_rs.radius);
      }

      const PosedSphere<T> s_pqr = Circumsphere(p, q, r);
      if (IsInside(s_pqr, s)) {
        EXPECT_LE(s_min_pqrs.radius, s_pqr.radius);
      }
      const PosedSphere<T> s_pqs = Circumsphere(p, q, s);
      if (IsInside(s_pqs, r)) {
        EXPECT_LE(s_min_pqrs.radius, s_pqs.radius);
      }
      const PosedSphere<T> s_prs = Circumsphere(p, r, s);
      if (IsInside(s_prs, q)) {
        EXPECT_LE(s_min_pqrs.radius, s_prs.radius);
      }
      const PosedSphere<T> s_qrs = Circumsphere(q, r, s);
      if (IsInside(s_qrs, p)) {
        EXPECT_LE(s_min_pqrs.radius, s_qrs.radius);
      }

      const PosedSphere<T> s_pqrs = Circumsphere(p, q, r, s);
      EXPECT_LE(s_min_pqrs.radius, s_pqrs.radius);
    }
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
