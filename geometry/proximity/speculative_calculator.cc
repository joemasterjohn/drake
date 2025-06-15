#include "drake/geometry/proximity/speculative_calculator.h"

#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

using Eigen::Vector3d;
using math::RotationMatrix;
using multibody::SpatialVelocity;

template <typename T>
AabbCalculator MovingBoundingSphereAabbCalculator(
    const std::vector<PosedSphere<double>>& mesh_bounding_spheres,
    const math::RigidTransform<T>& X_WG,
    const multibody::SpatialVelocity<T>& V_WG, double dt) {
  return [&mesh_bounding_spheres, &X_WG, &V_WG, dt](int e) -> Aabb {
    const PosedSphere<double>& sphere = mesh_bounding_spheres[e];
    // TODO(joemasterjohn): Change the sphere's cannonical frame name to
    // G for consistency. F == G here.
    const Vector3d p_WSo =
        ExtractDoubleOrThrow(X_WG * sphere.p_FSo.template cast<T>());
    // Since the sphere bounds the entire geometry, it also bounds all
    // possible rotations of the geometry about So. Thus a box of
    // half_width == radius centered at So is a valid bounding box for
    // **any** frame.
    const Vector3d half_width = Vector3d::Constant(sphere.radius);

    // Compute the translational velocity of the element at So, measured
    // and expressed in the world frame.
    const Vector3d p_GoSo_W = p_WSo - ExtractDoubleOrThrow(X_WG.translation());
    const Vector3d v_WSo =
        ExtractDoubleOrThrow(V_WG.Shift(p_GoSo_W).translational());

    // Since the Aabb of the element encompases all possible
    // rotations, we need not consider the angular velocity of the
    // element w_WSo. The bounding box of the sphere and it's
    // translation by dt*v_WSo covers all possible translations and
    // rotations of the element moving with constant spatial
    // velocity V_WSo over an interval of length dt.
    Vector3d min_corner = p_WSo - half_width;
    Vector3d max_corner = p_WSo + half_width;
    min_corner = min_corner.cwiseMin(min_corner + dt * v_WSo);
    max_corner = max_corner.cwiseMax(max_corner + dt * v_WSo);

    return Aabb((max_corner + min_corner) / 2, (max_corner - min_corner) / 2);
  };
}

template <typename T>
AabbCalculator TruncatedTaylorSeriesAabbCalculator(
    const VolumeMesh<double>& mesh, const math::RigidTransform<T>& X_WG,
    const multibody::SpatialVelocity<T>& V_WG, double dt) {
  const Vector3d p_WG = ExtractDoubleOrThrow(X_WG.translation());
  const Eigen::Matrix3d R_WG = ExtractDoubleOrThrow(X_WG.rotation().matrix());
  const Vector3d v_WG = ExtractDoubleOrThrow(V_WG.translational());
  const Vector3d w_WG = ExtractDoubleOrThrow(V_WG.rotational());

  // Start with a truncation of the vertex trajectories to quadratic.
  return [&mesh, p_WG, R_WG, v_WG, w_WG, dt](int e) -> Aabb {
    const VolumeElement& element = mesh.element(e);
    Vector3d min_corner = R_WG * mesh.vertex(element.vertex(0)) + p_WG;
    Vector3d max_corner = min_corner;

    for (int i = 0; i < 4; ++i) {
      const Vector3d p_GV_W = R_WG * mesh.vertex(element.vertex(i));
      const Vector3d w_x_p = (w_WG).cross(p_GV_W);
      const Vector3d w_x_w_x_p = (w_WG).cross(w_x_p);
      const Vector3d p_WV = p_WG + p_GV_W;
      const auto eval = [&](double t) {
        return p_WV + t * ((v_WG + w_x_p) + 0.5 * t * w_x_w_x_p);
      };
      const Vector3d p_WV_dt = eval(dt);
      min_corner = min_corner.cwiseMin(p_WV);
      min_corner = min_corner.cwiseMin(p_WV_dt);
      max_corner = max_corner.cwiseMax(p_WV);
      max_corner = max_corner.cwiseMax(p_WV_dt);
      const Vector3d t = (v_WG + w_x_p).cwiseQuotient(-w_x_w_x_p);
      for (int j = 0; j < 3; ++j) {
        if (t(j) > 0 && t(j) < dt) {
          const Vector3d p_WV_t = eval(t(j));
          min_corner = min_corner.cwiseMin(p_WV_t);
          max_corner = max_corner.cwiseMax(p_WV_t);
        }
      }
    }

    return Aabb((max_corner + min_corner) / 2, (max_corner - min_corner) / 2);
  };
}

template <typename T>
AabbCalculator StaticMeshAabbCalculator(const VolumeMesh<double>& mesh,
                                        const math::RigidTransform<T>& X_WG) {
  return [&mesh, &X_WG](int e) -> Aabb {
    VolumeElement tet = mesh.element(e);
    Vector3d min_corner = ExtractDoubleOrThrow(
        X_WG * mesh.vertex(tet.vertex(0)).template cast<T>());
    Vector3d max_corner = min_corner;
    for (int i = 1; i < 4; ++i) {
      const Vector3d p_WV = ExtractDoubleOrThrow(
          X_WG * mesh.vertex(tet.vertex(i)).template cast<T>());
      min_corner = min_corner.cwiseMin(p_WV);
      max_corner = max_corner.cwiseMax(p_WV);
    }

    return Aabb((max_corner + min_corner) / 2, (max_corner - min_corner) / 2);
  };
}

template <typename T>
void ComputeSpeculativeContactSurfaceByClosestPoints(
    GeometryId id_A, GeometryId id_B, const SoftGeometry& soft_A,
    const SoftGeometry& soft_B, const math::RigidTransform<T>& X_WA,
    const math::RigidTransform<T>& X_WB,
    const multibody::SpatialVelocity<T>& V_WA,
    const multibody::SpatialVelocity<T>& V_WB, const double dt,
    std::vector<SpeculativeContactSurface<T>>* speculative_surfaces) {
  constexpr double kEps = std::numeric_limits<double>::epsilon();
  // Data for each point of the speculative contact surface.
  std::vector<Vector3<T>> p_WC;
  std::vector<Vector3<T>> p_AC_W;
  std::vector<Vector3<T>> p_BC_W;
  std::vector<T> time_of_contact;
  std::vector<Vector3<T>> zhat_BA_W;
  std::vector<T> coefficients;
  std::vector<Vector3<T>> nhat_BA_W;
  std::vector<Vector3<T>> grad_eA_W;
  std::vector<Vector3<T>> grad_eB_W;
  std::vector<ClosestPointResult<T>> closest_points;
  std::vector<std::pair<int, int>> valid_element_pairs;
  std::vector<T> effective_radius;

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::pair<int, int>> element_pairs =
      soft_A.soft_mesh().mesh_dynamic_bvh().GetCollisionCandidates(
          soft_B.soft_mesh().mesh_dynamic_bvh());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration1 = end - start;

  fmt::print("  (gid({}) {} elements, gid({}) {} elements): {} pairs\n", id_A,
             soft_A.soft_mesh().mesh().num_elements(), id_B,
             soft_B.soft_mesh().mesh().num_elements(), ssize(element_pairs));

  fmt::print("    Broadphase:  {}s\n", duration1.count());

  // fmt::print("num_candidates: {} sA({}) sB({})\n", element_pairs.size(),
  //            soft_A.mesh().num_elements(), soft_B.mesh().num_elements());

  // Quick exit.
  if (ssize(element_pairs) == 0) return;
  start = std::chrono::high_resolution_clock::now();

  // Reserve memory for the surface data.
  p_WC.reserve(element_pairs.size());
  p_AC_W.reserve(element_pairs.size());
  p_BC_W.reserve(element_pairs.size());
  time_of_contact.reserve(element_pairs.size());
  zhat_BA_W.reserve(element_pairs.size());
  coefficients.reserve(element_pairs.size());
  nhat_BA_W.reserve(element_pairs.size());
  grad_eA_W.reserve(element_pairs.size());
  grad_eB_W.reserve(element_pairs.size());
  closest_points.reserve(element_pairs.size());
  valid_element_pairs.reserve(element_pairs.size());
  effective_radius.reserve(element_pairs.size());

  // For each element pair compute the necessary data.
  for (const auto& [tet_A, tet_B] : element_pairs) {
    const VolumeElement e_A = soft_A.mesh().element(tet_A);
    const VolumeElement e_B = soft_B.mesh().element(tet_B);
    std::array<Vector3<T>, 4> p_WA = {
        X_WA * soft_A.mesh().vertex(e_A.vertex(0)).cast<T>(),
        X_WA * soft_A.mesh().vertex(e_A.vertex(1)).cast<T>(),
        X_WA * soft_A.mesh().vertex(e_A.vertex(2)).cast<T>(),
        X_WA * soft_A.mesh().vertex(e_A.vertex(3)).cast<T>()};
    std::array<Vector3<T>, 4> p_WB = {
        X_WB * soft_B.mesh().vertex(e_B.vertex(0)).cast<T>(),
        X_WB * soft_B.mesh().vertex(e_B.vertex(1)).cast<T>(),
        X_WB * soft_B.mesh().vertex(e_B.vertex(2)).cast<T>(),
        X_WB * soft_B.mesh().vertex(e_B.vertex(3)).cast<T>()};

    // Ensure the tetrahedra do not intersect before proceeding.
    if (Intersects(p_WA, p_WB)) {
      continue;
    }

    // Compute the closest points of the two tetrahedra.
    closest_points.emplace_back(
        ClosestPointTetrahedronToTetrahedron(p_WA, p_WB));
    const ClosestPointResult<T>& result = closest_points.back();

    // Keep track of the witness point P on A, and witness point Q on B and
    // their respective velocities.
    const Vector3<T>& p_WAp = result.closest_A.p;
    const Vector3<T>& p_WBq = result.closest_B.p;
    const Vector3<T> p_AoAp_W = p_WAp - X_WA.translation();
    const SpatialVelocity<T> V_WAp = V_WA.Shift(p_AoAp_W);
    const Vector3<T> v_WAp = V_WAp.translational();
    //const Vector3<T> w_WAp = V_WAp.rotational();
    const Vector3<T> p_BoBq_W = p_WBq - X_WB.translation();
    const SpatialVelocity<T> V_WBq = V_WB.Shift(p_BoBq_W);
    const Vector3<T> v_WBq = V_WBq.translational();
    //const Vector3<T> w_WBq = V_WBq.rotational();

    const Vector3<T> p_BqAp_W = p_WAp - p_WBq;
    const T length_BqAp = p_BqAp_W.norm();
    const Vector3<T> n_hat_BqAp_W = p_BqAp_W / length_BqAp;

    // We define the contact point as the point along the line segment from
    // Bq to Ap such that at the time of contact tc:
    //
    //   p_WAp + tc ⋅ (n_hat_BqAp_W ⋅ v_WAp) ⋅ n_hat_BqAp_W =
    //   p_WBq + tc ⋅ (n_hat_BqAp_W ⋅ v_WBq) ⋅ n_hat_BqAp_W
    //
    // Which simplifies to:
    //
    // clang-format off
    //   => (p_WAp - p_WBq) + tc ⋅ n_hat_BqAp_W ⋅ (v_WAp - v_Bq) ⋅ n_hat_BqAp_W = 0
    //   => length_BqAp ⋅ n_hat_BqAp_W + tc ⋅ n_hat_BqAp_W ⋅ (v_WAp - v_Bq) ⋅ n_hat_BqAp_W = 0
    //   => n_hat_BqAp_W ⋅ [length_BqAp + tc ⋅ n_hat_BqAp_W ⋅ (v_WAp - v_Bq)] = 0
    //   => length_BqAp + tc ⋅ n_hat_BqAp_W ⋅ (v_WAp - v_Bq) = 0
    //   => tc = length_BqAp / (n_hat_BqAp_W ⋅ (v_WBq - v_WAp))
    // clang-format on
    //
    // TODO(joemasterjohn): Show that (n_hat_BqAp_W ⋅ (v_WBq - v_WAp) is
    // frame invariant.

    const T v_n = n_hat_BqAp_W.dot(v_WBq - v_WAp);

    if (abs(v_n) < kEps) {
      closest_points.pop_back();
      continue;
    }

    const T tc = length_BqAp / v_n;

    if (tc < 0 || tc > dt) {
      closest_points.pop_back();
      continue;
    }

    time_of_contact.emplace_back(tc);

    // TODO(joemasterjohn): Consider the case ot tc < 0. Or take as argument
    // to this query a range, such that tc ∈ [-k⁻⋅δt, k⁺⋅δt] for valid
    // constraints.

    // TEST
    // Do one step of advancement of the witness points to the time of contact.
    // Take their average as the contact point.
    // const Vector3<T> p_WAp_tc = p_WAp + tc*v_WAp;
    // const Vector3<T> p_WBq_tc = p_WBq + tc*v_WBq;
    // p_WC.emplace_back(0.5*(p_WAp_tc + p_WBq_tc));

    p_WC.emplace_back(p_WAp + tc * (n_hat_BqAp_W.dot(v_WAp)) * n_hat_BqAp_W);
    // p_WC.emplace_back(p_WAp);
    p_AC_W.emplace_back(p_AoAp_W);
    p_BC_W.emplace_back(p_BoBq_W);

    // Consider all of the closest point cases.
    // TODO(joemasterjohn): Document or reference written document for the
    // derivations of these coefficients and the algebraic simplifications.
    if (result.closest_A.type == ClosestPointType::Face) {
      DRAKE_ASSERT(result.closest_B.type == ClosestPointType::Vertex);
      const int vertex_B_index = result.closest_B.indices[0];
      // `indices` holds the 3 indices that make the face of A. By
      // convention the index of this face is the same as the index of
      // vertex opposite in the tet.
      const int face_A_index =
          6 - (result.closest_A.indices[0] + result.closest_A.indices[1] +
               result.closest_A.indices[2]);
      // Inward normals of the faces adjacent to the vertex of B.
      // The formulation uses the outward normals, but since they only show
      // up as cross products of 2 of these vectors, we can just use compute
      // on the inward normals and use references.
      const Vector3<T>& n_B0 =
          soft_B.mesh().inward_normal(tet_B, (vertex_B_index + 1) % 4);
      const Vector3<T>& n_B1 =
          soft_B.mesh().inward_normal(tet_B, (vertex_B_index + 2) % 4);
      const Vector3<T>& n_B2 =
          soft_B.mesh().inward_normal(tet_B, (vertex_B_index + 3) % 4);

      // The (inward) face normal of A is our volume normal zhat_BA.
      // Re-express it in world.
      zhat_BA_W.emplace_back(
          X_WA.rotation() *
          soft_A.mesh().inward_normal(tet_A, face_A_index).template cast<T>());
      // DRAKE_ASSERT(zhat_BA_W.back().dot(p_BqAp_W) > 0);
      //  n_A in the formulation is the (outward) face normal of A.
      //  Re-express in B for computation.
      const Vector3<T> n_A_B = -(X_WB.rotation().inverse() * zhat_BA_W.back());

      // TODO(joemasterjohn): Here is the meat of the formulation,
      // simplified to the smallest amount of flops I could get it to. The
      // cross products of the face normals are just the normalized edge
      // vectors, so if we pre-compute those we can avoid the cross products
      // here. Document the formulation or link to LaTeX docs.
      const Vector3<T> n_B0_x_n_B1 = n_B0.cross(n_B1);
      T numerator = n_B2.dot(n_B0_x_n_B1);
      numerator *= numerator;
      T denominator = 6 * n_A_B.dot(n_B0_x_n_B1);
      denominator *= n_A_B.dot(n_B1.cross(n_B2));
      denominator *= n_A_B.dot(n_B2.cross(n_B0));
      coefficients.emplace_back(abs(numerator / denominator));

      // Sanity check.
      DRAKE_ASSERT(coefficients.back() > 0);
    } else if (result.closest_B.type == ClosestPointType::Face) {
      DRAKE_ASSERT(result.closest_A.type == ClosestPointType::Vertex);
      const int vertex_A_index = result.closest_A.indices[0];
      // `indices` holds the 3 indices that make the face of B. By
      // convention the index of this face is the same as the index of
      // vertex opposite in the tet.
      const int face_B_index =
          6 - (result.closest_B.indices[0] + result.closest_B.indices[1] +
               result.closest_B.indices[2]);
      // Inward normals of the faces adjacent to the vertex of A.
      // The formulation uses the outward normals, but since they only show
      // up as cross products of 2 of these vectors, we can just use compute
      // on the inward normals and use references.
      const Vector3<T>& n_A0 =
          soft_A.mesh().inward_normal(tet_A, (vertex_A_index + 1) % 4);
      const Vector3<T>& n_A1 =
          soft_A.mesh().inward_normal(tet_A, (vertex_A_index + 2) % 4);
      const Vector3<T>& n_A2 =
          soft_A.mesh().inward_normal(tet_A, (vertex_A_index + 3) % 4);

      // The (outward) face normal of B is our volume normal zhat_BA.
      // Re-express it in world.
      zhat_BA_W.emplace_back(-(
          X_WB.rotation() *
          soft_B.mesh().inward_normal(tet_B, face_B_index).template cast<T>()));
      // DRAKE_ASSERT(zhat_BA_W.back().dot(p_BqAp_W) < 0);
      //  n_B in the formulation is the (outward) face normal of B.
      //  Re-express in A for computation.
      const Vector3<T> n_B_A = X_WA.rotation().inverse() * zhat_BA_W.back();

      // TODO(joemasterjohn): Here is the meat of the formulation,
      // simplified to the smallest amount of flops I could get it to. The
      // cross products of the face normals are just the normalized edge
      // vectors, so if we pre-compute those we can avoid the cross products
      // here. Document the formulation or link to LaTeX docs.
      const Vector3<T> n_A0_x_n_A1 = n_A0.cross(n_A1);
      T numerator = n_A2.dot(n_A0_x_n_A1);
      numerator *= numerator;
      T denominator = 6 * n_B_A.dot(n_A0_x_n_A1);
      denominator *= n_B_A.dot(n_A1.cross(n_A2));
      denominator *= n_B_A.dot(n_A2.cross(n_A0));
      coefficients.emplace_back(abs(numerator / denominator));

      DRAKE_ASSERT(coefficients.back() >= 0);

    } else {
      DRAKE_ASSERT(result.closest_A.type == ClosestPointType::Edge);
      DRAKE_ASSERT(result.closest_B.type == ClosestPointType::Edge);

      // This little trick takes an edge (i, j) to it's opposite edge (k, l)
      // such that {i, j, k, l} == {0, 1, 2, 3}.
      // clang-format off
          // e.g. (0) -> (0 + 1 + (3 - 0)%2)%4 -> (1 + 1)%4 -> (2)
          //      (3)    (3 + 1 + (3 - 0)%2)%4    (4 + 1)%4    (1)
      // clang-format on
      // This works for every possible edge (i, j) where i < j.
      const int v_A0 = result.closest_A.indices[0];
      const int v_A1 = result.closest_A.indices[1];
      const int f_A0 = (v_A0 + 1 + (v_A1 - v_A0) % 2) % 4;
      const int f_A1 = (v_A1 + 1 + (v_A1 - v_A0) % 2) % 4;

      const int v_B0 = result.closest_B.indices[0];
      const int v_B1 = result.closest_B.indices[1];
      const int f_B0 = (v_B0 + 1 + (v_B1 - v_B0) % 2) % 4;
      const int f_B1 = (v_B1 + 1 + (v_B1 - v_B0) % 2) % 4;

      // Outward normals of the faces adjacent to the closest edges,
      // expressed in world.
      const Vector3<T> n_A0 =
          -(X_WA.rotation() *
            soft_A.mesh().inward_normal(tet_A, f_A0).template cast<T>());
      const Vector3<T> n_A1 =
          -(X_WA.rotation() *
            soft_A.mesh().inward_normal(tet_A, f_A1).template cast<T>());
      const Vector3<T>& n_B0 =
          -(X_WB.rotation() *
            soft_B.mesh().inward_normal(tet_B, f_B0).template cast<T>());
      const Vector3<T>& n_B1 =
          -(X_WB.rotation() *
            soft_B.mesh().inward_normal(tet_B, f_B1).template cast<T>());

      const Vector3<T> n =
          (n_A1.cross(n_A0)).cross(n_B0.cross(n_B1)).normalized();

      // TODO(joemasterjohn): Play around with the algebra in the formulation
      // to see if this can be simplified.
      const T n_A0_dot_n = n_A0.dot(n);
      const T n_A1_dot_n = n_A1.dot(n);
      const Vector3<T> n_A1_x_n_B0 = n_A1.cross(n_B0);
      const Vector3<T> n_B0_x_n_A0 = n_B0.cross(n_A0);
      const Vector3<T> n_A1_x_n_B1 = n_A1.cross(n_B1);
      const Vector3<T> n_B1_x_n_A0 = n_B1.cross(n_A0);
      const Vector3<T> n_B0_x_n_B1 = n_B0.cross(n_B1);
      // TODO(joemasterjohn): it is not enough to assume that these vertices are
      // in general position. These denominators can be 0 in simple examples
      // where two faces are parallel to each other. Decide what to do in that
      // case.
      // clang-format off
      const Vector3<T> A = (-n_A0_dot_n*n_A1_x_n_B0 - n_A1_dot_n*n_B0_x_n_A0) / n_A0.dot(n_A1_x_n_B0);
      const Vector3<T> B = (-n_A0_dot_n*n_A1_x_n_B1 - n_A1_dot_n*n_B1_x_n_A0) / n_A0.dot(n_A1_x_n_B1);
      const Vector3<T> C = (-n_A0_dot_n*n_B0_x_n_B1) / n_A0.dot(n_B0_x_n_B1);
      const Vector3<T> D = (-n_A1_dot_n*n_B0_x_n_B1) / n_A1.dot(n_B0_x_n_B1);
      // clang-format on

      coefficients.emplace_back(abs(((D - A).dot((B - A).cross(C - A)))) / 6.0);

      // DRAKE_ASSERT(coefficients.back() >= 0);

      // n is aligned with the volume normal, but possibly in the wrong
      // direction. Set zhat_BA to either n or -n such that it points out of
      // B into A.
      if (n.dot(p_BqAp_W) < 0) {
        zhat_BA_W.emplace_back(-n);
      } else {
        zhat_BA_W.emplace_back(n);
      }
    }

    // Avoid nans in the coefficients. I think this is happening because of
    // co-planar faces in the volume formulation, but need to confirm. For now
    // just ignore the problematic pairs.
    using std::isinf;
    using std::isnan;
    // if (isinf(coefficients.back()) || isnan(coefficients.back()) ||
    //     isnan(time_of_contact.back()) ||
    //     zhat_BA_W.back().array().isNaN().any() ||
    //     p_WC.back().array().isNaN().any()) {
    if (isinf(coefficients.back()) || isnan(coefficients.back()) || coefficients.back() > 1e12) {
      closest_points.pop_back();
      time_of_contact.pop_back();
      p_WC.pop_back();
      p_AC_W.pop_back();
      p_BC_W.pop_back();
      zhat_BA_W.pop_back();
      coefficients.pop_back();
      continue;
    }

    // Calculate representative radii of tetrahdra based on their volume.
    using std::cbrt;
    const T vA = soft_A.mesh().CalcTetrahedronVolume(tet_A);
    const T vB = soft_B.mesh().CalcTetrahedronVolume(tet_B);
    const T rA = cbrt(3*ExtractDoubleOrThrow(vA) / (4*std::numbers::pi));
    const T rB = cbrt(3*ExtractDoubleOrThrow(vB) / (4*std::numbers::pi));
    // Calculate "effective" radius of the two representative sphere coming into contact.
    const T R = rA*rB / (rA + rB);
    effective_radius.emplace_back(R);

    // Get the gradient of the pressure field on each tet, and re-express in
    // world.
    grad_eA_W.emplace_back(
        X_WA.rotation() *
        soft_A.pressure_field().EvaluateGradient(tet_A).template cast<T>());
    grad_eB_W.emplace_back(
        X_WB.rotation() *
        soft_B.pressure_field().EvaluateGradient(tet_B).template cast<T>());
    // Calculate the contact normal, defined in the same manner as discrete
    // hydro.
    nhat_BA_W.emplace_back((grad_eA_W.back() - grad_eB_W.back()).normalized());

    // Add the element pair to the valid element pairs.
    valid_element_pairs.emplace_back(std::make_pair(tet_A, tet_B));
  }

  end = std::chrono::high_resolution_clock::now();
  duration1 = end - start;
  fmt::print("    Narrowphase: {}s\n", duration1.count());

  // Quick exit if no speculative contacts are found.
  if (ssize(p_WC) == 0) return;

  fmt::print("    num constraints: {}\n", ssize(p_WC));

  std::map<int, std::set<SortedTriplet<int>>> vertex0_face1_map;
  std::map<int, std::set<SortedTriplet<int>>> vertex1_face0_map;
  std::map<SortedPair<int>, std::set<SortedPair<int>>> edge0_edge1_map;
  std::map<SortedPair<int>, std::set<SortedPair<int>>> edge1_edge0_map;

  int num_duplicates = 0;

  for (int i = 0; i < ssize(p_WC); ++i) {
    const ClosestPointResult<T>& result = closest_points[i];
    const int tet0 = valid_element_pairs[i].first;
    const int tet1 = valid_element_pairs[i].second;
    const VolumeElement eA = soft_A.mesh().element(tet0);
    const VolumeElement eB = soft_B.mesh().element(tet1);

    if (result.closest_A.type == ClosestPointType::Vertex) {
      const int vA = eA.vertex(result.closest_A.indices[0]);
      const SortedTriplet<int> fB(eB.vertex(result.closest_B.indices[0]),
                                  eB.vertex(result.closest_B.indices[1]),
                                  eB.vertex(result.closest_B.indices[2]));
      if (vertex0_face1_map.contains(vA)) {
        std::set<SortedTriplet<int>>& faces = vertex0_face1_map[vA];
        if (faces.contains(fB)) {
          ++num_duplicates;
        } else {
          faces.insert(fB);
        }
      } else {
        vertex0_face1_map[vA] = {fB};
      }
    } else if (result.closest_A.type == ClosestPointType::Face) {
      const int vB = eB.vertex(result.closest_B.indices[0]);
      const SortedTriplet<int> fA(eA.vertex(result.closest_A.indices[0]),
                                  eA.vertex(result.closest_A.indices[1]),
                                  eA.vertex(result.closest_A.indices[2]));
      if (vertex1_face0_map.contains(vB)) {
        std::set<SortedTriplet<int>>& faces = vertex1_face0_map[vB];
        if (faces.contains(fA)) {
          ++num_duplicates;
        } else {
          faces.insert(fA);
        }
      } else {
        vertex1_face0_map[vB] = {fA};
      }
    } else {
      const SortedPair<int> edgeA(eA.vertex(result.closest_A.indices[0]),
                                  eA.vertex(result.closest_A.indices[1]));
      const SortedPair<int> edgeB(eB.vertex(result.closest_B.indices[0]),
                                  eB.vertex(result.closest_B.indices[1]));
      bool duplicate = false;
      if (edge0_edge1_map.contains(edgeA)) {
        std::set<SortedPair<int>> edges = edge0_edge1_map[edgeA];
        if (edges.contains(edgeB)) {
          duplicate = true;
        } else {
          edges.insert(edgeB);
        }
      } else {
        edge0_edge1_map[edgeA] = {edgeB};
      }

      if (edge1_edge0_map.contains(edgeB)) {
        std::set<SortedPair<int>> edges = edge1_edge0_map[edgeB];
        if (edges.contains(edgeA)) {
          duplicate = true;
        } else {
          edges.insert(edgeA);
        }
      } else {
        edge1_edge0_map[edgeB] = {edgeA};
      }

      if(duplicate) {
        ++num_duplicates;
      }
    }
  }

  fmt::print("    num duplicates: {}\n", num_duplicates);
  fmt::print("    num unique vA: {}\n", ssize(vertex0_face1_map));
  fmt::print("    num unique vB: {}\n", ssize(vertex1_face0_map));
  fmt::print("    num unique eA: {}\n", ssize(edge0_edge1_map));
  fmt::print("    num unique eB: {}\n", ssize(edge1_edge0_map));

  for(const auto [vA, fB] : vertex0_face1_map)

  speculative_surfaces->emplace_back(
      id_A, id_B, p_WC, p_AC_W, p_BC_W, time_of_contact, zhat_BA_W,
      coefficients, nhat_BA_W, grad_eA_W, grad_eB_W, closest_points,
      valid_element_pairs, effective_radius);
}

template <typename T>
void SpeculativeContactCalculator<T>::ComputeSpeculativeContactSurface(
    GeometryId id_A, GeometryId id_B,
    std::vector<SpeculativeContactSurface<T>>* speculative_surfaces) const {
  const SoftGeometry& soft_A = geometries_.soft_geometry(id_A);
  const SoftGeometry& soft_B = geometries_.soft_geometry(id_B);
  const math::RigidTransform<T>& X_WA = X_WGs_.at(id_A);
  const math::RigidTransform<T>& X_WB = X_WGs_.at(id_B);
  const multibody::SpatialVelocity<T>& V_WA = V_WGs_.at(id_A);
  const multibody::SpatialVelocity<T>& V_WB = V_WGs_.at(id_B);

  ComputeSpeculativeContactSurfaceByClosestPoints(id_A, id_B, soft_A, soft_B,
                                                  X_WA, X_WB, V_WA, V_WB, dt_,
                                                  speculative_surfaces);
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&MovingBoundingSphereAabbCalculator<T>,
     &TruncatedTaylorSeriesAabbCalculator<T>, &StaticMeshAabbCalculator<T>,
     &ComputeSpeculativeContactSurfaceByClosestPoints<T>));

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class SpeculativeContactCalculator);

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
