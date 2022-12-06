#include "drake/geometry/proximity/deformable_field_intersection.h"

#include <iostream>
#include <memory>
#include <unordered_set>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/geometry/proximity/mesh_intersection.h"
#include "drake/geometry/proximity/mesh_plane_intersection.h"
#include "drake/geometry/proximity/plane.h"
#include "drake/geometry/proximity/posed_half_space.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;
using MeshType = PolygonSurfaceMesh<double>;
using FieldType = MeshFieldLinear<double, MeshType>;
using MeshBuilder = PolyMeshBuilder<double>;

void IntersectFields(const VolumeMeshFieldLinear<double, double>& field0_M,
                     const Bvh<Aabb, VolumeMesh<double>>& bvh0_M,
                     const VolumeMeshFieldLinear<double, double>& field1_N,
                     const Bvh<Aabb, VolumeMesh<double>>& bvh1_N,
                     const math::RigidTransform<double>& X_MN,
                     std::unique_ptr<MeshType>* surface_01_M,
                     std::unique_ptr<FieldType>* e_01_M,
                     std::vector<Vector4<double>>* barycentric_coordinates_M,
                     std::vector<Vector4<double>>* barycentric_coordinates_N,
                     std::vector<Vector4<int>>* contact_vertex_indices_M,
                     std::vector<Vector4<int>>* contact_vertex_indices_N) {
  DRAKE_DEMAND(surface_01_M != nullptr);
  DRAKE_DEMAND(e_01_M != nullptr);
  DRAKE_DEMAND(barycentric_coordinates_M != nullptr);
  DRAKE_DEMAND(barycentric_coordinates_N != nullptr);
  DRAKE_DEMAND(contact_vertex_indices_M != nullptr);
  DRAKE_DEMAND(contact_vertex_indices_N != nullptr);
  surface_01_M->reset();
  e_01_M->reset();
  barycentric_coordinates_M->clear();
  barycentric_coordinates_N->clear();
  contact_vertex_indices_M->clear();
  contact_vertex_indices_N->clear();

  std::vector<std::pair<int, int>> candidate_tetrahedra;
  auto callback = [&candidate_tetrahedra](int tet0,
                                          int tet1) -> BvttCallbackResult {
    candidate_tetrahedra.emplace_back(tet0, tet1);
    return BvttCallbackResult::Continue;
  };
  bvh0_M.Collide(bvh1_N, X_MN, callback);
  const math::RigidTransform<double> X_NM = X_MN.inverse();

  MeshBuilder builder;
  // Here the contact polygon is represented as a list of vertex indices.
  std::vector<int> contact_polygon;
  // Each contact polygon has at most 8 vertices because it is the
  // intersection of the pressure-equilibrium plane and the two tetrahedra.
  // The plane intersects a tetrahedron into a convex polygon with at most four
  // vertices. That convex polygon intersects a tetrahedron into at most four
  // more vertices.
  contact_polygon.reserve(8);
  const math::RotationMatrix<double> R_NM = X_MN.rotation().inverse();
  for (const auto& [tet0, tet1] : candidate_tetrahedra) {
    // Initialize the plane with a non-zero-length normal vector
    // and an arbitrary point.
    Plane<double> equilibrium_plane_M{Vector3d::UnitZ(), Vector3d::Zero()};
    if (!CalcEquilibriumPlane(tet0, field0_M, tet1, field1_N, X_MN,
                              &equilibrium_plane_M)) {
      continue;
    }
    Vector3<double> polygon_nhat_M = equilibrium_plane_M.normal();
    if (!IsPlaneNormalAlongPressureGradient(polygon_nhat_M, tet0, field0_M)) {
      continue;
    }
    Vector3<double> reverse_polygon_nhat_N = R_NM * (-polygon_nhat_M);
    if (!IsPlaneNormalAlongPressureGradient(reverse_polygon_nhat_N, tet1,
                                            field1_N)) {
      continue;
    }
    const std::vector<Vector3<double>>& polygon_vertices_M =
        IntersectTetrahedra(tet0, field0_M.mesh(), tet1, field1_N.mesh(), X_MN,
                            equilibrium_plane_M);

    if (polygon_vertices_M.size() < 3) continue;

    // Add the vertices to the builder (with corresponding pressure values)
    // and construct index-based polygon representation.
    std::vector<int> polygon_vertex_indices;
    polygon_vertex_indices.reserve(polygon_vertices_M.size());
    for (const auto& p_MV : polygon_vertices_M) {
      polygon_vertex_indices.push_back(
          builder.AddVertex(p_MV, field0_M.EvaluateCartesian(tet0, p_MV)));
    }

    Vector4<int> v_M;
    Vector4<int> v_N;
    for (int i = 0; i < VolumeMesh<double>::kVertexPerElement; ++i) {
      v_M(i) = field0_M.mesh().element(tet0).vertex(i);
      v_N(i) = field1_N.mesh().element(tet1).vertex(i);
    }
    contact_vertex_indices_M->emplace_back(v_M);
    contact_vertex_indices_N->emplace_back(v_N);

    const Vector3<double>& grad_field0_M = field0_M.EvaluateGradient(tet0);
    builder.AddPolygon(polygon_vertex_indices, polygon_nhat_M, grad_field0_M);

    // TODO(xuchenhan-tri): Consider accessing the newly added polygon from
    //  the builder. Here we assume internal knowledge how the function
    //  AddPolygon works, i.e., the list of new vertices form the new polygon in
    //  that order.
    /* The centroid of the newly added polygon in the M frame. */
    const Vector3<double> p_MC = CalcPolygonCentroid(
        polygon_vertex_indices, polygon_nhat_M, builder.vertices());
    barycentric_coordinates_M->emplace_back(
        field0_M.mesh().CalcBarycentric(p_MC, tet0));
    barycentric_coordinates_N->emplace_back(
        field1_N.mesh().CalcBarycentric(X_NM * p_MC, tet1));
  }

  if (builder.num_faces() == 0) return;

  std::tie(*surface_01_M, *e_01_M) = builder.MakeMeshAndField();
}

void IntersectDeformableVolumes(
    GeometryId id0, const VolumeMeshFieldLinear<double, double>& field0_F,
    const Bvh<Aabb, VolumeMesh<double>>& bvh0_F,
    const math::RigidTransform<double>& X_WF, GeometryId id1,
    const VolumeMeshFieldLinear<double, double>& field1_G,
    const Bvh<Aabb, VolumeMesh<double>>& bvh1_G,
    const math::RigidTransform<double>& X_WG,
    DeformableContact<double>* deformable_contact) {
  DRAKE_DEMAND(deformable_contact != nullptr);
  const math::RigidTransform<double> X_FG = X_WF.InvertAndCompose(X_WG);

  // The computation will be in Frame F and then transformed to the world frame.
  std::unique_ptr<MeshType> surface01_F;
  std::unique_ptr<FieldType> field01_F;
  std::vector<Vector4<double>> barycentric_coordinates_0;
  std::vector<Vector4<double>> barycentric_coordinates_1;
  std::vector<Vector4<int>> contact_vertex_indices_0;
  std::vector<Vector4<int>> contact_vertex_indices_1;
  IntersectFields(field0_F, bvh0_F, field1_G, bvh1_G, X_FG, &surface01_F,
                  &field01_F, &barycentric_coordinates_0,
                  &barycentric_coordinates_1, &contact_vertex_indices_0,
                  &contact_vertex_indices_1);

  if (surface01_F == nullptr) return;

  const int num_faces = surface01_F->num_faces();
  std::unordered_set<int> participating_vertices_0;
  std::unordered_set<int> participating_vertices_1;
  participating_vertices_0.reserve(4 * num_faces);
  participating_vertices_1.reserve(4 * num_faces);
  for (const Vector4<int>& v : contact_vertex_indices_0) {
    for (int i = 0; i < 4; ++i) {
      participating_vertices_0.insert(v(i));
    }
  }
  for (const Vector4<int>& v : contact_vertex_indices_1) {
    for (int i = 0; i < 4; ++i) {
      participating_vertices_1.insert(v(i));
    }
  }

  /* Compute the penetration distance at the centroid of each contact polygon
   using the signed distance field. */
  std::vector<double> penetration_distances(num_faces);
  for (int i = 0; i < num_faces; ++i) {
    const Vector3<double>& contact_point_F = surface01_F->element_centroid(i);
    /* `field01_F` has a gradient, therefore `EvaluateCartesian()`
     should be cheap. */
    penetration_distances[i] = field01_F->EvaluateCartesian(i, contact_point_F);
  }
  /* Convert to world frame before reporting. */
  surface01_F->TransformVertices(X_WF);

  // The deformable contact surface is documented as having the normals pointing
  // *out* of the second geometry and *into* the first geometry. This code
  // creates a surface mesh with normals pointing out of field0's geometry into
  // field1's geometry, so we make sure the ids are ordered so that the field0
  // is the second id.
  // BE EXTREMELY CAREFUL HERE!!! The result is the OPPOSITE of the behavior of
  // hydroelastic. This is because the field for deformable is a signed distance
  // field instead of a pressure field. Therefore the field value decreases as
  // penetration goes deeper, whereas for hydro, the opposite is a true. As a
  // result, the CalcEquilibriumPlane function returns a plane with the opposite
  // normal as in the hydro case. Note the language in the doc of
  // CalcEquilibriumPlane "... n̂ points in the direction of increasing f₀ and
  // decreasing f₁."
  deformable_contact->AddDeformableDeformableContactSurface(
      id1, id0, participating_vertices_1, participating_vertices_0,
      std::move(*surface01_F), std::move(penetration_distances),
      std::move(contact_vertex_indices_1), std::move(contact_vertex_indices_0),
      std::move(barycentric_coordinates_1),
      std::move(barycentric_coordinates_0));
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
