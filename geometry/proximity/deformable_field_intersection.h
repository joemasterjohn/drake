#pragma once

#include <memory>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/plane.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/query_results/deformable_contact.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

// TODO(xuchenhan-tri): This is a (modified) copypasta from
// field_intersection.h/cc. We should combine/reuse code in these files,
// possibly with a pattern similar to SurfaceVolumeIntersector. Furthermore, we
// should also support calculation of the intersection of compliant hydro
// geometry and deformable geometry.

/* Creates the mesh and the field of the contact surface between two tetrahedral
 meshes with fields. The output surface mesh is posed in frame M of the first
 tetrahedral mesh.

 @param[in] field0_M  The first geometry represented as a tetrahedral mesh a
                      field, expressed in frame M.
 @param[in] bvh0_M    The bounding volume hierarchy built on the tetrahedral
                      mesh of `field0_M`.
 @param[in] field1_N  The second geometry represented as a tetrahedral mesh with
                      a field, expressed in frame N.
 @param[in] bvh1_N    The bounding volume hierarchy built on the tetrahedral
                      mesh of `field1_N`.
 @param[in] X_MN      The pose of frame N in frame M.
 @param[out] surface_01_M
                      The output mesh of the contact surface between the two
                      geometries of `field0_M` and `field1_N`, expressed in
                      frame M.
 @param[out] e_01_M   The pressure field on the contact surface, expressed in
                      frame M.
 @param[out] barycentric_coordinates_M
                      The barycentric coordindates of the centroid of each
                      polygon in the output mesh with respect to the tetrahedra
                      in M containing the centroid.
 @param[out] barycentric_coordinates_N
                      The barycentric coordindates of the centroid of each
                      polygon in the output mesh with respect to the tetrahedra
                      in N containing the centroid.
 @param[out] contact_vertex_indices_M
                      The vertex indices of the tetrahedron in M containing the
                      centroid of each polygon in the output mesh.
 @param[out] contact_vertex_indices_N
                      The vertex indices of the tetrahedron in N containing the
                      centroid of each polygon in the output mesh.
 @note  The output surface mesh may have duplicate vertices.
 */
void IntersectFields(
    const VolumeMeshFieldLinear<double, double>& field0_M,
    const Bvh<Aabb, VolumeMesh<double>>& bvh0_M,
    const VolumeMeshFieldLinear<double, double>& field1_N,
    const Bvh<Aabb, VolumeMesh<double>>& bvh1_N,
    const math::RigidTransform<double>& X_MN,
    std::unique_ptr<PolygonSurfaceMesh<double>>* surface_01_M,
    std::unique_ptr<MeshFieldLinear<double, PolygonSurfaceMesh<double>>>*
        e_01_M,
    std::vector<Vector4<double>>* barycentric_coordinates_M,
    std::vector<Vector4<double>>* barycentric_coordinates_N,
    std::vector<Vector4<int>>* contact_vertex_indices_M,
    std::vector<Vector4<int>>* contact_vertex_indices_N);

/* Computes the contact surface between two compliant hydroelastic geometries
 given a specific mesh-builder instance. The output contact surface is posed
 in World frame.

 @param[in] id0        ID of the first geometry.
 @param[in] field0_F   Pressure field on the tetrahedral mesh of the first
                       geometry, expressed in frame F.
 @param[in] bvh0_F     A bounding volume hierarchy built on the mesh of
                       `field0_F`.
 @param[in] X_WF       The pose of the first geometry in World.
 @param[in] id1        ID of the second geometry.
 @param[in] field1_G   Pressure field on the tetrahedral mesh of the second
                       geometry, expressed in frame G.
 @param[in] bvh1_G     A bounding volume hierarchy built on the mesh of
                       `field1_G`.
 @param[in] X_WG       The pose of the second geometry in World.
 @param[out] deformable_contact
                       The contact data between geometry 0 and 1 are added to
                       `deformable_contact`.
 */
void IntersectDeformableVolumes(
    GeometryId id0, const VolumeMeshFieldLinear<double, double>& field0_F,
    const Bvh<Aabb, VolumeMesh<double>>& bvh0_F,
    const math::RigidTransform<double>& X_WF, GeometryId id1,
    const VolumeMeshFieldLinear<double, double>& field1_G,
    const Bvh<Aabb, VolumeMesh<double>>& bvh1_G,
    const math::RigidTransform<double>& X_WG,
    DeformableContact<double>* deformable_contact);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
