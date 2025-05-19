#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sycl_hydroelastic_surface.h"
#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

/* This class implements a SYCL compatible version for performing geometric
 * _proximity_ queries. It is instantiated with geometric instances
 * (hydroelastic::Geometries) lazily when the contact surfaces are to be
 * computed (after ALL geometries of the scene have been instantiated).
 * Additionally, it requires a std::vector<SortedPair<GeometryId>> arrising from
 * the broad phase collision detection (FindCollisionCandidates()) that we
 * continue to keep on the host. Finally, this ProximityEngine only supports
 * CompliantCompliant collisions where the compliant bodies are not half spaces
 * and will throw if the hydroelastic::Geometry provided is not Compliant
 *
 * To provide geometric queries on the geometries, it provides a public member
 * function that takes a map of the geometry ID and its pose.
 *
 * TODO(huzaifa): MaybeMakeContactSurface looks at the type of collision of the
 * two bodies and then, from the hydroelastic::Geometries and if both the meshes
 * are "compliant-compliant", then, it extracts the "SoftGeometry" (indexing
 * with GeometryID). Then, within CalcCompliantCompliant, the soft geometries
 * are further checked for whether they are half-spaces --- if not, then the
 * soft mesh is extracted and that is what is used in
 * ComputeContactSurfaceFromCompliantVolumes. Understand if its possible to
 * instantiate this class with SoftMesh directly? What are the gains from this?
 * What are we losing out on?
 *
 * TODO part2(huzaifa): Even if we instantiate this class with
 * hydroelastic::Geometries, can we still only store the softMesh of the
 * candidates? This would mean being more lazily and only filling up memory
 * when we are looping over candidates.
 */

class SyclProximityEngine {
 public:
  /* @returns true iff the SYCL implementation is available. */
  static bool is_available();

  /* @param geometries The geometries to use for the proximity queries.
   * To be supplied lazily when contact surface is to be computed. */
  SyclProximityEngine(const hydroelastic::Geometries& geometries,
                      std::vector<SortedPair<GeometryId>> collision_candidates);

  ~SyclProximityEngine();
  SyclProximityEngine(const SyclProximityEngine& other);
  SyclProximityEngine& operator=(const SyclProximityEngine& other);

  /* @param X_WGs The poses of the geometries to compute the contact surface
   * for.
   * @returns A vector of SYCLHydroelasticSurfaces from each candidate collision
   * pair of geometries. The HydroelasticSurface itself holds the Id's of the
   * geometries that it belongs to.*/
  std::vector<SYCLHydroelasticSurface> ComputeSYCLHydroelasticSurface(
      const std::unordered_map<GeometryId, math::RigidTransform<double>>&
          X_WGs);

 private:
  // The queue to use for the SYCL operations.
  sycl::queue q_;

  // The collision candidates.
  std::vector<SortedPair<GeometryId>> collision_candidates_;

  /*
  A hydroelastic geometry only contains one mesh. Elements can only be
  tetrahedra.
  @param elements_ Elements of the mesh represented by 4 vertex indices that
  make up the tetrahedron. Query with element index.
  @param vertices_ Vertices of the mesh represented by 3D vectors. Query with
  vertex index. Expressed in mesh frame `M`. TODO(huzaifa): Should this be
  world frame `W`?
  @param inward_normals_ Inward normals of each face of the tetrahedron. Query
  with element index. Expressed in mesh frame `M`. TODO(huzaifa): Should this be
  world frame `W`?
  @param edge_vectors_ Edge vectors of each face of the tetrahedron. Query with
  element index. Expressed in mesh frame `M`. TODO(huzaifa): Should this be
  world frame `W`?
  @param num_elements_ Number of elements in the mesh. Query by GeometryId.
  @param num_vertices_ Number of vertices in the mesh. Query by GeometryId.

  @param pressures_ Pressure field on the mesh. Query by vertex index.
  @param min_pressures_ Minimum pressure on the mesh. Query by element index.
  @param max_pressures_ Maximum pressure on the mesh. Query by element index.
  @param gradients_ Gradient of pressure field in the domain of element `i`
  (indexed). Gradients are expressued in mesh frame `M`. TODO(huzaifa): Should
  this be world frame `W`?
  @param pressures_at_Mo_ Piecewise linear pressure field on element `i`
  evaluated Mo the origin of frame `M` of the mesh.
  */
  struct SYCLHydroelasticGeometries {
    // Volume mesh flattened
    std::array<int, 4>* elements_;
    Vector3<double>* vertices_;
    std::array<Vector3<double>, 4>* inward_normals_;
    std::array<Vector3<double>, 6>* edge_vectors_;
    size_t* num_elements_;
    size_t* num_vertices_;

    // VolumeMeshLinear -> MeshFieldLinear
    double* pressures_;
    double* min_pressures_;
    double* max_pressures_;
    Vector3<double>* gradients_;
    double* pressures_at_Mo_;
  };
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
