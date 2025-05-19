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
 * _proximity_ queries. It is instantiated with soft geometry instances lazily
 * when the contact surfaces are to be computed (after ALL geometries of the
 * scene have been instantiated).
 *
 * To provide geometric queries on the geometries, it provides a public member
 * function that takes a map of the geometry ID and its pose. This function
 * requires the collision candidates to be updated first and is the
 * responsibility of the client through UpdateCollisionCandidates().
 */

class SyclProximityEngine {
 public:
  /* @returns true iff the SYCL implementation is available. */
  static bool is_available();

  /* @param soft_geometries The soft geometries to use for the proximity
   * queries. To be supplied lazily when contact surface is to be computed. */
  SyclProximityEngine(
      const std::unordered_map<GeometryId, SoftGeometry>& soft_geometries);

  ~SyclProximityEngine();
  SyclProximityEngine(const SyclProximityEngine& other);
  SyclProximityEngine& operator=(const SyclProximityEngine& other);

  /* @param collision_candidates New vector of collision candidates after
   * broad phase collision detection. */
  void UpdateCollisionCandidates(
      const std::vector<SortedPair<GeometryId>>& collision_candidates);
  /* @param X_WGs The poses of the geometries to compute the contact surface
   * for.
   * @returns A vector of SYCLHydroelasticSurfaces from each candidate collision
   * pair of geometries. The HydroelasticSurface itself holds the Id's of the
   * geometries that it belongs to.*/
  std::vector<SYCLHydroelasticSurface> ComputeSYCLHydroelasticSurface(
      const std::unordered_map<GeometryId, math::RigidTransform<double>>&
          X_WGs);

 private:
  // We have a CPU queue for operations beneficial to perform on the host and a
  // device queue for operations beneficial to perform on the Accelerator.
  sycl::queue q_device_;
  sycl::queue q_host_;
  // The collision candidates.
  std::vector<SortedPair<GeometryId>> collision_candidates_;
  // GeometryIds of soft geometries (host-side)
  GeometryId* soft_geometry_ids_ = nullptr;
  // Number of geometries
  size_t num_geometries_ = 0;

  // SYCL shared arrays for geometry lookup
  size_t* sh_element_offsets_ = nullptr;  // Element offset for each geometry
  size_t* sh_vertex_offsets_ = nullptr;   // Vertex offset for each geometry
  size_t* sh_element_counts_ = nullptr;  // Number of elements for each geometry
  size_t* sh_vertex_counts_ = nullptr;   // Number of vertices for each geometry

  /*
  A hydroelastic geometry contains one mesh. Elements are tetrahedra.
  All data is stored in contiguous arrays, with each geometry's data
  at a specific offset in these arrays.
  */

  // Mesh element data - accessed by element_offset + local_element_index
  std::array<int, 4>* elements_ = nullptr;  // Elements as 4 vertex indices
  std::array<Vector3<double>, 4>* inward_normals_M_ =
      nullptr;  // Inward normals in mesh frame
  std::array<Vector3<double>, 4>* inward_normals_W_ =
      nullptr;  // Inward normals in world frame
  std::array<Vector3<double>, 6>* edge_vectors_M_ =
      nullptr;  // Edge vectors in mesh frame
  std::array<Vector3<double>, 6>* edge_vectors_W_ =
      nullptr;  // Edge vectors in world frame

  // Mesh vertex data - accessed by vertex_offset + local_vertex_index
  Vector3<double>* vertices_M_ = nullptr;  // Vertices in mesh frame
  Vector3<double>* vertices_W_ = nullptr;  // Vertices in world frame

  // Pressure field data - accessed by element_offset + local_element_index
  double* pressures_ = nullptr;      // Pressure values
  double* min_pressures_ = nullptr;  // Minimum pressure values
  double* max_pressures_ = nullptr;  // Maximum pressure values

  // Combined gradient and pressure values to optimize cache utilization
  // First 3 components are gradient, 4th component is pressure at origin
  Vector4<double>* gradient_M_pressure_at_Mo_ = nullptr;  // In mesh frame
  Vector4<double>* gradient_W_pressure_at_Wo_ = nullptr;  // In world frame
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake