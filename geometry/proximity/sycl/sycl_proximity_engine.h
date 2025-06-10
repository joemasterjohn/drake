#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "drake/common/sorted_pair.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"
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
 *
 * This class employs the PIMPL (Pointer to IMPLementation) idiom primarily
 * because it isolates SYCL-specific code and dependencies to the implementation
 * file, preventing SYCL header files from being transitively included in
 * client code.
 */
class SyclProximityEngine {
 public:
  // Explicitly declare copy constructor and assignment operator
  SyclProximityEngine(const SyclProximityEngine&);
  SyclProximityEngine& operator=(const SyclProximityEngine&);
  /* @returns true iff the SYCL implementation is available. */
  static bool is_available();

  /* @param soft_geometries The soft geometries to use for the proximity
   * queries. To be supplied lazily when contact surface is to be computed. */
  SyclProximityEngine(
      const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>&
          soft_geometries);

  /* Default constructor creates an empty engine. */
  SyclProximityEngine();

  ~SyclProximityEngine();

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
  // The implementation class
  class Impl;
  std::unique_ptr<Impl> impl_;
  friend class SyclProximityEngineTester;
  // Add attorney as friend
  friend class SyclProximityEngineAttorney;
};

// Attorney class for accessing private members of SyclProximityEngine and Impl
class SyclProximityEngineAttorney {
 public:
  static SyclProximityEngine::Impl* get_impl(SyclProximityEngine& engine);
  static const SyclProximityEngine::Impl* get_impl(
      const SyclProximityEngine& engine);

  static std::vector<uint8_t> get_collision_filter(
      SyclProximityEngine::Impl* impl);
  static std::vector<size_t> get_prefix_sum(SyclProximityEngine::Impl* impl);
  static std::vector<Vector3<double>> get_vertices_M(
      SyclProximityEngine::Impl* impl);
  static std::vector<Vector3<double>> get_vertices_W(
      SyclProximityEngine::Impl* impl);
  static std::vector<std::array<int, 4>> get_elements(
      SyclProximityEngine::Impl* impl);
  static double* get_pressures(SyclProximityEngine::Impl* impl);
  static Vector4<double>* get_gradient_M_pressure_at_Mo(
      SyclProximityEngine::Impl* impl);
  static Vector4<double>* get_gradient_W_pressure_at_Wo(
      SyclProximityEngine::Impl* impl);
  static size_t* get_collision_filter_host_body_index(
      SyclProximityEngine::Impl* impl);
  static size_t get_total_checks(SyclProximityEngine::Impl* impl);
  static size_t get_total_narrow_phase_checks(SyclProximityEngine::Impl* impl);
  static size_t get_total_polygons(SyclProximityEngine::Impl* impl);
  static std::vector<size_t> get_narrow_phase_check_indices(
      SyclProximityEngine::Impl* impl);
  static std::vector<size_t> get_valid_polygon_indices(
      SyclProximityEngine::Impl* impl);
  static std::vector<double> get_polygon_areas(SyclProximityEngine::Impl* impl);
  static std::vector<Vector3<double>> get_polygon_centroids(
      SyclProximityEngine::Impl* impl);
  static std::vector<double> get_debug_polygon_vertices(
      SyclProximityEngine::Impl* impl);
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake