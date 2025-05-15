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

#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

/* This class implements a SYCL compatible version for performing geometric
 * _proximity_ queries. It is instantiated with geometric instances (thus cannot
 * be used to create geometry instances) and thus is to be instantiated lazily
 * when the contact surfaces are to be computed (after ALL geometries of the
 * scene have been instantiated). To provide geometric queries on the
 * geometries, it provides a public member function that takes the poses of said
 * geometries.
 */

class SyclProximityEngine {
 public:
  /* @returns true iff the SYCL implementation is available. */
  static bool is_available();

  /* @param geometries The geometries to use for the proximity queries.
   * To be supplied lazily when contact surface is to be computed. */
  SyclProximityEngine(const hydroelastic::Geometries& geometries);

  ~SyclProximityEngine();
  SyclProximityEngine(const SyclProximityEngine& other);
  SyclProximityEngine& operator=(const SyclProximityEngine& other);

  /* @param X_WGs The poses of the geometries to compute the contact surface
   * for.
   * @returns A vectoor of SYCLHydroElasticSurfaces */

  // TODO(huzaifa): Think about how we could call FindCollisionCandidates() on
  // the geometries here Most likely, this function should also take the ID
  // pairs of the geometries that are potentially colliding.
  std::vector<SYCLHydroelasticSurface> ComputeSYCLHydroelasticSurface(
      const std::unordered_map<GeometryId, math::RigidTransform<double>>&
          X_WGs);

 private:
  // The queue to use for the SYCL operations.
  sycl::queue q_;

  struct SYCLHydroelasticGeometries {
    // TODO(huzaifa): Add the data from each hydroelastic geometry in a GPU
    // friendly manner here
  };
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
