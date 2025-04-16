#pragma once

#include <unordered_map>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/query_results/speculative_contact.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/math/spatial_algebra.h"

namespace drake {
namespace geometry {
namespace internal {
namespace hydroelastic {

template <typename T>
AabbCalculator MovingBoundingSphereAabbCalculator(
    const std::vector<PosedSphere<double>>& mesh_bounding_spheres,
    const math::RigidTransform<T>& X_WG,
    const multibody::SpatialVelocity<T>& V_WG, double dt);

template <typename T>
AabbCalculator StaticMeshAabbCalculator(const VolumeMesh<double>& mesh,
                                        const math::RigidTransform<T>& X_WG);

template <typename T>
void ComputeSpeculativeContactSurfaceByClosestPoints(
    GeometryId id_A, GeometryId id_B, const SoftGeometry& soft_A,
    const SoftGeometry& soft_B, const math::RigidTransform<T>& X_WA,
    const math::RigidTransform<T>& X_WB,
    const multibody::SpatialVelocity<T>& V_WA,
    const multibody::SpatialVelocity<T>& V_WB,
    const double dt,
    std::vector<SpeculativeContactSurface<T>>* speculative_surfaces);

/* Calculator for the shape-to-shape speculative contact results. It needs:

    - The T-valued poses of _all_ geometries in the corresponding SceneGraph,
      each indexed by its corresponding geometry's GeometryId.
    - The T-values spatial velocities of _all_ geometries in the corresponding
      SceneGraph, each indexed by its corresponding geometry's GeometryId.
      @note At the time of writing, this query only supports geometries with
      compliant representations.

 @tparam_nonsymbolic_scalar  */
template <typename T>
class SpeculativeContactCalculator {
 public:
  /* Constructs the fully-specified calculator. The values are as described in
   the class documentation. Some parameters (noted below) are aliased in the
   data and must remain valid at least as long as the
   SpeculativeContactCalculator instance.

   @param X_WGs                   The T-valued poses. Aliased.
   @param V_WGs                   The T-valued spatial velocities. Aliased.
   @param geometries              The set of all hydroelastic geometric
                                  representations. Aliased. */
  SpeculativeContactCalculator(
      const std::unordered_map<GeometryId, math::RigidTransform<T>>* X_WGs,
      const std::unordered_map<GeometryId, multibody::SpatialVelocity<T>>*
          V_WGs,
      const double dt,
      const Geometries* geometries)
      : X_WGs_(*X_WGs), V_WGs_(*V_WGs), dt_(dt), geometries_(*geometries) {
    DRAKE_DEMAND(X_WGs != nullptr);
    DRAKE_DEMAND(V_WGs != nullptr);
    DRAKE_DEMAND(geometries != nullptr);
  }

  /* Makes the contact surface (if it exists) between two potentially
     colliding geometries.

     @param id_A     Id of the first object in the pair (order insignificant).
     @param id_B     Id of the second object in the pair (order insignificant).
     @param speculative_surfaces Vector to add the computed surface to.  */
  void ComputeSpeculativeContactSurface(
      GeometryId id_A, GeometryId id_B,
      std::vector<SpeculativeContactSurface<T>>* speculative_surfaces) const;

 private:
  /* The T-valued poses of all geometries.  */
  const std::unordered_map<GeometryId, math::RigidTransform<T>>& X_WGs_;

  /* The T-valued spatial velocities of all geometries. */
  const std::unordered_map<GeometryId, multibody::SpatialVelocity<T>>& V_WGs_;

  /* Plant time step. */
  const double dt_;

  /* The hydroelastic geometric representations.  */
  const Geometries& geometries_;
};

}  // namespace hydroelastic
}  // namespace internal
}  // namespace geometry
}  // namespace drake
