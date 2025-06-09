#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "drake/geometry/geometry_ids.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

/*
  This class is used to store the hydroelastic collision surface for a given
  pair of geometries in a Structure of Arrays (SoA) format. It holds the id's of
  the two geometries that it belongs to. Additionally, it holds arrays (as
  std::vectors) for properties of each polygon that make up the hydroelastic
  surface.


*/
class SYCLHydroelasticSurface {
 public:
  /*
    Constructs a SYCLHydroelasticSurface between two geometries of ids
    `id_M` and `id_N` using a collection of polygons. A surface is not
    "constructed" in a traditional geometric sense but rather the relevant
    properties of each polygon are stored in arrays.

    @param centroids Vector of centroids of each polygon expressed in the world
    frame.
    @param areas Vector of areas of each polygon.
    @param pressure_Ws Pressure at the centroid of each polygon.
    @param grad_pressure_Ms Gradient of pressure in the domain of `tet0` from
    Mesh `M` expressed in the world frame.
    @param grad_pressure_Ns Gradient of pressure in the domain of `tet1` from
    Mesh `N` expressed in the world frame.

    * TODO(huzaifa): Do we need the grad_pressure's?
    * These gradients might not be needed because they are only used to compute
    * the scalar pressure gradients g_M and g_N. So we could just store g_M and
    * g_N directly.

    @param normal_Ws Vector of normal vectors of each polygon expressed in the
    world frame.
    @param id_M The id of the first geometry.
    @param id_N The id of the second geometry.
  */
  SYCLHydroelasticSurface(std::vector<Vector3<double>> centroids,
                          std::vector<double> areas,
                          std::vector<double> pressure_Ws,
                          std::vector<Vector3<double>> normal_Ws,
                          std::vector<double> g_M, std::vector<double> g_N,
                          GeometryId id_M, GeometryId id_N);

 private:
  std::vector<Vector3<double>> centroid_;
  std::vector<double> area_;
  std::vector<double> pressure_W_;
  std::vector<Vector3<double>> normal_W_;
  std::vector<double> g_M_;
  std::vector<double> g_N_;
  GeometryId id_M_{};
  GeometryId id_N_{};
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
