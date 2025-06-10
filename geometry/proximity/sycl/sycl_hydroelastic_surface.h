#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

/*
  This class is used to store the hydroelastic collision surface data for
  all geometry pairs in a Structure of Arrays (SoA) format. It holds arrays
  (as std::vectors) for properties of each polygon that make up the hydroelastic
  surfaces, along with the geometry IDs that each polygon belongs to.
*/
class SYCLHydroelasticSurface {
 public:
  /*
    Constructs a SYCLHydroelasticSurface containing all collision polygons.

    @param centroids Vector of centroids of each polygon expressed in the world
    frame.
    @param areas Vector of areas of each polygon.
    @param pressure_Ws Pressure at the centroid of each polygon.
    @param normal_Ws Vector of normal vectors of each polygon expressed in the
    world frame.
    @param g_M Vector of scalar pressure gradients g_M for each polygon.
    @param g_N Vector of scalar pressure gradients g_N for each polygon.
    @param geometry_ids_M Vector of geometry IDs for the first geometry of each
    polygon.
    @param geometry_ids_N Vector of geometry IDs for the second geometry of each
    polygon.
  */
  SYCLHydroelasticSurface(std::vector<Vector3<double>> centroids,
                          std::vector<double> areas,
                          std::vector<double> pressure_Ws,
                          std::vector<Vector3<double>> normal_Ws,
                          std::vector<double> g_M, std::vector<double> g_N,
                          std::vector<GeometryId> geometry_ids_M,
                          std::vector<GeometryId> geometry_ids_N);

  /*
    Factory method to create a SYCLHydroelasticSurface from device memory
    arrays. This method transfers data from device to host

    @param q_device SYCL queue for memory transfers
    @param polygon_centroids Device pointer to polygon centroids
    @param polygon_areas Device pointer to polygon areas
    @param polygon_pressure_W Device pointer to polygon pressures
    @param polygon_normals Device pointer to polygon normals
    @param polygon_g_M Device pointer to g_M values
    @param polygon_g_N Device pointer to g_N values
    @param polygon_geom_index_A Device pointer to geometry A indices
    @param polygon_geom_index_B Device pointer to geometry B indices
    @param narrow_phase_check_validity Device pointer to validity flags
    @param total_narrow_phase_checks Total number of narrow phase checks
    @returns SYCLHydroelasticSurface containing all valid polygons
  */
  static SYCLHydroelasticSurface CreateFromDeviceMemory(
      sycl::queue& q_device, const Vector3<double>* compacted_polygon_centroids,
      const double* compacted_polygon_areas,
      const double* compacted_polygon_pressure_W,
      const Vector3<double>* compacted_polygon_normals,
      const double* compacted_polygon_g_M, const double* compacted_polygon_g_N,
      const GeometryId* compacted_polygon_geom_index_A,
      const GeometryId* compacted_polygon_geom_index_B,
      const size_t total_polygons);

  // Accessors
  const std::vector<Vector3<double>>& centroids() const { return centroid_; }
  const std::vector<double>& areas() const { return area_; }
  const std::vector<double>& pressures() const { return pressure_W_; }
  const std::vector<Vector3<double>>& normals() const { return normal_W_; }
  const std::vector<double>& g_M() const { return g_M_; }
  const std::vector<double>& g_N() const { return g_N_; }
  const std::vector<GeometryId>& geometry_ids_M() const {
    return geometry_ids_M_;
  }
  const std::vector<GeometryId>& geometry_ids_N() const {
    return geometry_ids_N_;
  }

  // Utility methods
  size_t num_polygons() const { return area_.size(); }
  bool empty() const { return area_.empty(); }

 private:
  std::vector<Vector3<double>> centroid_;
  std::vector<double> area_;
  std::vector<double> pressure_W_;
  std::vector<Vector3<double>> normal_W_;
  std::vector<double> g_M_;
  std::vector<double> g_N_;
  std::vector<GeometryId> geometry_ids_M_;
  std::vector<GeometryId> geometry_ids_N_;
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
