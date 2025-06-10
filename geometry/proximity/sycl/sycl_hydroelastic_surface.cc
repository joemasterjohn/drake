#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"

#include <algorithm>
#include <map>
#include <utility>

#include "drake/common/drake_assert.h"
namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

SYCLHydroelasticSurface::SYCLHydroelasticSurface(
    std::vector<Vector3<double>> centroids, std::vector<double> areas,
    std::vector<double> pressure_Ws, std::vector<Vector3<double>> normal_Ws,
    std::vector<double> g_M, std::vector<double> g_N,
    std::vector<GeometryId> geometry_ids_M,
    std::vector<GeometryId> geometry_ids_N)
    : centroid_(std::move(centroids)),
      area_(std::move(areas)),
      pressure_W_(std::move(pressure_Ws)),
      normal_W_(std::move(normal_Ws)),
      g_M_(std::move(g_M)),
      g_N_(std::move(g_N)),
      geometry_ids_M_(std::move(geometry_ids_M)),
      geometry_ids_N_(std::move(geometry_ids_N)) {
  // Verify that all vectors have the same size
  DRAKE_THROW_UNLESS(centroid_.size() == area_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == pressure_W_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == normal_W_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == g_M_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == g_N_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == geometry_ids_M_.size());
  DRAKE_THROW_UNLESS(centroid_.size() == geometry_ids_N_.size());
}

SYCLHydroelasticSurface SYCLHydroelasticSurface::CreateFromDeviceMemory(
    sycl::queue& q_device, Vector3<double>* polygon_centroids,
    double* polygon_areas, double* polygon_pressure_W,
    Vector3<double>* polygon_normals, double* polygon_g_M, double* polygon_g_N,
    GeometryId* polygon_geom_index_A, GeometryId* polygon_geom_index_B,
    uint8_t* narrow_phase_check_validity, size_t total_narrow_phase_checks) {
  if (total_narrow_phase_checks == 0) {
    return SYCLHydroelasticSurface({}, {}, {}, {}, {}, {}, {}, {});
  }

  // Transfer data from device to host
  std::vector<Vector3<double>> host_centroids(total_narrow_phase_checks);
  std::vector<double> host_areas(total_narrow_phase_checks);
  std::vector<double> host_pressure_W(total_narrow_phase_checks);
  std::vector<Vector3<double>> host_normals(total_narrow_phase_checks);
  std::vector<double> host_g_M(total_narrow_phase_checks);
  std::vector<double> host_g_N(total_narrow_phase_checks);
  std::vector<GeometryId> host_geom_A(total_narrow_phase_checks);
  std::vector<GeometryId> host_geom_B(total_narrow_phase_checks);
  std::vector<uint8_t> host_validity(total_narrow_phase_checks);

  // Perform all memory transfers in parallel
  std::vector<sycl::event> transfer_events;
  transfer_events.push_back(
      q_device.memcpy(host_centroids.data(), polygon_centroids,
                      total_narrow_phase_checks * sizeof(Vector3<double>)));
  transfer_events.push_back(
      q_device.memcpy(host_areas.data(), polygon_areas,
                      total_narrow_phase_checks * sizeof(double)));
  transfer_events.push_back(
      q_device.memcpy(host_pressure_W.data(), polygon_pressure_W,
                      total_narrow_phase_checks * sizeof(double)));
  transfer_events.push_back(
      q_device.memcpy(host_normals.data(), polygon_normals,
                      total_narrow_phase_checks * sizeof(Vector3<double>)));
  transfer_events.push_back(
      q_device.memcpy(host_g_M.data(), polygon_g_M,
                      total_narrow_phase_checks * sizeof(double)));
  transfer_events.push_back(
      q_device.memcpy(host_g_N.data(), polygon_g_N,
                      total_narrow_phase_checks * sizeof(double)));
  transfer_events.push_back(
      q_device.memcpy(host_geom_A.data(), polygon_geom_index_A,
                      total_narrow_phase_checks * sizeof(GeometryId)));
  transfer_events.push_back(
      q_device.memcpy(host_geom_B.data(), polygon_geom_index_B,
                      total_narrow_phase_checks * sizeof(GeometryId)));
  transfer_events.push_back(
      q_device.memcpy(host_validity.data(), narrow_phase_check_validity,
                      total_narrow_phase_checks * sizeof(uint8_t)));

  // Wait for all transfers to complete
  sycl::event::wait_and_throw(transfer_events);

  return SYCLHydroelasticSurface(
      std::move(host_centroids), std::move(host_areas),
      std::move(host_pressure_W), std::move(host_normals), std::move(host_g_M),
      std::move(host_g_N), std::move(host_geom_A), std::move(host_geom_B));
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
