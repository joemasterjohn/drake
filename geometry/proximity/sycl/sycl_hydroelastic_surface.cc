#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

SYCLHydroelasticSurface::SYCLHydroelasticSurface(
    std::vector<Vector3<double>> centroids, std::vector<double> areas,
    std::vector<double> pressure_Ws, std::vector<Vector3<double>> normal_Ws,
    std::vector<double> g_M, std::vector<double> g_N, GeometryId id_M,
    GeometryId id_N)
    : centroid_(std::move(centroids)),
      area_(std::move(areas)),
      pressure_W_(std::move(pressure_Ws)),
      normal_W_(std::move(normal_Ws)),
      g_M_(std::move(g_M)),
      g_N_(std::move(g_N)),
      id_M_(id_M),
      id_N_(id_N) {}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
