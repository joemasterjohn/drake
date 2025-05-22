#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

SYCLHydroelasticSurface::SYCLHydroelasticSurface(
    std::vector<Vector3<double>> centroids, std::vector<double> areas,
    std::vector<double> pressure_Ws,
    std::vector<Vector3<double>> grad_pressure_Ms,
    std::vector<Vector3<double>> grad_pressure_Ns,
    std::vector<Vector3<double>> normal_Ws, GeometryId id_M, GeometryId id_N)
    : centroid_(std::move(centroids)),
      area_(std::move(areas)),
      pressure_W_(std::move(pressure_Ws)),
      grad_pressure_M_(std::move(grad_pressure_Ms)),
      grad_pressure_N_(std::move(grad_pressure_Ns)),
      normal_W_(std::move(normal_Ws)),
      id_M_(id_M),
      id_N_(id_N) {}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
