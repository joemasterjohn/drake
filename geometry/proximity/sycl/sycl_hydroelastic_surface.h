#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sycl/sycl.hpp>

#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

class SYCLHydroElasticSurface {
 public:
  SYCLHydroElasticSurface();

 private:
  Vector3<double> centroid_;
  double area_;
  VolumeMeshFieldLinear<double, double> pressure_W;
  Vector3<double> grad_pressure_M;
  Vector3<double> grad_pressure_N;
  Vector3<double> normal_W;
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
