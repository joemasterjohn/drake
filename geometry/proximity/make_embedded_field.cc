#include "drake/geometry/proximity/make_embedded_field.h"

#include <iostream>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/calc_distance_to_surface_mesh.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

template <typename T>
VolumeMeshFieldLinear<T, T> MakeEmbeddedPressureField(
    const Mesh& mesh, const VolumeMesh<T>* mesh_C,
    const T& hydroelastic_modulus, const T& depth) {
  DRAKE_DEMAND(hydroelastic_modulus > T(0));
  DRAKE_DEMAND(mesh_C != nullptr);

  const TriangleSurfaceMesh<double> surface_mesh =
      ReadObjToTriangleSurfaceMesh(mesh.filename(), mesh.scale());

  std::vector<T> pressure_values;
  pressure_values.reserve(mesh_C->num_vertices());

  for (auto v : mesh_C->vertices()) {
    Vector3<double> v_d = ExtractDoubleOrThrow(v);
    pressure_values.push_back(
        -hydroelastic_modulus *
        CalcSignedDistanceToSurfaceMesh(v_d, surface_mesh) / depth);
    // pressure_values.push_back(hydroelastic_modulus * (1 - v.norm()) / depth);
  }

  // std::cout << fmt::format("p_max: {}\n",
  // (*std::max_element(pressure_values.begin(), pressure_values.end())));

  return VolumeMeshFieldLinear<T, T>(std::move(pressure_values), mesh_C, true);
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS((
    &MakeEmbeddedPressureField<T>
))

}  // namespace internal
}  // namespace geometry
}  // namespace drake