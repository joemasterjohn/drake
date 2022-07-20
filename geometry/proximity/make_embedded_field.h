#pragma once

#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/proximity/calc_distance_to_surface_mesh.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

/*
 @pre This pressure field generation is highly dependent on the implementation
 of the convex mesh in MakeConvexVolumeMesh(). In particular it assumes the
 last vertex in its vertex list is the sole internal vertex and the rest of
 the vertices are boundary. If the implementation in MakeConvexVolumeMesh()
 were to change, this pressure field generation would also need to change.

 @param[in] mesh_C       A pointer to a tetrahedral mesh of a convex shape.
                         It is aliased in the returned pressure field and
                         must remain alive as long as the field.
 @param[in] hydroelastic_modulus  Scale extent to pressure.
 @return                 The pressure field defined on the tetrahedral mesh.
 @pre                    `hydroelastic_modulus` is strictly positive.
                         `mesh_C` is non-null.
 @tparam T               The scalar type for representing the mesh vertex
                         positions and the pressure value.
 */
template <typename T>
VolumeMeshFieldLinear<T, T> MakeEmbeddedPressureField(const Mesh& mesh,
    const VolumeMesh<T>* mesh_C, const T& hydroelastic_modulus, const T& depth) {
  DRAKE_DEMAND(hydroelastic_modulus > T(0));
  DRAKE_DEMAND(mesh_C != nullptr);

  unused(hydroelastic_modulus);
  unused(depth);

  const TriangleSurfaceMesh<double> surface_mesh =
      ReadObjToTriangleSurfaceMesh(mesh.filename(), mesh.scale());

  std::vector<T> pressure_values;
  pressure_values.reserve(mesh_C->num_vertices());

  for(auto v : mesh_C->vertices()) {
    pressure_values.push_back(CalcDistanceToSurfaceMesh(v, surface_mesh));
  }

  return VolumeMeshFieldLinear<T, T>(std::move(pressure_values), mesh_C);
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
