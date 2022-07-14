#pragma once

#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {


/* Constructed a tetrahedralized regular grid the contains the bounding box of
   the input `mesh` for use with the "embedded pressure field". */
template <typename T>
VolumeMesh<T> MakeEmbeddedVolumeMesh(const Mesh& mesh, int subdivisions, double margin);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
