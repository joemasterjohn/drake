#include <cmath>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/make_embedded_mesh.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;

namespace {}  // namespace

template <typename T>
VolumeMesh<T> MakeEmbeddedVolumeMesh(const Mesh& mesh, int subdivisions,
                                     double margin) {
  const TriangleSurfaceMesh<double> surface_mesh =
      ReadObjToTriangleSurfaceMesh(mesh.filename(), mesh.scale());

  std::vector<Vector3<T>> volume_mesh_vertices;
  volume_mesh_vertices.reserve((subdivisions + 1) * (subdivisions + 1) *
                               (subdivisions + 1));

  auto [center, size] = surface_mesh.CalcBoundingBox();
  double x_min = center[0] - 0.5 * size[0] - margin;
  double x_max = center[0] + 0.5 * size[0] + margin;
  double y_min = center[1] - 0.5 * size[1] - margin;
  double y_max = center[1] + 0.5 * size[1] + margin;
  double z_min = center[2] - 0.5 * size[2] - margin;
  double z_max = center[2] + 0.5 * size[2] + margin;

  for (int i = 0; i <= subdivisions; ++i) {
    double p = 1.0 * i / subdivisions;
    T x = (1 - p) * x_min + (p)*x_max;

    for (int j = 0; j <= subdivisions; ++j) {
      double q = 1.0 * j / subdivisions;
      T y = (1 - q) * y_min + (q)*y_max;

      for (int k = 0; k <= subdivisions; ++k) {
        double r = 1.0 * k / subdivisions;
        T z = (1 - r) * z_min + (r)*z_max;

        volume_mesh_vertices.emplace_back(x, y, z);

      }
    }
  }

  std::vector<VolumeElement> volume_mesh_elements;
  volume_mesh_elements.reserve(6 * volume_mesh_vertices.size());

  auto global_index = [subdivisions](int i, int j, int k) {
    return (i * (subdivisions+1) + j) * (subdivisions+1) + k;
  };

  for (int i = 0; i < subdivisions; ++i) {
    for (int j = 0; j < subdivisions; ++j) {
      for (int k = 0; k < subdivisions; ++k) {
        int v0 = global_index(i, j, k);
        int v1 = global_index(i + 1, j, k);
        int v2 = global_index(i + 1, j + 1, k);
        int v3 = global_index(i, j + 1, k);
        int v4 = global_index(i, j, k + 1);
        int v5 = global_index(i + 1, j, k + 1);
        int v6 = global_index(i + 1, j + 1, k + 1);
        int v7 = global_index(i, j + 1, k + 1);

        volume_mesh_elements.emplace_back(v0, v1, v2, v6);
        volume_mesh_elements.emplace_back(v0, v1, v5, v6);
        volume_mesh_elements.emplace_back(v0, v5, v6, v4);
        volume_mesh_elements.emplace_back(v0, v2, v3, v6);
        volume_mesh_elements.emplace_back(v0, v6, v3, v7);
        volume_mesh_elements.emplace_back(v0, v4, v6, v7);
      }
    }
  }

  return {std::move(volume_mesh_elements), std::move(volume_mesh_vertices)};
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&MakeEmbeddedVolumeMesh<T>))

}  // namespace internal
}  // namespace geometry
}  // namespace drake
