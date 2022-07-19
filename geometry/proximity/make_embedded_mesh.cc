#include <cmath>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/make_convex_mesh.h"
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

  auto [v_min, v_max] = surface_mesh.CalcBoundingBox();

  for (int i = 0; i <= subdivisions + 1; ++i) {
    double p = 1.0 * i / (subdivisions + 1);
    T x = (1 - p) * v_min[0] + (p)*v_max[0];

    for (int j = 0; j <= subdivisions + 1; ++j) {
      double q = 1.0 * j / (subdivisions + 1);
      T y = (1 - q) * v_min[1] + (q)*v_max[1];

      for (int k = 0; k <= subdivisions + 1; ++k) {
        double r = 1.0 * k / (subdivisions + 1);
        T z = (1 - r) * v_min[2] + (r)*v_max[2];

        volume_mesh_vertices.emplace_back(x, y, z);
      }
    }
  }

  std::vector<VolumeElement> volume_mesh_elements;
  volume_mesh_elements.reserve(6 * volume_mesh_vertices.size());

  auto global_index = [subdivisions](int i, int j, int k) {
    return (i * (subdivisions + 1) + j) * (subdivisions + 1) + k
  };

  for (int i = 0; i < subdivisions + 1; ++i) {
    for (int j = 0; j < subdivisions + 1; ++j) {
      for (int k = 0; k < subdivisions + 1; ++k) {
        int v0 = global_index(i, j, k);
        int v1 = global_index(i + 1, j, k);
        int v2 = global_index(i + 1, j + 1, k);
        int v3 = global_index(i, j + 1, k);
        int v4 = global_index(i, j, k + 1);
        int v5 = global_index(i + 1, j, k + 1);
        int v6 = global_index(i + 1, j + 1, k + 1);
        int v7 = global_index(i, j + 1, k + 1);

        volume_mesh_elements.emplace_back(v0, v1, v2, v5);
        volume_mesh_elements.emplace_back(v0, v1, v2, v6);
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
    (&MakeConvexVolumeMesh<T>))

}  // namespace internal
}  // namespace geometry
}  // namespace drake
