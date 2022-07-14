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

  auto swapX = [](std::vector<int>& v) {
    std::swap(v[0], v[1]);
    std::swap(v[2], v[3]);
    std::swap(v[4], v[5]);
    std::swap(v[6], v[7]);
  };

  auto swapY = [](std::vector<int>& v) {
    std::swap(v[0], v[3]);
    std::swap(v[1], v[2]);
    std::swap(v[4], v[7]);
    std::swap(v[5], v[6]);
  };

  auto swapZ = [](std::vector<int>& v) {
    std::swap(v[0], v[4]);
    std::swap(v[1], v[5]);
    std::swap(v[2], v[6]);
    std::swap(v[3], v[7]);
  };

  auto permute = [&subdivisions, &swapX, &swapY, &swapZ](int i, int j, int k, std::vector<int>& v) {
    if(i >= subdivisions/2) {
      swapX(v);
    }
    if(j >= subdivisions/2) {
      swapY(v);
    }
    if(k >= subdivisions/2) {
      swapZ(v);
    }
  };

  auto volume = [&volume_mesh_vertices](int v0, int v1, int v2, int v3) {
    const Vector3<T>& a = volume_mesh_vertices[v0];
    const Vector3<T>& b = volume_mesh_vertices[v1];
    const Vector3<T>& c = volume_mesh_vertices[v2];
    const Vector3<T>& d = volume_mesh_vertices[v3];
    return (d - a).dot((b - a).cross(c - a)) / T(6.0);
  };

  for (int i = 0; i < subdivisions; ++i) {
    for (int j = 0; j < subdivisions; ++j) {
      for (int k = 0; k < subdivisions; ++k) {

        std::vector<int> v;

        v.push_back(global_index(i, j, k));
        v.push_back(global_index(i + 1, j, k));
        v.push_back(global_index(i + 1, j + 1, k));
        v.push_back(global_index(i, j + 1, k));
        v.push_back(global_index(i, j, k + 1));
        v.push_back(global_index(i + 1, j, k + 1));
        v.push_back(global_index(i + 1, j + 1, k + 1));
        v.push_back(global_index(i, j + 1, k + 1));

        permute(i, j, k, v);

        if (volume(v[0], v[1], v[2], v[6]) > 0) {
          volume_mesh_elements.emplace_back(v[0], v[1], v[2], v[6]);
        } else {
          volume_mesh_elements.emplace_back(v[0], v[1], v[6], v[2]);
        }
        if (volume(v[0], v[2], v[3], v[6]) > 0) {
          volume_mesh_elements.emplace_back(v[0], v[2], v[3], v[6]);
        } else {
          volume_mesh_elements.emplace_back(v[0], v[2], v[6], v[3]);
        }
        if (volume(v[0], v[5], v[1], v[6]) > 0) {
          volume_mesh_elements.emplace_back(v[0], v[5], v[1], v[6]);
        } else {
          volume_mesh_elements.emplace_back(v[0], v[5], v[6], v[1]);
        }
        if (volume(v[0], v[4], v[5], v[6]) > 0) {
          volume_mesh_elements.emplace_back(v[0], v[4], v[5], v[6]);
        } else {
          volume_mesh_elements.emplace_back(v[0], v[4], v[6], v[5]);
        }
        if (volume(v[0], v[7], v[4], v[6]) > 0) {
          volume_mesh_elements.emplace_back(v[0], v[7], v[4], v[6]);
        } else {
          volume_mesh_elements.emplace_back(v[0], v[7], v[6], v[4]);
        }
        if (volume(v[0], v[3], v[7], v[6]) > 0) {
          volume_mesh_elements.emplace_back(v[0], v[3], v[7], v[6]);
        } else {
          volume_mesh_elements.emplace_back(v[0], v[3], v[6], v[7]);
        }
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
