#pragma once

#include <string>

#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/obb.h"
#

namespace drake {
namespace geometry {
namespace internal {

void WriteBVHToVtk(const std::string& file_name,
                   const Bvh<Obb, VolumeMesh<double>>& bvh,
                   const std::string& title);

void WriteBVHToVtk(const std::string& file_name,
                   const Bvh<Obb, TriangleSurfaceMesh<double>>& bvh,
                   const std::string& title);

void WriteObbToVtk(const std::string& file_name,
                   const Obb& obb,
                   const std::string& title);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
