#include "drake/geometry/proximity/bvh_to_vtk.h"

#include <fstream>
#include <iostream>
#include <vector>

#include <fmt/format.h>

#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;
using math::RigidTransformd;

void WriteVtkHeader(std::ofstream& out, const std::string& title) {
  out << "# vtk DataFile Version 3.0\n";
  out << title << std::endl;
  out << "ASCII\n";
  // An extra blank line makes the file more human readable.
  out << std::endl;
}

// Return the height of the `node`.
// depth = number of edges to the root node. root.depth = 0
// height = largest number of edges to a leaf. leaf.height = 0
template <typename MeshType>
int GetAllNodes(const BvNode<Obb, MeshType>& node, int depth,
                std::vector<const BvNode<Obb, MeshType>*>* all_nodes,
                std::vector<int>* all_depths, std::vector<int>* all_heights) {
  all_nodes->push_back(&node);
  all_depths->push_back(depth);
  // Remember the node index and initialize its entry to 0.
  int node_index = all_heights->size();
  all_heights->push_back(0);
  if (node.is_leaf()) {
    return 0;
  }
  int max_child_height = std::max(
      GetAllNodes(node.left(), depth + 1, all_nodes, all_depths, all_heights),
      GetAllNodes(node.right(), depth + 1, all_nodes, all_depths, all_heights));

  (*all_heights)[node_index] = max_child_height + 1;
  return (*all_heights)[node_index];
}

void WriteVtkUnstructuredGridFromObb(std::ofstream& out, const Obb& box_B) {
  std::vector<Vector3d> vertices;
  const Vector3d upper_B = box_B.half_width();
  const Vector3d lower_B = -box_B.half_width();
  const double x[2] = {lower_B.x(), upper_B.x()};
  const double y[2] = {lower_B.y(), upper_B.y()};
  const double z[2] = {lower_B.z(), upper_B.z()};
  // Order 8 box vertices (vertices of a hexahedral element) according to VTK.
  const Vector3d p_BVs[8] = {
      Vector3d(x[0], y[0], z[0]), Vector3d(x[1], y[0], z[0]),
      Vector3d(x[1], y[1], z[0]), Vector3d(x[0], y[1], z[0]),
      Vector3d(x[0], y[0], z[1]), Vector3d(x[1], y[0], z[1]),
      Vector3d(x[1], y[1], z[1]), Vector3d(x[0], y[1], z[1])};
  const RigidTransformd& X_MB = box_B.pose();
  for (const Vector3d& p_BV : p_BVs) {
    vertices.push_back(X_MB * p_BV);
  }

  out << "DATASET UNSTRUCTURED_GRID\n";
  out << "POINTS " << vertices.size() << " double\n";
  for (const auto& vertex : vertices) {
    out << fmt::format("{:12.8f} {:12.8f} {:12.8f}\n", vertex[0], vertex[1],
                       vertex[2]);
  }
  out << std::endl;

  const int num_elements = 1;
  constexpr int num_vertices_per_element = 8;
  const int num_integers = num_elements * (num_vertices_per_element + 1);
  out << "CELLS " << num_elements << " " << num_integers << std::endl;
  out << fmt::format("{}", num_vertices_per_element);
  out << fmt::format(" {:6d}", 0);
  out << fmt::format(" {:6d}", 1);
  out << fmt::format(" {:6d}", 2);
  out << fmt::format(" {:6d}", 3);
  out << fmt::format(" {:6d}", 4);
  out << fmt::format(" {:6d}", 5);
  out << fmt::format(" {:6d}", 6);
  out << fmt::format(" {:6d}", 7);
  out << std::endl;
  out << std::endl;

  constexpr int kVtkCellTypeHexahedron = 12;
  out << "CELL_TYPES " << num_elements << std::endl;
  out << fmt::format("{}\n", kVtkCellTypeHexahedron);
  out << std::endl;
}

// Write each BVH node as an independent hexahedral element. Vertices of
// hexahedral elements are not shared.
template <typename MeshType>
void WriteVtkUnstructuredGrid(
    std::ofstream& out,
    const std::vector<const BvNode<Obb, MeshType>*>& bvh_nodes) {
  std::vector<Vector3d> vertices;
  for (const BvNode<Obb, MeshType>* node : bvh_nodes) {
    const Obb& box_B = node->bv();
    const Vector3d upper_B = box_B.half_width();
    const Vector3d lower_B = -box_B.half_width();
    const double x[2] = {lower_B.x(), upper_B.x()};
    const double y[2] = {lower_B.y(), upper_B.y()};
    const double z[2] = {lower_B.z(), upper_B.z()};
    // Order 8 box vertices (vertices of a hexahedral element) according to VTK.
    const Vector3d p_BVs[8] = {
        Vector3d(x[0], y[0], z[0]), Vector3d(x[1], y[0], z[0]),
        Vector3d(x[1], y[1], z[0]), Vector3d(x[0], y[1], z[0]),
        Vector3d(x[0], y[0], z[1]), Vector3d(x[1], y[0], z[1]),
        Vector3d(x[1], y[1], z[1]), Vector3d(x[0], y[1], z[1])};
    const RigidTransformd& X_MB = box_B.pose();
    for (const Vector3d& p_BV : p_BVs) {
      vertices.push_back(X_MB * p_BV);
    }
  }

  out << "DATASET UNSTRUCTURED_GRID\n";
  out << "POINTS " << vertices.size() << " double\n";
  for (const auto& vertex : vertices) {
    out << fmt::format("{:12.8f} {:12.8f} {:12.8f}\n", vertex[0], vertex[1],
                       vertex[2]);
  }
  out << std::endl;

  const int num_elements = bvh_nodes.size();
  constexpr int num_vertices_per_element = 8;
  const int num_integers = num_elements * (num_vertices_per_element + 1);
  out << "CELLS " << num_elements << " " << num_integers << std::endl;
  for (int i = 0; i < num_elements; ++i) {
    out << fmt::format("{}", num_vertices_per_element);
    out << fmt::format(" {:6d}", 8 * i);
    out << fmt::format(" {:6d}", 8 * i + 1);
    out << fmt::format(" {:6d}", 8 * i + 2);
    out << fmt::format(" {:6d}", 8 * i + 3);
    out << fmt::format(" {:6d}", 8 * i + 4);
    out << fmt::format(" {:6d}", 8 * i + 5);
    out << fmt::format(" {:6d}", 8 * i + 6);
    out << fmt::format(" {:6d}", 8 * i + 7);
    out << std::endl;
  }
  out << std::endl;

  constexpr int kVtkCellTypeHexahedron = 12;
  out << "CELL_TYPES " << num_elements << std::endl;
  for (int i = 0; i < num_elements; ++i) {
    out << fmt::format("{}\n", kVtkCellTypeHexahedron);
  }
  out << std::endl;
}

void WriteVtkCellDataDepthHeightSerialIndex(std::ofstream& out,
                                            const std::vector<int>& depths,
                                            const std::vector<int>& heights) {
  const int num_nodes = depths.size();
  out << fmt::format("CELL_DATA {}\n", num_nodes);

  out << fmt::format("SCALARS BVH_node_depth int 1\n");
  out << "LOOKUP_TABLE default\n";
  for (const auto depth : depths) {
    out << fmt::format("{:6d}\n", depth);
  }
  out << std::endl;

  out << fmt::format("SCALARS BVH_node_height int 1\n");
  out << "LOOKUP_TABLE default\n";
  for (const auto height : heights) {
    out << fmt::format("{:6d}\n", height);
  }
  out << std::endl;

  out << fmt::format("SCALARS BVH_node_serialized_index int 1\n");
  out << "LOOKUP_TABLE default\n";
  for (int i = 0; i < num_nodes; ++i) {
    out << fmt::format("{:6d}\n", i);
  }
  out << std::endl;
}

template <typename MeshType>
void WriteToVtk(const std::string& file_name, const Bvh<Obb, MeshType>& bvh,
                const std::string& title) {
  std::ofstream file(file_name);
  if (file.fail()) {
    throw std::runtime_error(fmt::format("Cannot create file: {}.", file_name));
  }
  std::vector<const BvNode<Obb, MeshType>*> all_nodes;
  // depth = number of edges to the root node. root.depth = 0
  std::vector<int> all_depths;
  // height = largest number of edges to a leaf. leaf.height = 0
  std::vector<int> all_heights;
  GetAllNodes(bvh.root_node(), 0, &all_nodes, &all_depths, &all_heights);

  WriteVtkHeader(file, title);
  WriteVtkUnstructuredGrid(file, all_nodes);
  WriteVtkCellDataDepthHeightSerialIndex(file, all_depths, all_heights);
  file.close();
}

}  // namespace

void WriteBVHToVtk(const std::string& file_name,
                   const Bvh<Obb, VolumeMesh<double>>& bvh,
                   const std::string& title) {
  WriteToVtk(file_name, bvh, title);
}

void WriteBVHToVtk(const std::string& file_name,
                   const Bvh<Obb, TriangleSurfaceMesh<double>>& bvh,
                   const std::string& title) {
  WriteToVtk(file_name, bvh, title);
}

void WriteObbToVtk(const std::string& file_name, const Obb& obb,
                   const std::string& title) {
  std::ofstream file(file_name);
  if (file.fail()) {
    throw std::runtime_error(fmt::format("Cannot create file: {}.", file_name));
  }
  WriteVtkHeader(file, title);
  WriteVtkUnstructuredGridFromObb(file, obb);
  file.close();
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
