#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gflags/gflags.h>

#include "drake/geometry/proximity/aabb.h"
#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obb.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/query_object.h"
#include "drake/geometry/scene_graph_inspector.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_string(output_dir, "./hydroelastic_meshes",
              "Directory to save VTK mesh files");
DEFINE_int32(mesh_res, 5, "Mesh resolution hint for the spatula. In (mm)");
DEFINE_bool(use_aabb, false, "Use AABBs instead of OBBs for bounding volumes");

namespace drake {

using Eigen::Vector3d;
using geometry::SceneGraph;
using geometry::SceneGraphInspector;
using math::RigidTransform;
using math::RollPitchYaw;
using multibody::MultibodyPlant;
using multibody::MultibodyPlantConfig;
using systems::Context;

// Helper function to extract all bounding volumes from a BVH tree
template <typename BvType, typename MeshType>
std::vector<BvType> ExtractBoundingVolumesFromBvh(
    const geometry::internal::Bvh<BvType, MeshType>& bvh) {
  std::vector<BvType> bounding_volumes;

  // Recursive function to traverse the BVH tree
  std::function<void(const geometry::internal::BvNode<BvType, MeshType>&)>
      traverse = [&](const geometry::internal::BvNode<BvType, MeshType>& node) {
        // Add the current node's bounding volume
        if (node.is_leaf()) {
          bounding_volumes.push_back(node.bv());
        }

        // If not a leaf node, traverse children
        if (!node.is_leaf()) {
          traverse(node.left());
          traverse(node.right());
        }
      };

  traverse(bvh.root_node());
  return bounding_volumes;
}

// Helper function to transform OBBs to world coordinates
std::vector<geometry::internal::Obb> TransformObbsToWorld(
    const std::vector<geometry::internal::Obb>& obbs,
    const math::RigidTransformd& X_WG) {
  std::vector<geometry::internal::Obb> transformed_obbs;
  transformed_obbs.reserve(obbs.size());

  for (const auto& obb : obbs) {
    // Create a new OBB with the transformed pose
    const math::RigidTransformd transformed_pose = X_WG * obb.pose();
    transformed_obbs.emplace_back(transformed_pose, obb.half_width());
  }

  return transformed_obbs;
}

// Helper function to create AABBs from a transformed mesh
std::vector<geometry::internal::Aabb> CreateAabbsFromTransformedMesh(
    const geometry::VolumeMesh<double>& transformed_mesh) {
  // Create BVH with AABBs on the transformed mesh
  const geometry::internal::Bvh<geometry::internal::Aabb,
                                geometry::VolumeMesh<double>>
      bvhMesh(transformed_mesh);

  // Extract all AABBs from the BVH
  return ExtractBoundingVolumesFromBvh(bvhMesh);
}

// Function to write OBBs to VTK format as wireframe boxes
void WriteObbsToVtk(const std::string& file_name,
                    const std::vector<geometry::internal::Obb>& obbs,
                    const std::string& title) {
  std::ofstream file(file_name);
  if (file.fail()) {
    throw std::runtime_error(fmt::format("Cannot create file: {}.", file_name));
  }

  // Write VTK header
  file << "# vtk DataFile Version 3.0\n";
  file << title << std::endl;
  file << "ASCII\n";
  file << std::endl;

  // Calculate total number of points and lines
  const int num_obbs = obbs.size();
  const int points_per_box = 8;  // 8 vertices per box
  const int lines_per_box = 12;  // 12 edges per box
  const int total_points = num_obbs * points_per_box;
  const int total_lines = num_obbs * lines_per_box;

  file << "DATASET POLYDATA\n";
  file << "POINTS " << total_points << " double\n";

  // Write all box vertices
  for (const auto& obb : obbs) {
    const Vector3d& center = obb.center();
    const Vector3d& half_width = obb.half_width();
    const math::RigidTransformd& pose = obb.pose();

    // Generate the 8 vertices of the box in local coordinates
    std::vector<Vector3d> local_vertices = {
        Vector3d(half_width.x(), half_width.y(), half_width.z()),
        Vector3d(-half_width.x(), half_width.y(), half_width.z()),
        Vector3d(-half_width.x(), -half_width.y(), half_width.z()),
        Vector3d(half_width.x(), -half_width.y(), half_width.z()),
        Vector3d(half_width.x(), half_width.y(), -half_width.z()),
        Vector3d(-half_width.x(), half_width.y(), -half_width.z()),
        Vector3d(-half_width.x(), -half_width.y(), -half_width.z()),
        Vector3d(half_width.x(), -half_width.y(), -half_width.z())};

    // Transform vertices to world coordinates (OBBs are already in world frame)
    for (const auto& local_vertex : local_vertices) {
      Vector3d world_vertex = pose * local_vertex;
      file << fmt::format("{:12.8f} {:12.8f} {:12.8f}\n", world_vertex.x(),
                          world_vertex.y(), world_vertex.z());
    }
  }

  file << std::endl;
  file << "LINES " << total_lines << " " << (total_lines * 3) << std::endl;

  // Write line connectivity for each box
  for (int box_idx = 0; box_idx < num_obbs; ++box_idx) {
    const int base_vertex = box_idx * points_per_box;

    // Define the 12 edges of a box (each edge connects 2 vertices)
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  // bottom face
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  // top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // vertical edges
    };

    for (const auto& edge : edges) {
      file << fmt::format("2 {:6d} {:6d}\n", base_vertex + edge.first,
                          base_vertex + edge.second);
    }
  }

  file.close();
}

// Function to write AABBs to VTK format as wireframe boxes
void WriteAabbsToVtk(const std::string& file_name,
                     const std::vector<geometry::internal::Aabb>& aabbs,
                     const std::string& title) {
  std::ofstream file(file_name);
  if (file.fail()) {
    throw std::runtime_error(fmt::format("Cannot create file: {}.", file_name));
  }

  // Write VTK header
  file << "# vtk DataFile Version 3.0\n";
  file << title << std::endl;
  file << "ASCII\n";
  file << std::endl;

  // Calculate total number of points and lines
  const int num_aabbs = aabbs.size();
  const int points_per_box = 8;  // 8 vertices per box
  const int lines_per_box = 12;  // 12 edges per box
  const int total_points = num_aabbs * points_per_box;
  const int total_lines = num_aabbs * lines_per_box;

  file << "DATASET POLYDATA\n";
  file << "POINTS " << total_points << " double\n";

  // Write all box vertices
  for (const auto& aabb : aabbs) {
    const Vector3d& center = aabb.center();
    const Vector3d& half_width = aabb.half_width();

    // Generate the 8 vertices of the box (AABBs are axis-aligned)
    std::vector<Vector3d> vertices = {
        Vector3d(center.x() + half_width.x(), center.y() + half_width.y(),
                 center.z() + half_width.z()),
        Vector3d(center.x() - half_width.x(), center.y() + half_width.y(),
                 center.z() + half_width.z()),
        Vector3d(center.x() - half_width.x(), center.y() - half_width.y(),
                 center.z() + half_width.z()),
        Vector3d(center.x() + half_width.x(), center.y() - half_width.y(),
                 center.z() + half_width.z()),
        Vector3d(center.x() + half_width.x(), center.y() + half_width.y(),
                 center.z() - half_width.z()),
        Vector3d(center.x() - half_width.x(), center.y() + half_width.y(),
                 center.z() - half_width.z()),
        Vector3d(center.x() - half_width.x(), center.y() - half_width.y(),
                 center.z() - half_width.z()),
        Vector3d(center.x() + half_width.x(), center.y() - half_width.y(),
                 center.z() - half_width.z())};

    // Write vertices directly (AABBs are in world coordinates)
    for (const auto& vertex : vertices) {
      file << fmt::format("{:12.8f} {:12.8f} {:12.8f}\n", vertex.x(),
                          vertex.y(), vertex.z());
    }
  }

  file << std::endl;
  file << "LINES " << total_lines << " " << (total_lines * 3) << std::endl;

  // Write line connectivity for each box
  for (int box_idx = 0; box_idx < num_aabbs; ++box_idx) {
    const int base_vertex = box_idx * points_per_box;

    // Define the 12 edges of a box (each edge connects 2 vertices)
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  // bottom face
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  // top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // vertical edges
    };

    for (const auto& edge : edges) {
      file << fmt::format("2 {:6d} {:6d}\n", base_vertex + edge.first,
                          base_vertex + edge.second);
    }
  }

  file.close();
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create output directory
  std::filesystem::create_directories(FLAGS_output_dir);

  // Build the system
  systems::DiagramBuilder<double> builder;

  // Create plant and scene graph
  MultibodyPlantConfig plant_config;
  plant_config.time_step = 0.0;  // Continuous time
  plant_config.contact_model = "hydroelastic";

  auto [plant, scene_graph] =
      multibody::AddMultibodyPlant(plant_config, &builder);

  // Parse the models
  multibody::Parser parser(&builder);
  parser.SetAutoRenaming(true);

  // Add gripper model
  parser.AddModelsFromUrl(
      "package://drake_models/wsg_50_description/sdf/"
      "schunk_wsg_50_hydro_bubble.sdf");

  // Add spatula model
  parser.AddModelsFromUrl(
      "package://drake/examples/hydroelastic/spatula_slip_control/"
      "spatula" +
      std::to_string(FLAGS_mesh_res) + ".sdf");

  // Get model instances
  std::vector<multibody::ModelInstanceIndex> gripper_instances;
  std::vector<multibody::ModelInstanceIndex> spatula_instances;

  // Find the model instances for grippers and spatulas
  for (int i = 0; i < plant.num_model_instances(); ++i) {
    const auto model_instance = multibody::ModelInstanceIndex(i);
    const std::string& model_name = plant.GetModelInstanceName(model_instance);

    if (model_name.find("Schunk_Gripper") == 0) {
      gripper_instances.push_back(model_instance);
    } else if (model_name.find("spatula") == 0) {
      spatula_instances.push_back(model_instance);
    }
  }

  // Sort them to ensure consistent ordering
  std::sort(gripper_instances.begin(), gripper_instances.end());
  std::sort(spatula_instances.begin(), spatula_instances.end());

  // Position gripper in world coordinates (same as demo with num_instances=1)
  const RigidTransform<double> X_WGripper = RigidTransform<double>(
      RollPitchYaw(0.0, -1.57, 0.0), Eigen::Vector3d(0.0, 0.0, 0.25));
  plant.WeldFrames(plant.world_frame(),
                   plant.GetFrameByName("gripper", gripper_instances[0]),
                   X_WGripper);

  plant.Finalize();

  // Create a simulator to get the context
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);
  simulator.Initialize();

  // Get the context to set spatula position
  Context<double>& mutable_root_context = simulator.get_mutable_context();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, &mutable_root_context);

  // Position spatula in world coordinates (same as demo with num_instances=1)
  const RigidTransform<double> X_WSpatula = RigidTransform<double>(
      RollPitchYaw(-0.4, 0.0, 1.57), Eigen::Vector3d(0.35, 0.0, 0.25));
  const auto& spatula_body =
      plant.GetBodyByName("spatula", spatula_instances[0]);
  plant.SetFreeBodyPose(&plant_context, spatula_body, X_WSpatula);

  // Get the scene graph inspector to access hydroelastic meshes
  const SceneGraphInspector<double>& inspector = scene_graph.model_inspector();

  // Get the query object to access world poses
  const auto& query_object =
      scene_graph.get_query_output_port().Eval<geometry::QueryObject<double>>(
          diagram->GetSubsystemContext(scene_graph, mutable_root_context));

  std::cout
      << "Exporting hydroelastic meshes to VTK format in world coordinates..."
      << std::endl;
  std::cout << "Output directory: " << FLAGS_output_dir << std::endl;

  int mesh_count = 0;

  // Iterate through all geometries in the scene graph
  for (const auto& geometry_id : inspector.GetAllGeometryIds()) {
    const std::string& geometry_name = inspector.GetName(geometry_id);

    // Try to get the hydroelastic mesh for this geometry
    auto maybe_mesh = inspector.maybe_get_hydroelastic_mesh(geometry_id);

    if (std::holds_alternative<std::monostate>(maybe_mesh)) {
      // No hydroelastic mesh for this geometry
      continue;
    }

    // Get the pose of this geometry in world coordinates
    const RigidTransform<double> X_WG =
        query_object.GetPoseInWorld(geometry_id);

    std::string filename;
    std::string title;

    if (std::holds_alternative<const geometry::VolumeMesh<double>*>(
            maybe_mesh)) {
      // This is a volume mesh (typically for soft/compliant geometries)
      const auto* volume_mesh =
          std::get<const geometry::VolumeMesh<double>*>(maybe_mesh);

      // Export volume mesh
      filename = fmt::format("{}/{}_volume_world.vtk", FLAGS_output_dir,
                             geometry_name);
      title =
          fmt::format("Volume Mesh for {} (World Coordinates)", geometry_name);

      std::cout << "  Exporting volume mesh: " << geometry_name << " ("
                << volume_mesh->num_elements() << " tetrahedra)" << std::endl;

      // Transform the mesh to world coordinates
      geometry::VolumeMesh<double> transformed_mesh = *volume_mesh;
      transformed_mesh.TransformVertices(X_WG);

      geometry::internal::WriteVolumeMeshToVtk(filename, transformed_mesh,
                                               title);

      // Export bounding volumes (OBBs or AABBs)
      if (FLAGS_use_aabb) {
        // Use AABBs - create them from the transformed mesh
        std::vector<geometry::internal::Aabb> aabbs =
            CreateAabbsFromTransformedMesh(transformed_mesh);

        std::string aabb_filename = fmt::format(
            "{}/{}_aabbs_world.vtk", FLAGS_output_dir, geometry_name);
        std::string aabb_title =
            fmt::format("AABBs for {} (World Coordinates)", geometry_name);

        std::cout << "  Exporting AABBs: " << geometry_name << " ("
                  << aabbs.size() << " bounding boxes)" << std::endl;

        WriteAabbsToVtk(aabb_filename, aabbs, aabb_title);
      } else {
        // Use OBBs
        const geometry::internal::Bvh<geometry::internal::Obb,
                                      geometry::VolumeMesh<double>>
            bvhMesh(*volume_mesh);

        // Extract all OBBs from the BVH
        std::vector<geometry::internal::Obb> obbs =
            ExtractBoundingVolumesFromBvh(bvhMesh);

        // Transform OBBs to world coordinates
        obbs = TransformObbsToWorld(obbs, X_WG);

        std::string obb_filename = fmt::format("{}/{}_obbs_world.vtk",
                                               FLAGS_output_dir, geometry_name);
        std::string obb_title =
            fmt::format("OBBs for {} (World Coordinates)", geometry_name);

        std::cout << "  Exporting OBBs: " << geometry_name << " ("
                  << obbs.size() << " bounding boxes)" << std::endl;

        WriteObbsToVtk(obb_filename, obbs, obb_title);
      }

      mesh_count += 1;
    }
  }

  std::cout << "\nExport completed successfully!" << std::endl;
  std::cout << "Total meshes exported: " << mesh_count << std::endl;
  std::cout
      << "  - Volume meshes (*_volume_world.vtk): Use 'Volume' representation"
      << std::endl;
  if (FLAGS_use_aabb) {
    std::cout << "  - AABB wireframes (*_aabbs_world.vtk): Use 'Wireframe' "
                 "representation"
              << std::endl;
  } else {
    std::cout << "  - OBB wireframes (*_obbs_world.vtk): Use 'Wireframe' "
                 "representation"
              << std::endl;
  }
  std::cout << "\nFiles saved in: " << FLAGS_output_dir << std::endl;

  return 0;
}

}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::main(argc, argv);
}
