#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/scene_graph_inspector.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

// Load a model file and export all compliant mesh types to vtk files with
// pressure values.
int do_main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Parse the model give by the first command line argument and export all "
      "compliant mesh types to vtk files.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 2) {
    drake::log()->error("Missing input filename");
    return 1;
  }

  systems::DiagramBuilder<double> builder;
  multibody::MultibodyPlant<double>& plant =
      multibody::AddMultibodyPlantSceneGraph(&builder, 0.01);
  multibody::Parser parser(&plant);

  parser.AddModels(argv[1]);
  plant.Finalize();
  auto diagram = builder.Build();

  auto context = diagram->CreateDefaultContext();
  const systems::Context<double>& plant_context =
      plant.GetMyContextFromRoot(*context);

  const QueryObject<double> query_object =
      plant.get_geometry_query_input_port().Eval<QueryObject<double>>(
          plant_context);
  const SceneGraphInspector<double>& inspector = query_object.inspector();

  for (multibody::BodyIndex i{0}; i < plant.num_bodies(); ++i) {
    const std::vector<GeometryId>& ids =
        plant.GetCollisionGeometriesForBody(plant.get_body(i));
    for (const GeometryId id : ids) {
      const auto maybe_mesh_field =
          inspector.maybe_get_hydroelastic_mesh_field(id);
      const bool no_mesh_field = maybe_mesh_field.index() == 0;
      if (!no_mesh_field) {
        drake::log()->info(fmt::format("gid({})", id));
        std::string filename = fmt::format("{}_mesh.vtk", id);
        const drake::geometry::VolumeMeshFieldLinear<double, double>* pressure =
            std::get<
                const drake::geometry::VolumeMeshFieldLinear<double, double>*>(
                maybe_mesh_field);
        drake::geometry::internal::WriteVolumeMeshFieldLinearToVtk(
            filename, "pressure", *pressure, "pressure");
      }
    }
  }

  return 0;
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::geometry::internal::do_main(argc, argv);
}
