/** Masonary arch example from IPC paper.
 Note that in this example we use a larger beta (beta = 10.0) because the
 deformable bodies very large in scale and we can allow more penetration. Using
 the analysis in the sap paper, the penetration is on the order of
 β²*g*dt²/(4π²). With dt = 1e-2s and beta = 10, this is approximately 2.5e-3m,
 small compared to the 20m arch. Since the beta parameter is currently not
 exposed, we need to manually change this in sap_driver.cc. */

#include <iomanip>
#include <memory>
#include <sstream>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 600.0,
              "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.0, "Desired real time rate.");
DEFINE_double(time_step, 4.0e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 2e10, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.2, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 2300, "Mass density of the deformable body [kg/m³].");
DEFINE_double(stiffness_damping, 0.005,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(mu, 0.5, "Friction coefficient");
DEFINE_bool(with_gap, false,
            "Whether to leave a small gap among blocks of the arch at "
            "initialization. When run without gap, we lose the initial "
            "transiant and can take larger time steps (0.04s vs. 0.01s).");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::geometry::VolumeElement;
using drake::geometry::VolumeMesh;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace multibody {
namespace deformable_box {
namespace {

void AddArch(DeformableModel<double>* model,
             DeformableBodyConfig<double> config, std::string source,
             std::vector<DeformableBodyId>* body_ids) {
  const std::string vtk = FindResourceOrThrow(source);
  auto mesh = std::make_unique<Mesh>(vtk, 1.0);
  auto instance = std::make_unique<GeometryInstance>(
      RigidTransformd::Identity(), std::move(mesh), source);
  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  const CoulombFriction<double> surface_friction(FLAGS_mu, FLAGS_mu);
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  instance->set_proximity_properties(deformable_proximity_props);
  const auto body_id = model->RegisterDeformableBody(
      std::move(instance), config, /* unused resolution hint*/ 1.0);
  body_ids->emplace_back(body_id);
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  ProximityProperties rigid_proximity_props;
  const CoulombFriction<double> surface_friction(FLAGS_mu, FLAGS_mu);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                    geometry::internal::kRezHint, 1.0);
  /* Set up the bases. */
  Box base{10, 10, 10};
  const RigidTransformd X_WB1(Eigen::Vector3d{-36, 0, -6.4});
  const RigidTransformd X_WB2(Eigen::Vector3d{36, 0, -6.4});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WB1, base, "base1",
                                  rigid_proximity_props);
  plant.RegisterCollisionGeometry(plant.world_body(), X_WB2, base, "base2",
                                  rigid_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.7, 0.5, 0.4, 0.8));
  plant.RegisterVisualGeometry(plant.world_body(), X_WB1, base, "base1_visual",
                               illustration_props);
  plant.RegisterVisualGeometry(plant.world_body(), X_WB2, base, "base2_visual",
                               illustration_props);
  Box ground{100, 100, 40};
  const RigidTransformd X_WG(Eigen::Vector3d{0.0, 0, -31.4});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground, "ground",
                                  rigid_proximity_props);
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground, "ground",
                               illustration_props);

  /* Set up deformable blocks. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);

  DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(FLAGS_E);
  deformable_config.set_poissons_ratio(FLAGS_nu);
  deformable_config.set_mass_density(FLAGS_density);
  deformable_config.set_stiffness_damping_coefficient(FLAGS_stiffness_damping);

  std::vector<DeformableBodyId> body_ids;
  std::string dir = "drake/examples/multibody/arch/";
  for (int i = 1; i <= 25; ++i) {
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << i;
    std::string filename;
    if (FLAGS_with_gap) {
      filename = "arch_with_gap_" + ss.str() + ".vtk";
    } else {
      filename = "arch_without_gap_" + ss.str() + ".vtk";
    }
    AddArch(owned_deformable_model.get(), deformable_config, dir + filename,
            &body_ids);
  }

  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
  plant.AddPhysicalModel(std::move(owned_deformable_model));

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* It's essential to connect the vertex position port in DeformableModel to
   the source configuration port in SceneGraph when deformable bodies are
   present in the plant. */
  builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.AdvanceTo(FLAGS_simulation_time);

  /* Useful if we want to record the meshes in its configuration at the end of
   the last time step. */
  bool write_final_mesh = false;
  if (write_final_mesh) {
    int block = 1;
    for (auto body_id : body_ids) {
      const auto g_id = deformable_model->GetGeometryId(body_id);
      const VolumeMesh<double>* reference_mesh_ptr =
          scene_graph.model_inspector().GetReferenceMesh(g_id);
      DRAKE_DEMAND(reference_mesh_ptr != nullptr);
      std::vector<VolumeElement> elements = reference_mesh_ptr->tetrahedra();
      const int num_vertices = reference_mesh_ptr->num_vertices();
      std::vector<Vector3d> vertices;
      const auto state_index = deformable_model->GetDiscreteStateIndex(body_id);
      const Context<double>& plant_context =
          plant.GetMyContextFromRoot(simulator.get_context());
      const VectorXd& state =
          plant_context.get_discrete_state(state_index).value();
      DRAKE_DEMAND(state.size() == 9 * num_vertices);
      for (int i = 0; i < num_vertices; ++i) {
        vertices.emplace_back(state.segment<3>(3 * i));
      }
      VolumeMesh<double> final_mesh =
          VolumeMesh<double>(std::move(elements), std::move(vertices));

      std::stringstream ss;
      ss << std::setw(2) << std::setfill('0') << block;
      std::string file =
          "/home/xuchenhan/drake/examples/multibody/arch/final_mesh_" +
          ss.str() + ".vtk";
      geometry::internal::WriteVolumeMeshToVtk(file, final_mesh,
                                               "block_" + ss.str());
      ++block;
    }
  }

  return 0;
}

}  // namespace
}  // namespace deformable_box
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Launch meldis before running this example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::deformable_box::do_main();
}
