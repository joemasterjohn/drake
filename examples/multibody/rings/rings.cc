/** Rings example from IPC paper for benchmark against SOFA. */

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

DEFINE_int32(num_rings, 10, "Number of deformable rings.");
DEFINE_double(simulation_time, 60.0,
              "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 2.0e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e7, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 500, "Mass density of the deformable body [kg/m³].");
DEFINE_double(stiffness_damping, 0.05,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(mu, 0.0, "Friction coefficient");

using drake::geometry::AddCompliantHydroelasticProperties;
using drake::geometry::AddContactMaterial;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::geometry::VolumeElement;
using drake::geometry::VolumeMesh;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
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

void AddRing(DeformableModel<double>* model, const RigidTransformd& X_WR,
             DeformableBodyConfig<double> config, std::string source,
             std::string name, std::vector<DeformableBodyId>* body_ids,
             Vector4d rgba) {
  const std::string vtk = FindResourceOrThrow(source);
  auto mesh = std::make_unique<Mesh>(vtk, 1);
  auto instance =
      std::make_unique<GeometryInstance>(X_WR, std::move(mesh), name);
  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  const CoulombFriction<double> surface_friction(FLAGS_mu, FLAGS_mu);
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  instance->set_proximity_properties(deformable_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse", rgba);
  instance->set_illustration_properties(illustration_props);
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
  AddCompliantHydroelasticProperties(1.0, 1e6, &rigid_proximity_props);
  const CoulombFriction<double> surface_friction(FLAGS_mu, FLAGS_mu);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(1.0, 1.0, 1.0, 1.0));
  /* Set up the bases. */
  std::string dir = "drake/examples/multibody/rings/torus.vtk";
  std::string dir_visual = "drake/examples/multibody/rings/torus.obj";
  const std::string vtk = FindResourceOrThrow(dir);
  const std::string obj = FindResourceOrThrow(dir_visual);
  Mesh base(vtk, 1.);
  Mesh base_visual(obj, 1.);
  const RigidTransformd X_WG(RollPitchYawd(M_PI_2, M_PI_2, 0),
                             Eigen::Vector3d{0, 0, 0});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, base, "base",
                                  rigid_proximity_props);
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, base_visual, "base",
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
  const double kGap = -0.62;
  for (int i = 0; i < FLAGS_num_rings; ++i) {
    Vector4d rgba =
        (i % 2) ? Vector4d(0.333, 0.333, 0.333, 0.9) : Vector4d(1, 0, 0, 0.9);
    RigidTransformd X_WR(RollPitchYawd((i % 2) * M_PI_2, M_PI_2, 0),
                         Vector3d(0, 0, kGap * (i + 1)));
    AddRing(owned_deformable_model.get(), X_WR, deformable_config, dir,
            "ring_" + std::to_string(i), &body_ids, rgba);
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
