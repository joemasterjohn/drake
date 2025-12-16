#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/systems/analysis/convex_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(accuracy, 1e-3, "Integrator accuracy .");
DEFINE_double(max_step_size, 0.1, "Maximum time step size.");
DEFINE_double(E, 1e8, "Default hydroelastic modulus [Pa].");
DEFINE_double(margin, 1e-4, "Default margin [m].");
DEFINE_double(d, 0.0, "Default H&C dissipation [s/m].");
DEFINE_double(resolution, 0.01, "Default resolution hint [m].");
DEFINE_double(t, 1.0, "Default simulation time [s].");
DEFINE_double(z, 1.0, "Default z value.");
DEFINE_double(vz, 0.0, "Default vz value.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  drake::systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      drake::multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);

  drake::geometry::SceneGraphConfig sg_config;
  sg_config.default_proximity_properties.compliance_type = "compliant";
  sg_config.default_proximity_properties.hydroelastic_modulus = FLAGS_E;
  sg_config.default_proximity_properties.hunt_crossley_dissipation = FLAGS_d;
  sg_config.default_proximity_properties.margin = FLAGS_margin;
  sg_config.default_proximity_properties.resolution_hint = FLAGS_resolution;
  scene_graph.set_config(sg_config);

  drake::multibody::Parser parser(&plant);
  const std::string model_path = drake::FindResourceOrThrow(
      "drake/examples/integrators/ball_on_table.xml");
  parser.AddModels(model_path);
  plant.Finalize();

  auto diagram = builder.Build();

  auto context = diagram->CreateDefaultContext();
  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, context.get());

  Eigen::VectorXd initial_state(13);
  initial_state << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, FLAGS_z, 0.0, 0.0, 0.0, 0.0,
      0.0, FLAGS_vz;
  plant.SetPositionsAndVelocities(&plant_context, initial_state);
  auto simulator = drake::systems::internal::MakeSimulatorFromGflags(
      *diagram, std::move(context));

  drake::systems::SimulatorConfig config;
  config.integration_scheme = "convex";
  config.use_error_control = true;
  config.max_step_size = 0.1;
  config.publish_every_time_step = true;
  config.target_realtime_rate = 0.0;

  drake::systems::ApplySimulatorConfig(config, simulator.get());
  auto& ci = dynamic_cast<drake::systems::ConvexIntegrator<double>&>(
      simulator->get_mutable_integrator());
  ci.set_plant(&plant);

  simulator->Initialize();
  simulator->AdvanceTo(1.0);

  return 0;
}