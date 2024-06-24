#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/meshcat/contact_visualizer.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/trajectory_source.h"
#include "drake/visualization/visualization_config.h"
#include "drake/visualization/visualization_config_functions.h"

namespace drake {

using geometry::Meshcat;
using geometry::MeshcatVisualizer;
using geometry::MeshcatVisualizerParams;
using multibody::AddMultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::DiagramBuilder;
using systems::Simulator;
using systems::TrajectorySource;
using trajectories::PiecewisePolynomial;
using visualization::AddDefaultVisualization;

namespace examples {
namespace multibody {
namespace BarrettHand {
namespace {

DEFINE_double(target_realtime_rate, 1.0, "Desired rate relative to real time.");
DEFINE_double(simulation_time, 1.2,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 0.0002,
              "Time step for the plant. If 0, the plant is modeled as a "
              "continuous system.");
DEFINE_double(desired_velocity, 200, "Desired velocity of the motor in rad/s.");
DEFINE_bool(obstacles, false, "Add obstacles to the scene when true.");
DEFINE_bool(visualization, false, "Add Meshcat visualization to the diagram.");
DEFINE_bool(data_recording, false, "Record state and force data to file.");

int do_main() {
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = FLAGS_time_step;
  config.discrete_contact_approximation = "similar";
  config.stiction_tolerance = 1e-4;
  config.adjacent_bodies_collision_filters = false;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);
  Parser parser(&plant);
  auto model = parser.AddModels(FindResourceOrThrow(
      "drake/examples/multibody/BarrettHand/models/BarrettHand.sdf"));

  if (FLAGS_obstacles) {
    parser.AddModels(FindResourceOrThrow(
        "drake/examples/multibody/BarrettHand/models/obstacles.sdf"));
  }

  plant.Finalize();

  std::shared_ptr<Meshcat> meshcat;

  if (FLAGS_visualization) {
    meshcat = std::make_shared<Meshcat>();
    AddDefaultVisualization(&builder, meshcat);
  }

  Eigen::VectorXd breaks(3);
  breaks << 0, 0.15, FLAGS_simulation_time;
  Eigen::MatrixXd samples(1, 3);
  samples << 0, 0.15 * FLAGS_desired_velocity,
      0.15 * FLAGS_desired_velocity -
          (FLAGS_simulation_time - 0.15) * FLAGS_desired_velocity;
  auto motor_trajectory =
      PiecewisePolynomial<double>::FirstOrderHold(breaks, samples);

  auto motor_trajectory_source =
      builder.AddSystem<TrajectorySource>(motor_trajectory, 1);
  builder.Connect(motor_trajectory_source->get_output_port(),
                  plant.get_desired_state_input_port(model[0]));

  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();

  if (FLAGS_visualization) {
    meshcat->StartRecording();
  }

  simulator.AdvanceTo(FLAGS_simulation_time);

  if (FLAGS_visualization) {
    meshcat->StopRecording();
    meshcat->PublishRecording();
  }

  return 0;
}

}  // namespace
}  // namespace BarrettHand
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::BarrettHand::do_main();
}