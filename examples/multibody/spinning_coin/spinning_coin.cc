#include <iostream>
#include <fstream>

#include <fmt/format.h>
#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

using multibody::ContactModel;
using multibody::MultibodyPlant;
using multibody::Parser;
using multibody::SpatialVelocity;
using systems::Context;
using systems::DiagramBuilder;
using systems::Simulator;

namespace examples {
namespace multibody {
namespace spinning_coin {
namespace {

DEFINE_double(simulation_time, 5.0, "Duration of the simulation in seconds.");

DEFINE_double(mbt_dt, 1e-3,
              "Discrete time step. Defaults to 1e-3 for a discrete system.");

DEFINE_double(vy, 1, "y translational velocity");
DEFINE_double(wz, 1, "z rotational velocity");

DEFINE_string(output_filename, "spinning_coin_output", "Data output filename.");

DEFINE_bool(point_contact, false, "Select point contact mode.");
DEFINE_bool(low_res_contact_surface, false, "Select low res polygonal contact surfaces");

int do_main() {
  // Build a generic MultibodyPlant and SceneGraph.
  DiagramBuilder<double> builder;

  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(
      &builder, std::make_unique<MultibodyPlant<double>>(FLAGS_mbt_dt));

  // Make and add the coin model from an SDF model.
  const std::string coin_relative_name =
      "drake/examples/multibody/spinning_coin/models/coin.sdf";
  const std::string floor_relative_name =
      "drake/examples/multibody/spinning_coin/models/floor.sdf";
  const std::string coin_full_name = FindResourceOrThrow(coin_relative_name);
  const std::string floor_full_name = FindResourceOrThrow(floor_relative_name);

  Parser parser(&plant);
  parser.AddModelFromFile(floor_full_name);
  parser.AddModelFromFile(coin_full_name);

  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("floor").body_frame(),
                   math::RigidTransformd(Vector3d(25, 25, -1)));

  if(FLAGS_point_contact) {
    plant.set_contact_model(ContactModel::kPoint);
  } else {
    plant.set_contact_model(ContactModel::kHydroelastic);
  }

  plant.set_low_resolution_contact_surface(FLAGS_low_res_contact_surface);

  plant.Finalize();

  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);
  ConnectContactResultsToDrakeVisualizer(&builder, plant);

  auto diagram = builder.Build();

  // Create a context for the diagram and extract the context for the
  // model.
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());

  // Set the initial velocities of the coin.
  math::RigidTransformd X_WC(Vector3d(0.0, 0.0, 0.000875));
  plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("coin"), X_WC);


  const SpatialVelocity<double> V_WC_initial(Vector3d(0, 0, FLAGS_wz),
                                             Vector3d(0, FLAGS_vy, 0));
  plant.SetFreeBodySpatialVelocity(&plant_context, plant.GetBodyByName("coin"),
                                   V_WC_initial);

  // Create a simulator and run the simulation.
  std::unique_ptr<Simulator<double>> simulator =
      MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  std::ofstream output_file;
  output_file.open(FLAGS_output_filename);

  // Create a monitor for the ratio of angular to translational velocity
  simulator->set_monitor(
      [&plant, &output_file](const systems::Context<double>& root_context) {
        const SpatialVelocity<double> V_WC =
            plant.GetBodyByName("coin").EvalSpatialVelocityInWorld(
                plant.GetMyContextFromRoot(root_context));

        const double v = V_WC.translational().norm();
        const double w = V_WC.rotational().norm();

        const double curr_time = root_context.get_time();

        if (v > 1e-7 || w > 1e-7) {
          const double ratio = v / w;
          output_file << fmt::format("{} {} {} {}\n", curr_time, ratio, v, w);
        }

        return systems::EventStatus::Succeeded();
      });

  simulator->AdvanceTo(FLAGS_simulation_time);

  // Print some useful statistics.
  PrintSimulatorStatistics(*simulator);

  output_file.close();

  return 0;
}  // namespace

}  // namespace
}  // namespace spinning_coin
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("");

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::spinning_coin::do_main();
}
