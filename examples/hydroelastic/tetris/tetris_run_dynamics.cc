#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 100,
              "Desired duration of the simulation in seconds.");
// See MultibodyPlantConfig for the valid strings of contact_model.
DEFINE_string(contact_model, "hydroelastic",
              "Contact model. Options are: 'point', 'hydroelastic', "
              "'hydroelastic_with_fallback'.");
// See MultibodyPlantConfig for the valid strings of contact surface
// representation.
DEFINE_string(contact_surface_representation, "polygon",
              "Contact-surface representation for hydroelastics. "
              "Options are: 'triangle' or 'polygon'. Default is 'polygon'.");

DEFINE_double(mbp_dt, 0.01,
              "The fixed time step period (in seconds) of discrete updates "
              "for the multibody plant modeled as a discrete system. "
              "Strictly positive.");

// Ball's initial spatial velocity.
DEFINE_double(vx, 0.0,
              "Sphere's initial translational velocity in the x-axis in m/s.");
DEFINE_double(vy, 0.0,
              "Sphere's initial translational velocity in the y-axis in m/s.");
DEFINE_double(vz, 0.0,
              "Sphere's initial translational velocity in the z-axis in m/s.");
DEFINE_double(wx, 0.0,
              "Sphere's initial angular velocity in the x-axis in degrees/s.");
DEFINE_double(wy, 0.0,
              "Sphere's initial angular velocity in the y-axis in degrees/s.");
DEFINE_double(wz, 0.0,
              "Sphere's initial angular velocity in the z-axis in degrees/s.");

DEFINE_double(x0, 0, "Ball's initial position in the x-axis.");
DEFINE_double(y0, 0, "Ball's initial position in the y-axis.");
DEFINE_double(z0, 4, "Ball's initial position in the z-axis.");

DEFINE_double(gz, 9.8,
              "Gravity force in the negative z-axis direction in Newtons.");

namespace drake {
namespace examples {
namespace tetris {
namespace {

using drake::math::RigidTransformd;
using drake::multibody::CoulombFriction;
using drake::multibody::SpatialVelocity;
using Eigen::Vector3d;

int do_main() {
  systems::DiagramBuilder<double> builder;

  multibody::MultibodyPlantConfig config;
  // We allow only discrete systems.
  DRAKE_DEMAND(FLAGS_mbp_dt > 0.0);
  config.time_step = FLAGS_mbp_dt;
  config.penetration_allowance = 0.001;
  config.contact_model = FLAGS_contact_model;
  config.contact_surface_representation = FLAGS_contact_surface_representation;
  config.discrete_contact_solver = "sap";
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  // Add the dinner plate.
  multibody::Parser parser(&plant);
  std::string sphere_filename =
      FindResourceOrThrow("drake/examples/hydroelastic/tetris/nonconvex.sdf");

  std::string floor_filename =
      FindResourceOrThrow("drake/examples/hydroelastic/tetris/floor.sdf");
  parser.AddModelFromFile(floor_filename);
  auto sphere1 = parser.AddModelFromFile(sphere_filename, "sphere1");
  auto sphere2 = parser.AddModelFromFile(sphere_filename, "sphere2");

  // Gravity acting in the -z direction.
  plant.mutable_gravity_field().set_gravity_vector(Vector3d{0, 0, -FLAGS_gz});

  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("Floor").body_frame(),
                   RigidTransformd{Vector3d{30, 30, -5}});

  plant.Finalize();

  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);
  ConnectContactResultsToDrakeVisualizer(&builder, plant, scene_graph,
                                         /* lcm */ nullptr);

  auto diagram = builder.Build();
  auto simulator = MakeSimulatorFromGflags(*diagram);

  // Set the Sphere's initial pose.
  systems::Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(&simulator->get_mutable_context());
  // plant.SetFreeBodyPose(&plant_context,
  //                       plant.GetBodyByName("sphere", sphere_1_mi),
  //                       math::RigidTransformd{Vector3d(0, 0, 0)});
  plant.SetFreeBodyPose(
      &plant_context, plant.GetBodyByName("sphere", sphere1),
      math::RigidTransformd{Vector3d(FLAGS_x0, FLAGS_y0, FLAGS_z0)});
  plant.SetFreeBodySpatialVelocity(
      &plant_context, plant.GetBodyByName("sphere", sphere1),
      SpatialVelocity<double>{
          M_PI / 180.0 * Vector3d(FLAGS_wx, FLAGS_wy, FLAGS_wz),
          Vector3d(FLAGS_vx, FLAGS_vy, FLAGS_vz)});

  plant.SetFreeBodyPose(
      &plant_context, plant.GetBodyByName("sphere", sphere2),
      math::RigidTransformd{Vector3d(FLAGS_x0, FLAGS_y0 - 2.2, FLAGS_z0 + 2.2)});
  plant.SetFreeBodySpatialVelocity(
      &plant_context, plant.GetBodyByName("sphere", sphere2),
      SpatialVelocity<double>{
          M_PI / 180.0 * Vector3d(FLAGS_wx, FLAGS_wy, FLAGS_wz),
          Vector3d(FLAGS_vx, FLAGS_vy, FLAGS_vz)});

  simulator->AdvanceTo(FLAGS_simulation_time);
  systems::PrintSimulatorStatistics(*simulator);
  return 0;
}

}  // namespace
}  // namespace tetris
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("");
  FLAGS_simulator_publish_every_time_step = true;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::tetris::do_main();
}
