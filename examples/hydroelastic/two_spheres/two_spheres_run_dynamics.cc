#include <iostream>
#include <memory>
#include <chrono>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 1.4,
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
DEFINE_double(vx, 0.7,
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

DEFINE_double(roll, 0.9, "roll");
DEFINE_double(pitch, 0.4, "pitch");
DEFINE_double(yaw, 0.0, "yaw");

DEFINE_double(x0, 0, "Ball's initial position in the x-axis.");
DEFINE_double(y0, 0, "Ball's initial position in the y-axis.");
DEFINE_double(z0, 0.05, "Ball's initial position in the z-axis.");

DEFINE_double(gz, 9.8,
              "Gravity force in the negative z-axis direction in Newtons.");

DEFINE_bool(visualize, true, "Connects to visualizer when true.");

DEFINE_string(filename, "models/embedded/pepper5.sdf", "model filename");
namespace drake {
namespace examples {
namespace two_spheres {
namespace {

using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::multibody::ContactResults;
using drake::multibody::CoulombFriction;
using drake::multibody::SpatialVelocity;
using drake::multibody::PrismaticJoint;
using drake::multibody::Body;
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

  multibody::Parser parser(&plant);
  std::string sphere_filename = FindResourceOrThrow(
      "drake/examples/hydroelastic/two_spheres/" + FLAGS_filename);

//   std::string sphere_internal_filename = FindResourceOrThrow(
//       "drake/examples/hydroelastic/two_spheres/pancake_convex.sdf");
  std::string floor_filename =
      FindResourceOrThrow("drake/examples/hydroelastic/two_spheres/models/floor.sdf");
  auto sphere_embedded_mi =
      parser.AddModelFromFile(sphere_filename, "sphere_embedded");
//   auto sphere_internal_mi =
//       parser.AddModelFromFile(sphere_internal_filename, "sphere_internal");
  auto floor_mi = parser.AddModelFromFile(floor_filename, "floor");

  // Gravity acting in the -z direction.
  plant.mutable_gravity_field().set_gravity_vector(Vector3d{0, 0, -FLAGS_gz});

  plant.WeldFrames(plant.world_frame(),
                   plant.GetBodyByName("Floor", floor_mi).body_frame(),
                   RigidTransformd{Vector3d{30, 30, -5}});

  // plant.AddJoint<PrismaticJoint>(
  //     "world_embedded", plant.world_body(), RigidTransformd{Vector3d{0, 0, FLAGS_z0}},
  //     plant.GetBodyByName("pancake_embedded", sphere_embedded_mi), {},
  //     Vector3d{0, 0, 1});
  // plant.AddJoint<PrismaticJoint>(
  //     "world_convex", plant.world_body(), RigidTransformd{Vector3d{-0.5, 0, FLAGS_z0}},
  //     plant.GetBodyByName("pancake_convex", sphere_internal_mi), {},
  //     Vector3d{0, 0, 1});

  plant.Finalize();

  if (FLAGS_visualize) {
    geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);
    ConnectContactResultsToDrakeVisualizer(&builder, plant, scene_graph,
                                           /* lcm */ nullptr);
  }

  auto diagram = builder.Build();
  auto simulator = MakeSimulatorFromGflags(*diagram);

  // Set the Sphere's initial pose.
  systems::Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(&simulator->get_mutable_context());

  plant.SetFreeBodyPose(
      &plant_context, plant.GetBodyByName("pancake_embedded", sphere_embedded_mi),
      math::RigidTransformd{
          math::RollPitchYawd{FLAGS_roll, FLAGS_pitch, FLAGS_yaw},
          Vector3d(FLAGS_x0, FLAGS_y0, FLAGS_z0)});
  plant.SetFreeBodySpatialVelocity(
      &plant_context, plant.GetBodyByName("pancake_embedded", sphere_embedded_mi),
      SpatialVelocity<double>{
          M_PI / 180.0 * Vector3d(FLAGS_wx, FLAGS_wy, FLAGS_wz),
          Vector3d(FLAGS_vx, FLAGS_vy, FLAGS_vz)});

//   plant.SetFreeBodyPose(
//       &plant_context, plant.GetBodyByName("pancake_convex", sphere_internal_mi),
//       math::RigidTransformd{
//           math::RollPitchYawd{FLAGS_roll, FLAGS_pitch, FLAGS_yaw},
//           Vector3d(FLAGS_x0 - 50.5, FLAGS_y0, FLAGS_z0)});
//   plant.SetFreeBodySpatialVelocity(
//       &plant_context, plant.GetBodyByName("pancake_convex", sphere_internal_mi),
//       SpatialVelocity<double>{
//           M_PI / 180.0 * Vector3d(FLAGS_wx, FLAGS_wy, FLAGS_wz),
//           Vector3d(FLAGS_vx, FLAGS_vy, FLAGS_vz)});

//   simulator->set_monitor([&plant, &sphere_embedded_mi](
//                              const systems::Context<double>& root_context) {
//     const systems::Context<double>& context =
//         plant.GetMyContextFromRoot(root_context);

//     auto collect_contact_data = [&plant, &context](const Body<double>& body) {
//       double Fz_B = 0.0;
//       int total_contacts = 0;

//       auto geometries = plant.GetCollisionGeometriesForBody(body);
//       const ContactResults<double>& contact_results =
//           plant.get_contact_results_output_port().Eval<ContactResults<double>>(
//               context);
//       for (int i = 0; i < contact_results.num_hydroelastic_contacts(); ++i) {
//         const auto contact_info = contact_results.hydroelastic_contact_info(i);
//         const auto contact_surface = contact_info.contact_surface();
//         const bool contains_M =
//             std::find(geometries.begin(), geometries.end(),
//                       contact_surface.id_M()) != geometries.end();
//         const bool contains_N =
//             std::find(geometries.begin(), geometries.end(),
//                       contact_surface.id_N()) != geometries.end();

//         DRAKE_ASSERT(!contains_M || !contains_N);

//         if (contains_M || contains_N) {
//           //std::cout << fmt::format("Surface({}) Body({}) M({}) N({})\n", i, body.name(), contains_M, contains_N);
//           Fz_B +=
//               (contains_M ? 1 : -1) * contact_info.F_Ac_W().translational().z();
//           total_contacts += contact_surface.num_faces();
//         }
//       }
//       return std::tie(Fz_B, total_contacts);
//     };

//     const double t = context.get_time();

//     const Body<double>& Se = plant.GetBodyByName("pancake_embedded", sphere_embedded_mi);
//     const Body<double>& Si = plant.GetBodyByName("pancake_convex", sphere_internal_mi);
//     auto X_WSe = Se.EvalPoseInWorld(context);
//     auto X_WSi = Si.EvalPoseInWorld(context);

//     auto [Fz_Se, num_contacts_Se] = collect_contact_data(Se);

//     std::cout << fmt::format("{}\t{}\t{}\t{}\t{}\n", t, X_WSe.translation().z(),
//                              X_WSe.translation().z() - 0.01, Fz_Se,
//                              num_contacts_Se);


//     auto [Fz_Si, num_contacts_Si] = collect_contact_data(Si);
//     std::cout << fmt::format("{}\t{}\t{}\t{}\n", X_WSi.translation().z(),
//                              X_WSi.translation().z() - 0.01, Fz_Si,
//                              num_contacts_Si);

//     return systems::EventStatus::Succeeded();
//   });

  int max_contacts = 0;
  int num_timesteps = 0;
  int total_contacts = 0;

  simulator->set_monitor([&](const systems::Context<double>& root_context) {
    const systems::Context<double>& context =
        plant.GetMyContextFromRoot(root_context);

    const ContactResults<double>& contact_results =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            context);
    int num_contacts = 0;
    for (int i = 0; i < contact_results.num_hydroelastic_contacts(); ++i) {
      const auto contact_info = contact_results.hydroelastic_contact_info(i);
      const auto contact_surface = contact_info.contact_surface();
      num_contacts += contact_surface.num_faces();
    }

    max_contacts = std::max(max_contacts, num_contacts);
    total_contacts += num_contacts;
    ++num_timesteps;

    const Body<double>& Se =
        plant.GetBodyByName("pancake_embedded", sphere_embedded_mi);
    auto X_WSe = Se.EvalPoseInWorld(context);
    auto p = X_WSe.translation();
    auto r = X_WSe.rotation().ToQuaternion();

    std::cout << fmt::format("{}\t{}\t{}\t{}\t{}\t{}\t{}\n", p.x(), p.y(),
                             p.z(), r.x(), r.y(), r.z(), r.w());

    return systems::EventStatus::Succeeded();
  });

  auto start_time = std::chrono::system_clock::now();
  simulator->AdvanceTo(FLAGS_simulation_time);
  auto end_time = std::chrono::system_clock::now();

  const double elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time)
          .count() /
      1000.0;

  std::cout << elapsed_time << "\n";
  std::cout << max_contacts << "\n";
  std::cout << ((1.0 * total_contacts) / num_timesteps)
            << "\n";

  //   auto sphere_embedded_pose = plant.GetFreeBodyPose(
  //       plant_context, plant.GetBodyByName("pancake_embedded",
  //       sphere_embedded_mi));
  //   auto sphere_internal_pose = plant.GetFreeBodyPose(
  //       plant_context, plant.GetBodyByName("pancake_convex",
  //       sphere_internal_mi));

  //std::cout << fmt::format("Embedded final z: {}\n", sphere_embedded_pose);
  //std::cout << fmt::format("Internal final z: {}\n", sphere_internal_pose);

  return 0;
}

}  // namespace
}  // namespace two_spheres
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("");
  FLAGS_simulator_publish_every_time_step = true;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::two_spheres::do_main();
}
