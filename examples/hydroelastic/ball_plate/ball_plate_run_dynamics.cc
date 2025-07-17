#include <fstream>
#include <memory>
#include <algorithm>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/fmt_eigen.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/examples/hydroelastic/ball_plate/make_ball_plate_plant.h"
#include "drake/geometry/query_results/speculative_contact.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/geometry_contact_data.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/planar_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/visualization/visualization_config_functions.h"

DEFINE_double(simulation_time, 1.5,
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
DEFINE_double(hydroelastic_modulus, 1.0e7,
              "Hydroelastic modulus of the ball, [Pa].");
DEFINE_double(tile_modulus, 1.0e7, "Hydroelastic modulus of the tiles, [Pa].");
DEFINE_double(resolution_hint_factor, 0.3,
              "This scaling factor, [unitless], multiplied by the radius of "
              "the ball gives the target edge length of the mesh of the ball "
              "on the surface of its hydroelastic representation. The smaller "
              "number gives a finer mesh with more tetrahedral elements.");
DEFINE_double(dissipation, 20.0,
              "Hunt & Crossley dissipation, [s/m], for the ball");
DEFINE_double(friction_coefficient, 0.3,
              "coefficient for both static and dynamic friction, [unitless], "
              "of the ball.");
DEFINE_double(dt, 0.01,
              "The fixed time step period (in seconds) of discrete updates "
              "for the multibody plant modeled as a discrete system. "
              "Strictly positive.");

// Ball's initial spatial velocity.
DEFINE_double(vx, 0.0,
              "Ball's initial translational velocity in the x-axis in m/s.");
DEFINE_double(vy, 0.0,
              "Ball's initial translational velocity in the y-axis in m/s.");
DEFINE_double(vz, 0.0,
              "Ball's initial translational velocity in the z-axis in m/s.");
DEFINE_double(wx, 0,
              "Ball's initial angular velocity in the x-axis in degrees/s.");
DEFINE_double(wy, 0,
              "Ball's initial angular velocity in the y-axis in degrees/s.");
DEFINE_double(wz, 0,
              "Ball's initial angular velocity in the z-axis in degrees/s.");

// Ball's initial pose.
DEFINE_double(x0, 0.0, "Ball's initial position in the x-axis.");
DEFINE_double(y0, 0.0, "Ball's initial position in the y-axis.");
DEFINE_double(z0, 0.0, "Ball's initial position in the z-axis.");

// Real time rate override
DEFINE_double(rtr, 1.0, "Simulator target real time rate");

DEFINE_string(mode, "viz",
              "Which mode to run the simulation in: viz, data, timing");
DEFINE_string(data_file, "data.txt",
              "State data per timestep: t, q, v, Fcontact_Bcm_W");
DEFINE_string(
    surface_file, "surface.txt",
    "Contact surface data per timestep. Expect only one surface per timestep.");
DEFINE_int32(num_dofs, 6, "Number of dofs of the system: 3 or 6");

DEFINE_bool(use_speculative, false, "If true uses speculative consraints.");
DEFINE_int32(num_speculative, -1,
             "If use_speculative=true, sets the number of speculative "
             "constraints to use.");

namespace drake {
namespace examples {
namespace ball_plate {
namespace {

using drake::geometry::Meshcat;
using drake::geometry::Rgba;
using drake::geometry::Sphere;
using drake::geometry::internal::ClosestPointType;
using drake::geometry::internal::SpeculativeContactSurface;
using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
using drake::multibody::CoulombFriction;
using drake::multibody::PlanarJoint;
using drake::multibody::SpatialVelocity;
using drake::multibody::internal::GeometryContactData;
using Eigen::AngleAxisd;
using Eigen::Matrix3Xd;
using Eigen::Vector2d;
using Eigen::Vector3d;

int do_main() {
  systems::DiagramBuilder<double> builder;

  multibody::MultibodyPlantConfig config;
  // We allow only discrete systems.
  DRAKE_DEMAND(FLAGS_dt > 0.0);
  config.time_step = FLAGS_dt;
  config.penetration_allowance = 0.001;
  config.contact_model = FLAGS_contact_model;
  config.contact_surface_representation = FLAGS_contact_surface_representation;
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  // Ball's parameters.
  const double radius = 0.05;  // m
  const double mass = 0.1;     // kg
  AddRollingBallBodies(
      radius, mass, FLAGS_hydroelastic_modulus, FLAGS_tile_modulus,
      FLAGS_dissipation,
      CoulombFriction<double>{// static friction (unused in discrete systems)
                              FLAGS_friction_coefficient,
                              // dynamic friction
                              FLAGS_friction_coefficient},
      FLAGS_resolution_hint_factor, FLAGS_num_dofs, &plant);

  fmt::print("Use speculative: {}\n", FLAGS_use_speculative);

  plant.set_use_speculative(FLAGS_use_speculative);
  plant.set_num_speculative(FLAGS_num_speculative);
  plant.Finalize();

  // Set up visualization.
  std::shared_ptr<Meshcat> meshcat;
  if (FLAGS_mode == "viz") {
    meshcat = std::make_shared<Meshcat>();
    visualization::AddDefaultVisualization(&builder, meshcat);
  }

  auto diagram = builder.Build();
  auto simulator = MakeSimulatorFromGflags(*diagram);

  // Set the ball's initial state.
  systems::Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(&simulator->get_mutable_context());

  // Set the initial conditions.
  if (FLAGS_num_dofs == 6) {
    plant.SetFreeBodyPose(
        &plant_context, plant.GetBodyByName("Ball"),
        math::RigidTransformd{
            math::RollPitchYaw(M_PI * FLAGS_wx / 180.0, M_PI * FLAGS_wz / 180.0,
                               M_PI * FLAGS_wz / 180.0),
            Vector3d(FLAGS_x0, FLAGS_y0, FLAGS_z0)});
    plant.SetFreeBodySpatialVelocity(
        &plant_context, plant.GetBodyByName("Ball"),
        SpatialVelocity<double>{Vector3d(0, 0, 0),
                                Vector3d(FLAGS_vx, FLAGS_vy, FLAGS_vz)});
  } else if (FLAGS_num_dofs == 3) {
    const PlanarJoint<double>& xz_planar =
        plant.GetJointByName<PlanarJoint>("xz_planar");
    xz_planar.set_translation(&plant_context, Vector2d(FLAGS_x0, FLAGS_z0));
    xz_planar.set_translational_velocity(&plant_context,
                                         Vector2d(FLAGS_vx, FLAGS_vz));
    xz_planar.set_angular_velocity(&plant_context, FLAGS_wy);
  } else {
    throw std::logic_error("num_dofs must be 3 or 6.");
  }

  if (FLAGS_mode == "viz") {
    meshcat->StartRecording();
    simulator->Initialize();
    common::MaybePauseForUser();
    simulator->AdvanceTo(FLAGS_simulation_time);
    meshcat->StopRecording();
    meshcat->PublishRecording();
    common::MaybePauseForUser();
  } else if (FLAGS_mode == "data") {
    simulator->Initialize();

    std::ofstream out(FLAGS_data_file);
    std::ofstream surface_out(FLAGS_surface_file);
    if (!out) {
      throw std::runtime_error(
          fmt::format("Could not open file: {}", FLAGS_data_file));
    }

    if(!surface_out) {
      throw std::runtime_error(
        fmt::format("Could not open file: {}", FLAGS_surface_file));
    }

    double t = 0;
    while (t <= FLAGS_simulation_time) {
      const VectorX<double> qv = plant.GetPositionsAndVelocities(plant_context);
      const VectorX<double> F_contact =
          plant
              .get_generalized_contact_forces_output_port(
                  multibody::default_model_instance())
              .Eval(plant_context);

      out << fmt::format("{} {} {}\n", t, fmt_eigen(qv.transpose()),
                         fmt_eigen(F_contact.transpose()));

      // Save just the single smallest contact point to file.
      const GeometryContactData<double>& contact_data =
          plant.EvalGeometryContactData(plant_context);
      if (contact_data.get().speculative_surfaces.size() > 0) {
        const SpeculativeContactSurface<double>& speculative_surface =
            contact_data.get().speculative_surfaces[0];

        const size_t count =
            plant.num_speculative() >= 0
                ? static_cast<size_t>(1)
                : static_cast<size_t>(
                      speculative_surface.num_contact_points());
        std::vector<int> indices(speculative_surface.num_contact_points());
        std::iota(indices.begin(), indices.end(), 0);

        if (indices.size() > count) {
          std::partial_sort(indices.begin(),
                            indices.begin() + std::min(count, indices.size()),
                            indices.end(),
                            [&speculative_surface](size_t a, size_t b) {
                              return speculative_surface.time_of_contact()[a] <
                                     speculative_surface.time_of_contact()[b];
                            });
          indices.resize(count);
        }

        surface_out << fmt::format("{} ", t);
        for (int i : indices) {
          surface_out << speculative_surface.ToString(i);
        }
      }
      simulator->AdvanceTo(t);
      t += FLAGS_dt;
    }
  }

  // // Extra code to visualize truncated Taylor series for vertex trajectories
  // of
  // // bodies undergoing constant spatial velocity.
  // const Vector3d p_BP(0.05, 0.05, 0.05);
  // const Vector3d p_WB(0, 0, 0);
  // const Vector3d v_WB(1, 1, 1);
  // const Vector3d w_WB(0, 0, 1);
  // const double w = 20;
  // const int num_samples = 100;
  // const double dt = 0.005;
  // Matrix3Xd trajectory(3, num_samples);
  // Matrix3Xd quadratic_trajectory(3, num_samples);
  // const Vector3d w_x_p = (w * w_WB).cross(p_BP);
  // const Vector3d w_x_w_x_p = (w * w_WB).cross(w_x_p);
  // const Vector3d w_x_w_x_w_x_p = (w * w_WB).cross(w_x_w_x_p);

  // for (int i = 0; i < num_samples; ++i) {
  //   trajectory.col(i) =
  //       p_WB + i * dt * v_WB +
  //       RotationMatrixd(Eigen::AngleAxisd(i * dt * w, w_WB)) * p_BP;
  //   quadratic_trajectory.col(i) =
  //       p_WB + p_BP + i * dt * (v_WB + w_x_p) +
  //       0.5 * (i * i * dt * dt) * (w_x_w_x_p) +
  //       (i * i * i * dt * dt * dt / 6.0) * (w_x_w_x_w_x_p);
  // }

  // Vector3d critical_t[2];
  // for (int i = 0; i < 3; ++i) {
  //   const double a = w_x_w_x_w_x_p(i) / 2.0;
  //   const double b = w_x_w_x_p(i);
  //   const double c = (v_WB + w_x_p)(i);
  //   const double disc = b * b - 4 * a * c;
  //   if (disc >= 0) {
  //     critical_t[0](i) = (-b + std::sqrt(disc)) / (2 * a);
  //     critical_t[1](i) = (-b - std::sqrt(disc)) / (2 * a);
  //   } else {
  //     critical_t[0](i) = critical_t[1](i) =
  //         -std::numeric_limits<double>::infinity();
  //   }
  // }

  // meshcat->SetLine("trajectory", trajectory, 2.0);
  // meshcat->SetLine("quadratic trajectory", quadratic_trajectory, 2.0,
  //                  Rgba(0.8, 0.0, 0.0, 1.0));

  // meshcat->SetObject("quadratic_trajectory_q0", Sphere(0.001));
  // meshcat->SetTransform("quadratic_trajectory_q0",
  //                       RigidTransformd(quadratic_trajectory.col(0)));
  // meshcat->SetObject("quadratic_trajectory_qf", Sphere(0.001));
  // meshcat->SetTransform(
  //     "quadratic_trajectory_qf",
  //     RigidTransformd(quadratic_trajectory.col(num_samples - 1)));

  // for (int i = 0; i < 3; ++i) {
  //   for (int j = 0; j < 2; ++j) {
  //     if (critical_t[j](i) >= 0 && critical_t[j](i) <= dt * num_samples) {
  //       // const Vector3d q_critical =
  //       //     p_WB + p_BP + critical_t(i) * (v_WB + w_x_p) +
  //       //     0.5 * (critical_t(i) * critical_t(i)) * (w_x_w_x_p);
  //       const Vector3d q_critical =
  //           p_WB + p_BP + critical_t[j](i) * (v_WB + w_x_p) +
  //           0.5 * (critical_t[j](i) * critical_t[j](i)) * (w_x_w_x_p) +
  //           (critical_t[j](i) * critical_t[j](i) * critical_t[j](i) / 6.0) *
  //               (w_x_w_x_w_x_p);
  //       fmt::print("critical_t_{}_{}: {}\n", i, j, critical_t[j](i));
  //       fmt::print("q_critical({}, {}): {} {} {}\n", i, j, q_critical(0),
  //                  q_critical(1), q_critical(2));
  //       meshcat->SetObject(
  //           fmt::format("quadratic_trajectory_q_critical_{}_{}", i, j),
  //           Sphere(0.001));
  //       meshcat->SetTransform(
  //           fmt::format("quadratic_trajectory_q_critical_{}_{}", i, j),
  //           RigidTransformd(q_critical));
  //     }
  //   }
  // }
  // common::MaybePauseForUser();

  return 0;
}

}  // namespace
}  // namespace ball_plate
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(R"""(
This is an example of using the hydroelastic contact model with a non-convex
collision geometry loaded from an SDFormat file of a dinner plate. The ball,
the plate, and the floor are compliant, rigid, and compliant hydroelastic
respectively. Hence, The plate-ball, ball-floor, and plate-floor contacts are
rigid-compliant, compliant-compliant, and rigid-compliant respectively. The
hydroelastic contact model can work with non-convex shapes accurately without
resorting to their convex hulls. Launch meldis before running this example.
See the README.md file for more information.)""");
  FLAGS_simulator_publish_every_time_step = true;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_simulator_target_realtime_rate = FLAGS_rtr;
  return drake::examples::ball_plate::do_main();
}