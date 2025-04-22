#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/examples/hydroelastic/ball_plate/make_ball_plate_plant.h"
#include "drake/geometry/query_results/speculative_contact.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/geometry_contact_data.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/visualization/visualization_config_functions.h"

DEFINE_double(simulation_time, 3,
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
DEFINE_double(hydroelastic_modulus, 1.0e6,
              "Hydroelastic modulus of the ball, [Pa].");
DEFINE_double(resolution_hint_factor, 0.3,
              "This scaling factor, [unitless], multiplied by the radius of "
              "the ball gives the target edge length of the mesh of the ball "
              "on the surface of its hydroelastic representation. The smaller "
              "number gives a finer mesh with more tetrahedral elements.");
DEFINE_double(dissipation, 0.0,
              "Hunt & Crossley dissipation, [s/m], for the ball");
DEFINE_double(friction_coefficient, 0.3,
              "coefficient for both static and dynamic friction, [unitless], "
              "of the ball.");
DEFINE_double(mbp_dt, 0.01,
              "The fixed time step period (in seconds) of discrete updates "
              "for the multibody plant modeled as a discrete system. "
              "Strictly positive.");

// Ball's initial spatial velocity.
DEFINE_double(vx, 0.0,
              "Ball's initial translational velocity in the x-axis in m/s.");
DEFINE_double(vy, 0.0,
              "Ball's initial translational velocity in the y-axis in m/s.");
DEFINE_double(vz, -5.0,
              "Ball's initial translational velocity in the z-axis in m/s.");
DEFINE_double(wx, 0.001,
              "Ball's initial angular velocity in the x-axis in degrees/s.");
DEFINE_double(wy, 0.001,
              "Ball's initial angular velocity in the y-axis in degrees/s.");
DEFINE_double(wz, 0.001,
              "Ball's initial angular velocity in the z-axis in degrees/s.");

// Ball's initial pose.
DEFINE_double(z0, 0.10, "Ball's initial position in the z-axis.");
DEFINE_double(x0, 0.10, "Ball's initial position in the x-axis.");

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
using drake::multibody::SpatialVelocity;
using drake::multibody::internal::GeometryContactData;
using Eigen::AngleAxisd;
using Eigen::Matrix3Xd;
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
  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  // Ball's parameters.
  const double radius = 0.05;  // m
  const double mass = 0.1;     // kg
  AddBallPlateBodies(
      radius, mass, FLAGS_hydroelastic_modulus, FLAGS_dissipation,
      CoulombFriction<double>{// static friction (unused in discrete systems)
                              FLAGS_friction_coefficient,
                              // dynamic friction
                              FLAGS_friction_coefficient},
      FLAGS_resolution_hint_factor, &plant);

  plant.Finalize();

  // DRAKE_DEMAND(plant.num_velocities() == 12);
  // DRAKE_DEMAND(plant.num_positions() == 14);
  std::shared_ptr<Meshcat> meshcat = std::make_shared<Meshcat>();
  visualization::AddDefaultVisualization(&builder, meshcat);

  auto diagram = builder.Build();
  auto simulator = MakeSimulatorFromGflags(*diagram);

  // Set the ball's initial pose.
  systems::Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(&simulator->get_mutable_context());
  plant.SetFreeBodyPose(
      &plant_context, plant.GetBodyByName("Ball"),
      math::RigidTransformd{
          math::RollPitchYaw(M_PI * FLAGS_wx / 180.0, M_PI * FLAGS_wz / 180.0,
                             M_PI * FLAGS_wz / 180.0),
          Vector3d(FLAGS_x0, 0.0, FLAGS_z0)});
  plant.SetFreeBodySpatialVelocity(
      &plant_context, plant.GetBodyByName("Ball"),
      SpatialVelocity<double>{Vector3d(0, 0, 0),
                              Vector3d(FLAGS_vx, FLAGS_vy, FLAGS_vz)});

  simulator->Initialize();
  common::MaybePauseForUser();

  double t = 0;
  while (t < FLAGS_simulation_time) {
    // Visualize the speculative contact data.
    const GeometryContactData<double>& contact_data =
        plant.EvalGeometryContactData(plant_context);
    if (contact_data.get().speculative_surfaces.size() > 0) {
      const SpeculativeContactSurface<double>& speculative_surface =
          contact_data.get().speculative_surfaces[0];

      const int num_points = speculative_surface.num_contact_points();
      int num_VF = 0;
      int num_EE = 0;
      for (int i = 0; i < num_points; ++i) {
        if (speculative_surface.closest_points()[i].closest_A.type ==
            ClosestPointType::Edge) {
          ++num_EE;
        } else {
          ++num_VF;
        }
      }

      if (num_points > 0) {
        Eigen::Matrix3Xd start_VF = Eigen::Matrix3Xd::Zero(3, num_VF);
        Eigen::Matrix3Xd end_VF = Eigen::Matrix3Xd::Zero(3, num_VF);
        Eigen::Matrix3Xd start_EE = Eigen::Matrix3Xd::Zero(3, num_EE);
        Eigen::Matrix3Xd end_EE = Eigen::Matrix3Xd::Zero(3, num_EE);

        Rgba color_VF(0.3, 0.6, 0.3, 1.0);
        Rgba color_EE(0.3, 0.3, 0.6, 1.0);
        int index_VF = 0;
        int index_EE = 0;

        for (int i = 0; i < num_points; ++i) {
          if (speculative_surface.closest_points()[i].closest_A.type ==
              ClosestPointType::Edge) {
            start_EE.col(index_EE) =
                speculative_surface.closest_points()[i].closest_A.p;
            end_EE.col(index_EE) =
                speculative_surface.closest_points()[i].closest_B.p;
            ++index_EE;
          } else {
            start_VF.col(index_VF) =
                speculative_surface.closest_points()[i].closest_A.p;
            end_VF.col(index_VF) =
                speculative_surface.closest_points()[i].closest_B.p;
            ++index_VF;
          }
        }

        meshcat->SetLineSegments("speculative_surface_VF", start_VF, end_VF,
                                 2.0, color_VF);
        meshcat->SetLineSegments("speculative_surface_EE", start_EE, end_EE,
                                 2.0, color_EE);
      }
      // speculative_surface.SaveToFile(fmt::format("speculative_{}_{}.txt",
      //                                            speculative_surface.id_A(),
      //                                            speculative_surface.id_B()));
    }

    simulator->AdvanceTo(t);

    t += FLAGS_mbp_dt;
  }
  common::MaybePauseForUser();

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
  common::MaybePauseForUser();

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
  return drake::examples::ball_plate::do_main();
}
