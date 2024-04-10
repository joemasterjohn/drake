#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include <gflags/gflags.h>

#include "drake/common/nice_type_name.h"
#include "drake/common/temp_directory.h"
#include "drake/geometry/collision_filter_declaration.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/random_rotation.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/contact_solvers/sap/sap_solver.h"
#include "drake/multibody/meshcat/contact_visualizer.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/implicit_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"

// To profile with Valgrind run with (the defaults are good):
// valgrind --tool=callgrind --separate-callers=10 --instr-atstart=no
// bazel-bin/examples/multibody/clutter
#include <valgrind/callgrind.h>
namespace drake {
namespace multibody {
namespace examples {
namespace {

// Simulation parameters.
DEFINE_double(simulation_time, 10.0, "Simulation duration in seconds");
DEFINE_double(
    time_step, 1.0E-2,
    "If mbp_time_step > 0, the fixed-time step period (in seconds) of discrete "
    "updates for the plant (modeled as a discrete system). "
    "If mbp_time_step = 0, the plant is modeled as a continuous system "
    "and no contact forces are displayed.  mbp_time_step must be >= 0.");

// Visualization.
DEFINE_bool(visualize, false, "Whether to visualize (true) or not (false).");
DEFINE_bool(visualize_forces, false,
            "Whether to visualize forces (true) or not (false).");
DEFINE_double(viz_period, 1.0 / 60.0, "Viz period.");

// Discrete contact solver.
DEFINE_string(discrete_contact_solver, "sap",
              "Discrete contact solver. Options are: 'tamsi', 'sap'");
DEFINE_double(near_rigid_threshold, 1.0, "SAP near rigid threshold.");
DEFINE_int32(grid_size, 2, "Grid size");
DEFINE_bool(use_hydro, true, "If true use hydro, otherwise point");

using drake::geometry::CollisionFilterDeclaration;
using drake::math::RigidTransform;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::math::RotationMatrixd;
using drake::multibody::ContactResults;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using Eigen::Translation3d;
using Eigen::Vector3d;
using clock = std::chrono::steady_clock;
using drake::multibody::contact_solvers::internal::SapSolverParameters;

const std::string yellow_pepper(const std::string& number) {
  std::string proximity_properties = FLAGS_use_hydro ? R"""(
     <drake:compliant_hydroelastic/>
    <drake:hydroelastic_modulus>5.0e4</drake:hydroelastic_modulus>
    <!-- Most shapes (capsule, cylinder, ellipsoid, sphere) need
      drake:mesh_resolution_hint, but the resolution hint is no-op
      for the mesh geometry. That's why we do not set it here. -->
    <drake:hunt_crossley_dissipation>10</drake:hunt_crossley_dissipation>
    <!-- Both mu_dynamic and mu_static are used in Continuous system.
      Only mu_dynamic is used in Discrete system.  -->
    <drake:mu_dynamic> 1 </drake:mu_dynamic>
    <drake:mu_static> 1 </drake:mu_static>
  )"""
                                                     : "";
  return fmt::format(R"""(<?xml version="1.0"?>
    <sdf version="1.7" xmlns:drake="drake.mit.edu">
      <model name="yellow_pepper_{}">
        <link name="yellow_pepper_{}">
          <pose>0 0 0 0 0 0</pose>
          <inertial>
            <pose>0.000537 -0.00272 0.0384 0 0 0</pose>
            <mass>0.159</mass>
            <inertia>
              <ixx> 0.000101</ixx>
              <ixy>-0.000001</ixy>
              <ixz>-0.000004</ixz>
              <iyy> 0.000105</iyy>
              <iyz> 0.000007</iyz>
              <izz> 0.000107</izz>
            </inertia>
          </inertial>
          <visual name="yellow_bell_pepper_no_stem">
            <pose>0 0 0 0 0 0</pose>
            <geometry>
              <mesh>
                <uri>package://drake_models/veggies/assets/yellow_bell_pepper_no_stem_low.obj</uri>
                <scale>1 1 1</scale>
              </mesh>
            </geometry>
          </visual>
          <collision name="collision">
            <pose>0 0 0 0 0 0</pose>
            <geometry>
              <mesh>
                <uri>package://drake_models/veggies/assets/yellow_bell_pepper_no_stem_low.vtk</uri>
                <scale>1 1 1</scale>
              </mesh>
            </geometry>
            <drake:proximity_properties>
              {}
            </drake:proximity_properties>
          </collision>
        </link>
        <frame name="origin">
          <pose relative_to="yellow_pepper_{}">0 0 0 0 0 0</pose>
        </frame>
        <frame name="flush_bottom_center__z_up">
          <pose relative_to="yellow_pepper_{}">0 0 0 0 0 0</pose>
        </frame>
      </model>
    </sdf>
    )""",
                     number, number, proximity_properties, number, number);
}

const std::string rigid_box_sdf = R"""(<?xml version="1.0"?>
<sdf version="1.7">
  <model name="RigidBox">
    <link name="rigid_box_link">
      <visual name="visual">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>3.0 3.0 0.05</size>
          </box>
        </geometry>
        <material>
         <diffuse>0.9 0.8 0.7 0.5</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>3.0 3.0 0.05</size>
          </box>
        </geometry>
        <drake:proximity_properties>
          <drake:rigid_hydroelastic/>
          <drake:mu_dynamic>0.5</drake:mu_dynamic>
          <drake:hunt_crossley_dissipation>1.25</drake:hunt_crossley_dissipation>
          <drake:relaxation_time>0.1</drake:relaxation_time>
        </drake:proximity_properties>
      </collision>
    </link>
    <frame name="top_surface">
      <pose relative_to="rigid_box_link">0 0 0.025 0 0 0</pose>
    </frame>
  </model>
</sdf>
)""";

void AddScene(int grid_size, MultibodyPlant<double>* plant) {
  Parser parser(plant);

  // Add peppers
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      parser.AddModelsFromString(yellow_pepper(fmt::format("{}{}", i, j)),
                                 "sdf");
    }
  }

  // Add ground
  parser.AddModelsFromString(rigid_box_sdf, "sdf");
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("top_surface"));

  plant->Finalize();

  // Sett Peppers initial conditions.
  const double dx = 0.3;
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      plant->SetDefaultFreeBodyPose(
          plant->GetBodyByName(fmt::format("yellow_pepper_{}{}", i, j)),
          RigidTransformd(Vector3d(-0.15 + dx * i, -0.15 + dx * j, 0.20)));
    }
  }
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  plant_config.discrete_contact_solver = FLAGS_discrete_contact_solver;
  plant_config.sap_near_rigid_threshold = FLAGS_near_rigid_threshold;
  plant_config.contact_model = FLAGS_use_hydro ? "hydroelastic" : "point";
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlant(plant_config, &builder);
  AddScene(FLAGS_grid_size, &plant);

  fmt::print("Num positions: {:d}\n", plant.num_positions());
  fmt::print("Num velocities: {:d}\n", plant.num_velocities());

  // Publish contact results for visualization.
  std::shared_ptr<drake::geometry::Meshcat> meshcat;
  if (FLAGS_visualize) {
    meshcat = std::make_shared<drake::geometry::Meshcat>();
    drake::geometry::MeshcatVisualizerParams params;
    params.publish_period = FLAGS_viz_period;
    drake::geometry::MeshcatVisualizerd::AddToBuilder(
        &builder, scene_graph, meshcat, std::move(params));
  }
  if (FLAGS_visualize && FLAGS_visualize_forces) {
    drake::multibody::meshcat::ContactVisualizerParams cparams;
    cparams.newtons_per_meter = 60.0;
    drake::multibody::meshcat::ContactVisualizerd::AddToBuilder(
        &builder, plant, meshcat, std::move(cparams));
  }
  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());

  auto simulator =
      MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  // Monitor to save stats into a file.
  int num_surfaces = 0;
  int nc = 0;
  simulator->set_monitor([&simulator, &plant, &num_surfaces,
                          &nc](const systems::Context<double>& root_context) {
    const systems::Context<double>& plant_ctx =
        plant.GetMyContextFromRoot(root_context);
    // Compute delta time and update previous time.
    const auto& contact_results =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            plant_ctx);

    num_surfaces += contact_results.num_hydroelastic_contacts();
    for (int i = 0; i < contact_results.num_hydroelastic_contacts(); ++i) {
      const auto& info = contact_results.hydroelastic_contact_info(i);
      nc += info.contact_surface().num_faces();
    }

    return systems::EventStatus::Succeeded();
  });

  simulator->Initialize();
  if (FLAGS_visualize) {
    std::cout << "Press any key to continue ...\n";
    getchar();
  }

  if (FLAGS_visualize) {
    const double recording_frames_per_second = 1.0 / FLAGS_time_step;
    meshcat->StartRecording(recording_frames_per_second);
  }
  clock::time_point sim_start_time = clock::now();
  CALLGRIND_START_INSTRUMENTATION;
  simulator->AdvanceTo(FLAGS_simulation_time);
  CALLGRIND_STOP_INSTRUMENTATION;
  clock::time_point sim_end_time = clock::now();
  const double sim_time =
      std::chrono::duration<double>(sim_end_time - sim_start_time).count();
  std::cout << "AdvanceTo() time [sec]: " << sim_time << std::endl;
  if (FLAGS_visualize) {
    meshcat->StopRecording();
    meshcat->PublishRecording();
  }
  std::cout << fmt::format("num_surfaces = {}\n", num_surfaces);
  std::cout << fmt::format("num_contacts = {}\n", nc);

  PrintSimulatorStatistics(*simulator);

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "\nSimulation of a clutter of objects falling into a box container.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::examples::do_main();
}
