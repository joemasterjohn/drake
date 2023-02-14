#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"

DEFINE_double(simulation_time, 8.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.0, "Desired real time rate.");
DEFINE_double(time_step, 5.0e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(teddy_E, 5e2, "Young's modulus of the deformable body [Pa].");
DEFINE_double(teddy_nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(teddy_density, 100, "Mass density of the deformable body [kg/m³].");
DEFINE_double(teddy_beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(bubble_E, 3e2, "Young's modulus of the deformable body [Pa].");
DEFINE_double(bubble_nu, 0.2, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(bubble_density, 100, "Mass density of the deformable body [kg/m³].");
DEFINE_double(bubble_beta, 0.01,
              "Stiffness damping coefficient for the deformable body [1/s].");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::geometry::Sphere;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::PrismaticJoint;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace multibody {
namespace deformable_box {
namespace {

/* We create a leaf system that uses PD control to output a force signal to
 a gripper to follow a close-lift-open motion sequence. The signal is
 2-dimensional with the first element corresponding to the wrist degree of
 freedom and the second element corresponding to the left finger degree of
 freedom. This control is a time-based state machine, where forces change based
 on the context time. This is strictly for demo purposes and is not intended to
 generalize to other cases. There are four states: 0. The fingers are open in
 the initial state.
  1. The fingers are closed to secure a grasp.
  2. The gripper is lifted to a prescribed final height.
  3. The fingers are open to loosen a grasp.
 The desired state is interpolated between these states. */
class GripperPositionControl : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GripperPositionControl);

  /* Constructs a GripperPositionControl system with the given parameters.
   @param[in] open_width   The width between fingers in the open state. (meters)
   @param[in] closed_width The width between fingers in the closed state.
                           (meters)
   @param[in] height       The height of the gripper in the lifted state.
                           (meters) */
  GripperPositionControl(double open_width, double closed_width, double height)
      : initial_state_(0, -open_width / 2),
        closed_state_(0, -closed_width / 2),
        lifted_state_(height, -closed_width / 2),
        open_state_(height, -open_width / 2) {
    this->DeclareVectorOutputPort("gripper force", BasicVector<double>(2),
                                  &GripperPositionControl::SetAppliedForce);
    this->DeclareVectorInputPort("gripper state", BasicVector<double>(6));
  }

 private:
  void SetAppliedForce(const Context<double>& context,
                       BasicVector<double>* output) const {
    const VectorXd gripper_state =
        EvalVectorInput(context, GetInputPort("gripper state").get_index())
            ->get_value();
    /* There are 6 dofs in the state, corresponding to
     q_translate_joint, q_left_finger, q_right_finger,
     v_translate_joint, v_left_finger, v_right_finger. */
    /* The positions of the translate joint and the left finger. */
    const Vector2d measured_positions = gripper_state.segment<2>(0);
    /* The velocities of the translate joint and the left finger. */
    const Vector2d measured_velocities = gripper_state.segment<2>(3);
    const Vector2d desired_velocities(0, 0);
    Vector2d desired_positions;
    const double t = context.get_time();
    if (t < fingers_closed_time_) {
      const double end_time = fingers_closed_time_;
      const double theta = t / end_time;
      desired_positions =
          theta * closed_state_ + (1.0 - theta) * initial_state_;
    } else if (t < gripper_lifted_time_) {
      const double end_time = gripper_lifted_time_ - fingers_closed_time_;
      const double theta = (t - fingers_closed_time_) / end_time;
      desired_positions = theta * lifted_state_ + (1.0 - theta) * closed_state_;
    } else if (t < hold_time_) {
      desired_positions = lifted_state_;
    } else if (t < fingers_open_time_) {
      const double end_time = fingers_open_time_ - hold_time_;
      const double theta = (t - hold_time_) / end_time;
      desired_positions = theta * open_state_ + (1.0 - theta) * lifted_state_;
    } else {
      desired_positions = open_state_;
    }
    const Vector2d force = kp_ * (desired_positions - measured_positions) +
                           kd_ * (desired_velocities - measured_velocities);
    output->get_mutable_value() << force(1), force(0);
  }

  /* The time at which the fingers reach the desired closed state. */
  const double fingers_closed_time_{1.5};
  /* The time at which the gripper reaches the desired "lifted" state. */
  const double gripper_lifted_time_{3.0};
  const double hold_time_{5.5};
  /* The time at which the fingers reach the desired open state. */
  const double fingers_open_time_{7.0};
  Vector2d initial_state_;
  Vector2d closed_state_;
  Vector2d lifted_state_;
  Vector2d open_state_;
  const double kp_{1000};
  const double kd_{60.0};
};


int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Minimum required proximity properties for rigid bodies to interact with
   deformable bodies.
   1. A valid Coulomb friction coefficient, and
   2. A resolution hint. (Rigid bodies need to be tesselated so that collision
   queries can be performed against deformable geometries.) */
  ProximityProperties rigid_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> surface_friction(1.0, 1.0);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                    geometry::internal::kRezHint, 1.0);
  /* Set up a ground. */
  Box ground{4, 4, 4};
  const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -2.0});
  plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                  "ground_collision", rigid_proximity_props);
  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.3, 0.3, 0.3, 0.9));
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                               "ground_visual", std::move(illustration_props));

  // Parse the gripper model.
  Parser parser(&plant, &scene_graph);
  const std::string gripper_file = FindResourceOrThrow(
      "drake/examples/multibody/teddy/models/schunk_wsg_50.sdf");
  parser.AddModels(gripper_file);
  // Pose the gripper and weld it to the world.
  const math::RigidTransform<double> X_WF0 = math::RigidTransform<double>(
      math::RollPitchYaw(0.0, -1.57, 0.0), Eigen::Vector3d(0.06, 0.0, 0.0));
  const auto& translate_joint = plant.AddJoint<PrismaticJoint>(
      "translate_joint", plant.world_body(), {}, plant.GetBodyByName("gripper"),
      X_WF0, Vector3d::UnitZ());
  plant.AddJointActuator("slider", translate_joint);

  /* Set up a deformable sphere. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  DeformableModel<double>* deformable_model = owned_deformable_model.get();

  DeformableBodyConfig<double> teddy_config;
  teddy_config.set_youngs_modulus(FLAGS_teddy_E);
  teddy_config.set_poissons_ratio(FLAGS_teddy_nu);
  teddy_config.set_mass_density(FLAGS_teddy_density);
  teddy_config.set_stiffness_damping_coefficient(FLAGS_teddy_beta);

  DeformableBodyConfig<double> bubble_config;
  bubble_config.set_youngs_modulus(FLAGS_bubble_E);
  bubble_config.set_poissons_ratio(FLAGS_bubble_nu);
  bubble_config.set_mass_density(FLAGS_bubble_density);
  bubble_config.set_stiffness_damping_coefficient(FLAGS_bubble_beta);

  const std::string teddy_vtk =
      FindResourceOrThrow("drake/examples/multibody/teddy/teddy.vtk");
  auto teddy_mesh = std::make_unique<Mesh>(teddy_vtk, 0.1);
  const RigidTransformd X_WB(RollPitchYawd(1.57, 0, -1.57), Vector3d(-0.21, 0.0, 0));
  auto teddy_instance =
      std::make_unique<GeometryInstance>(X_WB, std::move(teddy_mesh), "teddy");
  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  teddy_instance->set_proximity_properties(deformable_proximity_props);
  IllustrationProperties deformable_illustration_props;
  deformable_illustration_props.AddProperty("phong", "diffuse",
                                            Vector4d(0.53, 0.42, 0.34, 1.0));
  teddy_instance->set_illustration_properties(deformable_illustration_props);

  deformable_model->RegisterDeformableBody(std::move(teddy_instance),
                                           teddy_config, 1.0);

  IllustrationProperties bubble_illustration_props;
  bubble_illustration_props.AddProperty("phong", "diffuse",
                                            Vector4d(0.9, 0.9, 0.9, 1.0));

  const std::string bubble_vtk =
      FindResourceOrThrow("drake/examples/multibody/teddy/bubble.vtk");
  auto left_bubble_mesh = std::make_unique<Mesh>(bubble_vtk);
  const RigidTransformd X_FB1(RollPitchYawd(0, 1.57, -1.57),
                              Vector3d(0.06, -0.095, 0.245));
  const RigidTransformd X_WB1 = X_WF0 * X_FB1;
  auto left_bubble_instance = std::make_unique<GeometryInstance>(
      X_WB1, std::move(left_bubble_mesh), "left bubble");
  left_bubble_instance->set_proximity_properties(deformable_proximity_props);
  left_bubble_instance->set_illustration_properties(bubble_illustration_props);
  DeformableBodyId left_bubble_id = deformable_model->RegisterDeformableBody(
      std::move(left_bubble_instance), bubble_config, 1.0);
  const Body<double>& left_finger = plant.GetBodyByName("left_finger_bubble");
  deformable_model->Weld(left_bubble_id, left_finger, X_WB1,
                         RigidTransformd(RollPitchYawd(0, 1.57, 0),
                                         Vector3d(-0.0725796, -0.065, 0.0599422)));

  auto right_bubble_mesh = std::make_unique<Mesh>(bubble_vtk);
  const RigidTransformd X_FB2(RollPitchYawd(0.0, 1.57, 1.57),
                              Vector3d(0.06, 0.095, 0.245));
  const RigidTransformd X_WB2 = X_WF0 * X_FB2;
  auto right_bubble_instance = std::make_unique<GeometryInstance>(
      X_WB2, std::move(right_bubble_mesh), "right bubble");
  right_bubble_instance->set_proximity_properties(deformable_proximity_props);
  right_bubble_instance->set_illustration_properties(bubble_illustration_props);
  DeformableBodyId right_bubble_id = deformable_model->RegisterDeformableBody(
      std::move(right_bubble_instance), bubble_config, 1.0);
  const Body<double>& right_finger = plant.GetBodyByName("right_finger_bubble");
  deformable_model->Weld(right_bubble_id, right_finger, X_WB2,
                         RigidTransformd(RollPitchYawd(0, 1.57, 0),
                                         Vector3d(-0.0725796, 0.065, 0.0599422)));

  plant.AddPhysicalModel(std::move(owned_deformable_model));

  PrismaticJoint<double>& right_joint =
      plant.GetMutableJointByName<PrismaticJoint>("right_finger_sliding_joint");
  PrismaticJoint<double>& left_joint =
      plant.GetMutableJointByName<PrismaticJoint>("left_finger_sliding_joint");
  plant.AddCouplerConstraint(left_joint, right_joint, -1.0);
  left_joint.set_default_damping(50.0);
  right_joint.set_default_damping(50.0);

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* It's essential to connect the vertex position port in DeformableModel to
   the source configuration port in SceneGraph when deformable bodies are
   present in the plant. */
  builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  /* Set the width between the fingers for open and closed states as well as the
   height to which the gripper lifts the deformable torus. */
  const double kL = 0.08;
  const double open_width = kL * 1.5;
  const double closed_width = kL * 0.2;
  const double lifted_height = 0.18;

  const auto& control = *builder.AddSystem<GripperPositionControl>(
      open_width, closed_width, lifted_height);
  builder.Connect(plant.get_state_output_port(), control.get_input_port());
  builder.Connect(control.get_output_port(), plant.get_actuation_input_port());

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  params.role = geometry::Role::kIllustration;
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr,
                                           params);

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  // Set the initial conditions for the spatula pose and the gripper finger
  // positions.
  Context<double>& mutable_root_context = simulator.get_mutable_context();
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, &mutable_root_context);

  // Set finger joint positions.
  left_joint.set_translation(&plant_context, -0.065);
  right_joint.set_translation(&plant_context, 0.065);

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
  gflags::SetUsageMessage(
      "This is a demo used to showcase deformable body simulations in Drake. "
      "A simple parallel gripper grasps a deformable box on the ground, lifts "
      "it up, and then drops it back on the ground. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::deformable_box::do_main();
}
