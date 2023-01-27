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
#include "drake/systems/primitives/vector_log_sink.h"

DEFINE_double(simulation_time, 8.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 1.0, "Desired real time rate.");
DEFINE_double(time_step, 1e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e4, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.0,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(t0, 0.0, "Time to start pressing [s].");
DEFINE_double(k, 10.0, "Slope of force [N/s].");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::PrismaticJoint;
using drake::multibody::SpatialInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::BasicVector;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace {

class Plot : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Plot);

  Plot() {
    this->DeclareVectorInputPort("force", BasicVector<double>(2));
    this->DeclareVectorInputPort("state", BasicVector<double>(2));
  }
};

/* A Leaf system that uses outputs a force signal to linear in time. */
class ForceControl : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceControl);

  /* Constructs a ForceControl system with the given parameters. The output
   force is the positive part of k*(t-t0). */
  ForceControl(double t0, double k) : t0_(t0), k_(k) {
    this->DeclareVectorOutputPort("gripper force", BasicVector<double>(2),
                                  &ForceControl::SetAppliedForce);
  }

 private:
  void SetAppliedForce(const Context<double>& context,
                       BasicVector<double>* output) const {
    Vector2d force = Vector2d::Zero();
    const double t = context.get_time();
    if (t > t0_) {
      force(0) = k_ * (t - t0_);
      force(1) = k_ * (t - t0_);
    }
    output->get_mutable_value() << force;
    std::cout << force.transpose() << std::endl;
  }

  double t0_{0.0};
  double k_{0.1};
};

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  SpatialInertia<double> unit_spatial_inertia =
      SpatialInertia<double>::MakeUnitary();
  const auto& top_body = plant.AddRigidBody("top", unit_spatial_inertia);
  const auto& bottom_body = plant.AddRigidBody("bottom", unit_spatial_inertia);

  const auto& top_joint = plant.AddJoint<PrismaticJoint>(
      "top_joint", plant.world_body(), {}, top_body,
      RigidTransformd::Identity(), -Vector3d::UnitZ());
  const auto& bottom_joint = plant.AddJoint<PrismaticJoint>(
      "bottom_joint", plant.world_body(), {}, bottom_body,
      RigidTransformd::Identity(), Vector3d::UnitZ());
  plant.AddJointActuator("top_joint_actuator", top_joint);
  plant.AddJointActuator("bottom_joint_actuator", bottom_joint);

  /* Set up collision and visualization geometries. */
  Box box{7, 7, 0.1};
  /* Minimum required proximity properties for rigid bodies to interact with
   deformable bodies.
   1. A valid Coulomb friction coefficient, and
   2. A resolution hint. (Rigid bodies need to be tesselated so that collision
   queries can be performed against deformable geometries.) */
  ProximityProperties rigid_proximity_props;
  const CoulombFriction<double> surface_friction(0, 0);
  AddContactMaterial({}, {}, surface_friction, &rigid_proximity_props);
  rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                    geometry::internal::kRezHint, 1.0);
  plant.RegisterCollisionGeometry(top_body, RigidTransformd::Identity(), box,
                                  "top_collision", rigid_proximity_props);
  plant.RegisterCollisionGeometry(bottom_body, RigidTransformd::Identity(), box,
                                  "bottom_collision", rigid_proximity_props);

  IllustrationProperties illustration_props;
  illustration_props.AddProperty("phong", "diffuse",
                                 Vector4d(0.7, 0.5, 0.4, 0.8));
  plant.RegisterVisualGeometry(top_body, RigidTransformd::Identity(), box,
                               "top_visual", illustration_props);
  plant.RegisterVisualGeometry(bottom_body, RigidTransformd::Identity(), box,
                               "bottom_visual", illustration_props);

  /* Set up a deformable cylinder. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);

  DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(FLAGS_E);
  deformable_config.set_poissons_ratio(FLAGS_nu);
  deformable_config.set_mass_density(FLAGS_density);
  deformable_config.set_stiffness_damping_coefficient(FLAGS_beta);

  const math::RigidTransform<double> X_WG = math::RigidTransform<double>(
      math::RollPitchYaw(0.0, -1.57, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0));
  const std::string cylinder_vtk = FindResourceOrThrow(
      "drake/examples/multibody/cylinder_press/cylinder.vtk");
  auto cylinder_mesh = std::make_unique<Mesh>(cylinder_vtk, 1.0);
  auto cylinder_instance = std::make_unique<GeometryInstance>(
      X_WG, std::move(cylinder_mesh), "deformable_cylinder");

  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  cylinder_instance->set_proximity_properties(deformable_proximity_props);
  const double unused_resolution_hint = 1.0;
  owned_deformable_model->RegisterDeformableBody(
      std::move(cylinder_instance), deformable_config, unused_resolution_hint);
  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
  plant.AddPhysicalModel(std::move(owned_deformable_model));
  plant.mutable_gravity_field().set_gravity_vector(Vector3d{0, 0, 0});
  // /* Viscous damping for the finger joints, in N⋅s/m. */
  // top_joint.set_default_damping(50.0);
  // bottom_joint.set_default_damping(50.0);

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

  const auto& control = *builder.AddSystem<ForceControl>(FLAGS_t0, FLAGS_k);
  builder.Connect(control.get_output_port(), plant.get_actuation_input_port());

  auto state_logger =
      systems::LogVectorOutput(plant.get_state_output_port(), &builder);
  auto force_logger =
      systems::LogVectorOutput(control.get_output_port(), &builder);

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Set initial conditions for the gripper. */
  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  top_joint.set_translation(&plant_context, -1.05);
  bottom_joint.set_translation(&plant_context, -1.05);

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.AdvanceTo(FLAGS_simulation_time);

  const auto& force_log = force_logger->FindLog(simulator.get_context());
  const auto& state_log = state_logger->FindLog(simulator.get_context());
  std::cout << "Forces" << std::endl;
  std::cout << force_log.data() << std::endl;
  std::cout << "States" << std::endl;
  std::cout << state_log.data() << std::endl;

  return 0;
}

}  // namespace
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase deformable body simulations in Drake. "
      "A simple parallel gripper grasps a deformable torus on the ground, "
      "lifts it up, and then drops it back on the ground. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
