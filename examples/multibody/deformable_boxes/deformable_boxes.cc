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
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_double(simulation_time, 8.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.0, "Desired real time rate.");
DEFINE_double(time_step, 1.0e-2,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e5, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1e3, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.005,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(resolution_hint, 0.05, "rezhint");

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
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using drake::systems::DiscreteStateIndex;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace multibody {
namespace deformable_box {
namespace {

void SetVelocity(DeformableBodyId id, const DeformableModel<double>& model,
                 Context<double>* plant_context, const Vector3d& v_WB) {
  const DiscreteStateIndex state_index = model.GetDiscreteStateIndex(id);
  VectorXd state = plant_context->get_discrete_state(state_index).value();
  DRAKE_DEMAND(state.size() % 3 == 0);  // q, v, a all the same length.
  const int num_dofs = state.size() / 3;
  DRAKE_DEMAND(num_dofs % 3 == 0);  // Each vertex needs a 3-vector.
  const int num_vertices = num_dofs / 3;
  VectorXd velocities(num_dofs);
  for (int i = 0; i < num_vertices; ++i) {
    velocities.segment<3>(3 * i) = v_WB;
  }
  state.segment(num_dofs, num_dofs) = velocities;
  plant_context->SetDiscreteState(state_index, state);
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Set up deformable boxes. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);

  DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(FLAGS_E);
  deformable_config.set_poissons_ratio(FLAGS_nu);
  deformable_config.set_mass_density(FLAGS_density);
  deformable_config.set_stiffness_damping_coefficient(FLAGS_beta);

  const std::string box_vtk =
      FindResourceOrThrow("drake/examples/multibody/deformable_boxes/box.vtk");
  auto box_mesh = std::make_unique<Mesh>(box_vtk, 1.0);
  const RigidTransformd X_WB(Vector3<double>(0.0, 0.0, 0.0));
  auto box_instance =
      std::make_unique<GeometryInstance>(X_WB, std::move(box_mesh), "box 0");

  auto box_mesh1 = std::make_unique<Mesh>(box_vtk, 1.0);
  const RigidTransformd X_WB1(RollPitchYawd(0, 0, 0.0),
                              Vector3<double>(-1.5, 0.0, 0.0));
  auto box_instance1 =
      std::make_unique<GeometryInstance>(X_WB1, std::move(box_mesh1), "box 1");

  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  const CoulombFriction<double> surface_friction(0.0, 0.0);
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  box_instance->set_proximity_properties(deformable_proximity_props);
  box_instance1->set_proximity_properties(deformable_proximity_props);

  /* Registration of all deformable geometries ostensibly requires a resolution
   hint parameter that dictates how the shape is tesselated. In the case of a
   `Mesh` shape, the resolution hint is unused because the shape is already
   tessellated. */
  // TODO(xuchenhan-tri): Though unused, we still asserts the resolution hint is
  // positive. Remove the requirement of a resolution hint for meshed shapes.
  const auto id0 = owned_deformable_model->RegisterDeformableBody(
      std::move(box_instance), deformable_config, FLAGS_resolution_hint);
  const auto id1 = owned_deformable_model->RegisterDeformableBody(
      std::move(box_instance1), deformable_config, FLAGS_resolution_hint);
  const DeformableModel<double>* deformable_model =
      owned_deformable_model.get();
  plant.AddPhysicalModel(std::move(owned_deformable_model));

  /* All deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* It's essential to connect the vertex position port in DeformableModel to
   the source configuration port in SceneGraph when deformable bodies are
   present in the plant. */
  builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(diagram_context.get());
  SetVelocity(id0, *deformable_model, &plant_context, Vector3d(-0.1, 0, 0));
  SetVelocity(id1, *deformable_model, &plant_context, Vector3d(0.1, 0, 0));

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
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
