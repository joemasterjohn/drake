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

DEFINE_double(simulation_time, 20.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.0, "Desired real time rate.");
DEFINE_double(time_step, 1.0e-3,
              "Discrete time step for the system [s]. Must be positive.");
DEFINE_double(E, 1e4, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 100, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.005,
              "Stiffness damping coefficient for the deformable body [1/s].");
DEFINE_double(resolution_hint, 0.05, "rezhint");

// Parameters for squeezing the spatula.
DEFINE_double(gripper_force, 2,
              "The baseline force to be applied by the gripper. [N].");
DEFINE_double(amplitude, 8,
              "The amplitude of the oscillations "
              "carried out by the gripper. [N].");
DEFINE_double(duty_cycle, 0.666666, "Duty cycle of the control signal.");
DEFINE_double(period, 6, "Period of the control signal. [s].");
DEFINE_double(phase, 0, "Phase of the control signal. [s].");

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

// We create a simple leaf system that outputs a square wave signal for our
// open loop controller. The Square system here supports an arbitrarily
// dimensional signal, but we will use a 2-dimensional signal for our gripper.
class Square final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Square)

  // Constructs a %Square system where different amplitudes, duty cycles,
  // periods, and phases can be applied to each square wave.
  //
  // @param[in] amplitudes the square wave amplitudes. (unitless)
  // @param[in] duty_cycles the square wave duty cycles.
  //                        (ratio of pulse duration to period of the waveform)
  // @param[in] periods the square wave periods. (seconds)
  // @param[in] phases the square wave phases. (radians)
  Square(const Eigen::VectorXd& amplitudes, const Eigen::VectorXd& duty_cycles,
         const Eigen::VectorXd& periods, const Eigen::VectorXd& phases)
      : amplitude_(amplitudes),
        duty_cycle_(duty_cycles),
        period_(periods),
        phase_(phases) {
    // Ensure the incoming vectors are all the same size.
    DRAKE_THROW_UNLESS(duty_cycles.size() == amplitudes.size());
    DRAKE_THROW_UNLESS(duty_cycles.size() == periods.size());
    DRAKE_THROW_UNLESS(duty_cycles.size() == phases.size());

    this->DeclareVectorOutputPort("Square Wave Output", duty_cycles.size(),
                                  &Square::CalcValueOutput);
  }

 private:
  void CalcValueOutput(const Context<double>& context,
                       BasicVector<double>* output) const {
    Eigen::VectorBlock<VectorX<double>> output_block =
        output->get_mutable_value();

    const double time = context.get_time();

    for (int i = 0; i < duty_cycle_.size(); ++i) {
      // Add phase offset.
      double t = time + (period_[i] * phase_[i] / (2 * M_PI));

      output_block[i] =
          amplitude_[i] *
          (t - floor(t / period_[i]) * period_[i] < duty_cycle_[i] * period_[i]
               ? 1
               : 0);
    }
  }

  const Eigen::VectorXd amplitude_;
  const Eigen::VectorXd duty_cycle_;
  const Eigen::VectorXd period_;
  const Eigen::VectorXd phase_;
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
                                 Vector4d(0.7, 0.5, 0.4, 0.8));
  plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                               "ground_visual", std::move(illustration_props));

  // Parse the gripper model.
  Parser parser(&plant, &scene_graph);
  const std::string gripper_file = FindResourceOrThrow(
      "drake/examples/multibody/teddy/models/schunk_wsg_50.sdf");
  parser.AddModels(gripper_file);
  // Pose the gripper and weld it to the world.
  const math::RigidTransform<double> X_WF0 = math::RigidTransform<double>(
      math::RollPitchYaw(0.0, -1.57, 0.0), Eigen::Vector3d(0, 0, 0.25));
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("gripper"), X_WF0);

  const std::string spatula_file =
      FindResourceOrThrow("drake/examples/multibody/teddy/models/spatula.sdf");
  parser.AddModels(spatula_file);

  /* Set up a deformable sphere. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  DeformableModel<double>* deformable_model = owned_deformable_model.get();

  DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(FLAGS_E);
  deformable_config.set_poissons_ratio(FLAGS_nu);
  deformable_config.set_mass_density(FLAGS_density);
  deformable_config.set_stiffness_damping_coefficient(FLAGS_beta);

  const std::string teddy_vtk =
      FindResourceOrThrow("drake/examples/multibody/teddy/teddy.vtk");
  auto teddy_mesh = std::make_unique<Mesh>(teddy_vtk, 0.1);
  const RigidTransformd X_WB(RollPitchYawd(1.57, 0, 0), Vector3d(0, 0, 0));
  auto teddy_instance =
      std::make_unique<GeometryInstance>(X_WB, std::move(teddy_mesh), "teddy");
  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  ProximityProperties deformable_proximity_props;
  AddContactMaterial({}, {}, surface_friction, &deformable_proximity_props);
  teddy_instance->set_proximity_properties(deformable_proximity_props);

  //   deformable_model->RegisterDeformableBody(std::move(teddy_instance),
  //                                            deformable_config, 1.0);

  const std::string bubble_vtk =
      FindResourceOrThrow("drake/examples/multibody/teddy/bubble.vtk");
  auto left_bubble_mesh = std::make_unique<Mesh>(bubble_vtk);
  const RigidTransformd X_FB1(RollPitchYawd(0, 1.5708, -1.5708),
                              Vector3d(0.0, -0.05, -0.182));
  const RigidTransformd X_WB1 = X_WF0 * X_FB1;
  auto left_bubble_instance = std::make_unique<GeometryInstance>(
      X_WB1, std::move(left_bubble_mesh), "left bubble");
  left_bubble_instance->set_proximity_properties(deformable_proximity_props);
  DeformableBodyId left_bubble_id = deformable_model->RegisterDeformableBody(
      std::move(left_bubble_instance), deformable_config, 1.0);
  const Body<double>& left_finger = plant.GetBodyByName("left_finger_bubble");
  deformable_model->Weld(left_bubble_id, left_finger, X_WB1,
                         RigidTransformd(RollPitchYawd(0, -1.57, 0),
                                         Vector3d(0.0725, -0.02, 0.249942)));

  auto right_bubble_mesh = std::make_unique<Mesh>(bubble_vtk);
  const RigidTransformd X_FB2(RollPitchYawd(0, 1.5708, 1.5708),
                              Vector3d(0.0, 0.05, -0.182));
  const RigidTransformd X_WB2 = X_WF0 * X_FB2;
  auto right_bubble_instance = std::make_unique<GeometryInstance>(
      X_WB2, std::move(right_bubble_mesh), "right bubble");
  right_bubble_instance->set_proximity_properties(deformable_proximity_props);
  DeformableBodyId right_bubble_id = deformable_model->RegisterDeformableBody(
      std::move(right_bubble_instance), deformable_config, 1.0);
  const Body<double>& right_finger = plant.GetBodyByName("right_finger_bubble");
  deformable_model->Weld(right_bubble_id, right_finger, X_WB2,
                         RigidTransformd(RollPitchYawd(0, -1.57, 0),
                                         Vector3d(0.0725, 0.02, 0.249942)));

  plant.AddPhysicalModel(std::move(owned_deformable_model));

  const PrismaticJoint<double>& right_joint =
      plant.GetJointByName<PrismaticJoint>("right_finger_sliding_joint");
  const PrismaticJoint<double>& left_joint =
      plant.GetJointByName<PrismaticJoint>("left_finger_sliding_joint");
  plant.AddCouplerConstraint(left_joint, right_joint, -1.0);

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* It's essential to connect the vertex position port in DeformableModel to
   the source configuration port in SceneGraph when deformable bodies are
   present in the plant. */
  builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  /* Construct the open loop square wave controller. To oscillate around a
   constant force, we construct a ConstantVectorSource and combine it with
   the square wave output using an Adder. */
  const double f0 = FLAGS_gripper_force;

  const Eigen::Vector2d amplitudes(FLAGS_amplitude, -FLAGS_amplitude);
  const Eigen::Vector2d duty_cycles(FLAGS_duty_cycle, FLAGS_duty_cycle);
  const Eigen::Vector2d periods(FLAGS_period, FLAGS_period);
  const Eigen::Vector2d phases(FLAGS_phase, FLAGS_phase);
  const auto& square_force =
      *builder.AddSystem<Square>(amplitudes, duty_cycles, periods, phases);
  const auto& constant_force =
      *builder.AddSystem<systems::ConstantVectorSource<double>>(
          Eigen::Vector2d(f0, -f0));
  const auto& adder = *builder.AddSystem<systems::Adder<double>>(2, 2);
  builder.Connect(square_force.get_output_port(), adder.get_input_port(0));
  builder.Connect(constant_force.get_output_port(), adder.get_input_port(1));

  /* Connect the output of the adder to the plant's actuation input. */
  builder.Connect(adder.get_output_port(0), plant.get_actuation_input_port());

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

  // Set spatula's free body pose.
  const math::RigidTransform<double> X_WF1 = math::RigidTransform<double>(
      math::RollPitchYaw(-0.4, 0.0, 1.57), Eigen::Vector3d(0.35, 0, 0.26));
  const auto& base_link = plant.GetBodyByName("spatula");
  plant.SetFreeBodyPose(&plant_context, base_link, X_WF1);

  // Set finger joint positions.
  left_joint.set_translation(&plant_context, -0.02);
  right_joint.set_translation(&plant_context, 0.02);

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
