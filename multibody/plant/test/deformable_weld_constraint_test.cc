#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/deformable_driver.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

using drake::geometry::Box;
using drake::geometry::GeometryId;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::ProximityProperties;
using drake::geometry::SceneGraph;
using drake::geometry::SceneGraphInspector;
using drake::geometry::Sphere;
using drake::geometry::VolumeMesh;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::multibody::contact_solvers::internal::ContactSolverResults;
using drake::multibody::fem::FemState;
using drake::systems::Context;
using drake::systems::Simulator;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using std::make_unique;
using std::move;
using std::unique_ptr;

namespace drake {
namespace multibody {
namespace internal {

/* Provides access to a selection of private functions in
 CompliantContactManager for testing purposes. */
class CompliantContactManagerTester {
 public:
  static const DeformableDriver<double>* deformable_driver(
      const CompliantContactManager<double>& manager) {
    return manager.deformable_driver_.get();
  }
};

/* Deformable body parameters.  */
constexpr double kYoungsModulus = 1e5;      // unit: N/m²
constexpr double kPoissonsRatio = 0.4;      // unitless.
constexpr double kMassDensity = 1e3;        // unit: kg/m³
constexpr double kStiffnessDamping = 0.01;  // unit: s
/* Time step (seconds). */
constexpr double kDt = 1e-2;
const CoulombFriction<double> kFriction{0.4, 0.4};

/* Sets up a deformable simulation with a deformable octahedron centered at the
 origin of the world frame and a rigid slope welded to the world body. The setup
 is used to test the discrete updates for deformable bodies in a simulator.
 Run:
   bazel run //tools:meldis -- --open-window
   bazel run //multibody/plant:deformable_weld_constraint_test
 to visualize the test. */
class DeformableWeldConstraintTest : public ::testing::Test {
 protected:
  /* Sets up a scene with a deformable cube sitting on the ground. */
  void SetUp() override {
    systems::DiagramBuilder<double> builder;
    std::tie(plant_, scene_graph_) = AddMultibodyPlantSceneGraph(&builder, kDt);

    auto deformable_model = make_unique<DeformableModel<double>>(plant_);
    body_id_ = RegisterDeformableOctahedron(
        deformable_model.get(), RigidTransformd(Vector3d(0, 0.15, 0)),
        "deformable");
    model_ = deformable_model.get();
    plant_->AddPhysicalModel(move(deformable_model));
    plant_->set_discrete_contact_solver(DiscreteContactSolver::kSap);

    const auto& box_body = plant_->AddRigidBody(
        "rigid_box",
        SpatialInertia(1.0, Vector3d(0, 0, 0),
                       UnitInertia<double>::SolidBox(0.1, 0.1, 0.1)));

    spec_.body_A = body_id_;
    spec_.body_B = box_body.index();
    spec_.vertex_index = 4;
    spec_.p_BQ = Vector3d(0, 0.1, 0);

    model_->AddWeldConstraint(spec_.body_A, spec_.vertex_index, box_body,
                              spec_.p_BQ);

    /* Register a rigid geometry that serves as an inclined plane. */
    ProximityProperties proximity_prop;
    geometry::AddContactMaterial({}, {}, kFriction, &proximity_prop);
    geometry::AddCompliantHydroelasticProperties(1.0, 1e6, &proximity_prop);
    IllustrationProperties illustration_props;
    illustration_props.AddProperty("phong", "diffuse",
                                   Vector4d(0.7, 0.5, 0.4, 0.8));

    const Box box(0.2, 0.2, 0.2);
    plant_->RegisterCollisionGeometry(box_body, RigidTransformd::Identity(),
                                      box, "box_collision", proximity_prop);
    plant_->RegisterVisualGeometry(box_body, RigidTransformd::Identity(), box,
                                   "box_visual", illustration_props);

    const Box ground(2, 2, 2);
    const RigidTransformd X_WG(Vector3d(0, 0, -1.15));
    plant_->RegisterCollisionGeometry(plant_->world_body(), X_WG, ground,
                                      "ground_collision", proximity_prop);
    plant_->RegisterVisualGeometry(plant_->world_body(), X_WG, ground,
                                   "ground_visual", illustration_props);
    plant_->Finalize();

    auto contact_manager = make_unique<CompliantContactManager<double>>();
    manager_ = contact_manager.get();
    plant_->SetDiscreteUpdateManager(move(contact_manager));
    driver_ = CompliantContactManagerTester::deformable_driver(*manager_);
    /* Connect visualizer. Useful for when this test is used for debugging. */
    geometry::DrakeVisualizerd::AddToBuilder(&builder, *scene_graph_);

    builder.Connect(model_->vertex_positions_port(),
                    scene_graph_->get_source_configuration_port(
                        plant_->get_source_id().value()));

    diagram_ = builder.Build();
  }

  /* Calls DeformableDriver::EvalFemState(). */
  const FemState<double>& EvalFemState(const Context<double>& context,
                                       DeformableBodyIndex index) const {
    return driver_->EvalFemState(context, index);
  }

  SceneGraph<double>* scene_graph_{nullptr};
  MultibodyPlant<double>* plant_{nullptr};
  DeformableModel<double>* model_{nullptr};
  const CompliantContactManager<double>* manager_{nullptr};
  const DeformableDriver<double>* driver_{nullptr};
  unique_ptr<systems::Diagram<double>> diagram_{nullptr};
  DeformableBodyId body_id_;
  internal::DeformableRigidWeldConstraintSpecs spec_;

 private:
  DeformableBodyId RegisterDeformableOctahedron(DeformableModel<double>* model,
                                                const RigidTransformd& X_WD,
                                                std::string name) {
    auto geometry = make_unique<GeometryInstance>(
        X_WD, make_unique<Sphere>(0.1), move(name));
    ProximityProperties props;
    geometry::AddContactMaterial({}, {}, kFriction, &props);
    geometry->set_proximity_properties(move(props));
    fem::DeformableBodyConfig<double> body_config;
    body_config.set_youngs_modulus(kYoungsModulus);
    body_config.set_poissons_ratio(kPoissonsRatio);
    body_config.set_mass_density(kMassDensity);
    body_config.set_stiffness_damping_coefficient(kStiffnessDamping);
    /* Make the resolution hint large enough so that we get an octahedron. */
    constexpr double kRezHint = 10.0;
    DeformableBodyId id =
        model->RegisterDeformableBody(move(geometry), body_config, kRezHint);
    /* Verify that the geometry has 7 vertices and is indeed an octahedron. */
    const SceneGraphInspector<double>& inspector =
        scene_graph_->model_inspector();
    GeometryId g_id = model->GetGeometryId(id);
    const VolumeMesh<double>* mesh_G = inspector.GetReferenceMesh(g_id);
    DRAKE_DEMAND(mesh_G != nullptr);
    DRAKE_DEMAND(mesh_G->num_vertices() == 7);
    return id;
  }
};

namespace {

TEST_F(DeformableWeldConstraintTest, SteadyState) {
  Simulator<double> simulator(*diagram_);
  /* Run simulation for long enough to reach steady state. */
  simulator.AdvanceTo(5.0);

  /* Verify the system has reached steady state. */
  const Context<double>& diagram_context = simulator.get_context();
  const Context<double>& plant_context =
      plant_->GetMyContextFromRoot(diagram_context);
  const FemState<double>& fem_state =
      EvalFemState(plant_context, DeformableBodyIndex(0));
  constexpr double kVelocityThreshold = 1e-4;      // unit: m/s.
  constexpr double kAccelerationThreshold = 1e-5;  // unit: m/s².
  const VectorXd& v = fem_state.GetVelocities();
  EXPECT_TRUE(CompareMatrices(v, VectorXd::Zero(v.size()), kVelocityThreshold));
  const VectorXd& a = fem_state.GetAccelerations();
  EXPECT_TRUE(
      CompareMatrices(a, VectorXd::Zero(a.size()), kAccelerationThreshold));
  const Vector3d p_WP =
      model_->GetVertexPosition(plant_context, body_id_, 4);
  const math::RigidTransformd& X_WB = plant_->EvalBodyPoseInWorld(
      plant_context, plant_->get_body(spec_.body_B));
  Vector3d p_WQ = X_WB * spec_.p_BQ;
  const Vector3d p_PQ_W = p_WQ - p_WP;
  EXPECT_NEAR(p_PQ_W.norm(), 0.0, 1e-5);
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
