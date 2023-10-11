#include <gtest/gtest.h>

#include <chrono>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/compute_numerical_gradient.h"
#include "drake/math/linear_solve.h"
#include "drake/multibody/contact_solvers/sap/sap_solver_results.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/test_utilities/rigid_body_on_compliant_ground.h"

constexpr double kEps = std::numeric_limits<double>::epsilon();

namespace drake {
namespace multibody {
namespace internal {

using namespace std::chrono;

class SapDriverTest {
 public:
  static const contact_solvers::internal::SapSolverResults<AutoDiffXd>&
  EvalSapSolverResults(const SapDriver<AutoDiffXd>& driver,
                       const Context<AutoDiffXd>& context) {
    return driver.EvalSapSolverResults(context);
  }

  static const internal::ContactProblemCache<AutoDiffXd>&
  EvalContactProblemCache(const SapDriver<AutoDiffXd>& driver,
                          const Context<AutoDiffXd>& context) {
    return driver.EvalContactProblemCache(context);
  }
};

void ValidateValueAndGradients(const VectorX<AutoDiffXd>& x,
                               const VectorX<double>& value_expected,
                               const MatrixX<double>& gradient_expected,
                               const double tolerance = kEps) {
  const VectorXd value = math::ExtractValue(x);
  const auto gradient = math::ExtractGradient(x);
  EXPECT_TRUE(CompareMatrices(value, value_expected, tolerance,
                              MatrixCompareType::relative));
  EXPECT_TRUE(CompareMatrices(gradient, gradient_expected, tolerance,
                              MatrixCompareType::relative));
}

// This test verifies gradients in the equilibrium state.
TEST_P(RigidBodyOnCompliantGround, GradientsOfGammaEquilibrium) {
  const VectorXd& x0 = plant_->GetPositionsAndVelocities(*plant_context_);
  VectorX<AutoDiffXd> x0_ad = math::InitializeAutoDiff(x0);
  plant_ad_->SetPositionsAndVelocities(plant_context_ad_, x0_ad);

  const contact_solvers::internal::SapSolverResults<AutoDiffXd>& results =
      SapDriverTest::EvalSapSolverResults(*sap_driver_ad_, *plant_context_ad_);

  // yz = -dt*(k*(phi_0 + dt*vz) + k*tau_d*vz)
  // yt = -vx / (sigma * wi)
  //
  // Value:
  //   vz = 0 => yz = -dt*k*phi0
  //
  // phi_0 = z0 - radius
  //
  // Gradient:
  //   dyz/z0 = -dt*k
  //
  //   dyz/dvz0 = dy/dx0 = dy/dvx0 = 0
  //   dyt/d* = 0
  const double phi0 = x0[0] - kPointContactSphereRadius_;
  const Vector3d gamma_expected{0, 0, -kTimeStep_ * kStiffness_ * phi0};
  // clang-format off
  const MatrixX<double> dgamma_dx0_expected =
      (MatrixX<double>(3, 4) <<                         0, 0, 0, 0,
                                                        0, 0, 0, 0,
                                -kTimeStep_ * kStiffness_, 0, 0, 0)
      .finished();
  // clang-format on

  ValidateValueAndGradients(results.gamma, gamma_expected, dgamma_dx0_expected);
}

// This test verifies gradients in the equilibrium state.
TEST_P(RigidBodyOnCompliantGround, GradientsOfStateEquilibrium) {
  const VectorXd& x0 = plant_->GetPositionsAndVelocities(*plant_context_);
  VectorX<AutoDiffXd> x0_ad = math::InitializeAutoDiff(x0);
  plant_ad_->SetPositionsAndVelocities(plant_context_ad_, x0_ad);

  systems::DiscreteValues<AutoDiffXd> values_ad(
      std::make_unique<systems::BasicVector<AutoDiffXd>>(4));
  MultibodyPlantTester::CalcDiscreteStep(*plant_ad_, *plant_context_ad_,
                                         &values_ad);
  const VectorX<AutoDiffXd>& x_ad = values_ad.value();

  // For a 2-dof point mass in contact with a plane, with state (z, x, vz, vx)
  // the balance of momentum for the point in equilibrium is:
  //
  // Normal component:
  //   m * (vz - vz*) = gz
  //   m * (vz - (vz0 + dt*g)) = -dt*k*((phi0 + dt*vz) + tau_d*vz)
  //   vz = vz0 + dt*g + (-k*dt/m) * (phi0 + (dt + tau_d)*vz)
  //
  // Where phi0 = z0 - radius
  //
  //   dvz/dz0 = (-k*dt/m) * (1 + (dt + tau_d)*dvz/dz0)
  //   dvz/dz0 + (k*dt*(dt + tau_d)/m)*dvz/dz0 = (-k*dt/m)
  //   dvz/dz0 = (-k*dt/m) / (1 + (k*dt*(dt + tau_d)/m))
  //   dvz/dz0 = (-k*dt) / (m + k*dt*(dt + tau_d))
  //
  //   dvz/dvz0 = 1 + (-k*dt/m)*(dt + tau_d)*dvz/dv0
  //   dvz/dvz0 + (k*dt*(dt + tau_d)/m)*dvz/dv0 = 1
  //   dvz/dvz0 = 1 / (1 + k*dt*(dt + tau_d)/m)
  //   dvz/dvz0 = m / (m + k*dt*(dt + tau_d))
  //
  // From the discrete update:
  //   z = z0 + dt*vz
  //
  //   dz/dz0 = 1 + dt*dvz/dz0
  //   dz/dvz0 = dt*dvz/dvz0
  //
  //   dz/dx0 = dz/dvx0 = dvz/dx0 = dvz/dvx0 = 0
  //
  // Tangential component:
  //   m * (vx - vx*) = gx
  //   m * (vx - vx0) = -vx / (sigma * wt)
  //   vx = vx0 - vx/(m*sigma*wt)
  //
  //   dvx/dx0 = 0
  //   dvx/dvx0 = 1 - dv/dvx0 / (m*sigma*wt)
  //   dvx/dvx0 = 1 / (1 + 1/(m*sigma*wt))
  //   dvx/dvx0 = (m*sigma*wt) / (m*sigma*wt + 1)
  //
  // From the discrete update:
  //   x = x0 + dt+vx
  //
  //   dx/dx0 = 1 + dt*dvx/dx0
  //   dx/dx0 = 1
  //
  //   dx/dvx0 = dt * dvx/dvx0
  //
  //   dx/dz0 = dx/dvz0 = dvx/dz0 = dvx/dvz0 = 0
  const double dvz_dz0 =
      (-kStiffness_ * kTimeStep_) /
      (kMass_ + kStiffness_ * kTimeStep_ * (kTimeStep_ + kRelaxationTime_));
  const double dvz_dvz0 =
      kMass_ /
      (kMass_ + (kStiffness_ * kTimeStep_ * (kTimeStep_ + kRelaxationTime_)));
  const double dz_dz0 = 1 + kTimeStep_ * dvz_dz0;
  const double dz_dvz0 = kTimeStep_ * dvz_dvz0;

  const double dz_dx0 = 0, dz_dvx0 = 0, dvz_dx0 = 0, dvz_dvx0 = 0;

  // Comes from internal knowledge of the Delassus diagonal estimation code.
  const double wt = std::sqrt(2 / (kMass_ * kMass_)) / 3;
  // Comes from internal knowledge of SapDriver.
  const double sigma = 1e-3;

  const double dvx_dx0 = 0;
  const double dvx_dvx0 = (kMass_ * sigma * wt) / (kMass_ * sigma * wt + 1);
  const double dx_dx0 = 1;
  const double dx_dvx0 = kTimeStep_ * dvx_dvx0;

  const double dx_dz0 = 0, dx_dvz0 = 0, dvx_dz0 = 0, dvx_dvz0 = 0;
  // clang-format off
  const MatrixX<double> dx_dx0_expected =
      (MatrixX<double>(4, 4) <<  dz_dz0,  dz_dx0,  dz_dvz0,  dz_dvx0,
                                 dx_dz0,  dx_dx0,  dx_dvz0,  dx_dvx0,
                                dvz_dz0, dvz_dx0, dvz_dvz0, dvz_dvx0,
                                dvx_dz0, dvx_dx0, dvx_dvz0, dvx_dvx0)
      .finished();
  // clang-format on

  ValidateValueAndGradients(x_ad, math::ExtractValue(x_ad), dx_dx0_expected);
}

// This test verifies gradients in the slip state.
TEST_P(RigidBodyOnCompliantGround, GradientsOfGammaSlip) {
  ApplyTangentialForceForBodyInSlip();

  const VectorXd& x0 = plant_->GetPositionsAndVelocities(*plant_context_);
  VectorX<AutoDiffXd> x0_ad = math::InitializeAutoDiff(x0);
  plant_ad_->SetPositionsAndVelocities(plant_context_ad_, x0_ad);

  const contact_solvers::internal::SapSolverResults<AutoDiffXd>& results =
      SapDriverTest::EvalSapSolverResults(*sap_driver_ad_, *plant_context_ad_);

  // const internal::ContactProblemCache<AutoDiffXd>& problem_cache =
  //     SapDriverTest::EvalContactProblemCache(*sap_driver_ad_,
  //                                            *plant_context_ad_);

  // drake::log()->info(fmt::format("R_WC: \n{}",
  // problem_cache.R_WC[0].matrix()));

  // drake::log()->info(
  //     fmt::format("J: \n{}", problem_cache.sap_problem->get_constraint(0)
  //                                .first_clique_jacobian()
  //                                .MakeDenseMatrix()));

  // yz = -dt*k*(phi_0 + (dt + tau_d)*vz)
  // yx = -vx / (sigma * wi)
  //
  // yr = SoftNorm(yt) = sqrt(yx*yx + eps)
  // We estimate this using yr = |yx|
  //
  // yr = |yx|
  // t_hat = yt / yr = yx / |yx|
  // mu_tilde = mu * sqrt(sigma * wt * dt * k * (dt + tau_d))
  // mu_hat = mu * sigma * wt * dt * k * (dt + tau_d)
  // f = 1 / (1 + mu_tilde*mu_tilde)
  //
  // gz = (yz + mu_hat*yr)*f
  // gx = mu * gz * t_hat
  //
  // phi_0 = z0 - radius
  //
  // Value:
  //   gz = (-dt*k*(phi0 + (dt + tau_d)*vz) + mu_hat*|vx|/(sigma*wt))*f
  //   gx = mu * gz * t_hat
  //
  // Gradient:
  //   dgz/dz0 = -dt*k*f
  //   dgx/dz0 = mu*dgz/dz0*t_hat
  //   dgz/dx0 = 0
  //   dgx/dx0 = 0
  //   dgz/dvz0 = dgz/dvx0 = dgx/dvz0 = dgx/dvx0 = 0

  // Comes from internal knowledge of the Delassus diagonal estimation code.
  const double wt = std::sqrt(2 / (kMass_ * kMass_)) / 3;
  // Comes from internal knowledge of SapDriver.
  const double sigma = 1e-3;

  const double s =
      sigma * wt * kTimeStep_ * kStiffness_ * (kTimeStep_ + kRelaxationTime_);
  const double mu_hat = kMu_ * s;
  const double mu_tilde = kMu_ * sqrt(s);
  const double f = 1 / (1 + mu_tilde * mu_tilde);

  const double phi0 = x0[0] - kPointContactSphereRadius_;
  const double vz = results.vc[2].value();
  const double vx = results.vc[1].value();

  const double yx = -vx / (sigma * wt);
  const double yr = std::abs(yx);
  const double t_hat = yx / yr;

  const double gz = f * (-kTimeStep_ * kStiffness_ *
                             (phi0 + (kTimeStep_ + kRelaxationTime_) * vz) +
                         mu_hat * yr);
  const double gx = kMu_ * gz * t_hat;

  const Vector3d gamma_expected{0, gx, gz};

  const double dgz_dz0 = -kTimeStep_ * kStiffness_ * f;
  const double dgx_dz0 = kMu_ * dgz_dz0 * t_hat;

  // clang-format off
  const MatrixX<double> dgamma_dx0_expected =
      (MatrixX<double>(3, 4) <<       0, 0, 0, 0,
                                dgx_dz0, 0, 0, 0,
                                dgz_dz0, 0, 0, 0)
      .finished();
  // clang-format on

  ValidateValueAndGradients(results.gamma, gamma_expected, dgamma_dx0_expected);
}

// This test verifies gradients in the slip state.
TEST_P(RigidBodyOnCompliantGround, GradientsOfStateSlip) {
  ApplyTangentialForceForBodyInSlip();

  const VectorXd& x0 = plant_->GetPositionsAndVelocities(*plant_context_);
  VectorX<AutoDiffXd> x0_ad = math::InitializeAutoDiff(x0);
  plant_ad_->SetPositionsAndVelocities(plant_context_ad_, x0_ad);

  // {
  //   const internal::ContactProblemCache<AutoDiffXd>& problem_cache =
  //       SapDriverTest::EvalContactProblemCache(*sap_driver_ad_,
  //                                              *plant_context_ad_);

  //   drake::log()->info(
  //       fmt::format("R_WC: \n{}", problem_cache.R_WC[0].matrix()));

  //   drake::log()->info(
  //       fmt::format("J: \n{}", problem_cache.sap_problem->get_constraint(0)
  //                                  .first_clique_jacobian()
  //                                  .MakeDenseMatrix()));
  // }

  systems::DiscreteValues<AutoDiffXd> values_ad(
      std::make_unique<systems::BasicVector<AutoDiffXd>>(4));
  MultibodyPlantTester::CalcDiscreteStep(*plant_ad_, *plant_context_ad_,
                                         &values_ad);
  const VectorX<AutoDiffXd>& x_ad = values_ad.value();

  plant_ad_->SetPositionsAndVelocities(plant_context_ad_, x_ad);
  const contact_solvers::internal::SapSolverResults<AutoDiffXd>& results =
      SapDriverTest::EvalSapSolverResults(*sap_driver_ad_, *plant_context_ad_);

  // {
  //   const internal::ContactProblemCache<AutoDiffXd>& problem_cache =
  //       SapDriverTest::EvalContactProblemCache(*sap_driver_ad_,
  //                                              *plant_context_ad_);

  //   drake::log()->info(
  //       fmt::format("R_WC: \n{}", problem_cache.R_WC[0].matrix()));
  //   drake::log()->info(
  //       fmt::format("J: \n{}", problem_cache.sap_problem->get_constraint(0)
  //                                  .first_clique_jacobian()
  //                                  .MakeDenseMatrix()));
  // }

  // clang-format off
  // Sliding Impulse:
  //
  //   yz = -dt*k*(phi_0 + (dt + tau_d)*vz)
  //   yx = -vx / (sigma * wi)
  //
  //   yr = SoftNorm(yx) = |yx| (by approximation)
  //   t_hat = yx / |yx| = sgn(yx)
  //   mu_tilde = mu * sqrt(sigma * wt * dt * k * (dt + tau_d))
  //   mu_hat = mu * sigma * wt * dt * k * (dt + tau_d)
  //   f = 1 / (1 + mu_tilde*mu_tilde)
  //
  //   gz = (yz + mu_hat*yr)*f
  //   gx = mu * gz * t_hat
  //
  // Where phi_0 = z0 - radius
  //
  //   gz = f * (-dt*k*(phi0 + (dt + tau_d)*vz) + mu_hat*|-vx|/(sigma*wt))
  //   gx = mu * gz * t_hat
  //
  // For a 2-dof point mass in contact with a plane, with state (z, x, vz, vx)
  // the balance of momentum for the point in slip is:
  //
  // Normal component:
  //   m * (vz - vz*) = gz
  //   m * (vz - (vz0 + dt*g)) = f*(-dt*k*(phi0 + (dt + tau_d)*vz) + mu_hat*|-vx|/(sigma*wt))
  //   vz = vz0 + dt*g + (f/m)*(-dt*k*(phi0 + (dt + tau_d)*vz) + mu_hat*|-vx|/(sigma*wt))
  //
  // Tangential component:
  //   m * (vx - vx*) = gx
  //   m * (vx - (vx0 + dt*(1.05*mu*g))) = mu*t_hat*f*(-dt*k*(phi0 + (dt + tau_d)*vz) + mu_hat*|-vx|/(sigma*wt))
  //   vx = vx0 + dt*(1.05*mu*g) + (mu*t_hat*f/m)*(-dt*k*(phi0 + (dt + tau_d)*vz) + mu_hat*|-vx|/(sigma*wt))
  //
  // Taking derivatives wrt x0:
  //
  // Using d|-vx|_dx0 = (-vx/|-vx|)*d(-vx)_dx0 = -sgn(-vx)*dvx_dx0 = -t_hat*dvx_dx0
  //
  //  dvz_dx0 = dvz0_dx0 + (f/m)*(-dt*k*(dphi0_dx0 + (dt + tau_d)*dvz_dx0) + (-t_hat*mu_hat/(sigma*wt))*dvx_dx0)
  //  dvx_dx0 = dvx0_dx0 + (mu*t_hat*f/m)*(-dt*k*(dphi0_dx0 + (dt + tau_d)*dvz_dx0) + (-t_hat*mu_hat/(sigma*wt))*dvx_dx0)
  //
  // We can rearrange these two equations into the system:
  //
  // [     1 + (f*dt*k*(dt + tau_d))/m                 (f*t_hat*mu_hat)/(m*sigma*wt)] [dvz_dx0] = [dvz0_dx0 - (f*dt*k/m)*dphi0_dx0]
  // [(mu*t_hat*f*dt*k*(dt + tau_d))/m    1 + (mu*t_hat*f*t_hat*mu_hat)/(m*sigma*wt)] [dvx_dx0] = [dvx0_dx0 - ((mu*t_hat*f*dt*k)/m)*dphi0_dx0]
  //
  // A*dv_dx0 = db_dx0
  // clang-format on

  // Comes from internal knowledge of the Delassus diagonal estimation code.
  const double wt = std::sqrt(2 / (kMass_ * kMass_)) / 3;
  // Comes from internal knowledge of SapDriver.
  const double sigma = 1e-3;

  // The contact kinematics may sometimes choose a frame such that vt = -vx,
  // but we can express everything analytically using vx itself.
  const double vx = results.v[1].value();
  const double yx = -vx / (sigma * wt);
  const double yr = std::abs(yx);
  const double t_hat = yx / yr;

  const double s =
      sigma * wt * kTimeStep_ * kStiffness_ * (kTimeStep_ + kRelaxationTime_);
  const double mu_hat = kMu_ * s;
  const double mu_tilde = kMu_ * sqrt(s);
  const double f = 1 / (1 + mu_tilde * mu_tilde);

  const double mu = kMu_;
  const double dt = kTimeStep_;
  const double k = kStiffness_;
  const double tau_d = kRelaxationTime_;
  const double m = kMass_;

  // clang-format off
  const MatrixX<double> A = (MatrixX<double>(2, 2) <<
       1 + f*dt*k*(dt + tau_d)/m,                (f*t_hat*mu_hat)/(m*sigma*wt),
    mu*t_hat*f*dt*k*(dt+tau_d)/m,  (1 + (mu*t_hat*f*t_hat*mu_hat)/(m*sigma*wt))
  ).finished();

  const MatrixX<double> db_dx0 = (MatrixX<double>(2, 4) <<
             -f*dt*k/m, 0, 1, 0,
    -mu*t_hat*f*dt*k/m, 0, 0, 1
  ).finished();
  // clang-format on

  const math::LinearSolver<Eigen::LDLT, MatrixX<double>> A_ldlt(A);
  const MatrixX<double> dv_dx0 = A_ldlt.Solve(db_dx0);

  // From the discrete update:
  //   z = z0 + dt*vz
  //   x = x0 + dt*vx
  //
  //   dz/dz0  = 1 + dt*dvz/dz0
  //   dz/dx0  = dt*dvz/dx0
  //   dz/dvz0 = dt*dvz/dvz0
  //   dz/dvx0 = dt*dvz/dvx0
  //
  //   dx/dz0  = dt*dvx/dz0
  //   dx/dx0  = 1 + dt*dvx/dx0
  //   dx/dvz0 = dt*dvx/dvz0
  //   dx/dvx0 = dt*dvx/dvx0

  MatrixX<double> dq_dx0 = dt * dv_dx0;
  dq_dx0(0, 0) += 1;
  dq_dx0(1, 1) += 1;

  const MatrixX<double> dx_dx0_expected =
      (MatrixX<double>(4, 4) << dq_dx0, dv_dx0).finished();

  // Because of the approximation of SoftNorm, we expect a slight error about
  // the size of soft_tolerance = 1e-7.
  ValidateValueAndGradients(x_ad, math::ExtractValue(x_ad), dx_dx0_expected,
                            1e-7);
}

// Setup test cases using point and hydroelastic contact.
std::vector<ContactTestConfig> MakeTestCases() {
  return std::vector<ContactTestConfig>{
      {.description = "PointContact",
       .point_contact = true,
       .contact_model = ContactModel::kPoint,
       .contact_solver = DiscreteContactSolver::kSap},
  };
}

INSTANTIATE_TEST_SUITE_P(SapDriverGradientsTest, RigidBodyOnCompliantGround,
                         testing::ValuesIn(MakeTestCases()),
                         testing::PrintToStringParamName());

struct SapDriverNumericalGradientsTestConfig {
  // This is a gtest test suffix; no underscores or spaces.
  std::string description;
  VectorX<double> x0;
  double sovler_rel_tolerance;
  double expected_error;
  double perturbation;
  DiscreteContactSolver solver{DiscreteContactSolver::kSap};
};

// This provides the suffix for each test parameter: the test config
// description.
std::ostream& operator<<(std::ostream& out,
                         const SapDriverNumericalGradientsTestConfig& c) {
  out << c.description;
  return out;
}

class SapDriverNumericalGradientsTest
    : public ::testing::TestWithParam<SapDriverNumericalGradientsTestConfig> {
 public:
  void SetUp() override {
  const SapDriverNumericalGradientsTestConfig& config = GetParam();

    const std::string kArmSdfPath = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        "iiwa7_with_sphere_collision.sdf");

    DiagramBuilder<double> builder;
    auto items = AddMultibodyPlantSceneGraph(&builder, kTimeStep_);
    plant_ = &items.plant;

    Parser parser(plant_);
    parser.AddModels(kArmSdfPath);

    // Ground geometry.
    geometry::ProximityProperties ground_props;
    geometry::AddContactMaterial(kHcDissipation_, kStiffness_,
                                 CoulombFriction<double>(kMu_, kMu_),
                                 &ground_props);
    ground_props.AddProperty(geometry::internal::kMaterialGroup,
                             geometry::internal::kRelaxationTime,
                             kRelaxationTime_ / 2);
    geometry::AddCompliantHydroelasticPropertiesForHalfSpace(
        kGroundThickness_, kHydroelasticModulus_, &ground_props);
    plant_->RegisterCollisionGeometry(
        plant_->world_body(),
        geometry::HalfSpace::MakePose(Vector3d::UnitZ(), Vector3d::Zero()),
        geometry::HalfSpace(), "ground_collision", ground_props);

    plant_->WeldFrames(plant_->world_frame(),
                       plant_->GetBodyByName("iiwa_link_0").body_frame(),
                       math::RigidTransformd::Identity());

    plant_->set_discrete_contact_solver(config.solver);
    plant_->set_sap_near_rigid_threshold(0.0);
    plant_->Finalize();

    diagram_ = builder.Build();

    context_ = diagram_->CreateDefaultContext();
    plant_context_ =
        &diagram_->GetMutableSubsystemContext(*plant_, context_.get());

    // Scalar-convert the model and create a default context for it.
    diagram_ad_ =
        dynamic_pointer_cast<Diagram<AutoDiffXd>>(diagram_->ToAutoDiffXd());
    plant_ad_ = static_cast<const MultibodyPlant<AutoDiffXd>*>(
        &(diagram_ad_->GetSubsystemByName(plant_->get_name())));
    context_ad_ = diagram_ad_->CreateDefaultContext();
    context_ad_->SetTimeStateAndParametersFrom(*context_);
    plant_context_ad_ =
        &diagram_ad_->GetMutableSubsystemContext(*plant_ad_, context_ad_.get());

    // Fix input ports.
    const VectorX<double> tau =
        VectorX<double>::Zero(plant_->num_actuated_dofs());
    const VectorX<AutoDiffXd> tau_ad = tau;
    plant_->get_actuation_input_port().FixValue(plant_context_, tau);
    plant_ad_->get_actuation_input_port().FixValue(plant_context_ad_, tau_ad);

    if (config.solver == DiscreteContactSolver::kSap) {
      // When using the SAP solver, the solver convergence tolerance must be set
      // accordingly to the level of high accuracy used in these tests, dictated
      // by the fixture's parameter sovler_rel_tolerance.
      auto& manager = MultibodyPlantTester::manager(*plant_);
      auto& manager_ad = MultibodyPlantTester::manager(*plant_ad_);
      contact_solvers::internal::SapSolverParameters sap_parameters;
      sap_parameters.rel_tolerance = config.sovler_rel_tolerance;
      manager.set_sap_solver_parameters(sap_parameters);
      manager_ad.set_sap_solver_parameters(sap_parameters);
    }
  }

  void SetState(const VectorX<double>& x) {
    plant_->SetPositionsAndVelocities(plant_context_, x);
  }

  void SetState(const VectorX<AutoDiffXd>& x) {
    plant_ad_->SetPositionsAndVelocities(plant_context_ad_, x);
  }

  void CalcNextState(const VectorX<double>& x0, VectorX<double>* x) {
    SetState(x0);
    systems::DiscreteValues<double> values(
        std::make_unique<systems::BasicVector<double>>(
            plant_->num_positions() + plant_->num_velocities()));
    MultibodyPlantTester::CalcDiscreteStep(*plant_, *plant_context_, &values);
    *x = values.value();
  }

  void CalcNextState(const VectorX<AutoDiffXd>& x0, VectorX<AutoDiffXd>* x) {
    SetState(x0);
    systems::DiscreteValues<AutoDiffXd> values(
        std::make_unique<systems::BasicVector<AutoDiffXd>>(
            plant_ad_->num_positions() + plant_ad_->num_velocities()));
    MultibodyPlantTester::CalcDiscreteStep(*plant_ad_, *plant_context_ad_,
                                           &values);
    *x = values.value();
  }

 protected:
  MultibodyPlant<double>* plant_{nullptr};
  std::unique_ptr<Diagram<double>> diagram_{nullptr};
  std::unique_ptr<Context<double>> context_{nullptr};
  Context<double>* plant_context_{nullptr};

  // AutoDiffXd model to compute automatic derivatives:
  std::unique_ptr<Diagram<AutoDiffXd>> diagram_ad_{nullptr};
  const MultibodyPlant<AutoDiffXd>* plant_ad_{nullptr};
  std::unique_ptr<Context<AutoDiffXd>> context_ad_{nullptr};
  Context<AutoDiffXd>* plant_context_ad_{nullptr};

  // Parameters of the problem.
  const double kTimeStep_{0.001};  // Discrete time step of the plant.
  const double kGravity_{10.0};    // Acceleration of gravity, in m/s².
  const double kMass_{10.0};       // Mass of the rigid body, in kg.
  const double kPointContactSphereRadius_{0.02};  // In m.
  const double kStiffness_{1.0e4};                // In N/m.
  const double kHydroelasticModulus_{250.0};      // In Pa.
  const double kHcDissipation_{0.2};              // In s/m.
  const double kGroundThickness_{0.1};            // In m.
  const double kMu_{0.5};                         // Coefficient of friction.
  const double kRelaxationTime_{0.1};             // In s.
};

// Setup test cases using point and hydroelastic contact.
std::vector<SapDriverNumericalGradientsTestConfig>
MakeNumericalGradientsTestCases() {
  return std::vector<SapDriverNumericalGradientsTestConfig>{
      {
          .description = "NoContactSAP",
          .x0 = VectorX<double>::Zero(14),
          .sovler_rel_tolerance = 1e-15,
          .expected_error = 1e-8,
          .perturbation = 1e-15,
          .solver = DiscreteContactSolver::kSap,
      },
      {
          .description = "OneContactStictionSAP",
          .x0 = (VectorX<double>(14) << 0, 1.17, 0, -1.33, 0, 0.58, 0,  // q
                                        0,    0, 0,     0, 0,    0, 0   // v
                 )
                    .finished(),
          .sovler_rel_tolerance = 1e-15,
          .expected_error = 1e-4,
          .perturbation = 1e-15,
          .solver = DiscreteContactSolver::kSap,
      },
      {
          .description = "OneContactSlipSAP",
          .x0 = (VectorX<double>(14) << 0, 1.17, 0, -1.33, 0, 0.58, 0,  // q
                                        0, -0.1, 0,  -0.2, 0,    0, 0   // v
                 )
                    .finished(),
          .sovler_rel_tolerance = 1e-15,
          .expected_error = 1e-4,
          .perturbation = 1e-15,
          .solver = DiscreteContactSolver::kSap,
      },
  };
}

INSTANTIATE_TEST_SUITE_P(SapDriverGradientsTest,
                         SapDriverNumericalGradientsTest,
                         testing::ValuesIn(MakeNumericalGradientsTestCases()),
                         testing::PrintToStringParamName());

TEST_P(SapDriverNumericalGradientsTest, CompareNumericalGradients) {
  const SapDriverNumericalGradientsTestConfig& config = GetParam();

  const VectorX<double> x0 = config.x0;
  VectorX<double> x(x0.size());

  auto start = high_resolution_clock::now();
  CalcNextState(x0, &x);
  auto duration = duration_cast<microseconds>(high_resolution_clock::now() - start);
  drake::log()->info("Discrete Update<double>: {}s", duration.count() / 1e6);

  std::function<void(const VectorX<double>& x, VectorX<double>*)> next_state =
      [&](const VectorX<double>& X, VectorX<double>* Y) -> void {
    SetState(X);
    systems::DiscreteValues<double> values(
        std::make_unique<systems::BasicVector<double>>(
            plant_->num_positions() + plant_->num_velocities()));
    MultibodyPlantTester::CalcDiscreteStep(*plant_, *plant_context_, &values);
    *Y = values.value();
  };

  start = high_resolution_clock::now();
  MatrixX<double> dx_dx0 = math::ComputeNumericalGradient(
      next_state, x0,
      math::NumericalGradientOption(math::NumericalGradientMethod::kCentral,
                                    config.perturbation));
  duration = duration_cast<microseconds>(high_resolution_clock::now() - start);
  drake::log()->info("Finite Differences: {}s", duration.count() / 1e6);

  const VectorX<AutoDiffXd> x0_ad = math::InitializeAutoDiff(x0);
  VectorX<AutoDiffXd> x_ad(x0_ad.size());

  start = high_resolution_clock::now();
  CalcNextState(x0_ad, &x_ad);
  duration = duration_cast<microseconds>(high_resolution_clock::now() - start);
  drake::log()->info("Discrete Update<AutoDiffXd>: {}s", duration.count() / 1e6);

  ValidateValueAndGradients(x_ad, x, dx_dx0, config.expected_error);
}

}  // namespace drake
}  // namespace multibody
}  // namespace drake
