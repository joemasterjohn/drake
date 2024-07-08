
#include <vector>

#include "stubdev/cuda_onestepsap_vd.h"
#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

// This tests checks if the solver converges to the expected answer in the
// simple QP setting
GTEST_TEST(KernelTest, FullSapSolve) {
  int num_velocities = 2;
  int num_contacts = 1;
  int num_problems = 4;
  std::vector<SAPCPUData> sap_cpu_data;

  for (int i = 0; i < num_problems; i++) {
    SAPCPUData sap_data;

    sap_data.num_contacts = num_contacts;
    sap_data.num_velocities = num_velocities;
    sap_data.num_problems = num_problems;

    sap_data.dynamics_matrix =
        Eigen::MatrixXd::Identity(num_velocities, num_velocities);

    if (i == 0) {
      // i == 0: feasible region, far
      Eigen::Matrix<double, 2, 1> vector(8, 3);
      sap_data.v_star = vector;
      Eigen::Matrix<double, 2, 1> vector2(5, 10);
      sap_data.v_guess = vector2;
    } else if (i == 1) {
      // i == 1: feasible region, close
      Eigen::Matrix<double, 2, 1> vector(8, 3);
      sap_data.v_star = vector;
      Eigen::Matrix<double, 2, 1> vector2(5, 3);
      sap_data.v_guess = vector2;
    } else if (i == 2) {
      // i == 2: infeasible region, far
      Eigen::Matrix<double, 2, 1> vector(8, 3);
      sap_data.v_star = vector;
      Eigen::Matrix<double, 2, 1> vector2(10, 10);
      sap_data.v_guess = vector2;
    } else {
      // i == 3: infeasible region, close
      Eigen::Matrix<double, 2, 1> vector(8, 3);
      sap_data.v_star = vector;
      Eigen::Matrix<double, 2, 1> vector2(9, 3);
      sap_data.v_guess = vector2;
    }

    // J is contact 3nc x num_velocities, G is 3nc x 3nc
    // this conforms the size of the Hessian matrix H = A + J^T * G * J

    Eigen::MatrixXd J(3, 2);
    J << 0, 0, 0, 0, 1, 0;

    sap_data.constraint_data.J = J;

    // initialize impulse data vector and regularization matrices
    for (int j = 0; j < num_contacts; j++) {
      sap_data.gamma.push_back(Eigen::Vector3d::Zero(3, 1));
      sap_data.R.push_back(Eigen::Vector3d::Random(3, 1));
      Eigen::MatrixXd G_temp = Eigen::MatrixXd::Zero(3, 3);
      sap_data.constraint_data.G.push_back(G_temp.transpose() * G_temp);
    }

    sap_cpu_data.push_back(sap_data);
  }

  std::vector<Eigen::MatrixXd> v_solved;
  std::vector<int> v_iteration_counter;

  TestOneStepSapGPU(sap_cpu_data, v_solved, v_iteration_counter, num_velocities,
                    num_contacts, num_problems);

  // Check v_solved
  for (int i = 0; i < num_problems; i++) {
    Eigen::Matrix<double, 2, 1> v_solved_cpu(6, 3);
    EXPECT_LT(abs((v_solved_cpu - v_solved[i]).norm()), 1e-2);
  }
}

// This tests checks if the expected termination condition in various initial
// guesses in the simple QP setting
GTEST_TEST(KernelTest, IterationCounter) {
  int num_velocities = 2;
  int num_contacts = 1;
  int num_problems = 1;
  std::vector<SAPCPUData> sap_cpu_data;

  for (int i = 0; i < num_problems; i++) {
    SAPCPUData sap_data;

    sap_data.num_contacts = num_contacts;
    sap_data.num_velocities = num_velocities;
    sap_data.num_problems = num_problems;

    sap_data.dynamics_matrix =
        Eigen::MatrixXd::Identity(num_velocities, num_velocities);

    // i == 1: feasible region, close
    Eigen::Matrix<double, 2, 1> vector(8, 3);
    sap_data.v_star = vector;
    Eigen::Matrix<double, 2, 1> vector2(5, 3);
    sap_data.v_guess = vector2;

    // J is contact 3nc x num_velocities, G is 3nc x 3nc
    // this conforms the size of the Hessian matrix H = A + J^T * G * J

    Eigen::MatrixXd J(3, 2);
    J << 0, 0, 0, 0, 1, 0;

    sap_data.constraint_data.J = J;

    // initialize impulse data vector and regularization matrices
    for (int j = 0; j < num_contacts; j++) {
      sap_data.gamma.push_back(Eigen::Vector3d::Zero(3, 1));
      sap_data.R.push_back(Eigen::Vector3d::Random(3, 1));
      Eigen::MatrixXd G_temp = Eigen::MatrixXd::Zero(3, 3);
      sap_data.constraint_data.G.push_back(G_temp.transpose() * G_temp);
    }

    sap_cpu_data.push_back(sap_data);
  }

  std::vector<Eigen::MatrixXd> v_solved;
  std::vector<int> v_iteration_counter;

  TestOneStepSapGPU(sap_cpu_data, v_solved, v_iteration_counter, num_velocities,
                    num_contacts, num_problems);

  // Check v_solved and number of iterations
  for (int i = 0; i < num_problems; i++) {
    Eigen::Matrix<double, 2, 1> v_solved_cpu(6, 3);
    // check final solution
    EXPECT_LT(abs((v_solved_cpu - v_solved[i]).norm()), 1e-2);
    // we expect 1 step convergence, actually 2, due to termination condition
    EXPECT_LT(abs(v_iteration_counter[i] - 2), 1e-8);
  }
}

// This tests checks the cost evaluation and cholesky solution for Hx=-grad
// one step of the SAP problem
GTEST_TEST(KernelTest, CostEvalAndCholSolve) {
  int num_velocities = 2;
  int num_contacts = 1;
  int num_problems = 2;
  std::vector<SAPCPUData> sap_cpu_data;

  for (int i = 0; i < num_problems; i++) {
    SAPCPUData sap_data;

    sap_data.num_contacts = num_contacts;
    sap_data.num_velocities = num_velocities;
    sap_data.num_problems = num_problems;

    sap_data.dynamics_matrix =
        Eigen::MatrixXd::Identity(num_velocities, num_velocities);

    // i == 0: feasible region
    if (i == 0) {
      Eigen::Matrix<double, 2, 1> vector(8, 3);
      sap_data.v_star = vector;
      Eigen::Matrix<double, 2, 1> vector2(5, 3);
      sap_data.v_guess = vector2;
    }
    // i == 1: infeasible region
    else {
      Eigen::Matrix<double, 2, 1> vector(8, 3);
      sap_data.v_star = vector;
      Eigen::Matrix<double, 2, 1> vector2(7, 3);
      sap_data.v_guess = vector2;
    }

    // J is contact 3nc x num_velocities, G is 3nc x 3nc
    // this conforms the size of the Hessian matrix H = A + J^T * G * J

    Eigen::MatrixXd J(3, 2);
    J << 0, 0, 0, 0, 1, 0;

    sap_data.constraint_data.J = J;

    // initialize impulse data vector and regularization matrices
    for (int j = 0; j < num_contacts; j++) {
      sap_data.gamma.push_back(Eigen::Vector3d::Zero(3, 1));
      sap_data.R.push_back(Eigen::Vector3d::Random(3, 1));
      Eigen::MatrixXd G_temp = Eigen::MatrixXd::Zero(3, 3);
      sap_data.constraint_data.G.push_back(G_temp.transpose() * G_temp);
    }

    sap_cpu_data.push_back(sap_data);
  }

  std::vector<double> momentum_cost;
  std::vector<double> regularizer_cost;
  std::vector<Eigen::MatrixXd> hessian;
  std::vector<Eigen::MatrixXd> neg_grad;
  std::vector<Eigen::MatrixXd> chol_x;
  std::vector<Eigen::MatrixXd> v_solved;
  std::vector<Eigen::MatrixXd> chol_l;
  std::vector<Eigen::MatrixXd> G;

  TestCostEvalAndSolveSapGPU(sap_cpu_data, momentum_cost, regularizer_cost,
                             hessian, neg_grad, chol_x, chol_l, G,
                             num_velocities, num_contacts, num_problems);

  for (int i = 0; i < num_problems; i++) {
    if (i == 0) {
      // check hessian
      EXPECT_LT(abs((hessian[i] - Eigen::MatrixXd::Identity(2, 2)).norm()),
                1e-6);

      // check cost
      EXPECT_LT(abs(momentum_cost[i] - 4.5), 1e-6);
      EXPECT_LT(abs(regularizer_cost[i] - 0.0), 1e-6);

      // check G is [0, 0; 0, 0], as the contraint is not active
      EXPECT_LT(abs(G[i](0, 0) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](0, 1) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](0, 2) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](1, 0) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](1, 1) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](1, 2) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](2, 0) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](2, 1) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](2, 2) - 0.0), 1e-6);

      // check that chol_l * chol_l.transpose is identity
      EXPECT_LT(abs((chol_l[i] * chol_l[i].transpose() -
                     Eigen::MatrixXd::Identity(2, 2))
                        .norm()),
                1e-6);

      // check if chol_x is (-3, 0)
      EXPECT_LT(abs((chol_x[i] - Eigen::Matrix<double, 2, 1>(3, 0)).norm()),
                1e-6);
    }

    if (i == 1) {
      // check if hessian is [10001, 1; 0, 1]
      EXPECT_LT(abs(hessian[i](0, 0) - 10001), 1e-6);
      EXPECT_LT(abs(hessian[i](0, 1) - 0), 1e-6);
      EXPECT_LT(abs(hessian[i](1, 0) - 0), 1e-6);
      EXPECT_LT(abs(hessian[i](1, 1) - 1), 1e-6);

      // check cost
      EXPECT_LT(abs(momentum_cost[i] - 0.5), 1e-6);
      EXPECT_LT(abs(regularizer_cost[i] - 5000.0), 1e-6);

      // check G is [0.0, 0.0, 0.0; 0.0, 0.0, 10000.0], as the contraint is
      // active
      EXPECT_LT(abs(G[i](0, 0) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](0, 1) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](0, 2) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](1, 0) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](1, 1) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](1, 2) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](2, 0) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](2, 1) - 0.0), 1e-6);
      EXPECT_LT(abs(G[i](2, 2) - 1e4), 1e-6);

      // check chol_x
      EXPECT_LT(abs((chol_x[i] - Eigen::Matrix<double, 2, 1>(-1, 0)).norm()),
                1e-3);
    }
  }
}

}  // namespace
}  // namespace drake
