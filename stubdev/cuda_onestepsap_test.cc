
#include "stubdev/cuda_onestepsap.h"

#include <vector>

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, OneStepSAP_GPU) {
  std::cout << "Kernel Test: OneStepSAP_GPU" << std::endl;

  int num_velocities = 50;
  int num_contacts = 35;
  int num_problems = 500;
  std::vector<SAPCPUData> sap_cpu_data;

  for (int i = 0; i < num_problems; i++) {
    SAPCPUData sap_data;

    sap_data.num_contacts = num_contacts;
    sap_data.num_velocities = num_velocities;
    sap_data.num_problems = num_problems;

    sap_data.dynamics_matrix =
        Eigen::MatrixXd::Random(num_velocities, num_velocities);
    sap_data.dynamics_matrix = sap_data.dynamics_matrix.transpose() *
                               sap_data.dynamics_matrix;  // ensure A is SPD

    sap_data.v_star =
        Eigen::MatrixXd::Random(num_velocities, 1);  // free motion velocity
    sap_data.v_guess = Eigen::MatrixXd::Random(num_velocities, 1);

    // J is contact 3nc x num_velocities, G is 3nc x 3nc
    // this conforms the size of the Hessian matrix H = A + J^T * G * J

    sap_data.constraint_data.J =
        Eigen::MatrixXd::Random(3 * num_contacts, num_velocities);

    // initialize impulse data vector and regularization matrices
    for (int j = 0; j < num_contacts; j++) {
      sap_data.gamma.push_back(Eigen::Vector3d::Random(3, 1));
      sap_data.R.push_back(Eigen::Vector3d::Random(3, 1));
      Eigen::MatrixXd G_temp = Eigen::MatrixXd::Random(3, 3);
      sap_data.constraint_data.G.push_back(G_temp.transpose() * G_temp);
    }

    sap_cpu_data.push_back(sap_data);
  }

  std::vector<double> momentum_cost;
  std::vector<double> regularizer_cost;
  std::vector<Eigen::MatrixXd> hessian;
  std::vector<Eigen::MatrixXd> neg_grad;
  momentum_cost.resize(num_problems);
  regularizer_cost.resize(num_problems * num_contacts);

  TestOneStepSapGPU(sap_cpu_data, momentum_cost, regularizer_cost, hessian,
                    neg_grad, num_velocities, num_contacts, num_problems);

  // Check momentum cost
  for (int i = 0; i < num_problems; i++) {
    Eigen::MatrixXd delta_v = sap_cpu_data[i].v_guess - sap_cpu_data[i].v_star;
    Eigen::MatrixXd delta_P = sap_cpu_data[i].dynamics_matrix * delta_v;
    Eigen::MatrixXd lambda_m_cpu = 0.5 * delta_v.transpose() * delta_P;
    EXPECT_LT(abs(momentum_cost[i] - lambda_m_cpu(0, 0)), 1e-10);
  }

  // Check regularizer cost
  for (int i = 0; i < num_problems; i++) {
    double lambda_r_sum = 0;
    for (int j = 0; j < num_contacts; j++) {
      lambda_r_sum +=
          0.5 * sap_cpu_data[i].gamma[j].dot(sap_cpu_data[i].R[j].cwiseProduct(
                    sap_cpu_data[i].gamma[j]));
    }

    EXPECT_LT(abs(regularizer_cost[i] - lambda_r_sum), 1e-10);
  }

  // Check hessian
  for (int i = 0; i < num_problems; i++) {
    Eigen::MatrixXd G_J_cpu =
        Eigen::MatrixXd::Zero(num_contacts * 3, num_velocities);
    for (int j = 0; j < num_contacts; j++) {
      G_J_cpu.block(j * 3, 0, 3, num_velocities) =
          sap_cpu_data[i].constraint_data.G[j] *
          sap_cpu_data[i].constraint_data.J.block(j * 3, 0, 3, num_velocities);
    }
    Eigen::MatrixXd H_cpu =
        sap_cpu_data[i].dynamics_matrix +
        (sap_cpu_data[i].constraint_data.J.transpose() * G_J_cpu);
    EXPECT_LT(abs((H_cpu - hessian[i]).norm()), 1e-10);
  }

  // Check -gradient
  for (int i = 0; i < num_problems; i++) {
    Eigen::MatrixXd gamma_full = Eigen::MatrixXd::Zero(num_contacts * 3, 1);
    for (int j = 0; j < num_contacts; j++) {
      gamma_full.block(j * 3, 0, 3, 1) = sap_cpu_data[i].gamma[j];
    }
    Eigen::MatrixXd neg_grad_cpu =
        (sap_cpu_data[i].dynamics_matrix *
         (sap_cpu_data[i].v_guess - sap_cpu_data[i].v_star)) -
        (sap_cpu_data[i].constraint_data.J.transpose() * gamma_full);
    neg_grad_cpu = -neg_grad_cpu;
    EXPECT_LT(abs((neg_grad_cpu - neg_grad[i]).norm()), 1e-10);
  }

  // Check - cholesky solve
}

}  // namespace
}  // namespace drake
