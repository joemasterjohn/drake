
#include <vector>

#include "stubdev/cuda_onestepsap_vd.h"
#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

// This tests checks if the solver converges to free motion velocity when there
// is no constraint imposed
GTEST_TEST(KernelTest, OneStepSAP_GPU) {
  std::cout << "Kernel Test: OneStepSap" << std::endl;

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

    Eigen::Matrix<double, 2, 1> vector(8, 3);
    sap_data.v_star = vector;
    Eigen::Matrix<double, 2, 1> vector2(5, 4);
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

  TestOneStepSapGPU(sap_cpu_data, v_solved, num_velocities, num_contacts,
                    num_problems);

  // Check v_solved
  for (int i = 0; i < num_problems; i++) {
    Eigen::Matrix<double, 2, 1> v_solved_cpu(6, 3);
    EXPECT_LT(abs((v_solved_cpu - v_solved[i]).norm()), 1e-1);
  }
}

// // This tests checks the cost evaluation and cholesky solution for Hx=-grad
// for
// // one step of the SAP problem
// GTEST_TEST(KernelTest, CostEvalAndSolve_GPU) {
//   std::cout << "Kernel Test: CostEvalAndSolve" << std::endl;

//   int num_velocities = 30;
//   int num_contacts = 10;
//   int num_problems = 100;
//   std::vector<SAPCPUData> sap_cpu_data;

//   for (int i = 0; i < num_problems; i++) {
//     SAPCPUData sap_data;

//     sap_data.num_contacts = num_contacts;
//     sap_data.num_velocities = num_velocities;
//     sap_data.num_problems = num_problems;

//     sap_data.dynamics_matrix =
//         Eigen::MatrixXd::Random(num_velocities, num_velocities);
//     sap_data.dynamics_matrix = sap_data.dynamics_matrix.transpose() *
//                                sap_data.dynamics_matrix;  // ensure A is SPD

//     sap_data.v_star =
//         Eigen::MatrixXd::Random(num_velocities, 1);  // free motion velocity
//     sap_data.v_guess = Eigen::MatrixXd::Zero(num_velocities, 1);

//     // J is contact 3nc x num_velocities, G is 3nc x 3nc
//     // this conforms the size of the Hessian matrix H = A + J^T * G * J

//     sap_data.constraint_data.J =
//         Eigen::MatrixXd::Random(3 * num_contacts, num_velocities);

//     // initialize impulse data vector and regularization matrices
//     for (int j = 0; j < num_contacts; j++) {
//       sap_data.gamma.push_back(Eigen::Vector3d::Random(3, 1));
//       sap_data.R.push_back(Eigen::Vector3d::Random(3, 1));
//       Eigen::MatrixXd G_temp = Eigen::MatrixXd::Random(3, 3);
//       sap_data.constraint_data.G.push_back(G_temp.transpose() * G_temp);
//     }

//     sap_cpu_data.push_back(sap_data);
//   }

//   std::vector<double> momentum_cost;
//   std::vector<double> regularizer_cost;
//   std::vector<Eigen::MatrixXd> hessian;
//   std::vector<Eigen::MatrixXd> neg_grad;
//   std::vector<Eigen::MatrixXd> chol_x;
//   std::vector<Eigen::MatrixXd> v_solved;

//   TestCostEvalAndSolveSapGPU(sap_cpu_data, momentum_cost, regularizer_cost,
//                              hessian, neg_grad, chol_x, num_velocities,
//                              num_contacts, num_problems);

//   // Check momentum cost
//   for (int i = 0; i < num_problems; i++) {
//     Eigen::MatrixXd delta_v = sap_cpu_data[i].v_guess -
//     sap_cpu_data[i].v_star; Eigen::MatrixXd delta_P =
//     sap_cpu_data[i].dynamics_matrix * delta_v; Eigen::MatrixXd lambda_m_cpu =
//     0.5 * delta_v.transpose() * delta_P; EXPECT_LT(abs(momentum_cost[i] -
//     lambda_m_cpu(0, 0)), 1e-8);
//   }

//   // Check regularizer cost
//   for (int i = 0; i < num_problems; i++) {
//     double lambda_r_sum = 0;
//     for (int j = 0; j < num_contacts; j++) {
//       lambda_r_sum +=
//           0.5 *
//           sap_cpu_data[i].gamma[j].dot(sap_cpu_data[i].R[j].cwiseProduct(
//                     sap_cpu_data[i].gamma[j]));
//     }

//     EXPECT_LT(abs(regularizer_cost[i] - lambda_r_sum), 1e-8);
//   }

//   // Check hessian
//   for (int i = 0; i < num_problems; i++) {
//     Eigen::MatrixXd G_J_cpu =
//         Eigen::MatrixXd::Zero(num_contacts * 3, num_velocities);
//     for (int j = 0; j < num_contacts; j++) {
//       G_J_cpu.block(j * 3, 0, 3, num_velocities) =
//           sap_cpu_data[i].constraint_data.G[j] *
//           sap_cpu_data[i].constraint_data.J.block(j * 3, 0, 3,
//           num_velocities);
//     }
//     Eigen::MatrixXd H_cpu =
//         sap_cpu_data[i].dynamics_matrix +
//         (sap_cpu_data[i].constraint_data.J.transpose() * G_J_cpu);
//     EXPECT_LT(abs((H_cpu - hessian[i]).norm()), 1e-8);
//   }

//   // Check -gradient
//   for (int i = 0; i < num_problems; i++) {
//     Eigen::MatrixXd gamma_full = Eigen::MatrixXd::Zero(num_contacts * 3, 1);
//     for (int j = 0; j < num_contacts; j++) {
//       gamma_full.block(j * 3, 0, 3, 1) = sap_cpu_data[i].gamma[j];
//     }
//     Eigen::MatrixXd neg_grad_cpu =
//         (sap_cpu_data[i].dynamics_matrix *
//          (sap_cpu_data[i].v_guess - sap_cpu_data[i].v_star)) -
//         (sap_cpu_data[i].constraint_data.J.transpose() * gamma_full);
//     neg_grad_cpu = -neg_grad_cpu;
//     EXPECT_LT(abs((neg_grad_cpu - neg_grad[i]).norm()), 1e-8);
//   }

//   // Check - cholesky solve
//   for (int i = 0; i < num_problems; i++) {
//     Eigen::VectorXd x_cpu = hessian[i].ldlt().solve(neg_grad[i]);
//     EXPECT_LT(abs((x_cpu - chol_x[i]).norm()), 1e-8);
//   }
// }

}  // namespace
}  // namespace drake
