
#include "stubdev/cuda_onestepsap.h"

#include <vector>

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

// GTEST_TEST(KernelTest, OneStepSAP_CPU) {
//   // assume the problem contains 12 bodies, so A 12*3 by 12*3, v_m 12*3 by 1
//   // etc..
//   int num_velocities = 12;
//   int num_contacts = 3;
//   int num_problems = 100;
//   std::vector<SAPCPUData> v_sap_data;
//   std::vector<Eigen::MatrixXd> v_guess;

//   for (int i = 0; i < num_problems; i++) {
//     Eigen::MatrixXd guess = Eigen::MatrixXd::Random(num_velocities * 3, 1);
//     v_guess.push_back(guess);

//     SAPCPUData sap_data;

//     sap_data.num_contacts = num_contacts;
//     sap_data.num_velocities = num_velocities;
//     sap_data.num_problems = num_problems;

//     sap_data.A =
//         Eigen::MatrixXd::Random(num_velocities * 3, num_velocities * 3);
//     sap_data.A = sap_data.A.transpose() * sap_data.A;  // ensure A is SPD

//     sap_data.v_star =
//         Eigen::MatrixXd::Random(num_velocities * 3, 1);  // free motion
//         velocity

//     // J is contact 3nc x num_velocities, G is 3nc x 3nc
//     // this conforms the size of the Hessian matrix H = A + J^T * G * J

//     sap_data.constraint_data.J =
//         Eigen::MatrixXd::Random(3 * num_contacts, num_velocities);
//     sap_data.constraint_data.G =
//         Eigen::MatrixXd::Random(3 * num_contacts, 3 * num_contacts);
//     sap_data.constraint_data.G =
//         sap_data.constraint_data.G.transpose() * sap_data.constraint_data.G;

//     // initialize impulse data vector and regularization matrices
//     for (int j = 0; j < num_contacts; j++) {
//       sap_data.v_gamma.push_back(Eigen::MatrixXd::Random(3, 1));
//       sap_data.v_R.push_back(Eigen::MatrixXd::Random(3, 3));
//     }

//     v_sap_data.push_back(sap_data);
//   }

//   // variable to store the results on CPU, passed as a reference to the
//   // TestOneStepSap function call
//   std::vector<double> v_lambda_m;
//   std::vector<double> v_lambda_r;
//   v_lambda_m.resize(num_problems);
//   v_lambda_r.resize(num_problems * num_contacts);

//   TestOneStepSap(v_guess, v_sap_data, v_lambda_m, v_lambda_r, num_velocities,
//                  num_contacts, num_problems);

//   for (int i = 0; i < num_problems; i++) {
//     Eigen::MatrixXd delta_v = v_guess[i] - v_sap_data[i].v_star;
//     Eigen::MatrixXd delta_P = v_sap_data[i].A * delta_v;
//     Eigen::MatrixXd lambda_m_cpu = 0.5 * delta_v.transpose() * delta_P;
//     EXPECT_LT(v_lambda_m[i] - lambda_m_cpu(0, 0), 1e-10);

//     double lambda_r_sum = 0;
//     for (int j = 0; j < num_contacts; j++) {
//       Eigen::MatrixXd lambda_r_cpu =
//           0.5 * v_sap_data[i].v_gamma[j].transpose() * v_sap_data[i].v_R[j] *
//           v_sap_data[i].v_gamma[j];
//       lambda_r_sum += lambda_r_cpu(0, 0);
//     }
//     EXPECT_LT(v_lambda_r[i] - lambda_r_sum, 1e-10);
//   }
// }

GTEST_TEST(KernelTest, OneStepSAP_GPU) {
  std::cout << "Kernel Test: OneStepSAP_GPU" << std::endl;

  // assume the problem contains 12 bodies, so A 12*3 by 12*3, v_m 12*3 by 1
  // etc..
  int num_velocities = 12;
  int num_contacts = 3;
  int num_problems = 500;
  std::vector<SAPCPUData> v_sap_data;

  for (int i = 0; i < num_problems; i++) {
    SAPCPUData sap_data;

    sap_data.num_contacts = num_contacts;
    sap_data.num_velocities = num_velocities;
    sap_data.num_problems = num_problems;

    sap_data.A =
        Eigen::MatrixXd::Random(num_velocities * 3, num_velocities * 3);
    sap_data.A = sap_data.A.transpose() * sap_data.A;  // ensure A is SPD

    sap_data.v_star =
        Eigen::MatrixXd::Random(num_velocities * 3, 1);  // free motion velocity
    sap_data.v_guess = Eigen::MatrixXd::Random(num_velocities * 3, 1);

    // J is contact 3nc x num_velocities, G is 3nc x 3nc
    // this conforms the size of the Hessian matrix H = A + J^T * G * J

    sap_data.constraint_data.J =
        Eigen::MatrixXd::Random(3 * num_contacts, num_velocities);
    sap_data.constraint_data.G =
        Eigen::MatrixXd::Random(3 * num_contacts, 3 * num_contacts);
    sap_data.constraint_data.G =
        sap_data.constraint_data.G.transpose() * sap_data.constraint_data.G;

    // initialize impulse data vector and regularization matrices
    for (int j = 0; j < num_contacts; j++) {
      sap_data.v_gamma.push_back(Eigen::Vector3d::Random(3, 1));
      sap_data.v_R.push_back(Eigen::Vector3d::Random(3, 1));
    }

    v_sap_data.push_back(sap_data);
  }

  // variable to store the results on CPU, passed as a reference to the
  // TestOneStepSap function call
  std::vector<double> v_lambda_m;
  std::vector<double> v_lambda_r;
  v_lambda_m.resize(num_problems);
  v_lambda_r.resize(num_problems * num_contacts);

  TestOneStepSapGPU(v_sap_data, v_lambda_m, v_lambda_r, num_velocities,
                    num_contacts, num_problems);
}

}  // namespace
}  // namespace drake
