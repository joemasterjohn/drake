
#include "stubdev/cuda_onestepsap.h"

#include <vector>

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, OneStepSAP) {
  // assume the problem contains 12 bodies, so A 12*3 by 12*3, v_m 12*3 by 1
  // etc..
  int num_rbodies = 12;
  int num_contacts = 3;
  int num_equations = 500;
  std::vector<SAPGPUData> v_sap_data;
  std::vector<Eigen::MatrixXd> v_guess;

  for (int i = 0; i < num_equations; i++) {
    Eigen::MatrixXd guess = Eigen::MatrixXd::Random(num_rbodies * 3, 1);
    v_guess.push_back(guess);

    SAPGPUData sap_data;

    sap_data.A = Eigen::MatrixXd::Random(num_rbodies * 3, num_rbodies * 3);
    sap_data.A = sap_data.A.transpose() * sap_data.A;  // ensure A is SPD

    sap_data.v_star =
        Eigen::MatrixXd::Random(num_rbodies * 3, 1);  // free motion velocity

    // J is contact 3nc x num_rbodies, G is 3nc x 3nc
    // this confirms the size of the Hessian matrix H = A + J^T * G * J

    sap_data.constraint_data.J =
        Eigen::MatrixXd::Random(3 * num_contacts, num_rbodies);
    sap_data.constraint_data.G =
        Eigen::MatrixXd::Random(3 * num_contacts, 3 * num_contacts);
    sap_data.constraint_data.G =
        sap_data.constraint_data.G.transpose() * sap_data.constraint_data.G;

    v_sap_data.push_back(sap_data);
  }

  // variable to store the results on CPU, passed as a reference to the
  // test_onestep_sap function call
  std::vector<double> v_lambda_m;
  v_lambda_m.resize(num_equations);

  test_onestep_sap(v_guess, v_sap_data, v_lambda_m, num_rbodies, num_contacts,
                   num_equations);

  for (int i = 0; i < num_equations; i++) {
    Eigen::MatrixXd delta_v = v_guess[i] - v_sap_data[i].v_star;
    Eigen::MatrixXd delta_P = v_sap_data[i].A * delta_v;
    Eigen::MatrixXd lambda_cpu = 0.5 * delta_v.transpose() * delta_P;
    EXPECT_LT(v_lambda_m[i] - lambda_cpu(0, 0), 1e-10);
  }
}

}  // namespace
}  // namespace drake
