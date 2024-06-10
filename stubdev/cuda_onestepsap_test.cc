
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
    // this conforms the size of the Hessian matrix H = A + J^T * G * J

    sap_data.constraint_data.J =
        Eigen::MatrixXd::Random(3 * num_contacts, num_rbodies);
    sap_data.constraint_data.G =
        Eigen::MatrixXd::Random(3 * num_contacts, 3 * num_contacts);
    sap_data.constraint_data.G =
        sap_data.constraint_data.G.transpose() * sap_data.constraint_data.G;

    // initialize impulse data vector and regularization matrices
    for (int j = 0; j < num_contacts; j++) {
      sap_data.v_gamma.push_back(Eigen::MatrixXd::Random(3, 1));
      sap_data.v_R.push_back(Eigen::MatrixXd::Random(3, 3));
    }

    v_sap_data.push_back(sap_data);
  }

  // variable to store the results on CPU, passed as a reference to the
  // test_onestep_sap function call
  std::vector<double> v_lambda_m;
  std::vector<double> v_lambda_r;
  v_lambda_m.resize(num_equations);
  v_lambda_r.resize(num_equations * num_contacts);

  test_onestep_sap(v_guess, v_sap_data, v_lambda_m, v_lambda_r, num_rbodies,
                   num_contacts, num_equations);

  for (int i = 0; i < num_equations; i++) {
    Eigen::MatrixXd delta_v = v_guess[i] - v_sap_data[i].v_star;
    Eigen::MatrixXd delta_P = v_sap_data[i].A * delta_v;
    Eigen::MatrixXd lambda_m_cpu = 0.5 * delta_v.transpose() * delta_P;
    EXPECT_LT(v_lambda_m[i] - lambda_m_cpu(0, 0), 1e-10);

    double lambda_r_sum = 0;
    for (int j = 0; j < num_contacts; j++) {
      Eigen::MatrixXd lambda_r_cpu =
          0.5 * v_sap_data[i].v_gamma[j].transpose() * v_sap_data[i].v_R[j] *
          v_sap_data[i].v_gamma[j];
      lambda_r_sum += lambda_r_cpu(0, 0);
    }
    EXPECT_LT(v_lambda_r[i] - lambda_r_sum, 1e-10);
  }
}

}  // namespace
}  // namespace drake
