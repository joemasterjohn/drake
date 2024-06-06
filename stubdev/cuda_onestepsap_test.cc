
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
  int num_equations = 200;
  std::vector<Eigen::MatrixXd> v_m;
  std::vector<Eigen::MatrixXd> v_A;
  std::vector<Eigen::MatrixXd> v_J;
  std::vector<Eigen::MatrixXd> v_gamma;
  std::vector<Eigen::MatrixXd> v_dv;

  // initialize the problem to be 64
  // initialize fake v_m
  // note that contact jacobian is 3*nc by num_rbodies*3, as it maps to
  // num_rbodies generalized velocities gamma is impulse and a vector, so it is
  // 3*nc by 1, since each contact has 3 components
  for (int i = 0; i < num_equations; i++) {
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(num_rbodies * 3, 1);
    Eigen::MatrixXd A =
        Eigen::MatrixXd::Random(num_rbodies * 3, num_rbodies * 3);
    Eigen::MatrixXd J =
        Eigen::MatrixXd::Random(num_rbodies * 3, num_rbodies * 3);
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Random(num_rbodies * 3, 1);
    Eigen::MatrixXd dv = Eigen::MatrixXd::Random(num_rbodies * 3, 1);
    dv.setZero();

    v_m.push_back(m);
    v_A.push_back(A);
    v_J.push_back(J);
    v_gamma.push_back(gamma);
    v_dv.push_back(dv);
  }

  test_onestep_sap(v_A, v_J, v_gamma, v_m, dv, num_equations);
}

}  // namespace
}  // namespace drake
