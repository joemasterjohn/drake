#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>
// define a constraint data structure
struct ConstraintData {
  Eigen::MatrixXd J;
  Eigen::MatrixXd G;
};

// define a SAP data strucutre
struct SAPGPUData {
  Eigen::MatrixXd A;
  Eigen::MatrixXd v_star;
  ConstraintData constraint_data;
  std::vector<Eigen::MatrixXd> v_gamma;  // impulse data vector
  std::vector<Eigen::MatrixXd> v_R;      // regularization matrix vector
};

void test_onestep_sap(std::vector<Eigen::MatrixXd>& v_guess,
                      std::vector<SAPGPUData>& v_sap_data,
                      std::vector<double>& v_lambda_m,
                      std::vector<double>& v_lambda_r, int num_rbodies,
                      int num_contacts, int num_equations);