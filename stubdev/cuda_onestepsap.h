#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>
// define a constraint data structure
struct ConstraintData {
  Eigen::MatrixXd J;
  Eigen::MatrixXd G;
};

struct SAPCPUData {
  Eigen::MatrixXd dynamics_matrix;  // Dynamics matrix A
  Eigen::MatrixXd v_star;           // Free motion velocity v*
  Eigen::MatrixXd v_guess;
  ConstraintData constraint_data;
  std::vector<Eigen::Vector3d> gamma;  // impulse data vector
  std::vector<Eigen::Vector3d> R;      // regularization matrix vector

  int num_contacts;
  int num_velocities;
  int num_problems;
};

void TestOneStepSapGPU(std::vector<SAPCPUData>& sap_cpu_data,
                       std::vector<double>& momentum_cost,
                       std::vector<double>& regularizer_cost,
                       int num_velocities, int num_contacts, int num_problems);