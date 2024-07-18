#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

#ifndef dt
#define dt 0.01
#endif

#ifndef gravity
#define gravity -9.81
#endif

// define a constraint data structure
struct ConstraintData {
  Eigen::MatrixXd J;
  std::vector<Eigen::MatrixXd> G;

  Eigen::VectorXd phi0;
  Eigen::VectorXd contact_stiffness;
  Eigen::VectorXd contact_damping;

  int num_active_contacts;
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
