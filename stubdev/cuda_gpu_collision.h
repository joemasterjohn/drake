#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <eigen3/Eigen/Dense>

// Define sphere geometry
struct Sphere {
  Eigen::Vector3d center;
  double radius;
  double mass;

  Eigen::Vector3d velocity;

  // material properties
  double stiffness;
  double damping;
};

// Structure to hold collision data
struct CollisionData {
  bool isColliding;
  Eigen::Vector3d p_WC;       // Collision point on object A
  Eigen::Vector3d nhat_BA_W;  // Collision normal, we follow the convention,
                              // pointing from B to A

  double phi0;        // overlap distance
  Eigen::Matrix3d R;  // rotation matrix
  Eigen::Vector3d
      gamma;  // contact impulse, expressed in local frame, we follow the
              // convention, gamma is applied on A, then -gamma is applied on B
  Eigen::Vector3d gamma_W;  // contact impulse, expressed in world frame, we
                            // follow the convention, gamma is applied on A,
                            // then -gamma is applied on B

  double vn;  // normal relative velocity, positive if two spheres approaching,
              // negative if separating; on the direction of contact normal

  // variables for current step

  // history variables
  double f_0;  // contact force history, f0 = k * x0
};

// Collision check
void CollisionEngine(Sphere* h_spheres, const int numProblems,
                     const int numSpheres,
                     CollisionData* h_collisionMatrixSpheres,
                     double* h_jacobian, double* h_gamma, int* h_num_collisions,
                     double* h_dynamic_matrix, double* h_velocity_vector,
                     double* h_v_star, double* h_phi0);
