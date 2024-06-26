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

  Eigen::Vector3d velocity;

  // material properties
  double stiffness;
  double damping;
};

// Structure to hold collision data
struct CollisionData {
  bool isColliding;
  Eigen::Vector3d p_WC;       // Collision point on object A
  Eigen::Vector3d nhat_BA_W;  // Collision normal

  double phi0;        // overlap distance
  Eigen::Matrix3d R;  // rotation matrix

  double vn;  // normal relative velocity

  // variables for current step

  // history variables
  double f_0;  // contact force history, f0 = k * x0
};

// Collision check
void CollisionEngine(Sphere* h_spheres, const int numProblems,
                     const int numSpheres,
                     CollisionData* h_collisionMatrixSpheres);

void EvaluateAntiderivative(CollisionData* h_collisionMatrixSpheres);