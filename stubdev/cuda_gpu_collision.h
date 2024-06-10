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
};

// Structure to hold collision data
struct CollisionData {
  bool isColliding;
  Eigen::Vector3d p_WC;       // Collision point on object A
  Eigen::Vector3d nhat_BA_W;  // Collision normal

  double phi0;        // overlap distance
  Eigen::Matrix3d R;  // rotation matrix
};

// Collision check
void CollisionEngine(Sphere* h_spheres, const int numProblems,
                     const int numSpheres,
                     CollisionData* h_collisionMatrixSpheres);