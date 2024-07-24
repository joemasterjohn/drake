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

  double vn;  // normal relative velocity, positive if two spheres approaching,
              // negative if separating; on the direction of contact normal

  // variables for current step

  // history variables
  double f_0;  // contact force history, f0 = k * x0
};

// define a SAP data strucutre
struct CollisionGPUData {
  void Initialize(Sphere* h_spheres, int m_num_problems, int m_num_spheres);
  void Update();
  void Destroy();

  // CPU data retrival, assuming the CPU data is already allocated
  void RetrieveCollisionDataToCPU(CollisionData* h_collisionMatrixSpheres);
  void RetrieveJacobianToCPU(double* h_jacobian);
  void RetrieveNumCollisionsToCPU(int* h_num_collisions);
  void RetrieveDynamicMatrixToCPU(double* h_dynamic_matrix);
  void RetrieveVelocityVectorToCPU(double* h_velocity_vector);
  void RetrieveVStarToCPU(double* h_v_star);
  void RetrievePhi0ToCPU(double* h_phi0);
  void RetrieveContactStiffnessToCPU(double* h_contact_stiffness);
  void RetrieveContactDampingToCPU(double* h_contact_damping);
  void RetieveSphereDataToCPU(Sphere* h_spheres);

  Sphere* GetSpherePtr() { return d_spheres; }

  CollisionData* GetCollisionMatrixPtr() { return d_collisionMatrixSpheres; }

  double* GetJacobianPtr() { return d_jacobian; }

  int* GetNumCollisionsPtr() { return d_num_collisions; }

  double* GetDynamicMatrixPtr() { return d_dynamic_matrix; }

  double* GetVelocityVectorPtr() { return d_velocity_vector; }

  double* GetPhi0Ptr() { return d_phi0; }

  double* GetVStarPtr() { return d_v_star; }

  double* GetContactStiffnessPtr() { return d_contact_stiffness; }

  double* GetContactDampingPtr() { return d_contact_damping; }

  void CollisionEngine(const int numProblems, const int numSpheres);

  void IntegrateExplicitEuler(const int numProblems, const int numSpheres);

 private:
  Sphere* d_spheres;
  CollisionData* d_collisionMatrixSpheres;
  double* d_jacobian;
  int* d_num_collisions;
  double* d_dynamic_matrix;
  double* d_velocity_vector;
  double* d_phi0;
  double* d_v_star;

  double* d_contact_stiffness;
  double* d_contact_damping;

  int num_problems = 0;
  int num_spheres = 0;
};
