#include <stdio.h>

#include <iostream>

#include "cuda_gpu_collision.cuh"
#include "cuda_gpu_collision.h"

#ifndef dt
#define dt 0.01
#endif

#ifndef gravity
#define gravity -9.81
#endif

// Device function to check Sphere-Sphere collision
__device__ CollisionData CheckSphereCollision(const Sphere& a,
                                              const Sphere& b) {
  CollisionData data = {
      false, {0, 0, 0}, {0, 0, 0}, 0, Eigen::Matrix3d::Zero()};

  Eigen::Vector3d dist = a.center - b.center;
  double distSquared =
      dist(0) * dist(0) + dist(1) * dist(1) + dist(2) * dist(2);
  double distLength = sqrt(distSquared);
  double radiusSum = a.radius + b.radius;

  dist.normalize();

  if (distSquared <= (radiusSum * radiusSum)) {
    data.isColliding = true;
    // Calculate collision normal
    data.nhat_BA_W = dist;
    // Normalize the collision normal

    data.nhat_BA_W.normalize();
    // Calculate collision points
    Eigen::Vector3d midpoint;
    midpoint(0) = (a.center(0) + b.center(0)) / 2;
    midpoint(1) = (a.center(1) + b.center(1)) / 2;
    midpoint(2) = (a.center(2) + b.center(2)) / 2;

    data.phi0 = -(distLength - radiusSum);  // sign convention

    data.p_WC = midpoint;

    // Get collision frame matrix
    // Random vector v is default to {1.0, 1.0, 1.0}
    Eigen::Vector3d v(1.0, 1.0, 1.0);
    v.normalize();

    double y_hat_temp = v.dot(data.nhat_BA_W);
    Eigen::Vector3d y_hat = v - y_hat_temp * data.nhat_BA_W;
    y_hat.normalize();
    Eigen::Vector3d x_hat = y_hat.cross(data.nhat_BA_W);

    data.R(0, 0) = x_hat(0);           // x of x-axis
    data.R(0, 1) = x_hat(1);           // y of x-axis
    data.R(0, 2) = x_hat(2);           // z of x-axis
    data.R(1, 0) = y_hat(0);           // x of y-axis
    data.R(1, 1) = y_hat(1);           // y of y-axis
    data.R(1, 2) = y_hat(2);           // z of y-axis
    data.R(2, 0) = data.nhat_BA_W(0);  // x of z-axis
    data.R(2, 1) = data.nhat_BA_W(1);  // y of z-axis
    data.R(2, 2) = data.nhat_BA_W(2);  // z of z-axis

    data.vn = -(a.velocity - b.velocity)
                   .dot(data.nhat_BA_W);  // negative for departing, positive
                                          // for approaching
  } else {
    data.isColliding = false;
  }

  return data;
}

// Kernel to detect collisions between Spheres
__global__ void DetectSphereCollisions(const Sphere* spheres, int numProblems,
                                       int numSpheres,
                                       CollisionData* collisionMatrix) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    if (j < numSpheres) {
      for (int k = j + 1; k < numSpheres; k++) {
        collisionMatrix[(p_idx * numSpheres * numSpheres) + j * numSpheres +
                        k] =
            CheckSphereCollision(spheres[p_idx * numSpheres + j],
                                 spheres[p_idx * numSpheres + k]);
      }
    }
  }
  __syncwarp();
}

// Kernel to detect collisions between Spheres
__global__ void ConstructJacobianGamma(
    const Sphere* spheres, int numProblems, int numSpheres,
    CollisionData* collisionMatrix, double* d_jacobian, int* d_num_collisions,
    double* d_phi0, double* d_contact_stiffness, double* d_contact_damping) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  Eigen::Map<Eigen::MatrixXd> full_jacobian(
      d_jacobian +
          blockIdx.x * (numSpheres * 3) * (numSpheres * numSpheres * 3),
      numSpheres * numSpheres * 3, numSpheres * 3);
  Eigen::Map<Eigen::VectorXd> contact_stiffness(
      d_contact_stiffness + blockIdx.x * numSpheres * numSpheres,
      numSpheres * numSpheres, 1);
  Eigen::Map<Eigen::VectorXd> contact_damping(
      d_contact_damping + blockIdx.x * numSpheres * numSpheres,
      numSpheres * numSpheres, 1);

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    if (j < numSpheres) {
      for (int k = j + 1; k < numSpheres; k++) {
        if (collisionMatrix[(p_idx * numSpheres * numSpheres) + j * numSpheres +
                            k]
                .isColliding) {
          int collision_idx = atomicAdd(&d_num_collisions[p_idx], 1);

          // update the harmonic mean of contact stiffness
          contact_stiffness[collision_idx] =
              (2 * spheres[p_idx * numSpheres + j].stiffness *
               spheres[p_idx * numSpheres + k].stiffness) /
              (spheres[p_idx * numSpheres + j].stiffness +
               spheres[p_idx * numSpheres + k].stiffness);

          // update the harmonic mean of contact damping
          contact_damping[collision_idx] =
              (2 * spheres[p_idx * numSpheres + j].damping *
               spheres[p_idx * numSpheres + k].damping) /
              (spheres[p_idx * numSpheres + j].damping +
               spheres[p_idx * numSpheres + k].damping);

          // construct Jacobian matrix
          full_jacobian.block<3, 3>(collision_idx * 3, j * 3) =
              collisionMatrix[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                  .R *
              Eigen::MatrixXd::Identity(3, 3);
          full_jacobian.block<3, 3>(collision_idx * 3, k * 3) =
              collisionMatrix[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                  .R *
              -Eigen::MatrixXd::Identity(3, 3);

          // add data to phi0
          d_phi0[p_idx * numSpheres * numSpheres + collision_idx] =
              collisionMatrix[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                  .phi0;
        }
      }
    }
  }
  __syncwarp();
}

__global__ void ConstructDynamicMatrixVelocityVector(
    const Sphere* spheres, int numProblems, int numSpheres,
    double* d_dynamic_matrix, double* d_velocity_vector) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  Eigen::Map<Eigen::MatrixXd> dynamic_matrix(
      d_dynamic_matrix + blockIdx.x * numSpheres * 3 * numSpheres * 3,
      numSpheres * 3, numSpheres * 3);
  Eigen::Map<Eigen::VectorXd> velocity_vector(
      d_velocity_vector + blockIdx.x * numSpheres * 3, numSpheres * 3, 1);

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    if (j < numSpheres) {
      dynamic_matrix.block<3, 3>(j * 3, j * 3) =
          spheres[p_idx * numSpheres + j].mass *
          Eigen::MatrixXd::Identity(3, 3);
      velocity_vector.block<3, 1>(j * 3, 0) =
          spheres[p_idx * numSpheres + j].velocity;
    }
  }

  __syncwarp();
}

__global__ void CalculateFreeMotionVelocity(const Sphere* spheres,
                                            int numProblems, int numSpheres,
                                            double* d_velocity_vector,
                                            double* d_v_star) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  Eigen::Map<Eigen::VectorXd> velocity_vector(
      d_velocity_vector + blockIdx.x * numSpheres * 3, numSpheres * 3, 1);
  Eigen::Map<Eigen::VectorXd> v_star(d_v_star + blockIdx.x * numSpheres * 3,
                                     numSpheres * 3, 1);

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    if (j < numSpheres) {
      v_star.block<3, 1>(j * 3, 0) =
          velocity_vector.block<3, 1>(j * 3, 0) + dt * Eigen::Vector3d(0, 0, 0);
    }
  }

  __syncwarp();
}

void CollisionGPUData::CollisionEngine(const int numProblems,
                                       const int numSpheres) {
  // Kernel launches
  int threadsPerBlock = 32;
  int blocksPerGridSpheres = numProblems;
  DetectSphereCollisions<<<blocksPerGridSpheres, threadsPerBlock>>>(
      this->GetSpherePtr(), numProblems, numSpheres,
      this->GetCollisionMatrixPtr());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Construct Jacobian matrix and Gamma vector
  ConstructJacobianGamma<<<blocksPerGridSpheres, threadsPerBlock>>>(
      this->GetSpherePtr(), numProblems, numSpheres,
      this->GetCollisionMatrixPtr(), this->GetJacobianPtr(),
      this->GetNumCollisionsPtr(), this->GetPhi0Ptr(),
      this->GetContactStiffnessPtr(), this->GetContactDampingPtr());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Construct Dynamic matrix
  ConstructDynamicMatrixVelocityVector<<<blocksPerGridSpheres,
                                         threadsPerBlock>>>(
      this->GetSpherePtr(), numProblems, numSpheres,
      this->GetDynamicMatrixPtr(), this->GetVelocityVectorPtr());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Calculate free motion velocity vector Dynamic matrix
  CalculateFreeMotionVelocity<<<blocksPerGridSpheres, threadsPerBlock>>>(
      this->GetSpherePtr(), numProblems, numSpheres,
      this->GetVelocityVectorPtr(), this->GetVStarPtr());
  HANDLE_ERROR(cudaDeviceSynchronize());
}

__global__ void IntegrateExplicitEulerKernel(Sphere* spheres, int numProblems,
                                             int numSpheres,
                                             double* d_velocity_vector) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  Eigen::Map<Eigen::VectorXd> velocity_vector(
      d_velocity_vector + blockIdx.x * numSpheres * 3, numSpheres * 3, 1);

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    spheres[p_idx * numSpheres + j].velocity =
        velocity_vector.block<3, 1>(j * 3, 0);
    spheres[p_idx * numSpheres + j].center =
        spheres[p_idx * numSpheres + j].center +
        dt * velocity_vector.block<3, 1>(j * 3, 0);
  }

  __syncwarp();
}

// an eplicit euler to update the position based on velocity_vector
void CollisionGPUData::IntegrateExplicitEuler(const int numProblems,
                                              const int numSpheres) {
  // Kernel launches
  int threadsPerBlock = 32;
  int blocksPerGridSpheres = numProblems;
  IntegrateExplicitEulerKernel<<<blocksPerGridSpheres, threadsPerBlock>>>(
      d_spheres, numProblems, numSpheres, this->GetVelocityVectorPtr());
  HANDLE_ERROR(cudaDeviceSynchronize());
}