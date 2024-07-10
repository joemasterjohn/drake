#include <stdio.h>

#include <iostream>

#include "cuda_gpu_collision.h"

#ifndef dt
#define dt 0.001
#endif

#ifndef gravity
#define gravity -9.81
#endif

// CUDA error handeling
// =====================
static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
// =====================

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

__device__ void CalculateIndividualGamma(CollisionData& data, const Sphere& a,
                                         const Sphere& b) {
  // harmonic mean for stiffness and damping
  double stiffness_avg =
      (2 * a.stiffness * b.stiffness) / (a.stiffness + b.stiffness);
  double damping_avg = (2 * a.damping * b.damping) / (a.damping + b.damping);

  // calculate gamma in the local frame
  double gamma_z =
      (stiffness_avg * (data.phi0)) * (fmax(0, 1 + damping_avg * data.vn));
  Eigen::Vector3d gamma_temp(0.0, 0.0, gamma_z);
  data.gamma = gamma_temp;

  // calculate gamma in the global frame
  data.gamma_W = dt * data.R.transpose() * data.gamma;
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

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    if (j < numSpheres) {
      for (int k = j + 1; k < numSpheres; k++) {
        CalculateIndividualGamma(
            collisionMatrix[(p_idx * numSpheres * numSpheres) + j * numSpheres +
                            k],
            spheres[p_idx * numSpheres + j], spheres[p_idx * numSpheres + k]);
      }
    }
  }

  __syncwarp();
}

// d_spheres, numProblems, numSpheres, d_collisionMatrixSpheres, d_jacobian,
//      d_gamma, d_num_collisions

// Kernel to detect collisions between Spheres
__global__ void ConstructJacobianGamma(const Sphere* spheres, int numProblems,
                                       int numSpheres,
                                       CollisionData* collisionMatrix,
                                       double* d_jacobian, double* d_gamma,
                                       int* d_num_collisions, double* d_phi0,
                                       double* d_contact_stiffness,
                                       double* d_contact_damping) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  Eigen::Map<Eigen::VectorXd> full_gamma(
      d_gamma + blockIdx.x * numSpheres * numSpheres * 3,
      numSpheres * numSpheres * 3, 1);
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

          // construct full gamma vector
          full_gamma(collision_idx * 3 + 0) =
              collisionMatrix[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                  .gamma_W(0);
          full_gamma(collision_idx * 3 + 1) =
              collisionMatrix[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                  .gamma_W(1);
          full_gamma(collision_idx * 3 + 2) =
              collisionMatrix[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                  .gamma_W(2);

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
      v_star.block<3, 1>(j * 3, 0) = velocity_vector.block<3, 1>(j * 3, 0) +
                                     dt * Eigen::Vector3d(0, 0, gravity);
    }
  }

  __syncwarp();
}

void CollisionEngine(Sphere* h_spheres, const int numProblems,
                     const int numSpheres,
                     CollisionData* h_collisionMatrixSpheres,
                     double* h_jacobian, double* h_gamma, int* h_num_collisions,
                     double* h_dynamic_matrix, double* h_velocity_vector,
                     double* h_v_star, double* h_phi0,
                     double* h_contact_stiffness, double* h_contact_damping) {
  // Device memory allocations
  Sphere* d_spheres;
  CollisionData* d_collisionMatrixSpheres;

  int* d_num_collisions;
  double* d_jacobian;
  double* d_dynamic_matrix;  // for now, we deal with 3DOF per body, so A matrix
                             // is 3*numsphere x 3*numsphere
  double* d_velocity_vector;  // for now, we deal with 3DOF per body, so
                              // velocity vector is 3*numsphere x 1
  double* d_gamma;
  double* d_v_star;
  double* d_phi0;
  double* d_contact_stiffness;
  double* d_contact_damping;

  HANDLE_ERROR(cudaMalloc((void**)&d_spheres,
                          numProblems * numSpheres * sizeof(Sphere)));
  HANDLE_ERROR(cudaMalloc(
      (void**)&d_collisionMatrixSpheres,
      numProblems * numSpheres * numSpheres * sizeof(CollisionData)));
  HANDLE_ERROR(cudaMalloc((void**)&d_jacobian,
                          numProblems * sizeof(double) * (numSpheres * 3) *
                              numSpheres * numSpheres * 3));
  HANDLE_ERROR(cudaMalloc((void**)&d_gamma, numProblems * sizeof(double) *
                                                numSpheres * numSpheres * 3));
  HANDLE_ERROR(
      cudaMalloc((void**)&d_num_collisions, numProblems * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(
      (void**)&d_dynamic_matrix,
      numProblems * sizeof(double) * numSpheres * 3 * numSpheres * 3));
  HANDLE_ERROR(cudaMalloc((void**)&d_velocity_vector,
                          numProblems * sizeof(double) * numSpheres * 3));
  HANDLE_ERROR(cudaMalloc((void**)&d_v_star,
                          numProblems * sizeof(double) * numSpheres * 3));
  HANDLE_ERROR(cudaMalloc(
      (void**)&d_phi0, numProblems * sizeof(double) * numSpheres * numSpheres));
  HANDLE_ERROR(
      cudaMalloc((void**)&d_contact_stiffness,
                 numProblems * sizeof(double) * numSpheres * numSpheres));
  HANDLE_ERROR(
      cudaMalloc((void**)&d_contact_damping,
                 numProblems * sizeof(double) * numSpheres * numSpheres));

  // Copy data to device
  HANDLE_ERROR(cudaMemcpy(d_spheres, h_spheres,
                          numProblems * numSpheres * sizeof(Sphere),
                          cudaMemcpyHostToDevice));

  // Set jacobian, num_collisions, full gamma vector to zero
  HANDLE_ERROR(cudaMemset(d_jacobian, 0,
                          numProblems * sizeof(double) * (numSpheres * 3) *
                              numSpheres * numSpheres * 3));
  HANDLE_ERROR(cudaMemset(d_num_collisions, 0, numProblems * sizeof(int)));
  HANDLE_ERROR(cudaMemset(
      d_gamma, 0, numProblems * sizeof(double) * numSpheres * numSpheres * 3));
  HANDLE_ERROR(cudaMemset(
      d_dynamic_matrix, 0,
      numProblems * sizeof(double) * numSpheres * 3 * numSpheres * 3));
  HANDLE_ERROR(cudaMemset(d_velocity_vector, 0,
                          numProblems * sizeof(double) * numSpheres * 3));
  HANDLE_ERROR(
      cudaMemset(d_v_star, 0, numProblems * sizeof(double) * numSpheres * 3));
  HANDLE_ERROR(cudaMemset(
      d_phi0, 0, numProblems * sizeof(double) * numSpheres * numSpheres));
  HANDLE_ERROR(
      cudaMemset(d_contact_stiffness, 0,
                 numProblems * sizeof(double) * numSpheres * numSpheres));
  HANDLE_ERROR(
      cudaMemset(d_contact_damping, 0,
                 numProblems * sizeof(double) * numSpheres * numSpheres));

  // Kernel launches
  int threadsPerBlock = 32;
  int blocksPerGridSpheres = numProblems;
  DetectSphereCollisions<<<blocksPerGridSpheres, threadsPerBlock>>>(
      d_spheres, numProblems, numSpheres, d_collisionMatrixSpheres);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Construct Jacobian matrix and Gamma vector
  ConstructJacobianGamma<<<blocksPerGridSpheres, threadsPerBlock>>>(
      d_spheres, numProblems, numSpheres, d_collisionMatrixSpheres, d_jacobian,
      d_gamma, d_num_collisions, d_phi0, d_contact_stiffness,
      d_contact_damping);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Construct Dynamic matrix
  ConstructDynamicMatrixVelocityVector<<<blocksPerGridSpheres,
                                         threadsPerBlock>>>(
      d_spheres, numProblems, numSpheres, d_dynamic_matrix, d_velocity_vector);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Calculate free motion velocity vector Dynamic matrix
  CalculateFreeMotionVelocity<<<blocksPerGridSpheres, threadsPerBlock>>>(
      d_spheres, numProblems, numSpheres, d_velocity_vector, d_v_star);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Copy results back to host
  HANDLE_ERROR(
      cudaMemcpy(h_collisionMatrixSpheres, d_collisionMatrixSpheres,
                 numProblems * numSpheres * numSpheres * sizeof(CollisionData),
                 cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_jacobian, d_jacobian,
                          numProblems * sizeof(double) * (numSpheres * 3) *
                              numSpheres * numSpheres * 3,
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(h_gamma, d_gamma,
                 numProblems * sizeof(double) * numSpheres * numSpheres * 3,
                 cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_num_collisions, d_num_collisions,
                          numProblems * sizeof(int), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(h_dynamic_matrix, d_dynamic_matrix,
                 numProblems * sizeof(double) * numSpheres * 3 * numSpheres * 3,
                 cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_velocity_vector, d_velocity_vector,
                          numProblems * sizeof(double) * numSpheres * 3,
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_v_star, d_v_star,
                          numProblems * sizeof(double) * numSpheres * 3,
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(
      h_phi0, d_phi0, numProblems * sizeof(double) * numSpheres * numSpheres,
      cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(h_contact_stiffness, d_contact_stiffness,
                 numProblems * sizeof(double) * numSpheres * numSpheres,
                 cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(h_contact_damping, d_contact_damping,
                 numProblems * sizeof(double) * numSpheres * numSpheres,
                 cudaMemcpyDeviceToHost));

  // Free device memory
  cudaFree(d_spheres);
  cudaFree(d_collisionMatrixSpheres);
  cudaFree(d_jacobian);
  cudaFree(d_num_collisions);
  cudaFree(d_gamma);
}
