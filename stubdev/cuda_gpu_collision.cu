#include <stdio.h>

#include <iostream>

#include "cuda_gpu_collision.h"
#include <cuda_runtime.h>

#ifndef dt
#define dt 0.01
#endif

#ifndef gravity
#define gravity -9.81
#endif

#ifndef HANDLE_ERROR_MACRO
#define HANDLE_ERROR_MACRO
static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
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

void CollisionGPUData::Initialize(Sphere* h_spheres, int m_num_problems,
                                  int m_num_spheres) {
  // update problem size
  this->num_problems = m_num_problems;
  this->num_spheres = m_num_spheres;

  // allocate memory
  HANDLE_ERROR(cudaMalloc((void**)&d_spheres,
                          num_problems * num_spheres * sizeof(Sphere)));
  HANDLE_ERROR(cudaMalloc(
      (void**)&d_collisionMatrixSpheres,
      num_problems * num_spheres * num_spheres * sizeof(CollisionData)));
  HANDLE_ERROR(cudaMalloc((void**)&d_jacobian,
                          num_problems * sizeof(double) * (num_spheres * 3) *
                              num_spheres * num_spheres * 3));
  HANDLE_ERROR(
      cudaMalloc((void**)&d_num_collisions, num_problems * sizeof(int)));
  HANDLE_ERROR(cudaMalloc(
      (void**)&d_dynamic_matrix,
      num_problems * sizeof(double) * num_spheres * 3 * num_spheres * 3));
  HANDLE_ERROR(cudaMalloc((void**)&d_velocity_vector,
                          num_problems * sizeof(double) * num_spheres * 3));
  HANDLE_ERROR(cudaMalloc((void**)&d_v_star,
                          num_problems * sizeof(double) * num_spheres * 3));
  HANDLE_ERROR(cudaMalloc((void**)&d_phi0, num_problems * sizeof(double) *
                                               num_spheres * num_spheres));
  HANDLE_ERROR(
      cudaMalloc((void**)&d_contact_stiffness,
                 num_problems * sizeof(double) * num_spheres * num_spheres));
  HANDLE_ERROR(
      cudaMalloc((void**)&d_contact_damping,
                 num_problems * sizeof(double) * num_spheres * num_spheres));

  // copy data to device
  HANDLE_ERROR(cudaMemcpy(d_spheres, h_spheres,
                          num_problems * num_spheres * sizeof(Sphere),
                          cudaMemcpyHostToDevice));

  // set data to 0
  HANDLE_ERROR(cudaMemset(d_jacobian, 0,
                          num_problems * sizeof(double) * (num_spheres * 3) *
                              num_spheres * num_spheres * 3));
  HANDLE_ERROR(cudaMemset(d_num_collisions, 0, num_problems * sizeof(int)));
  HANDLE_ERROR(cudaMemset(
      d_dynamic_matrix, 0,
      num_problems * sizeof(double) * num_spheres * 3 * num_spheres * 3));
  HANDLE_ERROR(cudaMemset(d_velocity_vector, 0,
                          num_problems * sizeof(double) * num_spheres * 3));
  HANDLE_ERROR(
      cudaMemset(d_v_star, 0, num_problems * sizeof(double) * num_spheres * 3));
  HANDLE_ERROR(cudaMemset(
      d_phi0, 0, num_problems * sizeof(double) * num_spheres * num_spheres));
  HANDLE_ERROR(
      cudaMemset(d_contact_stiffness, 0,
                 num_problems * sizeof(double) * num_spheres * num_spheres));
  HANDLE_ERROR(
      cudaMemset(d_contact_damping, 0,
                 num_problems * sizeof(double) * num_spheres * num_spheres));
}

void CollisionGPUData::Update() {
  // set data to 0
  HANDLE_ERROR(cudaMemset(d_jacobian, 0,
                          num_problems * sizeof(double) * (num_spheres * 3) *
                              num_spheres * num_spheres * 3));
  HANDLE_ERROR(cudaMemset(d_num_collisions, 0, num_problems * sizeof(int)));
  HANDLE_ERROR(cudaMemset(
      d_dynamic_matrix, 0,
      num_problems * sizeof(double) * num_spheres * 3 * num_spheres * 3));
  HANDLE_ERROR(cudaMemset(d_velocity_vector, 0,
                          num_problems * sizeof(double) * num_spheres * 3));
  HANDLE_ERROR(
      cudaMemset(d_v_star, 0, num_problems * sizeof(double) * num_spheres * 3));
  HANDLE_ERROR(cudaMemset(
      d_phi0, 0, num_problems * sizeof(double) * num_spheres * num_spheres));
  HANDLE_ERROR(
      cudaMemset(d_contact_stiffness, 0,
                 num_problems * sizeof(double) * num_spheres * num_spheres));
  HANDLE_ERROR(
      cudaMemset(d_contact_damping, 0,
                 num_problems * sizeof(double) * num_spheres * num_spheres));
}

void CollisionGPUData::Destroy() {
  HANDLE_ERROR(cudaFree(d_spheres));
  HANDLE_ERROR(cudaFree(d_collisionMatrixSpheres));
  HANDLE_ERROR(cudaFree(d_jacobian));
  HANDLE_ERROR(cudaFree(d_num_collisions));
  HANDLE_ERROR(cudaFree(d_dynamic_matrix));
  HANDLE_ERROR(cudaFree(d_velocity_vector));
  HANDLE_ERROR(cudaFree(d_phi0));
  HANDLE_ERROR(cudaFree(d_v_star));
  HANDLE_ERROR(cudaFree(d_contact_stiffness));
  HANDLE_ERROR(cudaFree(d_contact_damping));
}

void CollisionGPUData::RetrieveCollisionDataToCPU(
    CollisionData* h_collisionMatrixSpheres) {
  HANDLE_ERROR(cudaMemcpy(
      h_collisionMatrixSpheres, d_collisionMatrixSpheres,
      num_problems * num_spheres * num_spheres * sizeof(CollisionData),
      cudaMemcpyDeviceToHost));
}

void CollisionGPUData::RetrieveJacobianToCPU(double* h_jacobian) {
  HANDLE_ERROR(cudaMemcpy(h_jacobian, d_jacobian,
                          num_problems * sizeof(double) * (num_spheres * 3) *
                              num_spheres * num_spheres * 3,
                          cudaMemcpyDeviceToHost));
}

void CollisionGPUData::RetrieveNumCollisionsToCPU(int* h_num_collisions) {
  HANDLE_ERROR(cudaMemcpy(h_num_collisions, d_num_collisions,
                          num_problems * sizeof(int), cudaMemcpyDeviceToHost));
}

void CollisionGPUData::RetrieveDynamicMatrixToCPU(double* h_dynamic_matrix) {
  HANDLE_ERROR(cudaMemcpy(
      h_dynamic_matrix, d_dynamic_matrix,
      num_problems * sizeof(double) * num_spheres * 3 * num_spheres * 3,
      cudaMemcpyDeviceToHost));
}

void CollisionGPUData::RetrieveVelocityVectorToCPU(double* h_velocity_vector) {
  HANDLE_ERROR(cudaMemcpy(h_velocity_vector, d_velocity_vector,
                          num_problems * sizeof(double) * num_spheres * 3,
                          cudaMemcpyDeviceToHost));
}

void CollisionGPUData::RetrieveVStarToCPU(double* h_v_star) {
  HANDLE_ERROR(cudaMemcpy(h_v_star, d_v_star,
                          num_problems * sizeof(double) * num_spheres * 3,
                          cudaMemcpyDeviceToHost));
}

void CollisionGPUData::RetrievePhi0ToCPU(double* h_phi0) {
  HANDLE_ERROR(cudaMemcpy(
      h_phi0, d_phi0, num_problems * sizeof(double) * num_spheres * num_spheres,
      cudaMemcpyDeviceToHost));
}

void CollisionGPUData::RetrieveContactStiffnessToCPU(
    double* h_contact_stiffness) {
  HANDLE_ERROR(
      cudaMemcpy(h_contact_stiffness, d_contact_stiffness,
                 num_problems * sizeof(double) * num_spheres * num_spheres,
                 cudaMemcpyDeviceToHost));
}

void CollisionGPUData::RetrieveContactDampingToCPU(double* h_contact_damping) {
  HANDLE_ERROR(
      cudaMemcpy(h_contact_damping, d_contact_damping,
                 num_problems * sizeof(double) * num_spheres * num_spheres,
                 cudaMemcpyDeviceToHost));
}

void CollisionGPUData::RetieveSphereDataToCPU(Sphere* h_spheres) {
  HANDLE_ERROR(cudaMemcpy(h_spheres, d_spheres,
                          num_problems * num_spheres * sizeof(Sphere),
                          cudaMemcpyDeviceToHost));
}