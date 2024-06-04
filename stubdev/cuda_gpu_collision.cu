#include <stdio.h>

#include <iostream>

#include "cuda_gpu_collision.h"

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
__host__ __device__ CollisionData checkSphereCollision(const Sphere& a,
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

    data.phi0 = distLength - radiusSum;

    data.p_WC = midpoint;

    // Get collision frame matrix
    // Random vector v is default to {1.0, 1.0, 1.0}
    Eigen::Vector3d v(1.0, 1.0, 1.0);
    v.normalize();

    double y_hat_temp = v.dot(data.nhat_BA_W);
    Eigen::Vector3d y_hat = v - y_hat_temp * data.nhat_BA_W;
    y_hat.normalize();
    Eigen::Vector3d x_hat = y_hat.cross(data.nhat_BA_W);

    data.R(0, 0) = x_hat(0);
    data.R(0, 1) = x_hat(1);
    data.R(0, 2) = x_hat(2);
    data.R(1, 0) = y_hat(0);
    data.R(1, 1) = y_hat(1);
    data.R(1, 2) = y_hat(2);
    data.R(2, 0) = data.nhat_BA_W(0);
    data.R(2, 1) = data.nhat_BA_W(1);
    data.R(2, 2) = data.nhat_BA_W(2);
  }

  return data;
}

// Kernel to detect collisions between Spheres
__global__ void detectSphereCollisions(const Sphere* spheres, int numProblems,
                                       int numSpheres,
                                       CollisionData* collisionMatrix,
                                       int offset) {
  int idx = threadIdx.x + offset;
  int p_idx = blockIdx.x;

  if (idx >= numSpheres) return;
  if (p_idx >= numProblems) return;

  for (int j = idx; j < numSpheres; j++) {
    if (idx != j) {
      collisionMatrix[(p_idx * numSpheres * numSpheres) + idx * numSpheres +
                      j] =
          checkSphereCollision(spheres[p_idx * numSpheres + idx],
                               spheres[p_idx * numSpheres + j]);
    }
  }
}

void collision_engine(Sphere* h_spheres, const int numProblems,
                      const int numSpheres,
                      CollisionData* h_collisionMatrixSpheres) {
  // Device memory allocations
  Sphere* d_spheres;
  CollisionData* d_collisionMatrixSpheres;

  HANDLE_ERROR(cudaMalloc((void**)&d_spheres,
                          numProblems * numSpheres * sizeof(Sphere)));
  HANDLE_ERROR(cudaMalloc(
      (void**)&d_collisionMatrixSpheres,
      numProblems * numSpheres * numSpheres * sizeof(CollisionData)));
  // Copy data to device
  HANDLE_ERROR(cudaMemcpy(d_spheres, h_spheres,
                          numProblems * numSpheres * sizeof(Sphere),
                          cudaMemcpyHostToDevice));

  // Kernel launches
  int threadsPerBlock = 32;
  int blocksPerGridSpheres = numProblems;

  int offset = 0;
  while (offset < numSpheres) {
    detectSphereCollisions<<<blocksPerGridSpheres, threadsPerBlock>>>(
        d_spheres, numProblems, numSpheres, d_collisionMatrixSpheres, offset);
    offset += 32;
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  // Copy results back to host
  HANDLE_ERROR(
      cudaMemcpy(h_collisionMatrixSpheres, d_collisionMatrixSpheres,
                 numProblems * numSpheres * numSpheres * sizeof(CollisionData),
                 cudaMemcpyDeviceToHost));

  // Free device memory
  cudaFree(d_spheres);
  cudaFree(d_collisionMatrixSpheres);
}
