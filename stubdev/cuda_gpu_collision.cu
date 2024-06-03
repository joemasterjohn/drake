#include <stdio.h>

#include <iostream>

#include "cuda_gpu_collision.h"

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
    // step 1 - generate a random vector using eigen
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
__global__ void detectSphereCollisions(const Sphere* spheres, int numSpheres,
                                       CollisionData* collisionMatrix,
                                       int offset) {
  int idx = threadIdx.x + offset;
  if (idx < numSpheres) {
    for (int j = idx; j < numSpheres; j++) {
      if (idx != j) {
        collisionMatrix[idx * numSpheres + j] =
            checkSphereCollision(spheres[idx], spheres[j]);
      }
    }
  }
}

void collision_engine(Sphere* h_spheres, const int numSpheres,
                      CollisionData* h_collisionMatrixSpheres) {
  // Device memory allocations
  Sphere* d_spheres;
  CollisionData* d_collisionMatrixSpheres;

  cudaMalloc((void**)&d_spheres, numSpheres * sizeof(Sphere));
  cudaMalloc((void**)&d_collisionMatrixSpheres,
             numSpheres * numSpheres * sizeof(CollisionData));
  // Copy data to device
  cudaMemcpy(d_spheres, h_spheres, numSpheres * sizeof(Sphere),
             cudaMemcpyHostToDevice);

  // Kernel launches
  int threadsPerBlock = 32;
  int blocksPerGridSpheres = 1;

  int offset = 0;
  while (offset < numSpheres) {
    detectSphereCollisions<<<blocksPerGridSpheres, threadsPerBlock>>>(
        d_spheres, numSpheres, d_collisionMatrixSpheres, offset);
    offset += 32;
    cudaDeviceSynchronize();
  }

  // Copy results back to host
  cudaMemcpy(h_collisionMatrixSpheres, d_collisionMatrixSpheres,
             numSpheres * numSpheres * sizeof(CollisionData),
             cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_spheres);
  cudaFree(d_collisionMatrixSpheres);
}
