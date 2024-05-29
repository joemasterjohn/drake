// CollisionGPUData definition
// Geometry data structure definition
#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <eigen3/Eigen/Dense>

#ifndef dt
#define dt 0.01
#endif

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

// Define a plane geometry
struct Plane {
  Eigen::Vector3d n;
  Eigen::Vector3d p1;
  Eigen::Vector3d p2;

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
#if defined(__CUDACC__)
  __device__ const Sphere* get_spheres() const { return d_spheres; }

  __device__ Sphere* get_spheres() { return d_spheres; }

  __device__ const Plane* get_planes() const { return d_planes; }

  __device__ Plane* get_planes() { return d_planes; }

  __device__ const CollisionData* get_collision_matrix() const {
    return d_collisionMatrixSpheres;
  }

  __device__ CollisionData* get_collision_matrix() {
    return d_collisionMatrixSpheres;
  }

  __device__ const CollisionData* get_collision_matrix_sphere_plane() const {
    return d_collisionMatrixSpherePlane;
  }

  __device__ CollisionData* get_collision_matrix_sphere_plane() {
    return d_collisionMatrixSpherePlane;
  }

  __device__ const double* get_jacobian() const { return d_jacobian; }

  __device__ double* get_jacobian() { return d_jacobian; }

  __device__ const int* get_num_collisions() const { return d_num_collisions; }

  __device__ int* get_num_collisions() { return d_num_collisions; }

  __device__ const double* get_dynamic_matrix() const {
    return d_dynamic_matrix;
  }

  __device__ double* get_dynamic_matrix() { return d_dynamic_matrix; }

  __device__ const double* get_velocity_vector() const {
    return d_velocity_vector;
  }

  __device__ double* get_velocity_vector() { return d_velocity_vector; }

  __device__ const double* get_phi0() const { return d_phi0; }

  __device__ double* get_phi0() { return d_phi0; }

  __device__ const double* get_v_star() const { return d_v_star; }

  __device__ double* get_v_star() { return d_v_star; }

  __device__ const double* get_contact_stiffness() const {
    return d_contact_stiffness;
  }

  __device__ double* get_contact_stiffness() { return d_contact_stiffness; }

  __device__ const double* get_contact_damping() const {
    return d_contact_damping;
  }

  __device__ double* get_contact_damping() { return d_contact_damping; }

  __device__ int get_num_problems() const { return num_problems; }

  __device__ int get_num_spheres() const { return num_spheres; }

  __device__ int get_num_planes() const { return num_planes; }

  __device__ CollisionGPUData* get_collision_gpu_data() {
    return d_collision_gpu_data;
  }

#endif

  void Initialize(Sphere* h_spheres, int m_num_problems, int m_num_spheres);
  void InitializePlane(Plane* h_planes, int m_num_planes);
  void CopyStructToGPU();
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

  Plane* GetPlanePtr() { return d_planes; }

  CollisionData* GetCollisionMatrixPtr() { return d_collisionMatrixSpheres; }

  CollisionData* GetCollisionMatrixSpherePlanePtr() {
    return d_collisionMatrixSpherePlane;
  }

  double* GetJacobianPtr() { return d_jacobian; }

  int* GetNumCollisionsPtr() { return d_num_collisions; }

  double* GetDynamicMatrixPtr() { return d_dynamic_matrix; }

  double* GetVelocityVectorPtr() { return d_velocity_vector; }

  double* GetPhi0Ptr() { return d_phi0; }

  double* GetVStarPtr() { return d_v_star; }

  double* GetContactStiffnessPtr() { return d_contact_stiffness; }

  double* GetContactDampingPtr() { return d_contact_damping; }

  CollisionGPUData* GetCollisionGPUDataPtr() { return d_collision_gpu_data; }

  void CollisionEngine(const int numProblems, const int numSpheres,
                       const int numPlanes);

 private:
  Sphere* d_spheres;
  Plane* d_planes;
  CollisionData* d_collisionMatrixSpheres;
  CollisionData* d_collisionMatrixSpherePlane;
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
  int num_planes = 0;

  CollisionGPUData* d_collision_gpu_data;  // Storing GPU copy of SAPGPUData
};

#if defined(__CUDACC__)
// CUDA device function related to gpu collision engine

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

  if (distSquared < (radiusSum * radiusSum)) {
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
__device__ void DetectSphereCollisions(const Sphere* spheres, int numProblems,
                                       int numSpheres,
                                       CollisionData* collisionMatrix) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    for (int k = j + 1; k < numSpheres; k++) {
      collisionMatrix[(p_idx * numSpheres * numSpheres) + j * numSpheres + k] =
          CheckSphereCollision(spheres[p_idx * numSpheres + j],
                               spheres[p_idx * numSpheres + k]);
    }
  }
  __syncwarp();
}

// Device function to check Sphere-Sphere collision
__device__ CollisionData CheckSpherePlaneCollision(const Plane& p,
                                                   const Sphere& s) {
  CollisionData data = {
      false, {0, 0, 0}, {0, 0, 0}, 0, Eigen::Matrix3d::Zero()};

  double distLength = (s.center - p.p1).dot(p.n);

  if (distLength < s.radius) {
    data.isColliding = true;
    data.phi0 = -(distLength - s.radius);  // sign convention
    data.nhat_BA_W = p.n;
    data.nhat_BA_W.normalize();
    data.p_WC = s.center - data.nhat_BA_W * data.phi0;

    // Get collision frame matrix
    // Random vector v is default to {1.0, 1.0, 1.0}
    // This matrix is normal-dependent
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

    data.vn = -s.velocity.dot(data.nhat_BA_W);  // negative for departing,
                                                // positive for approaching
  } else {
    data.isColliding = false;
  }

  return data;
}

__device__ void DetectSpherePlaneCollisions(const Sphere* spheres,
                                            const Plane* planes,
                                            int numProblems, int numSpheres,
                                            int numPlanes,
                                            CollisionData* collisionMatrix) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    for (int k = 0; k < numPlanes; k++) {
      collisionMatrix[(p_idx * numPlanes * numSpheres) + k * numSpheres + j] =
          CheckSpherePlaneCollision(planes[p_idx * numPlanes + k],
                                    spheres[p_idx * numSpheres + j]);
    }
  }
  __syncwarp();
}

// Kernel to detect collisions between Spheres
__device__ void ConstructJacobianGamma(
    const Sphere* spheres, const Plane* planes, int numProblems, int numSpheres,
    int numPlanes, CollisionData* collisionMatrixSS,
    CollisionData* collisionMatrixSP, double* d_jacobian, int* d_num_collisions,
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

  // Construct Jacobian matrix for Sphere-Sphere collision
  for (int j = idx; j < numSpheres; j += blockDim.x) {
    for (int k = j + 1; k < numSpheres; k++) {
      if (collisionMatrixSS[(p_idx * numSpheres * numSpheres) + j * numSpheres +
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
            collisionMatrixSS[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                .R *
            Eigen::MatrixXd::Identity(3, 3);
        full_jacobian.block<3, 3>(collision_idx * 3, k * 3) =
            collisionMatrixSS[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                .R *
            -Eigen::MatrixXd::Identity(3, 3);

        // add data to phi0
        d_phi0[p_idx * numSpheres * numSpheres + collision_idx] =
            collisionMatrixSS[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                .phi0;
      }
    }
  }

  __syncwarp();

  // Construct Jacobian matrix for Sphere-Plane collision
  for (int j = idx; j < numSpheres; j += blockDim.x) {
    for (int k = 0; k < numPlanes; k++) {
      if (collisionMatrixSP[(p_idx * numSpheres * numPlanes) + k * numSpheres +
                            j]
              .isColliding) {
        int collision_idx = atomicAdd(&d_num_collisions[p_idx], 1);

        // update the harmonic mean of contact stiffness
        contact_stiffness[collision_idx] =
            (2 * spheres[p_idx * numSpheres + j].stiffness *
             planes[p_idx * numPlanes + k].stiffness) /
            (spheres[p_idx * numSpheres + j].stiffness +
             planes[p_idx * numPlanes + k].stiffness);

        // update the harmonic mean of contact damping
        contact_damping[collision_idx] =
            (2 * spheres[p_idx * numSpheres + j].damping *
             planes[p_idx * numPlanes + k].damping) /
            (spheres[p_idx * numSpheres + j].damping +
             planes[p_idx * numPlanes + k].damping);

        // construct Jacobian matrix
        full_jacobian.block<3, 3>(collision_idx * 3, j * 3) =
            collisionMatrixSP[(p_idx * numSpheres * numPlanes) +
                              k * numSpheres + j]
                .R *
            Eigen::MatrixXd::Identity(3, 3);

        // add data to phi0
        d_phi0[p_idx * numSpheres * numSpheres + collision_idx] =
            collisionMatrixSP[(p_idx * numSpheres * numPlanes) +
                              k * numSpheres + j]
                .phi0;
      }
    }
  }

  __syncwarp();
}

__device__ void ConstructDynamicMatrixVelocityVector(
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

__device__ void CalculateFreeMotionVelocity(const Sphere* spheres,
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
                                     dt * Eigen::Vector3d(0, -4.905, 0);
    }
  }

  __syncwarp();
}

__device__ void OneStepCollision(CollisionGPUData* data) {
  // update to reset d_jacobian and d_num_collisions to 0
  if (threadIdx.x == 0) data->get_num_collisions()[blockIdx.x] = 0;

  for (int i = threadIdx.x;
       i < data->get_num_spheres() * 3 * data->get_num_spheres() *
               data->get_num_spheres() * 3;
       i += blockDim.x) {
    data->get_jacobian()[blockIdx.x * data->get_num_spheres() * 3 *
                             data->get_num_spheres() * data->get_num_spheres() *
                             3 +
                         i] = 0.0;
  }

  __syncwarp();

  DetectSphereCollisions(data->get_spheres(), data->get_num_problems(),
                         data->get_num_spheres(), data->get_collision_matrix());
  DetectSpherePlaneCollisions(data->get_spheres(), data->get_planes(),
                              data->get_num_problems(), data->get_num_spheres(),
                              data->get_num_planes(),
                              data->get_collision_matrix_sphere_plane());
  ConstructJacobianGamma(
      data->get_spheres(), data->get_planes(), data->get_num_problems(),
      data->get_num_spheres(), data->get_num_planes(),
      data->get_collision_matrix(), data->get_collision_matrix_sphere_plane(),
      data->get_jacobian(), data->get_num_collisions(), data->get_phi0(),
      data->get_contact_stiffness(), data->get_contact_damping());
  ConstructDynamicMatrixVelocityVector(
      data->get_spheres(), data->get_num_problems(), data->get_num_spheres(),
      data->get_dynamic_matrix(), data->get_velocity_vector());
  CalculateFreeMotionVelocity(data->get_spheres(), data->get_num_problems(),
                              data->get_num_spheres(),
                              data->get_velocity_vector(), data->get_v_star());
  __syncwarp();
}

#endif