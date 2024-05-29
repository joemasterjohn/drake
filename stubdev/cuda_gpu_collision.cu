// GPU collision engine
// The class CollisionGPUData is used to perform data mapping, storage, and
// operations on GPU collision data
#include <stdio.h>

#include <iostream>

#include "cuda_gpu_collision.cuh"
#include <cuda_runtime.h>

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

void CollisionGPUData::InitializePlane(Plane* h_planes, int m_num_planes) {
  // update problem size
  this->num_planes = m_num_planes;

  // allocte memory for planes
  HANDLE_ERROR(
      cudaMalloc((void**)&d_planes, num_problems * num_planes * sizeof(Plane)));
  HANDLE_ERROR(cudaMalloc(
      (void**)&d_collisionMatrixSpherePlane,
      num_problems * num_planes * num_spheres * sizeof(CollisionData)));

  // copy data to device
  HANDLE_ERROR(cudaMemcpy(d_planes, h_planes,
                          num_problems * num_planes * sizeof(Plane),
                          cudaMemcpyHostToDevice));
}

void CollisionGPUData::CopyStructToGPU() {
  // copy struct to device
  HANDLE_ERROR(cudaMalloc(&d_collision_gpu_data, sizeof(CollisionGPUData)));
  HANDLE_ERROR(cudaMemcpy(d_collision_gpu_data, this, sizeof(CollisionGPUData),
                          cudaMemcpyHostToDevice));
}

void CollisionGPUData::Destroy() {
  HANDLE_ERROR(cudaFree(d_spheres));
  HANDLE_ERROR(cudaFree(d_planes));
  HANDLE_ERROR(cudaFree(d_collisionMatrixSpheres));
  HANDLE_ERROR(cudaFree(d_collisionMatrixSpherePlane));
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