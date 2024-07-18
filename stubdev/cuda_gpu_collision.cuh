#pragma once

#include "cuda_gpu_collision.h"
// CUDA error handeling
// =====================
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
// =====================

//
// define a SAP data strucutre
struct CollisionGPUData {
  void Initialize(Sphere* h_spheres, int m_num_problems, int m_num_spheres) {
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
    HANDLE_ERROR(cudaMemset(d_v_star, 0,
                            num_problems * sizeof(double) * num_spheres * 3));
    HANDLE_ERROR(cudaMemset(
        d_phi0, 0, num_problems * sizeof(double) * num_spheres * num_spheres));
    HANDLE_ERROR(
        cudaMemset(d_contact_stiffness, 0,
                   num_problems * sizeof(double) * num_spheres * num_spheres));
    HANDLE_ERROR(
        cudaMemset(d_contact_damping, 0,
                   num_problems * sizeof(double) * num_spheres * num_spheres));
  }

  void Update() {
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
    HANDLE_ERROR(cudaMemset(d_v_star, 0,
                            num_problems * sizeof(double) * num_spheres * 3));
    HANDLE_ERROR(cudaMemset(
        d_phi0, 0, num_problems * sizeof(double) * num_spheres * num_spheres));
    HANDLE_ERROR(
        cudaMemset(d_contact_stiffness, 0,
                   num_problems * sizeof(double) * num_spheres * num_spheres));
    HANDLE_ERROR(
        cudaMemset(d_contact_damping, 0,
                   num_problems * sizeof(double) * num_spheres * num_spheres));
  }

  void Destroy() {
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

  // CPU data retrival, assuming the CPU data is already allocated
  void RetrieveCollisionDataToCPU(CollisionData* h_collisionMatrixSpheres) {
    HANDLE_ERROR(cudaMemcpy(
        h_collisionMatrixSpheres, d_collisionMatrixSpheres,
        num_problems * num_spheres * num_spheres * sizeof(CollisionData),
        cudaMemcpyDeviceToHost));
  }

  void RetrieveJacobianToCPU(double* h_jacobian) {
    HANDLE_ERROR(cudaMemcpy(h_jacobian, d_jacobian,
                            num_problems * sizeof(double) * (num_spheres * 3) *
                                num_spheres * num_spheres * 3,
                            cudaMemcpyDeviceToHost));
  }

  void RetrieveNumCollisionsToCPU(int* h_num_collisions) {
    HANDLE_ERROR(cudaMemcpy(h_num_collisions, d_num_collisions,
                            num_problems * sizeof(int),
                            cudaMemcpyDeviceToHost));
  }

  void RetrieveDynamicMatrixToCPU(double* h_dynamic_matrix) {
    HANDLE_ERROR(cudaMemcpy(
        h_dynamic_matrix, d_dynamic_matrix,
        num_problems * sizeof(double) * num_spheres * 3 * num_spheres * 3,
        cudaMemcpyDeviceToHost));
  }

  void RetrieveVelocityVectorToCPU(double* h_velocity_vector) {
    HANDLE_ERROR(cudaMemcpy(h_velocity_vector, d_velocity_vector,
                            num_problems * sizeof(double) * num_spheres * 3,
                            cudaMemcpyDeviceToHost));
  }

  void RetrieveVStarToCPU(double* h_v_star) {
    HANDLE_ERROR(cudaMemcpy(h_v_star, d_v_star,
                            num_problems * sizeof(double) * num_spheres * 3,
                            cudaMemcpyDeviceToHost));
  }

  void RetrievePhi0ToCPU(double* h_phi0) {
    HANDLE_ERROR(
        cudaMemcpy(h_phi0, d_phi0,
                   num_problems * sizeof(double) * num_spheres * num_spheres,
                   cudaMemcpyDeviceToHost));
  }

  void RetrieveContactStiffnessToCPU(double* h_contact_stiffness) {
    HANDLE_ERROR(
        cudaMemcpy(h_contact_stiffness, d_contact_stiffness,
                   num_problems * sizeof(double) * num_spheres * num_spheres,
                   cudaMemcpyDeviceToHost));
  }

  void RetrieveContactDampingToCPU(double* h_contact_damping) {
    HANDLE_ERROR(
        cudaMemcpy(h_contact_damping, d_contact_damping,
                   num_problems * sizeof(double) * num_spheres * num_spheres,
                   cudaMemcpyDeviceToHost));
  }

  void RetieveSphereDataToCPU(Sphere* h_spheres) {
    HANDLE_ERROR(cudaMemcpy(h_spheres, d_spheres,
                            num_problems * num_spheres * sizeof(Sphere),
                            cudaMemcpyDeviceToHost));
  }

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
