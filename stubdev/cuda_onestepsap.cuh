#pragma once

#include "cuda_onestepsap.h"

// define a SAP data strucutre
struct SAPGPUData {
  __device__ Eigen::Ref<Eigen::MatrixXd> A() {
    int row_size = 3 * num_rbodies;
    int col_size = 3 * num_rbodies;
    return Eigen::Map<Eigen::MatrixXd>(
        A_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Ref<Eigen::MatrixXd> v_star() {
    int row_size = 3 * num_rbodies;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_star_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Ref<Eigen::MatrixXd> J() {
    int row_size = 3 * num_contacts;
    int col_size = num_rbodies;
    return Eigen::Map<Eigen::MatrixXd>(
        J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Ref<Eigen::MatrixXd> G() {
    int row_size = 3 * num_contacts;
    int col_size = 3 * num_contacts;
    return Eigen::Map<Eigen::MatrixXd>(
        G_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Ref<Eigen::MatrixXd> v_gamma() {
    int row_size = 3;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_gamma_global +
            (blockIdx.x * blockDim.x + threadIdx.x) * row_size * col_size,
        row_size, col_size);
  }

  __device__ Eigen::Ref<Eigen::MatrixXd> v_R() {
    int row_size = 3;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXd>(
        v_R_global +
            (blockIdx.x * blockDim.x + threadIdx.x) * row_size * col_size,
        row_size, col_size);
  }

  void MakeSAPGPUData(std::vector<SAPCPUData> data) {
    this->num_contacts = data[0].num_contacts;
    this->num_rbodies = data[0].num_rbodies;
    this->num_equations = data.size();

    // Malloc for all pointers
    cudaMalloc(&A_global, num_equations * 3 * num_rbodies * 3 * sizeof(double));
    cudaMalloc(&v_star_global,
               num_equations * num_rbodies * 3 * sizeof(double));
    cudaMalloc(&J_global,
               num_equations * num_contacts * 3 * num_rbodies * sizeof(double));
    cudaMalloc(&G_global, num_equations * num_contacts * 3 * num_contacts * 3 *
                              sizeof(double));
    cudaMalloc(&v_gamma_global,
               num_equations * num_contacts * 3 * sizeof(double));
    cudaMalloc(&v_R_global,
               num_equations * num_contacts * 3 * 3 * sizeof(double));
    cudaMalloc(&delta_p_global, num_equations * 3 * num_rbodies * 3 *
                                    num_rbodies * sizeof(double));
    cudaMalloc(&lambdam_global, num_equations * sizeof(double));  // 1D vector
    cudaMalloc(&v_lambdar_intermediate_global,
               num_equations * num_contacts * 3 * sizeof(double));
    cudaMalloc(&v_lambdar_global,
               num_equations * num_contacts * sizeof(double));
    cudaMalloc(&lambdar_global, num_equations * sizeof(double));

    // Copy data to GPU
    for (int i = 0; i < num_equations; i++) {
      cudaMemcpy(A_global + i * 3 * num_rbodies * 3 * num_rbodies,
                 data[i].A.data(),
                 3 * num_rbodies * 3 * num_rbodies * sizeof(double),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(v_star_global + i * num_rbodies * 3, data[i].v_star.data(),
                 num_rbodies * 3 * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(J_global + i * num_contacts * 3 * num_rbodies,
                 data[i].constraint_data.J.data(),
                 num_contacts * 3 * num_rbodies * sizeof(double),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(G_global + i * num_contacts * 3 * num_contacts * 3,
                 data[i].constraint_data.G.data(),
                 num_contacts * 3 * num_contacts * 3 * sizeof(double),
                 cudaMemcpyHostToDevice);

      for (int j = 0; j < num_contacts; j++) {
        cudaMemcpy(v_gamma_global + i * num_contacts * 3 + j * 3,
                   data[i].v_gamma[j].data(), 3 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(v_R_global + i * num_contacts * 3 * 3 + j * 3 * 3,
                   data[i].v_R[j].data(), 3 * 3 * sizeof(double),
                   cudaMemcpyHostToDevice);
      }
    }
  }

 private:
  double* A_global;        // Global memory A matrix for all sims
  double* v_star_global;   // Global memory v_star for all sims
  double* J_global;        // Global memory J matrix for all sims
  double* G_global;        // Global memory G matrix for all sims
  double* v_gamma_global;  // Global memory v_gamma for all sims
  double* v_R_global;      // Global memory v_R for all sims

  double* delta_p_global;  // Global memory delta_p for all sims
  double* lambdam_global;  // Global memory lambda_m for all sims

  double* v_lambdar_intermediate_global;  // Global memory
                                          // v_lambdar_intermediate for all sims
  double* v_lambdar_global;  // Global memory v_lambdar for all sims
  double* lambdar_global;    // Global memory lambda_r for all sims

  int num_contacts;
  int num_equations;
  int num_rbodies;
};