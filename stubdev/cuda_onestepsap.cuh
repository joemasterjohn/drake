#pragma once

#include "cuda_onestepsap.h"

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
//
// define a SAP data strucutre
struct SAPGPUData {
  // Mutable get functions
  __device__ Eigen::Map<Eigen::MatrixXd> A() {
    int row_size = 3 * num_rbodies;
    int col_size = 3 * num_rbodies;
    return Eigen::Map<Eigen::MatrixXd>(
        A_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  // Const get functions
  __device__ const Eigen::Map<Eigen::MatrixXd> A() const {
    int row_size = 3 * num_rbodies;
    int col_size = 3 * num_rbodies;
    return Eigen::Map<Eigen::MatrixXd>(
        A_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> v_guess() {
    int row_size = 3 * num_rbodies;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_guess_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> v_guess() const {
    int row_size = 3 * num_rbodies;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_guess_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> v_star() {
    int row_size = 3 * num_rbodies;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_star_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> v_star() const {
    int row_size = 3 * num_rbodies;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_star_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> delta_v() {
    int row_size = 3 * num_rbodies;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> delta_v() const {
    int row_size = 3 * num_rbodies;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> delta_v_transpose() {
    int row_size = 1;
    int col_size = 3 * num_rbodies;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> delta_v_transpose() const {
    int row_size = 1;
    int col_size = 3 * num_rbodies;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> delta_p() {
    int row_size = 3;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> delta_p() const {
    int row_size = 3;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> J() {
    int row_size = 3 * num_contacts;
    int col_size = num_rbodies;
    return Eigen::Map<Eigen::MatrixXd>(
        J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> J() const {
    int row_size = 3 * num_contacts;
    int col_size = num_rbodies;
    return Eigen::Map<Eigen::MatrixXd>(
        J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> G() {
    int row_size = 3 * num_contacts;
    int col_size = 3 * num_contacts;
    return Eigen::Map<Eigen::MatrixXd>(
        G_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> G() const {
    int row_size = 3 * num_contacts;
    int col_size = 3 * num_contacts;
    return Eigen::Map<Eigen::MatrixXd>(
        G_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> v_gamma(int constraint_index) {
    int row_size = 3;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_gamma_global + (blockIdx.x * num_contacts + constraint_index) *
                             row_size * col_size,
        row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> v_gamma(
      int constraint_index) const {
    int row_size = 3;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_gamma_global + (blockIdx.x * num_contacts + constraint_index) *
                             row_size * col_size,
        row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> v_gamma_transpose() {
    int row_size = 1;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXd>(
        v_gamma_global +
            (blockIdx.x * blockDim.x + threadIdx.x) * row_size * col_size,
        row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> v_gamma_transpose() const {
    int row_size = 1;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXd>(
        v_gamma_global +
            (blockIdx.x * blockDim.x + threadIdx.x) * row_size * col_size,
        row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> v_R() {
    int row_size = 3;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXd>(
        v_R_global +
            (blockIdx.x * blockDim.x + threadIdx.x) * row_size * col_size,
        row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::Vector3d> v_R(int constraint_index) const {
    return Eigen::Map<Eigen::Vector3d>(
        v_R_global + (blockIdx.x * num_contacts + constraint_index) * 3, 3, 1);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> v_lambdar_intermediate() {
    int row_size = 1;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXd>(
        v_lambdar_intermediate_global +
            (blockIdx.x * blockDim.x + threadIdx.x) * row_size * col_size,
        row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> v_lambdar_intermediate() const {
    int row_size = 1;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXd>(
        v_lambdar_intermediate_global +
            (blockIdx.x * blockDim.x + threadIdx.x) * row_size * col_size,
        row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> lambda_m() {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        lambdam_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> lambda_m() const {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        lambdam_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> v_lambda_r() {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_lambdar_global + (blockIdx.x) * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> v_lambda_r() const {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_lambdar_global + (blockIdx.x) * row_size * col_size, row_size,
        col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> lambda_r() {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        lambdar_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> lambda_r() const {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        lambdar_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __host__ __device__ const int NumRBodies() const { return num_rbodies; }
  __host__ __device__ const int NumContacts() const { return num_contacts; }
  __host__ __device__ const int NumEquations() const { return num_equations; }

  // Retrival functions - copy data back to CPU
  void GetLambdaM(std::vector<double>& lambdam) {
    lambdam.resize(num_equations);
    cudaMemcpy(lambdam.data(), lambdam_global, num_equations * sizeof(double),
               cudaMemcpyDeviceToHost);
  }

  //   // Retrival functions - copy data back to CPU
  //   void GetDeltaP(std::vector<Eigen::MatrixXd>& deltap) {
  //     deltap.resize(num_equations);
  //     for (int i = 0; i < num_equations; i++) {
  //       Eigen::MatrixXd temp(3 * num_rbodies, 1);
  //       cudaMemcpy(temp.data(), delta_p_global + i * 3 * num_rbodies,
  //                  3 * num_rbodies * sizeof(double), cudaMemcpyDeviceToHost);
  //       deltap[i] = temp;
  //     }
  //   }

  //   // Retrival functions - copy data back to CPU
  //   void GetDeltaV(std::vector<Eigen::MatrixXd>& deltav) {
  //     deltav.resize(num_equations);
  //     for (int i = 0; i < num_equations; i++) {
  //       Eigen::MatrixXd temp(3 * num_rbodies, 1);
  //       cudaMemcpy(temp.data(), delta_v_global + i * 3 * num_rbodies,
  //                  3 * num_rbodies * sizeof(double), cudaMemcpyDeviceToHost);
  //       deltav[i] = temp;
  //     }
  //   }

  void MakeSAPGPUData(std::vector<SAPCPUData> data) {
    this->num_contacts = data[0].num_contacts;
    this->num_rbodies = data[0].num_rbodies;
    this->num_equations = data.size();

    // Malloc for all pointers
    HANDLE_ERROR(cudaMalloc(&A_global, num_equations * 3 * num_rbodies * 3 *
                                           num_rbodies * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&v_guess_global,
                            num_equations * 3 * num_rbodies * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&v_star_global,
                            num_equations * 3 * num_rbodies * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&delta_v_global,
                            num_equations * 3 * num_rbodies * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&J_global, num_equations * num_contacts * 3 *
                                           num_rbodies * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&G_global, num_equations * num_contacts * 3 *
                                           num_contacts * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&v_gamma_global,
                            num_equations * num_contacts * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(
        &v_R_global, num_equations * num_contacts * 3 * 3 * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(
        &delta_p_global,
        num_equations * 3 * num_rbodies * 3 * num_rbodies * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&lambdam_global,
                            num_equations * sizeof(double)));  // 1D vector
    HANDLE_ERROR(cudaMalloc(&v_lambdar_intermediate_global,
                            num_equations * num_contacts * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&v_lambdar_global,
                            num_equations * num_contacts * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&lambdar_global, num_equations * sizeof(double)));

    // Copy data to GPU
    for (int i = 0; i < num_equations; i++) {
      HANDLE_ERROR(cudaMemcpy(
          A_global + i * 3 * num_rbodies * 3 * num_rbodies, data[i].A.data(),
          3 * num_rbodies * 3 * num_rbodies * sizeof(double),
          cudaMemcpyHostToDevice));
      HANDLE_ERROR(
          cudaMemcpy(v_star_global + i * num_rbodies * 3, data[i].v_star.data(),
                     num_rbodies * 3 * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(
          v_guess_global + i * num_rbodies * 3, data[i].v_guess.data(),
          num_rbodies * 3 * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(J_global + i * num_contacts * 3 * num_rbodies,
                              data[i].constraint_data.J.data(),
                              num_contacts * 3 * num_rbodies * sizeof(double),
                              cudaMemcpyHostToDevice));
      HANDLE_ERROR(
          cudaMemcpy(G_global + i * num_contacts * 3 * num_contacts * 3,
                     data[i].constraint_data.G.data(),
                     num_contacts * 3 * num_contacts * 3 * sizeof(double),
                     cudaMemcpyHostToDevice));

      for (int j = 0; j < num_contacts; j++) {
        HANDLE_ERROR(cudaMemcpy(v_gamma_global + i * num_contacts * 3 + j * 3,
                                data[i].v_gamma[j].data(), 3 * sizeof(double),
                                cudaMemcpyHostToDevice));
        HANDLE_ERROR(
            cudaMemcpy(v_R_global + i * num_contacts * 3 * 3 + j * 3 * 3,
                       data[i].v_R[j].data(), 3 * 3 * sizeof(double),
                       cudaMemcpyHostToDevice));
      }
    }
  }

 private:
  double* A_global;        // Global memory A matrix for all sims
  double* v_star_global;   // Global memory v_star for all sims
  double* v_guess_global;  // Global memory v_guess for all sims
  double* J_global;        // Global memory J matrix for all sims
  double* G_global;        // Global memory G matrix for all sims
  double* v_gamma_global;  // Global memory v_gamma for all sims
  double* v_R_global;      // Global memory v_R for all sims

  double* delta_v_global;  // Global memory delta_v for all sims
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