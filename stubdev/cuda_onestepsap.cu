
#include <iostream>

#include "cuda_cholesky.h"
#include "cuda_gauss_seidel.h"
#include "cuda_gpu_collision.h"
#include "cuda_matmul.cuh"
#include "cuda_matmul.h"
#include "cuda_onestepsap.h"

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

void test_onestep_sap(std::vector<Eigen::MatrixXd>& v_guess,
                      std::vector<SAPGPUData>& v_sap_data,
                      std::vector<double>& v_lambda_m, int num_rbodies,
                      int num_contacts, int num_equations) {
  std::cout << "test_onestep_sap called with " << num_equations << " equations"
            << std::endl;
  int threadsPerBlock = 32;
  // allocate GPU memory to calculate v_guess - v_star
  double* d_v_guess;
  double* d_v_star;
  double* d_delta_v;

  size_t size_v_guess = num_equations * num_rbodies * 3 * sizeof(double);
  size_t size_v_star = num_equations * num_rbodies * 3 * sizeof(double);
  size_t size_delta_v = num_equations * num_rbodies * 3 * sizeof(double);

  HANDLE_ERROR(cudaMalloc((void**)&d_v_guess, size_v_guess));
  HANDLE_ERROR(cudaMalloc((void**)&d_v_star, size_v_star));
  HANDLE_ERROR(cudaMalloc((void**)&d_delta_v, size_delta_v));

  // Copy data to device
  for (int i = 0; i < num_equations; i++) {
    HANDLE_ERROR(cudaMemcpy(d_v_guess + i * num_rbodies * 3, v_guess[i].data(),
                            num_rbodies * 3 * sizeof(double),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_v_star + i * num_rbodies * 3, v_sap_data[i].v_star.data(),
                   num_rbodies * 3 * sizeof(double), cudaMemcpyHostToDevice));
  }

  // envoke substraction kernel
  int num_strides = (3 * num_rbodies + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  // Note: d_v_guess, d_v_star, and d_delta_v are all 3 * num_rbodies * 1
  matrixLinOp_32thd_Kernel<<<num_equations, threadsPerBlock>>>(
      d_v_guess, d_v_star, d_delta_v, 3 * num_rbodies, 1, LinOpType::SUB,
      num_strides, num_equations);
  cudaDeviceSynchronize();

  // // TEST CODE
  // // allocate space for delta_v on CPU
  // std::vector<Eigen::MatrixXd> delta_v(num_equations,
  //                                      Eigen::MatrixXd(num_rbodies * 3, 1));

  // // map to eigen
  // for (int i = 0; i < num_equations; i++) {
  //   // copy from d_delta_v to delta_v
  //   HANDLE_ERROR(cudaMemcpy(delta_v[i].data(), d_delta_v + i * num_rbodies *
  //   3,
  //                           num_rbodies * 3 * sizeof(double),
  //                           cudaMemcpyDeviceToHost));

  //   Eigen::MatrixXd delta_v_check = v_guess[i] - v_sap_data[i].v_star;

  //   // std::cout << "delta_v: " << std::endl;
  //   // std::cout << delta_v[i] << std::endl;
  //   // std::cout << "=================" << std::endl;

  //   std::cout << "error: " << (delta_v[i] - delta_v_check).norm() <<
  //   std::endl;
  // }

  // reserve space for delta_P on GPU
  // Note: delta_P is a 3 * num_contacts * 1 matrix
  double* d_delta_P;
  size_t size_delta_P = num_equations * num_rbodies * 3 * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_delta_P, size_delta_P));

  // calculate delta_P = A * delta_v
  // reserve space for A on GPU
  // Note: A is a 3 * num_rbodies by 3 * num_rbodies matrix
  double* d_A;
  size_t size_A =
      num_equations * num_rbodies * 3 * num_rbodies * 3 * sizeof(double);
  int A_rows = num_rbodies * 3;
  int A_cols = num_rbodies * 3;
  // copy A to device
  HANDLE_ERROR(cudaMalloc((void**)&d_A, size_A));
  for (int i = 0; i < num_equations; i++) {
    HANDLE_ERROR(cudaMemcpy(d_A + i * A_rows * A_cols, v_sap_data[i].A.data(),
                            A_rows * A_cols * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  // calculate delta_P = A * delta_v
  int stride = (A_rows + threadsPerBlock - 1) / threadsPerBlock;
  matrixMultiply_32thd_Kernel<<<num_equations, 32>>>(
      d_A, d_delta_v, d_delta_P, 3 * num_rbodies, 3 * num_rbodies, 1, stride,
      num_equations);
  cudaDeviceSynchronize();

  // //  TEST_CODE
  // // allocate space for P on CPU
  // double* P = new double[num_equations * num_rbodies * 3];
  // // copy from d_delta_P to P
  // HANDLE_ERROR(cudaMemcpy(P, d_delta_P, size_delta_P,
  // cudaMemcpyDeviceToHost));
  // // map to eigen
  // std::vector<Eigen::MatrixXd> P_eigen;
  // for (int i = 0; i < num_equations; i++) {
  //   Eigen::MatrixXd P_i = Eigen::Map<Eigen::MatrixXd>(P + i * num_rbodies *
  //   3,
  //                                                     3 * num_rbodies, 1);
  //   P_eigen.push_back(P_i);
  //   std::cout
  //       << "error: "
  //       << (P_i - v_sap_data[i].A * (v_guess[i] -
  //       v_sap_data[i].v_star)).norm()
  //       << std::endl;
  // }

  // calculate lambda = 0.5 * d_delta_v.transpose * d_delta_P
  // reserve space for lambda on GPU
  // Note: lambda is a scalar
  double* d_lambda;
  size_t size_lambda = num_equations * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_lambda, size_lambda));

  stride = 1;
  matrixMultiply_32thd_Kernel<<<num_equations, 32>>>(
      d_delta_v, d_delta_P, d_lambda, 1, 3 * num_rbodies, 1, stride,
      num_equations);
  cudaDeviceSynchronize();

  // copy results back
  HANDLE_ERROR(cudaMemcpy(v_lambda_m.data(), d_lambda, size_lambda,
                          cudaMemcpyDeviceToHost));

  // multiply the GPU computed results by 0.5 to finish the computation of
  // 0.5 * delta_v.transpose() * delta_p
  for (int i = 0; i < num_equations; i++) {
    v_lambda_m[i] *= 0.5;
  }
}