
#include <iostream>

#include "cuda_cholesky.h"
#include "cuda_gauss_seidel.h"
#include "cuda_gpu_collision.h"
#include "cuda_matmul.h"
#include "cuda_onestepsap.h"

static void HandleError(cudaError_t err, const char* file, int line) {
  // CUDA error handeling from the "CUDA by example" book
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

void test_onestep_sap(std::vector<Eigen::MatrixXd>& v_guess,
                      std::vector<SAPGPUData>& v_sap_data, int num_rbodies,
                      int num_contacts, int num_equations) {
  std::cout << "test_onestep_sap called with " << num_equations << " equations"
            << std::endl;
  int threadsPerBlock = 32;
  int numBlocks = num_equations;

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
  int num_strides =
      (3 * num_rbodies * 1 + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  matrixLinOp_32thd_Kernel<<<numBlocks, threadsPerBlock>>>(
      d_v_guess, d_v_star, d_delta_v, 3 * num_rbodies, 1, LinOpType::SUB,
      num_strides, num_equations);
  cudaDeviceSynchronize();

  // reserve space for delta_P on GPU
  double* d_delta_P;
  size_t size_delta_P = num_equations * num_rbodies * 3 * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_delta_P, size_delta_P));

  // calculate delta_P = A * delta_v
  // reserve space for A on GPU
  double* d_A;
  size_t size_A =
      num_equations * num_contacts * 3 * num_contacts * 3 * sizeof(double);
  int A_rows = num_contacts * 3;
  int A_cols = num_contacts * 3;
  // copy A to device
  HANDLE_ERROR(cudaMalloc((void**)&d_A, size_A));
  for (int i = 0; i < num_equations; i++) {
    HANDLE_ERROR(cudaMemcpy(d_A + i * A_rows * A_cols, v_sap_data[i].A.data(),
                            A_rows * A_cols * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  int stride = (A_rows + threadsPerBlock - 1) / threadsPerBlock;
  matrixMultiply_32thd_Kernel<<<num_equations, 32>>>(
      d_A, d_delta_v, d_delta_P, 3 * num_contacts, 3 * num_contacts, 1, stride,
      num_equations);
}