#include <stdio.h>

#include <iostream>

#include "cuda_matmul.h"

static void HandleError(cudaError_t err, const char* file, int line) {
  // CUDA error handeling from the "CUDA by example" book
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// ==========================================================================
// Matrix to Matrix multiplication using 32 threads per block
// Note: the sizes of two matrices are variable
// it can be matrix-matrix or matrix-vector
// ==========================================================================

__global__ void matrixMultiply_32hd_Kernel(double* v_A, double* v_B,
                                           double* v_C, int A_row, int A_col,
                                           int B_col, int stride,
                                           int num_eqations) {
  extern __shared__ double sums[];

  int thread_idx = threadIdx.x;
  int equ_idx = blockIdx.x;

  Eigen::Map<Eigen::MatrixXd> d_A(v_A + equ_idx * A_row * A_col, A_row, A_col);
  Eigen::Map<Eigen::VectorXd> d_B(v_B + equ_idx * A_col * B_col, A_col, B_col);
  Eigen::Map<Eigen::MatrixXd> d_C(v_C + equ_idx * A_row * B_col, A_row, B_col);

  for (int k = 0; k < B_col; k++) {
    for (int j = 0; j < A_col; j++) {
      for (int i = 0; i < stride; i++) {
        int row = i * 32 + thread_idx;
        int col = j;

        if (row < A_row && col < A_col) {
          if (col == 0) {
            sums[row] = 0.0;
          }

          if (row < A_row) {
            sums[row] += d_A(row, col) * d_B(col, k);
          }

          if (col == A_col - 1) {
            d_C(row, k) = sums[row];
          }
        }
      }
    }
  }
}

void matrixMultiply_32thd(std::vector<Eigen::MatrixXd>& v_A,
                          std::vector<Eigen::MatrixXd>& v_B,
                          std::vector<Eigen::MatrixXd>& v_C,
                          int num_equations) {
  int M = v_A[0].rows();
  int N = v_A[0].cols();
  int K = v_B[0].cols();

  size_t size_vA = num_equations * M * N * sizeof(double);
  size_t size_vB = num_equations * N * K * sizeof(double);
  size_t size_vC = num_equations * M * K * sizeof(double);

  double* d_vA;
  double* d_vB;
  double* d_vC;

  // Allocate device memory
  HANDLE_ERROR(cudaMalloc((void**)&d_vA, size_vA));
  HANDLE_ERROR(cudaMalloc((void**)&d_vB, size_vB));
  HANDLE_ERROR(cudaMalloc((void**)&d_vC, size_vC));

  // Copy data to device
  for (int i = 0; i < num_equations; i++) {
    HANDLE_ERROR(cudaMemcpy(d_vA + i * M * N, v_A[i].data(),
                            M * N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_vB + i * N * K, v_B[i].data(),
                            N * K * sizeof(double), cudaMemcpyHostToDevice));
  }

  // Define block and grid sizes
  int threadsPerBlock = 32;
  int numBlocks = num_equations;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  int stride = (M + threadsPerBlock - 1) / threadsPerBlock;
  // Launch kernel
  matrixMultiply_32hd_Kernel<<<numBlocks, threadsPerBlock,
                               2048 * sizeof(double)>>>(
      d_vA, d_vB, d_vC, M, N, K, stride, num_equations);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for matrixMultiply_32hd_Kernel: " << milliseconds
            << " ms\n";

  // Copy result back to host
  for (int i = 0; i < num_equations; i++) {
    HANDLE_ERROR(cudaMemcpy(v_C[i].data(), d_vC + i * M * K,
                            M * K * sizeof(double), cudaMemcpyDeviceToHost));
  }

  // Free device memory
  HANDLE_ERROR(cudaFree(d_vA));
  HANDLE_ERROR(cudaFree(d_vB));
  HANDLE_ERROR(cudaFree(d_vC));
}

// ==========================================================================
// Matrix to Matrix addition using 32 threads per block
// Note: the sizes of two matrices need to be the same to add them
// ==========================================================================

__global__ void matrixLinOp_32thd_Kernel(double* v_A, double* v_B,
                                         double* v_Res, int row, int col,
                                         LinOpType op, int num_strides,
                                         int num_eqations) {
  int thread_idx = threadIdx.x;
  int equ_idx = blockIdx.x;

  Eigen::Map<Eigen::MatrixXd> d_A(v_A + equ_idx * row * col, row, col);
  Eigen::Map<Eigen::VectorXd> d_B(v_B + equ_idx * row * col, row, col);
  Eigen::Map<Eigen::MatrixXd> d_Res(v_Res + equ_idx * row * col, row, col);

  for (int i = 0; i < num_strides; i++) {
    int cur_idx = i * 32 + thread_idx;
    int cur_row = cur_idx / 32;
    int cur_col = cur_idx % 32;

    if (cur_row < row && cur_col < col) {
      d_Res(cur_row, cur_col) =
          (op == LinOpType::ADD)
              ? d_A(cur_row, cur_col) + d_B(cur_row, cur_col)
              : d_A(cur_row, cur_col) - d_B(cur_row, cur_col);
    }
  }
}

void matrixLinOp_32hd(std::vector<Eigen::MatrixXd>& v_A,
                      std::vector<Eigen::MatrixXd>& v_B,
                      std::vector<Eigen::MatrixXd>& v_Res, LinOpType op,
                      int num_equations) {
  int M = v_A[0].rows();
  int N = v_A[0].cols();

  size_t size_vA = num_equations * M * N * sizeof(double);
  size_t size_vB = num_equations * M * N * sizeof(double);
  size_t size_vRes = num_equations * M * N * sizeof(double);

  double* d_vA;
  double* d_vB;
  double* d_vRes;

  // Allocate device memory
  HANDLE_ERROR(cudaMalloc((void**)&d_vA, size_vA));
  HANDLE_ERROR(cudaMalloc((void**)&d_vB, size_vB));
  HANDLE_ERROR(cudaMalloc((void**)&d_vRes, size_vRes));

  // Copy data to device
  for (int i = 0; i < num_equations; i++) {
    HANDLE_ERROR(cudaMemcpy(d_vA + i * M * N, v_A[i].data(),
                            M * N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_vB + i * M * N, v_B[i].data(),
                            N * N * sizeof(double), cudaMemcpyHostToDevice));
  }

  // Define block and grid sizes
  int threadsPerBlock = 32;
  int numBlocks = num_equations;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  int num_strides = (M * N + threadsPerBlock - 1) / threadsPerBlock;
  // Launch kernel
  matrixLinOp_32thd_Kernel<<<numBlocks, threadsPerBlock>>>(
      d_vA, d_vB, d_vRes, M, N, LinOpType::ADD, num_strides, num_equations);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for matrixLinOp_32thd_Kernel: " << milliseconds
            << " ms\n";

  // Copy result back to host
  for (int i = 0; i < num_equations; i++) {
    HANDLE_ERROR(cudaMemcpy(v_Res[i].data(), d_vRes + i * M * N,
                            M * N * sizeof(double), cudaMemcpyDeviceToHost));
  }

  // Free device memory
  HANDLE_ERROR(cudaFree(d_vA));
  HANDLE_ERROR(cudaFree(d_vB));
  HANDLE_ERROR(cudaFree(d_vRes));
}

// // ==========================================================================
// // Evaluate the gradient of the primal cost l_p
// // ==========================================================================
// void evalute_l_p_grad(std::vector<Eigen::MatrixXd>& v_A,
//                       std::vector<Eigen::MatrixXd>& v_v,
//                       std::vector<Eigen::MatrixXd>& v_v_m,
//                       std::vector<Eigen::MatrixXd>& v_J,
//                       std::vector<Eigen::MatrixXd>& v_gamma,
//                       std::vector<Eigen::MatrixXd>& v_grad_l_p,  // result
//                       int num_equations) {
//   // Allocate device memory
//   double *d_A, *d_v, *d_v_m, *d_J, *d_gamma, *d_grad_l_p;
// }
