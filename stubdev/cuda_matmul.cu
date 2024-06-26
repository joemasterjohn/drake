#include <stdio.h>

#include <iostream>

#include "cuda_matmul.cuh"
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
__device__ void MatrixMultiply32thdFunc(double* v_A, double* v_B, double* v_C,
                                        double* sums, int thread_idx,
                                        int equ_idx, int A_row, int A_col,
                                        int B_col, int stride) {
  Eigen::Map<Eigen::MatrixXd> d_A(v_A + equ_idx * A_row * A_col, A_row, A_col);
  Eigen::Map<Eigen::VectorXd> d_B(v_B + equ_idx * A_col * B_col, A_col, B_col);
  Eigen::Map<Eigen::MatrixXd> d_C(v_C + equ_idx * A_row * B_col, A_row, B_col);

  for (int k = 0; k < B_col; k++) {
    for (int j = 0; j < A_col; j++) {
      for (int i = 0; i < stride; i++) {
        int row = i * 32 + thread_idx;
        int col = j;
        if (row < A_row) {
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

// Kernel function serving as a wrapper
__global__ void MatrixMultiply32thdKernel(double* v_A, double* v_B, double* v_C,
                                          int A_row, int A_col, int B_col,
                                          int stride, int num_eqations) {
  extern __shared__ double sums[];

  int thread_idx = threadIdx.x;
  int equ_idx = blockIdx.x;

  if (equ_idx >= num_eqations) return;

  // Call the device function
  MatrixMultiply32thdFunc(v_A, v_B, v_C, sums, thread_idx, equ_idx, A_row,
                          A_col, B_col, stride);
}

// __global__ void MatrixMultiply32thdKernel(double* v_A, double* v_B, double*
// v_C,
//                                           int A_row, int A_col, int B_col,
//                                           int stride, int num_eqations) {
//   extern __shared__ double sums[];

//   int thread_idx = threadIdx.x;
//   int equ_idx = blockIdx.x;

//   if (equ_idx >= num_eqations) return;

//   Eigen::Map<Eigen::MatrixXd> d_A(v_A + equ_idx * A_row * A_col, A_row,
//   A_col); Eigen::Map<Eigen::VectorXd> d_B(v_B + equ_idx * A_col * B_col,
//   A_col, B_col); Eigen::Map<Eigen::MatrixXd> d_C(v_C + equ_idx * A_row *
//   B_col, A_row, B_col);

//   for (int k = 0; k < B_col; k++) {
//     for (int j = 0; j < A_col; j++) {
//       for (int i = 0; i < stride; i++) {
//         int row = i * 32 + thread_idx;
//         int col = j;
//         if (row < A_row) {
//           if (col == 0) {
//             sums[row] = 0.0;
//           }

//           if (row < A_row) {
//             sums[row] += d_A(row, col) * d_B(col, k);
//           }

//           if (col == A_col - 1) {
//             d_C(row, k) = sums[row];
//           }
//         }
//       }
//     }
//   }
// }

void MatrixMultiply32thdHost(std::vector<Eigen::MatrixXd>& v_A,
                             std::vector<Eigen::MatrixXd>& v_B,
                             std::vector<Eigen::MatrixXd>& v_C,
                             int num_problems) {
  int M = v_A[0].rows();
  int N = v_A[0].cols();
  int K = v_B[0].cols();

  size_t size_vA = num_problems * M * N * sizeof(double);
  size_t size_vB = num_problems * N * K * sizeof(double);
  size_t size_vC = num_problems * M * K * sizeof(double);

  double* d_vA;
  double* d_vB;
  double* d_vC;

  // Allocate device memory
  HANDLE_ERROR(cudaMalloc((void**)&d_vA, size_vA));
  HANDLE_ERROR(cudaMalloc((void**)&d_vB, size_vB));
  HANDLE_ERROR(cudaMalloc((void**)&d_vC, size_vC));

  // Copy data to device
  for (int i = 0; i < num_problems; i++) {
    HANDLE_ERROR(cudaMemcpy(d_vA + i * M * N, v_A[i].data(),
                            M * N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_vB + i * N * K, v_B[i].data(),
                            N * K * sizeof(double), cudaMemcpyHostToDevice));
  }

  // Define block and grid sizes
  int threadsPerBlock = 32;
  int numBlocks = num_problems;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  int stride = (M + threadsPerBlock - 1) / threadsPerBlock;
  // Launch kernel
  MatrixMultiply32thdKernel<<<numBlocks, threadsPerBlock,
                              2048 * sizeof(double)>>>(d_vA, d_vB, d_vC, M, N,
                                                       K, stride, num_problems);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for matrixMultiply_32hd_Kernel: " << milliseconds
            << " ms\n";

  // Copy result back to host
  for (int i = 0; i < num_problems; i++) {
    HANDLE_ERROR(cudaMemcpy(v_C[i].data(), d_vC + i * M * K,
                            M * K * sizeof(double), cudaMemcpyDeviceToHost));
  }

  // Free device memory
  HANDLE_ERROR(cudaFree(d_vA));
  HANDLE_ERROR(cudaFree(d_vB));
  HANDLE_ERROR(cudaFree(d_vC));
}

// ==========================================================================
// Matrix to Matrix addition and subtraction using 32 threads per block
// Note: the sizes of two matrices need to be the same to add them
// ==========================================================================

// Device function to perform linear operations (addition or subtraction)
__device__ void MatrixLinOp32thdFunc(double* v_A, double* v_B, double* v_Res,
                                     int thread_idx, int equ_idx, int row,
                                     int col, LinOpType op, int num_strides) {
  Eigen::Map<Eigen::MatrixXd> d_A(v_A + equ_idx * row * col, row, col);
  Eigen::Map<Eigen::VectorXd> d_B(v_B + equ_idx * row * col, row, col);
  Eigen::Map<Eigen::MatrixXd> d_Res(v_Res + equ_idx * row * col, row, col);

  for (int i = 0; i < num_strides; i++) {
    int cur_idx = i * 32 + thread_idx;
    if (cur_idx >= row * col) continue;
    int cur_col = cur_idx / row;
    int cur_row = cur_idx % row;

    if (cur_row < row && cur_col < col) {
      d_Res(cur_row, cur_col) =
          (op == LinOpType::ADD)
              ? d_A(cur_row, cur_col) + d_B(cur_row, cur_col)
              : d_A(cur_row, cur_col) - d_B(cur_row, cur_col);
    }
  }
}
// Kernel function serving as a wrapper
__global__ void MatrixLinOp32thdKernel(double* v_A, double* v_B, double* v_Res,
                                       int row, int col, LinOpType op,
                                       int num_strides, int num_eqations) {
  int thread_idx = threadIdx.x;
  int equ_idx = blockIdx.x;

  if (equ_idx >= num_eqations) return;

  // Call the device function
  MatrixLinOp32thdFunc(v_A, v_B, v_Res, thread_idx, equ_idx, row, col, op,
                       num_strides);
}

void MatrixLinOp32thdHost(std::vector<Eigen::MatrixXd>& v_A,
                          std::vector<Eigen::MatrixXd>& v_B,
                          std::vector<Eigen::MatrixXd>& v_Res, LinOpType op,
                          int num_problems) {
  int M = v_A[0].rows();
  int N = v_A[0].cols();

  size_t size_vA = num_problems * M * N * sizeof(double);
  size_t size_vB = num_problems * M * N * sizeof(double);
  size_t size_vRes = num_problems * M * N * sizeof(double);

  double* d_vA;
  double* d_vB;
  double* d_vRes;

  // Allocate device memory
  HANDLE_ERROR(cudaMalloc((void**)&d_vA, size_vA));
  HANDLE_ERROR(cudaMalloc((void**)&d_vB, size_vB));
  HANDLE_ERROR(cudaMalloc((void**)&d_vRes, size_vRes));

  // Copy data to device
  for (int i = 0; i < num_problems; i++) {
    HANDLE_ERROR(cudaMemcpy(d_vA + i * M * N, v_A[i].data(),
                            M * N * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_vB + i * M * N, v_B[i].data(),
                            M * N * sizeof(double), cudaMemcpyHostToDevice));
  }

  // Define block and grid sizes
  int threadsPerBlock = 32;
  int numBlocks = num_problems;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  int num_strides = (M * N + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  MatrixLinOp32thdKernel<<<numBlocks, threadsPerBlock>>>(
      d_vA, d_vB, d_vRes, M, N, op, num_strides, num_problems);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for MatrixLinOp32thdKernel: " << milliseconds
            << " ms\n";

  // Copy result back to host
  for (int i = 0; i < num_problems; i++) {
    HANDLE_ERROR(cudaMemcpy(v_Res[i].data(), d_vRes + i * M * N,
                            M * N * sizeof(double), cudaMemcpyDeviceToHost));
  }

  // Free device memory
  HANDLE_ERROR(cudaFree(d_vA));
  HANDLE_ERROR(cudaFree(d_vB));
  HANDLE_ERROR(cudaFree(d_vRes));
}
