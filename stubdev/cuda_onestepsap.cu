
#include <iostream>

#include "cuda_matmul.cuh"
#include "cuda_onestepsap.cuh"
#include "cuda_reduce.cuh"
#include <cuda_runtime.h>

// ========================================================================
// OneStepSapGPU Kernels and Functions with new data struct
// ========================================================================

// Device function to calculate alpha*A + B = C
// A and B are const inputs, C is mutable.
__device__ void SAXPY(double alpha, const Eigen::Map<Eigen::MatrixXd> A,
                      const Eigen::Map<Eigen::MatrixXd> B,
                      Eigen::Map<Eigen::MatrixXd> C) {
  int thread_idx = threadIdx.x;
  int row = A.rows();
  int col = A.cols();

  int num_strides = (A.rows() + 31) / 32;

  for (int i = 0; i < num_strides; i++) {
    int cur_idx = i * 32 + thread_idx;
    if (cur_idx >= row * col) continue;
    int cur_col = cur_idx / row;
    int cur_row = cur_idx % row;

    if (cur_row < row && cur_col < col) {
      C(cur_row, cur_col) = alpha * A(cur_row, cur_col) + B(cur_row, cur_col);
    }
  }
}

// Device function to calculate alpha*(A*B) = C
// A and B are const inputs, C is mutable.
__device__ void MMultiply(double alpha, const Eigen::Map<Eigen::MatrixXd> A,
                          const Eigen::Map<Eigen::MatrixXd> B,
                          Eigen::Map<Eigen::MatrixXd> C, double* sums) {
  int A_row = A.rows();
  int A_col = A.cols();
  int B_col = B.cols();
  int stride = (A_row + 31) / 32;
  int thread_idx = threadIdx.x;

  for (int k = 0; k < B_col; k++) {
    for (int j = 0; j < A_col; j++) {
      for (int i = 0; i < stride; i++) {
        int row = i * 32 + thread_idx;
        int col = j;
        if (row < A_row) {
          if (j == 0) {
            sums[row] = 0.0;
          }

          sums[row] += A(row, col) * B(col, k);

          if (col == A_col - 1) {
            C(row, k) = alpha * sums[row];
          }
        }
      }
    }
  }
}

// Sets lambda_r = 0.5 * gamma.transpose() * R * gamma by modifying `data`
__device__ void CalcRegularizationCost(SAPGPUData* data) {
  double sum = 0.0;
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    sum += 0.5 * data->gamma(i).dot(data->R(i).cwiseProduct(data->gamma(i)));
  }
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
  if (threadIdx.x == 0) {
    data->regularizer_cost()(0, 0) = sum;
  }
}

// Kernel function serving as a wrapper
__global__ void CalcMomentumCostKernel(SAPGPUData* data) {
  extern __shared__ double sums[];
  int thread_idx = threadIdx.x;
  int equ_idx = blockIdx.x;
  int num_problems = data->NumProblems();

  if (equ_idx >= num_problems) return;

  // Calculate velocity gain
  SAXPY(-1.0, data->v_star(), data->v_guess(), data->velocity_gain());

  // Calculate momentum gain
  MMultiply(1.0, data->dynamics_matrix(), data->velocity_gain(),
            data->momentum_gain(), sums);

  // Calculate momentum cost
  MMultiply(0.5, data->velocity_gain_transpose(), data->momentum_gain(),
            data->momentum_cost(), sums);
}

__global__ void CalcRegularizerCostKernel(SAPGPUData* data) {
  extern __shared__ double sums[];
  int equ_idx = blockIdx.x;
  int num_problems = data->NumProblems();
  int num_contacts = data->NumContacts();

  if (equ_idx >= num_problems) return;

  // Calculate regularization cost
  CalcRegularizationCost(data);
}

__device__ void CalculateHessian(SAPGPUData* data) {
  int num_stride =
      ((3 * data->NumVelocities() * 3 * data->NumVelocities()) + 31) / 32;
  for (int i = 0; i < num_stride; i++) {
    int cur_idx = i * 32 + threadIdx.x;
    if (cur_idx >= 3 * data->NumVelocities() * 3 * data->NumVelocities())
      return;
    int cur_col = cur_idx / (3 * data->NumVelocities());
    int cur_row = cur_idx % (3 * data->NumVelocities());

    if (cur_row < 3 * data->NumVelocities() &&
        cur_col < 3 * data->NumVelocities()) {
      data->H()(cur_row, cur_col) =
          data->J().col(cur_row).dot(data->G_J().col(cur_col));
    }
  }
}

__global__ void CalcHessianKernel(SAPGPUData* data) {
  extern __shared__ double sums[];
  int equ_idx = blockIdx.x;
  int num_problems = data->NumProblems();
  int num_contacts = data->NumContacts();

  if (equ_idx >= num_problems) return;

  // Calculate G*J
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    int J_row = i * 3;
    // do a simple matrix multiplication of 3x3 multiplied by 3 by
    // 3*num_velocities
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3 * data->NumVelocities(); b++) {
        data->G_J()(J_row + a, b) = 0;
        for (int c = 0; c < 3; c++) {
          data->G_J()(J_row + a, b) +=
              data->G(i)(a, c) * data->J()(J_row + c, b);
        }
      }
    }
  }

  // Calculate data->H() = J_transpose * (G*J)
  CalculateHessian(data);
}

__global__ void CalcNegGradKernel(SAPGPUData* data) {
  extern __shared__ double sums[];
  int equ_idx = blockIdx.x;
  int num_problems = data->NumProblems();
  int num_contacts = data->NumContacts();

  if (equ_idx >= num_problems) return;

  for (int i = threadIdx.x; i < 3 * data->NumVelocities(); i += blockDim.x) {
    double sum = 0.0;
    for (int j = 0; j < 3 * data->NumContacts(); j++) {
      sum += data->J()(j, i) * data->gamma_full()(j);
    }
    data->neg_grad()(i, 0) = -(data->momentum_gain()(i, 0) - sum);
  }
}

// ==========================================================================

void TestOneStepSapGPU(std::vector<SAPCPUData>& sap_cpu_data,
                       std::vector<double>& momentum_cost,
                       std::vector<double>& regularizer_cost,
                       std::vector<Eigen::MatrixXd>& hessian,
                       std::vector<Eigen::MatrixXd>& neg_grad,
                       int num_velocities, int num_contacts, int num_problems) {
  std::cout << "TestOneStepSapGPU with GPU called with " << num_problems
            << " problems" << std::endl;
  SAPGPUData sap_gpu_data;
  sap_gpu_data.MakeSAPGPUData(sap_cpu_data);

  // copy SAPGPUData to GPU
  SAPGPUData* d_sap_gpu_data;
  HANDLE_ERROR(cudaMalloc(&d_sap_gpu_data, sizeof(SAPGPUData)));
  HANDLE_ERROR(cudaMemcpy(d_sap_gpu_data, &sap_gpu_data, sizeof(SAPGPUData),
                          cudaMemcpyHostToDevice));

  int threadsPerBlock = 32;

  // Evaluate Cost

  CalcMomentumCostKernel<<<num_problems, threadsPerBlock,
                           2048 * sizeof(double)>>>(d_sap_gpu_data);

  CalcRegularizerCostKernel<<<num_problems, threadsPerBlock,
                              2048 * sizeof(double)>>>(d_sap_gpu_data);

  HANDLE_ERROR(cudaDeviceSynchronize());

  sap_gpu_data.RetriveMomentumCostToCPU(momentum_cost);
  sap_gpu_data.RetriveRegularizerCostToCPU(regularizer_cost);

  // Assemble Hessian
  // Calculate G*J
  CalcHessianKernel<<<num_problems, threadsPerBlock, 2048 * sizeof(double)>>>(
      d_sap_gpu_data);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Assemble -grad
  CalcNegGradKernel<<<num_problems, threadsPerBlock, 2048 * sizeof(double)>>>(
      d_sap_gpu_data);

  HANDLE_ERROR(cudaDeviceSynchronize());

  sap_gpu_data.RetriveHessianToCPU(hessian);
  sap_gpu_data.RetriveNegGradToCPU(neg_grad);
}

// ===========================================================================

// ===========================================================================
// Joe's Notes
// ===========================================================================
// Sets vc = J * v by modifying `data` __device__ void CalcConstraintVelocity(
//                   SAPGPUData * data) {
//   // vc = J*v
//   for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
//     data->v_vc(i) = data->v_J(i) * data->v_guess();
//   }
// }
// ===========================================================================