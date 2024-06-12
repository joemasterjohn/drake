
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
          if (col == 0) {
            sums[row] = 0.0;
          }

          if (row < A_row) {
            sums[row] += A(row, col) * B(col, k);
          }

          if (col == A_col - 1) {
            C(row, k) = alpha * sums[row];
          }
        }
      }
    }
  }
}

__device__ void ReduceByEquation(Eigen::Map<Eigen::MatrixXd> arr,
                                 Eigen::Map<Eigen::MatrixXd> arr_reduced,
                                 int num_problems, int num_contacts) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int equ_idx = global_idx / num_problems;
  int contact_idx = global_idx % num_problems;

  if (global_idx >= num_problems * num_contacts) return;

  atomicAdd(&arr_reduced(0, 0), arr(0, 0));
}

// Sets lambda_r = 0.5 * gamma.transpose() * R * gamma by modifying `data`
__device__ void CalcRegularizationCost(SAPGPUData* data) {
  double sum = 0.0;
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    sum +=
        0.5 * data->v_gamma(i).dot(data->v_R(i).cwiseProduct(data->v_gamma(i)));
  }
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
  if (threadIdx.x == 0) {
    data->lambda_r()(0, 0) = sum;
  }
}

// Sets vc = J * v by modifying `data` __device__ void CalcConstraintVelocity(
//                   SAPGPUData * data) {
//   // vc = J*v
//   for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
//     data->v_vc(i) = data->v_J(i) * data->v_guess();
//   }
// }
// ===========================================================================

// Kernel function serving as a wrapper
__global__ void CalcMomentumCostKernel(SAPGPUData data) {
  extern __shared__ double sums[];
  int thread_idx = threadIdx.x;
  int equ_idx = blockIdx.x;
  int num_problems = data.NumEquations();

  if (equ_idx >= num_problems) return;

  // Computation of Lambda_m
  // Step 1 - delta_v = v_guess - v_star
  SAXPY(-1.0, data.v_star(), data.v_guess(), data.delta_v());

  // Step 2 - delta_p = A * delta_v
  MMultiply(1.0, data.A(), data.delta_v(), data.delta_p(), sums);

  // Step 3 - lambda_m = 0.5 * delta_v.transpose() * delta_p
  MMultiply(0.5, data.delta_v_transpose(), data.delta_p(), data.lambda_m(),
            sums);
}

__global__ void CalcImpulseCostKernel(SAPGPUData data) {
  extern __shared__ double sums[];
  int equ_idx = blockIdx.x;
  int num_problems = data.NumEquations();
  int num_contacts = data.NumContacts();

  if (equ_idx >= num_problems) return;

  CalcRegularizationCost(&data);
}

// ==========================================================================

void TestOneStepSapGPU(std::vector<SAPCPUData>& v_sap_data,
                       std::vector<double>& v_lambda_m,
                       std::vector<double>& v_lambda_r, int num_velocities,
                       int num_contacts, int num_problems) {
  std::cout << "TestOneStepSapGPU with GPU called with " << num_problems
            << " equations" << std::endl;
  SAPGPUData sap_gpu_data;

  sap_gpu_data.MakeSAPGPUData(v_sap_data);

  int threadsPerBlock = 32;
  CalcMomentumCostKernel<<<num_problems, threadsPerBlock,
                           2048 * sizeof(double)>>>(sap_gpu_data);

  HANDLE_ERROR(cudaDeviceSynchronize());

  CalcImpulseCostKernel<<<num_problems, threadsPerBlock,
                          2048 * sizeof(double)>>>(sap_gpu_data);

  HANDLE_ERROR(cudaDeviceSynchronize());

  sap_gpu_data.GetLambdaM(v_lambda_m);
  for (int i = 0; i < num_problems; i++) {
    Eigen::MatrixXd delta_v = v_sap_data[i].v_guess - v_sap_data[i].v_star;
    Eigen::MatrixXd delta_P = v_sap_data[i].A * delta_v;
    Eigen::MatrixXd lambda_m_cpu = 0.5 * delta_v.transpose() * delta_P;
    std::cout << " err lambda m: " << v_lambda_m[i] - lambda_m_cpu(0, 0)
              << std::endl;
  }

  sap_gpu_data.GetLambdaR(v_lambda_r);
  for (int i = 0; i < num_problems; i++) {
    double lambda_r_sum = 0;
    for (int j = 0; j < num_contacts; j++) {
      lambda_r_sum +=
          0.5 * v_sap_data[i].v_gamma[j].dot(v_sap_data[i].v_R[j].cwiseProduct(
                    v_sap_data[i].v_gamma[j]));
    }

    std::cout << " err lambda r: " << v_lambda_r[i] - lambda_r_sum << std::endl;
  }
}

void TestOneStepSap(std::vector<Eigen::MatrixXd>& v_guess,
                    std::vector<SAPCPUData>& v_sap_data,
                    std::vector<double>& v_lambda_m,
                    std::vector<double>& v_lambda_r, int num_velocities,
                    int num_contacts, int num_problems) {
  std::cout << "TestOneStepSap with CPU called with " << num_problems
            << " equations" << std::endl;

  // calculation of lambda_m -> momentum cost
  int threadsPerBlock = 32;
  // allocate GPU memory to calculate v_guess - v_star
  double* d_v_guess;
  double* d_v_star;
  double* d_delta_v;

  size_t size_v_guess = num_problems * num_velocities * 3 * sizeof(double);
  size_t size_v_star = num_problems * num_velocities * 3 * sizeof(double);
  size_t size_delta_v = num_problems * num_velocities * 3 * sizeof(double);

  HANDLE_ERROR(cudaMalloc((void**)&d_v_guess, size_v_guess));
  HANDLE_ERROR(cudaMalloc((void**)&d_v_star, size_v_star));
  HANDLE_ERROR(cudaMalloc((void**)&d_delta_v, size_delta_v));

  // Copy data to device
  for (int i = 0; i < num_problems; i++) {
    HANDLE_ERROR(cudaMemcpy(
        d_v_guess + i * num_velocities * 3, v_guess[i].data(),
        num_velocities * 3 * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(
        d_v_star + i * num_velocities * 3, v_sap_data[i].v_star.data(),
        num_velocities * 3 * sizeof(double), cudaMemcpyHostToDevice));
  }

  // envoke substraction kernel
  int num_strides =
      (3 * num_velocities + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  // Note: d_v_guess, d_v_star, and d_delta_v are all 3 * num_velocities * 1
  MatrixLinOp32thdKernel<<<num_problems, threadsPerBlock>>>(
      d_v_guess, d_v_star, d_delta_v, 3 * num_velocities, 1, LinOpType::SUB,
      num_strides, num_problems);
  cudaDeviceSynchronize();

  // // TEST CODE
  // // allocate space for delta_v on CPU
  // std::vector<Eigen::MatrixXd> delta_v(num_problems,
  //                                      Eigen::MatrixXd(num_velocities * 3,
  //                                      1));

  // // map to eigen
  // for (int i = 0; i < num_problems; i++) {
  //   // copy from d_delta_v to delta_v
  //   HANDLE_ERROR(cudaMemcpy(delta_v[i].data(), d_delta_v + i * num_velocities
  //   * 3,
  //                           num_velocities * 3 * sizeof(double),
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
  size_t size_delta_P = num_problems * num_velocities * 3 * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_delta_P, size_delta_P));

  // calculate delta_P = A * delta_v
  // reserve space for A on GPU
  // Note: A is a 3 * num_velocities by 3 * num_velocities matrix
  double* d_A;
  size_t size_A =
      num_problems * num_velocities * 3 * num_velocities * 3 * sizeof(double);
  int A_rows = num_velocities * 3;
  int A_cols = num_velocities * 3;
  // copy A to device
  HANDLE_ERROR(cudaMalloc((void**)&d_A, size_A));
  for (int i = 0; i < num_problems; i++) {
    HANDLE_ERROR(cudaMemcpy(d_A + i * A_rows * A_cols, v_sap_data[i].A.data(),
                            A_rows * A_cols * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  // calculate delta_P = A * delta_v
  int stride = (A_rows + threadsPerBlock - 1) / threadsPerBlock;
  MatrixMultiply32thdKernel<<<num_problems, 32>>>(
      d_A, d_delta_v, d_delta_P, 3 * num_velocities, 3 * num_velocities, 1,
      stride, num_problems);
  cudaDeviceSynchronize();

  // //  TEST_CODE
  // // allocate space for P on CPU
  // double* P = new double[num_problems * num_velocities * 3];
  // // copy from d_delta_P to P
  // HANDLE_ERROR(cudaMemcpy(P, d_delta_P, size_delta_P,
  // cudaMemcpyDeviceToHost));
  // // map to eigen
  // std::vector<Eigen::MatrixXd> P_eigen;
  // for (int i = 0; i < num_problems; i++) {
  //   Eigen::MatrixXd P_i = Eigen::Map<Eigen::MatrixXd>(P + i * num_velocities
  //   * 3,
  //                                                     3 * num_velocities, 1);
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
  size_t size_lambda = num_problems * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_lambda, size_lambda));

  stride = 1;
  MatrixMultiply32thdKernel<<<num_problems, 32>>>(
      d_delta_v, d_delta_P, d_lambda, 1, 3 * num_velocities, 1, stride,
      num_problems);
  cudaDeviceSynchronize();

  // copy results back
  HANDLE_ERROR(cudaMemcpy(v_lambda_m.data(), d_lambda, size_lambda,
                          cudaMemcpyDeviceToHost));

  // multiply the GPU computed results by 0.5 to finish the computation of
  // 0.5 * delta_v.transpose() * delta_p
  for (int i = 0; i < num_problems; i++) {
    v_lambda_m[i] *= 0.5;
  }

  // calculation of lambda_r -> impulse cost
  // reserve space for lambda_r on GPU, size is num_problems

  // copy impulse and R to GPU
  double* d_impulse;
  double* d_R;

  size_t size_impulse = num_problems * num_contacts * 3 * sizeof(double);
  size_t size_R = num_problems * num_contacts * 3 * 3 * sizeof(double);

  HANDLE_ERROR(cudaMalloc((void**)&d_impulse, size_impulse));
  HANDLE_ERROR(cudaMalloc((void**)&d_R, size_R));

  for (int i = 0; i < num_problems; i++) {
    for (int j = 0; j < num_contacts; j++) {
      HANDLE_ERROR(cudaMemcpy(d_R + i * num_contacts * 3 * 3 + j * 3 * 3,
                              v_sap_data[i].v_R[j].data(),
                              3 * 3 * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(d_impulse + i * num_contacts * 3 + j * 3,
                              v_sap_data[i].v_gamma[j].data(),
                              3 * sizeof(double), cudaMemcpyHostToDevice));
    }
  }

  // allocate GPU space for intermediate results for impulse.transpose() * R
  double* d_intermediate;
  size_t size_intermediate = num_problems * num_contacts * 3 * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_intermediate, size_intermediate));

  // calculate lambda_r = 0.5 * impulse.transpose() * R * impulse
  stride = 1;  // d_impulse is a column vector, so tranpose is a row vector
  MatrixMultiply32thdKernel<<<num_problems * num_contacts, 32>>>(
      d_impulse, d_R, d_intermediate, 1, 3, 3, stride,
      num_problems * num_contacts);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // allocate GPU space for final results for lambda_r
  double* d_lambda_r;
  size_t size_lambda_r = num_problems * num_contacts * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_lambda_r, size_lambda_r));

  stride = 1;  // intermediate is 1 by 3, and impulse vector is 3 x 1.
  MatrixMultiply32thdKernel<<<num_problems * num_contacts, 32>>>(
      d_intermediate, d_impulse, d_lambda_r, 1, 3, 1, stride,
      num_problems * num_contacts);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // allocate post reduce results for lambda_r
  double* d_lambda_r_reduced;
  size_t size_lambda_r_reduced = num_problems * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_lambda_r_reduced, size_lambda_r_reduced));

  // call reduce kernel, to sum up the lambda_r for each equation

  ReduceByProblemKernel<<<num_problems, 32>>>(d_lambda_r, d_lambda_r_reduced,
                                              num_problems, num_contacts);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // copy d_lambda_reduced back to v_lambda_r
  HANDLE_ERROR(cudaMemcpy(v_lambda_r.data(), d_lambda_r_reduced,
                          size_lambda_r_reduced, cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_problems; i++) {
    v_lambda_r[i] *= 0.5;
  }
}