
#include <iostream>

#include "cuda_cholesky.cuh"
#include "cuda_onestepsap.cuh"
#include <cuda_runtime.h>

// ========================================================================
// OneStepSapGPU Kernels and Functions with new data struct
// ========================================================================

// ========================================================================
// Device Functions
// ========================================================================
// Device helper function to perform SAPXY operation
// Matrix A, B and C are represented by Eigen::MatrixXd
// C = alpha * A + B
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

// Device helper function to calculate alpha*(A*B) = C
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

  __syncwarp();

  if (threadIdx.x == 0) {
    data->regularizer_cost()(0, 0) = sum;
  }

  __syncwarp();
}

// Contruct Hessian, H = dynamics_matrix + J * G * J^T
__device__ void CalculateHessian(SAPGPUData* data) {
  int num_stride = ((data->NumVelocities() * data->NumVelocities()) + 31) / 32;
  for (int i = 0; i < num_stride; i++) {
    int cur_idx = i * 32 + threadIdx.x;
    if (cur_idx < data->NumVelocities() * data->NumVelocities()) {
      int cur_col = cur_idx / (data->NumVelocities());
      int cur_row = cur_idx % (data->NumVelocities());

      if (cur_row < data->NumVelocities() && cur_col < data->NumVelocities()) {
        data->H()(cur_row, cur_col) =
            data->dynamics_matrix()(cur_row, cur_col) +
            data->J().col(cur_row).dot(data->G_J().col(cur_col));
      }
    }
  }
}

// Calculate momentum gain and momentum cost
// This function will overwrite the global memory
// This function assumes the v_guess is properly set and assumes it is the input
__device__ void CalcMomentumCost(SAPGPUData* data, double* sums) {
  // Calculate velocity gain
  SAXPY(-1.0, data->v_star(), data->v_guess(), data->velocity_gain());

  __syncwarp();

  // Calculate momentum gain
  MMultiply(1.0, data->dynamics_matrix(), data->velocity_gain(),
            data->momentum_gain(), sums);

  __syncwarp();

  // Calculate momentum cost
  MMultiply(0.5, data->velocity_gain_transpose(), data->momentum_gain(),
            data->momentum_cost(), sums);

  __syncwarp();
}

// Calculate for the search direction, this direction will then be scaled by
// alpha in the line search section
__device__ void CalcSearchDirection(SAPGPUData* data, double* sums) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  // Calculate Momentum Cost
  CalcMomentumCost(data, sums);

  // Calculate Regularization Cost
  CalcRegularizationCost(data);

  // Calculate and assemble Hessian
  // Calculate G*J
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    int J_row = i * 3;
    // do a simple matrix multiplication of 3x3 multiplied by 3 by
    // 3*num_velocities
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < data->NumVelocities(); b++) {
        data->G_J()(J_row + a, b) = 0;
        for (int c = 0; c < 3; c++) {
          data->G_J()(J_row + a, b) +=
              data->G(i)(a, c) * data->J()(J_row + c, b);
        }
      }
    }
  }

  __syncwarp();

  CalculateHessian(data);

  // Calculate negative gradient

  for (int i = threadIdx.x; i < data->NumVelocities(); i += blockDim.x) {
    double sum = 0.0;
    for (int j = 0; j < 3 * data->NumContacts(); j++) {
      sum += data->J()(j, i) * data->gamma_full()(j);
    }
    data->neg_grad()(i, 0) = -(data->momentum_gain()(i, 0) - sum);
  }

  __syncwarp();

  // Cholesky Factorization and Solve for search direction

  int num_stride = (data->NumVelocities() + 31) / 32;

  Eigen::Map<Eigen::MatrixXd> d_M = data->H();
  Eigen::Map<Eigen::MatrixXd> d_L = data->chol_L();
  Eigen::Map<Eigen::VectorXd> d_b = data->neg_grad();
  Eigen::Map<Eigen::VectorXd> d_x = data->chol_x();
  Eigen::Map<Eigen::VectorXd> d_y = data->chol_y();

  CholeskyFactorizationFunc(d_M, d_L, equ_idx, thread_idx,
                            data->NumVelocities(), num_stride);
  __syncwarp();
  CholeskySolveForwardFunc(d_L, d_b, d_y, equ_idx, thread_idx,
                           data->NumVelocities(), num_stride);
  __syncwarp();
  CholeskySolveBackwardFunc(d_L, d_y, d_x, equ_idx, thread_idx,
                            data->NumVelocities(), num_stride);
  __syncwarp();
}

// Device function to evaluate the cost function at alpha
__device__ void SAPLineSearchEvalCost(SAPGPUData* data, double alpha, double* f,
                                      double* sums,
                                      Eigen::Map<Eigen::MatrixXd> dv_alpha,
                                      Eigen::Map<Eigen::MatrixXd> v_alpha) {
  Eigen::Map<Eigen::MatrixXd> x_dir(data->chol_x().data(),
                                    data->NumVelocities(), 1);
  SAXPY(alpha, x_dir, data->v_guess(), v_alpha);

  if (threadIdx.x < data->NumVelocities()) {
    data->v_guess()(threadIdx.x, 0) = v_alpha(threadIdx.x, 0);
  }
  __syncwarp();

  CalcMomentumCost(data, sums);

  __syncwarp();

  CalcRegularizationCost(data);

  __syncwarp();

  if (threadIdx.x == 0) {
    *f = data->momentum_cost()(0, 0) + data->regularizer_cost()(0, 0);
  }

  __syncwarp();
}

// Device function to evaluate the 1st derivative of the cost function w.r.t.
// alpha
__device__ void SAPLineSearchEvalDer(SAPGPUData* data, double alpha, double* d,
                                     double* sums,
                                     Eigen::Map<Eigen::MatrixXd> dv_alpha,
                                     Eigen::Map<Eigen::MatrixXd> v_alpha,
                                     Eigen::Map<Eigen::MatrixXd> delta_v_c,
                                     double* dmid_1, double* dmid_2) {
  // calculate dmid_1 = momentum_gain.transpose() * dv(alpha)
  double res = 0.0;
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    res += 0.5 * data->momentum_gain()(i, 0) * dv_alpha(i, 0);
  }
  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  if (threadIdx.x == 0) {
    *dmid_1 = res;
  }

  __syncwarp();

  // calculate dmid_2 = v_c.transpose() * gamma_full
  res = 0.0;
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    res += delta_v_c(i, 0) * data->gamma_full()(i);
  }
  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  __syncwarp();

  if (threadIdx.x == 0) {
    *dmid_2 = res;
    *d = *dmid_1 + *dmid_2;
  }

  __syncwarp();
}

// Device function to evaluate the 2nd derivative of the cost function w.r.t.
// alpha
__device__ void SAPLineSearchEval2Der(SAPGPUData* data, double alpha,
                                      double* d2, double* sums,
                                      Eigen::Map<Eigen::MatrixXd> dv_alpha,
                                      Eigen::Map<Eigen::MatrixXd> v_alpha,
                                      Eigen::Map<Eigen::MatrixXd> delta_v_c,
                                      double* ddmid_1, double* ddmid_2) {
  double res = 0.0;

  // calculate ddmid_1 = dv_alpha.transpose() * momentum_gain
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    res += dv_alpha(i, 0) * data->momentum_gain()(i, 0);
  }
  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  if (threadIdx.x == 0) {
    *ddmid_1 = res;
  }

  __syncwarp();

  //  calculate ddmid_2
  res = 0.0;
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    double vec_0 = delta_v_c(3 * i, 0);
    double vec_1 = delta_v_c(3 * i + 1, 0);
    double vec_2 = delta_v_c(3 * i + 2, 0);

    for (int j = 0; j < 3; j++) {
      // vector formed by [vec_0, vec_1, vec_2] multiply by G[i].col(j)
      for (int k = 0; k < 3; k++) {
        res += delta_v_c(3 * i + j) *
               (vec_0 * data->G(i)(0, k) + vec_1 * data->G(i)(1, k) +
                vec_2 * data->G(i)(2, k));
      }
    }
  }

  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  if (threadIdx.x == 0) {
    *ddmid_2 = res;
    *d2 = *ddmid_1 + *ddmid_2;
  }

  __syncwarp();
}

__device__ void SAPLineSearch(SAPGPUData* data, double* buff) {
  // scratch space for each block (each problem)
  // 0 - l_alpha
  // 1 - r_alpha
  // 2 - mid_alpha
  // 3 - fl
  // 4 - fr
  // 5 - fmid
  // 6 - dmid
  // 7 - d2mid
  // 8 - dmid_1 (momentum_gain.transpose() * dv(alpha))
  // 9 - dmid_2 ((J*dv(alpha)).transpose() * gamma(alpha))
  // 10 - ddmid_1 (dv_alpha.transpose() * momentum_gain)
  // 11 - ddmid_2 (J*dv_alpha).transpose() * G * (J*dv_alpha)
  // 12 - prev_alpha (for termination logic)
  // 13 - flag (0 for termination, 1 for continuation)

  // [14, (num_velocities + 14)) - dv(alpha)

  // [(num_velocities + 14), (2*num_velocities + 14)) - v(alpha)

  // [(2*num_velocities+14),(2*num_velocities+14+3*num_contacts)) -
  // delta_v_c
  size_t buff_arr_size = 2;
  size_t buff_arr_offset = 14;

  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (thread_idx == 0) {
    buff[0] = data->l_alpha();
    buff[1] = data->r_alpha();
    buff[2] = (buff[0] + buff[1]) / 2.0;
    buff[12] = 0.0;
    buff[13] = 1.0;
  }

  __syncwarp();

  double* sums =
      buff + (buff_arr_size * data->NumVelocities() + buff_arr_offset);

  Eigen::Map<Eigen::MatrixXd> dv_alpha(
      buff + buff_arr_offset, data->NumVelocities(),
      1);  // scratch space, needs to be calculated on the fly
  Eigen::Map<Eigen::MatrixXd> v_alpha(
      buff + buff_arr_offset + data->NumVelocities(), data->NumVelocities(), 1);
  Eigen::Map<Eigen::MatrixXd> delta_v_c(
      buff + buff_arr_offset + 2 * data->NumVelocities(),
      3 * data->NumContacts(), 1);

  double* l_alpha = buff;
  double* r_alpha = buff + 1;
  double* mid_alpha = buff + 2;
  double* fl = buff + 3;
  double* fr = buff + 4;
  double* fmid = buff + 5;
  double* dmid = buff + 6;
  double* d2mid = buff + 7;
  double* dmid_1 = buff + 8;
  double* dmid_2 = buff + 9;
  double* ddmid_1 = buff + 10;
  double* ddmid_2 = buff + 11;
  double* prev_alpha = buff + 12;
  double* flag = buff + 13;

  // SAP line search loop
  while (*flag == 1.0) {
    // evaluate the cost function at l_alpha and r_alpha
    // TODO: Update G and gamma
    SAPLineSearchEvalCost(data, buff[0], &buff[3], sums, dv_alpha, v_alpha);
    // TODO: Update G and gamma
    SAPLineSearchEvalCost(data, buff[1], &buff[4], sums, dv_alpha, v_alpha);

    // we evaluate fmid the last as cache will be left in the global memory
    // TODO: Update G and gamma
    SAPLineSearchEvalCost(data, buff[2], &buff[5], sums, dv_alpha, v_alpha);

    // derivative evaluation for newton-raphson
    MMultiply(1.0, data->J(), dv_alpha, delta_v_c, sums);

    // evaluate the first derivative of mid_alpha
    SAPLineSearchEvalDer(data, buff[2], &buff[6], sums, dv_alpha, v_alpha,
                         delta_v_c, dmid_1, dmid_2);

    // evaluate the second derivative of mid_alpha
    SAPLineSearchEval2Der(data, buff[2], &buff[7], sums, dv_alpha, v_alpha,
                          delta_v_c, ddmid_1, ddmid_2);

    if (threadIdx.x == 0) {
      // bisect if newton out of range
      if ((((*mid_alpha) - (*r_alpha)) * (*dmid) - (*fmid)) *
              ((*mid_alpha - *l_alpha) * (*dmid) - (*fmid)) >
          0.0) {
        *(prev_alpha) = *(mid_alpha);

        if ((*fmid) * (*fl) < 0.0) {
          *r_alpha = *mid_alpha;
        } else {
          *l_alpha = *mid_alpha;
        }

        // update *mid_alpha
        *mid_alpha = (*l_alpha + *r_alpha) / 2.0;
      } else {
        // newton is in range
        *(prev_alpha) = *(mid_alpha);

        // calculate descent distance
        double dx = *fmid / *dmid;
        // update *mid_alpha
        *mid_alpha -= dx;
      }

      // assumption: assume machine prescision
      if (abs(*prev_alpha - *mid_alpha) == 0) {
        *flag = 0.0;
      }
    }
  }

  __syncwarp();
}

// ========================================================================
// Kernels
// ========================================================================

__global__ void SolveWithGuessImplKernel(SAPGPUData* data) {
  extern __shared__ double sums[];

  // SAP Iteration flag
  double* flag = sums + 1;
  double* first_iter = sums + 2;
  Eigen::Map<Eigen::MatrixXd> prev_v_guess(
      sums + 2, data->NumVelocities(),
      1);  // previous v_guess, used for termination logic

  if (threadIdx.x == 0) *flag = 1.0;  // thtread 0 initializes the flag

  // SAP Iteration loop
  while (*flag == 1.0) {
    // calculate search direction
    // we add offset to shared memory to avoid SoveWithGuessImplKernel
    // __shared__ varibales being overwritten
    CalcSearchDirection(data, sums + 2 + data->NumVelocities());

    __syncwarp();

    // perform line search
    // we add offset to shared memory to avoid SoveWithGuessImplKernel
    // __shared__ varibales being overwritten
    SAPLineSearch(data, sums + 2 + data->NumVelocities());

    __syncwarp();

    // Thread 0 registers first results or check residual if the current
    // iteration is not 0, if necessary, continue
    if (threadIdx.x == 0) {
      if (*first_iter == 0.0) {
        *first_iter = 1.0;
      } else {
        // TODO: check tolerance
        // now set it to a very large value so that the program cam properly
        // terminate
        double tolerance = 10000.0;
        if (data->momentum_cost()(0, 0) + data->regularizer_cost()(0, 0) <=
            tolerance) {
          *flag = 0.0;
        }
      }
    }

    __syncwarp();
  }
}

// ==========================================================================
// Driver function to invoke the SAP solve
void TestOneStepSapGPU(std::vector<SAPCPUData>& sap_cpu_data,
                       std::vector<double>& momentum_cost,
                       std::vector<double>& regularizer_cost,
                       std::vector<Eigen::MatrixXd>& hessian,
                       std::vector<Eigen::MatrixXd>& neg_grad,
                       std::vector<Eigen::VectorXd>& chol_x, int num_velocities,
                       int num_contacts, int num_problems) {
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

  SolveWithGuessImplKernel<<<num_problems, threadsPerBlock,
                             4096 * sizeof(double)>>>(d_sap_gpu_data);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Transfer back to CPU for gtest validation
  sap_gpu_data.RetriveMomentumCostToCPU(momentum_cost);
  sap_gpu_data.RetriveRegularizerCostToCPU(regularizer_cost);
  sap_gpu_data.RetriveHessianToCPU(hessian);
  sap_gpu_data.RetriveNegGradToCPU(neg_grad);
  sap_gpu_data.RetriveCholXToCPU(chol_x);
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