
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
  Eigen::Map<Eigen::MatrixXd> d_b = data->neg_grad();
  Eigen::Map<Eigen::MatrixXd> d_x = data->chol_x();
  Eigen::Map<Eigen::MatrixXd> d_y = data->chol_y();

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
__device__ void SAPLineSearchEvalCost(
    SAPGPUData* data, double alpha, double& f, double* sums,
    Eigen::Map<Eigen::MatrixXd> v_alpha,
    Eigen::Map<Eigen::MatrixXd> v_guess_prev) {
  SAXPY(alpha, data->chol_x(), v_guess_prev, v_alpha);

  // if (threadIdx.x == 0) {
  //   printf("====================================\n");
  //   printf("v_guess_prev:\n");
  //   for (int i = 0; i < v_guess_prev.rows(); i++) {
  //     printf(" %f\n", v_guess_prev(i, 0));
  //   }

  //   printf("v_alpha:\n");
  //   for (int i = 0; i < v_alpha.rows(); i++) {
  //     printf(" %f\n", v_alpha(i, 0));
  //   }

  //   printf("chol_x:\n");
  //   for (int i = 0; i < data->chol_x().rows(); i++) {
  //     printf(" %f\n", data->chol_x()(i, 0));
  //   }

  //   printf("v star:\n");
  //   for (int i = 0; i < data->v_star().rows(); i++) {
  //     printf(" %f\n", data->v_star()(i, 0));
  //   }
  // }

  __syncwarp();

  if (threadIdx.x == 0) {
    for (int i = 0; i < data->NumVelocities(); i++) {
      data->v_guess()(i, 0) = v_alpha(i, 0);
    }
  }
  __syncwarp();

  CalcMomentumCost(data, sums);

  __syncwarp();

  CalcRegularizationCost(data);

  __syncwarp();

  if (threadIdx.x == 0) {
    f = data->momentum_cost()(0, 0) + data->regularizer_cost()(0, 0);
  }

  __syncwarp();
}

// Device function to evaluate the 1st derivative of the cost function w.r.t.
// alpha
__device__ void SAPLineSearchEvalDer(SAPGPUData* data, double alpha, double& d,
                                     double* sums,
                                     Eigen::Map<Eigen::MatrixXd> v_alpha,
                                     Eigen::Map<Eigen::MatrixXd> delta_v_c,
                                     Eigen::Map<Eigen::MatrixXd> delta_p) {
  // chol_x(search direction) * momentum_gain(computed in eval device function)
  double res = 0.0;
  double d_temp = 0.0;
  for (int i = threadIdx.x; i < data->NumVelocities(); i += blockDim.x) {
    res += data->chol_x()(i, 0) * data->momentum_gain()(i, 0);
  }
  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  if (threadIdx.x == 0) {
    d_temp = res;
  }

  __syncwarp();

  // TODO: check formulation
  // ((J*dv(alpha)).transpose() * gamma(alpha))
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
    d_temp += res;
    d = d_temp;
  }

  __syncwarp();
}

// Device function to evaluate the 2nd derivative of the cost function w.r.t.
// alpha
__device__ void SAPLineSearchEval2Der(SAPGPUData* data, double alpha,
                                      double& d2, double* sums,
                                      Eigen::Map<Eigen::MatrixXd> v_alpha,
                                      Eigen::Map<Eigen::MatrixXd> delta_v_c,
                                      Eigen::Map<Eigen::MatrixXd> delta_p) {
  double res = 0.0;
  double d_temp = 0.0;

  // chol_x(search direction) * delta_p (A * chol_x (search direction))
  for (int i = threadIdx.x; i < data->NumVelocities(); i += blockDim.x) {
    res += data->chol_x()(i, 0) * delta_p(i, 0);  // change this momentum gain
  }
  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  if (threadIdx.x == 0) {
    d_temp = res;
  }

  __syncwarp();

  // TODO: check formulation
  // (J*dv_alpha).transpose() * G * (J*dv_alpha)
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
    d_temp += res;
    d2 = d_temp;
  }

  __syncwarp();
}

__device__ double SAPLineSearch(SAPGPUData* data, double* buff) {
  // scratch space for each problem (per block)

  size_t buff_arr_offset = 14;  // 14 doubles for local variables

  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  __syncwarp();

  double* sums = buff + buff_arr_offset + 3 * data->NumVelocities() +
                 3 * data->NumContacts();

  Eigen::Map<Eigen::MatrixXd> v_alpha(buff + buff_arr_offset,
                                      data->NumVelocities(), 1);
  Eigen::Map<Eigen::MatrixXd> delta_v_c(
      buff + buff_arr_offset + 1 * data->NumVelocities(),
      3 * data->NumContacts(), 1);
  Eigen::Map<Eigen::MatrixXd> delta_p(buff + buff_arr_offset +
                                          1 * data->NumVelocities() +
                                          3 * data->NumContacts(),
                                      data->NumVelocities(), 1);
  Eigen::Map<Eigen::MatrixXd> v_guess_prev(buff + buff_arr_offset +
                                               2 * data->NumVelocities() +
                                               3 * data->NumContacts(),
                                           data->NumVelocities(), 1);

  double& l_alpha = buff[0];      // alpha left bound
  double& r_alpha = buff[1];      // alpha right bound
  double& guess_alpha = buff[2];  // alpha guess value
  double& f_l = buff[3];          // evaluated function val at l_alpha
  double& f_r = buff[4];          // evaluated function val at r_alpha
  double& f_guess = buff[5];      // evaluated function val at guess_alpha

  double& d_l = buff[6];       // derivative at l_alpha
  double& d_r = buff[7];       // derivative at r_alpha
  double& d_guess = buff[8];   // derivative at guess_alpha
  double& d2_guess = buff[9];  // 2nd derivative at guess_alpha

  double& prev_alpha = buff[10];        // previous alpha record
  double& dx_negative_prev = buff[11];  // previous x_negative record
  double& flag = buff[12];              // loop flag
  double& first_iter = buff[13];        // first iteration flag

  // scratch space variable initialization
  if (thread_idx == 0) {
    l_alpha = 0.0;
    r_alpha = 1.0;
    guess_alpha = (l_alpha + r_alpha) / 2.0;
    flag = 1.0;
    first_iter = 1.0;
    for (int i = 0; i < data->NumVelocities(); i++) {
      v_guess_prev(i, 0) = data->v_guess()(i, 0);
    }
  }

  __syncwarp();

  // evaluate the cost function at l_alpha and r_alpha
  // TODO: Update G and gamma
  SAPLineSearchEvalCost(data, l_alpha, f_l, sums, v_alpha, v_guess_prev);
  MMultiply(1.0, data->J(), data->chol_x(), delta_v_c, sums);
  MMultiply(1.0, data->dynamics_matrix(), data->chol_x(), delta_p, sums);
  SAPLineSearchEvalDer(data, l_alpha, d_l, sums, v_alpha, delta_v_c, delta_p);

  // TODO: Update G and gamma
  SAPLineSearchEvalCost(data, r_alpha, f_r, sums, v_alpha, v_guess_prev);
  MMultiply(1.0, data->J(), data->chol_x(), delta_v_c, sums);
  MMultiply(1.0, data->dynamics_matrix(), data->chol_x(), delta_p, sums);
  SAPLineSearchEvalDer(data, r_alpha, d_r, sums, v_alpha, delta_v_c, delta_p);

  // we evaluate fmid the last as cache will be left in the global memory
  // TODO: Update G and gamma
  SAPLineSearchEvalCost(data, guess_alpha, f_guess, sums, v_alpha,
                        v_guess_prev);
  MMultiply(1.0, data->J(), data->chol_x(), delta_v_c, sums);
  MMultiply(1.0, data->dynamics_matrix(), data->chol_x(), delta_p, sums);
  SAPLineSearchEvalDer(data, guess_alpha, d_guess, sums, v_alpha, delta_v_c,
                       delta_p);
  // evaluate the second derivative of mid_alpha
  SAPLineSearchEval2Der(data, guess_alpha, d2_guess, sums, v_alpha, delta_v_c,
                        delta_p);

  // TODO: replace this while loop to a for loop
  // SAP line search loop
  // TODO: set max_iteration somewhere else
  int max_iteration = 100;
  for (int i = 0; i < max_iteration; i++) {
    if (flag == 1.0) {
      if (threadIdx.x == 0) {
        double root_guess = 0.0;
        bool newton_is_slow = false;
        double dx_negative = 0.0;

        if (first_iter == 0.0) {
          newton_is_slow =
              2.0 * abs(d_guess) > abs(d2_guess * dx_negative_prev);
        } else {
          newton_is_slow = false;
          first_iter = 0.0;
        }

        // return logic
        if (abs(d_guess) < 1e-6) {
          flag = 0.0;
        }

        if (abs(guess_alpha - 1.0) <= 1e-6 || abs(guess_alpha - 0.0) <= 1e-6) {
          flag = 0.0;
        }

        // TODO: remove this debugging output
        // printf("l_alpha: %f, r_alpha: %f, guess: %f, d_guess: %f \n",
        // l_alpha,
        //        r_alpha, guess_alpha, d_guess);
        if (newton_is_slow) {
          dx_negative = 0.5 * (l_alpha - r_alpha);
          root_guess = l_alpha - dx_negative;
        } else {
          dx_negative = d_guess / d2_guess;
          root_guess = guess_alpha - dx_negative;
          // newton_is_out_of_range
          if (root_guess < l_alpha || root_guess > r_alpha) {
            dx_negative = 0.5 * (l_alpha - r_alpha);
            root_guess = l_alpha - dx_negative;
          }
        }

        // update guess_alpha
        // record dx_negative at this iteration
        guess_alpha = root_guess;
        dx_negative_prev = dx_negative;
      }

      // we evaluate fguess the last as cache will be left in the global memory
      // TODO: Update G and gamma
      SAPLineSearchEvalCost(data, guess_alpha, f_guess, sums, v_alpha,
                            v_guess_prev);
      MMultiply(1.0, data->J(), data->chol_x(), delta_v_c, sums);
      SAPLineSearchEvalDer(data, guess_alpha, d_guess, sums, v_alpha, delta_v_c,
                           delta_p);
      // evaluate the second derivative of guess_alpha
      SAPLineSearchEval2Der(data, guess_alpha, d2_guess, sums, v_alpha,
                            delta_v_c, delta_p);

      __syncwarp();

      // based on eval, shrink the interval
      // update l_alpha and r_alpha to update intervals
      if (threadIdx.x == 0) {
        // printf(" dl*d_guess: %f, dr*d_guess: %f \n", d_l * d_guess,
        //        d_r * d_guess);
        if (d_l * d_guess > 0.0) {
          l_alpha = guess_alpha;
          f_l = f_guess;
          d_l = d_guess;
        } else {
          r_alpha = guess_alpha;
          f_r = f_guess;
          d_r = d_guess;
        }
      }

      __syncwarp();
    }
  }

  return guess_alpha;
}

// device function to evaluate the 2-norm of the neg_grad for each iteration
// this is used to termination logic: ||delta_l_p|| < epsilon_alpha + epsilon_r
__device__ double NegGradCostEval(SAPGPUData* data) {
  double sum = 0.0;
  for (int i = threadIdx.x; i < data->NumVelocities(); i += blockDim.x) {
    sum += data->neg_grad()(i, 0) * data->neg_grad()(i, 0);
  }
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

  return sum;
}

// ========================================================================
// Kernels
// ========================================================================

__global__ void SolveWithGuessKernel(SAPGPUData* data) {
  extern __shared__ double sums[];

  // SAP Iteration flag
  double& flag = sums[0];
  double& first_iter = sums[1];
  Eigen::Map<Eigen::MatrixXd> prev_v_guess(
      sums + 2, data->NumVelocities(),
      1);  // previous v_guess, used for termination logic

  if (threadIdx.x == 0) {
    flag = 1.0;  // thread 0 initializes the flag
    first_iter = 1.0;
  }

  // TODO: might need replace this while loop to a for loop
  // SAP Iteration loop
  while (flag == 1.0) {
    // calculate search direction
    // we add offset to shared memory to avoid SoveWithGuessImplKernel
    // __shared__ varibales being overwritten
    CalcSearchDirection(data, sums + 2 + data->NumVelocities());

    __syncwarp();

    // perform line search
    // we add offset to shared memory to avoid SoveWithGuessImplKernel
    // __shared__ varibales being overwritten
    double alpha = SAPLineSearch(data, sums + 2 + data->NumVelocities());

    __syncwarp();

    double neg_grad_norm = NegGradCostEval(data);
    if (threadIdx.x == 0) printf("neg_grad_norm: %f\n", neg_grad_norm);

    // Thread 0 registers first results or check residual if the current
    // iteration is not 0, if necessary, continue
    if (threadIdx.x == 0) {
      if (first_iter == 1.0) {
        first_iter = 0.0;
      } else {
        // TODO: check tolerance
        // now the tolerance is fixed and set to 1e-6
        double tolerance = 1e-6;
        if (neg_grad_norm <= tolerance) {
          flag = 0.0;
        }
      }

      // assign previous
      for (int i = 0; i < data->NumVelocities(); i++) {
        prev_v_guess(i, 0) = data->v_guess()(i, 0);
      }

      // debugging output print v_alpha
      printf("v_alpha: \n");
      for (int i = 0; i < data->NumVelocities(); i++) {
        printf("%f\n", data->v_guess()(i, 0));
      }

      // debugging output print v_star
      printf("v_star: \n");
      for (int i = 0; i < data->NumVelocities(); i++) {
        printf("%f\n", data->v_star()(i, 0));
      }
      printf("this is a step of cholsolve\n");
      printf("alpha at this step is %f\n", alpha);
    }

    __syncwarp();
  }

  // TODO: remove this debugging output
  if (threadIdx.x == 0) printf("SAP Converged!\n");
  __syncwarp();
}

// ==========================================================================
// Driver function to invoke the SAP solve
void TestOneStepSapGPU(std::vector<SAPCPUData>& sap_cpu_data,
                       std::vector<double>& momentum_cost,
                       std::vector<double>& regularizer_cost,
                       std::vector<Eigen::MatrixXd>& hessian,
                       std::vector<Eigen::MatrixXd>& neg_grad,
                       std::vector<Eigen::MatrixXd>& chol_x, int num_velocities,
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

  SolveWithGuessKernel<<<num_problems, threadsPerBlock,
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