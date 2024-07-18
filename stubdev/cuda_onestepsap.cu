
#include <iostream>

#include "cuda_cholesky.cuh"
#include "cuda_onestepsap.cuh"
#include <cuda_runtime.h>

#define alpha_max 1.5
#define max_iteration 100
#define f_tolerance 1.0e-8
#define abs_tolerance 1.0e-14
#define rel_tolerance 1.0e-6
#define cost_abs_tolerance 1.0e-30
#define cost_rel_tolerance 1.0e-15

// ========================================================================
// OneStepSapGPU Kernels and Functions with new data struct
// ========================================================================

// ========================================================================
// Data Access Functions
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

  int num_strides = (A.rows() * A.cols() + 31) / 32;

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

__device__ void CalcConstraintCost(SAPGPUData* data) {
  double sum = 0.0;
  for (int i = threadIdx.x; i < data->num_active_contacts(); i += blockDim.x) {
    double k = data->contact_stiffness(i)(0, 0);
    double d = data->contact_damping(i)(0, 0);
    double phi_0_i = data->phi0(i)(0, 0);
    double v_d = 1.0 / (d + 1.0e-20);
    double v_x = data->phi0(i)(0, 0) * k / dt / (k + 1.0e-20);
    double v_hat = min(v_x, v_d);
    double v_n = (data->J().row(i * 3 + 2) * data->v_guess())(0, 0);
    double v = min(v_n, v_hat);  // clamped

    double df = -dt * k * v;

    double N = dt * (v * (phi_0_i * k + 1.0 / 2.0 * df) -
                     d * v * v / 2.0 * (phi_0_i * k + 2.0 / 3.0 * df));

    sum += -N;
  }
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

  __syncwarp();

  if (threadIdx.x == 0) {
    data->constraint_cost()(0, 0) = sum;
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

// Depends on vc (constraint velocity) already being computed.
__device__ void UpdateGammaG(SAPGPUData* data) {
  // set all gamma to 0
  for (int i = threadIdx.x; i < data->NumContacts() * 3; i += blockDim.x) {
    data->gamma_full()(i) = 0.0;
  }

  // set all G to 0
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    data->G(i).setZero();
  }

  __syncwarp();

  for (int i = threadIdx.x; i < data->num_active_contacts() * 3;
       i += blockDim.x) {
    if (i % 3 == 2) {
      // compute corresponding v_c
      // contact velocity on the z direction in the contact local frame
      double v_contact_z = -(data->J().row(i) * data->v_guess())(0, 0);
      double phi_0 = data->phi0(int(i / 3))(0, 0);

      // update each gamma
      // first-order approximation of the penetration
      // max(0.0, phi_0 + dt * v_contact_z)
      double pen_approx = phi_0 + dt * v_contact_z;
      if (pen_approx <= 0.0) pen_approx = 0.0;

      double damping_term =
          1.0 + data->contact_damping(int(i / 3))(0, 0) * v_contact_z;
      if (damping_term <= 0.0) damping_term = 0.0;

      data->gamma(int(i / 3))(2) = dt *
                                   data->contact_stiffness(int(i / 3))(0, 0) *
                                   pen_approx * damping_term;

      // update each G
      // impulse gradiant
      double np =
          -dt *
          ((data->contact_stiffness(int(i / 3))(0, 0) * dt * damping_term) +
           data->contact_damping(int(i / 3))(0, 0) *
               data->contact_stiffness(int(i / 3))(0, 0) * pen_approx);
      // assign -np to G(2,2)
      data->G(int(i / 3))(2, 2) = -np;
    }
  }
  __syncwarp();
}

__device__ void CalculateDlDalpha0(SAPGPUData* data, double* sums) {
  double sum = 0.0;
  for (int i = threadIdx.x; i < data->NumVelocities(); i += blockDim.x) {
    sum += (-data->neg_grad()(i, 0)) * data->chol_x()(i, 0);
  }

  sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

  __syncwarp();

  if (threadIdx.x == 0) {
    data->dl_dalpha0()(0, 0) = sum;
  }

  __syncwarp();
}

// Calculate for the search direction, this direction will then be scaled by
// alpha in the line search section
__device__ void CalcSearchDirection(SAPGPUData* data, double* sums) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  UpdateGammaG(data);

  // Calculate Momentum Cost
  CalcMomentumCost(data, sums);

  // Calculate Constraint Cost
  CalcConstraintCost(data);

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

  // Calculate the dl_dalpha0 for each problem
  // dℓ/dα(α = 0) = ∇ᵥℓ(α = 0)⋅Δv.
  // also -neg_grad.dot(chol_x)
  CalculateDlDalpha0(data, sums);

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

  UpdateGammaG(data);

  __syncwarp();

  CalcMomentumCost(data, sums);

  __syncwarp();

  CalcConstraintCost(data);

  __syncwarp();

  if (threadIdx.x == 0) {
    f = data->momentum_cost()(0, 0) + data->constraint_cost()(0, 0);
  }

  __syncwarp();
}

// Device function to evaluate the 1st derivative of the cost function w.r.t.
// alpha
__device__ void SAPLineSearchEvalDer(SAPGPUData* data, double alpha,
                                     double& dell_dalpha, double* sums,
                                     Eigen::Map<Eigen::MatrixXd> v_alpha,
                                     Eigen::Map<Eigen::MatrixXd> delta_v_c,
                                     Eigen::Map<Eigen::MatrixXd> delta_p) {
  // chol_x(search direction) * momentum_gain(computed in eval device
  // function)
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

  res = 0.0;
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    res += delta_v_c.block<3, 1>(3 * i, 0).dot(data->gamma(i));
  }
  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  __syncwarp();

  if (threadIdx.x == 0) {
    d_temp = d_temp - res;
    dell_dalpha = d_temp;
  }

  __syncwarp();
}

// Device function to evaluate the 2nd derivative of the cost function w.r.t.
// alpha
__device__ void SAPLineSearchEval2Der(SAPGPUData* data, double alpha,
                                      double& d2ell_dalpha2, double* sums,
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

  res = 0.0;
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    double vec_0 = delta_v_c(3 * i, 0);
    double vec_1 = delta_v_c(3 * i + 1, 0);
    double vec_2 = delta_v_c(3 * i + 2, 0);

    // vector formed by [vec_0, vec_1, vec_2] multiply by G[i].col(j)
    for (int k = 0; k < 3; k++) {
      res += delta_v_c(3 * i + k, 0) *
             (vec_0 * data->G(i)(0, k) + vec_1 * data->G(i)(1, k) +
              vec_2 * data->G(i)(2, k));
    }
  }

  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  __syncwarp();

  if (threadIdx.x == 0) {
    d_temp += res;
    d2ell_dalpha2 = d_temp;
  }

  __syncwarp();
}

__device__ double SAPLineSearch(SAPGPUData* data, double* buff) {
  // scratch space for each problem (per block)

  size_t buff_arr_offset = 14;  // 15 doubles for local variables

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

  double& alpha_l = buff[0];      // alpha left bound
  double& alpha_r = buff[1];      // alpha right bound
  double& alpha_guess = buff[2];  // alpha guess value
  double& ell_l = buff[3];        // evaluated function val at alpha_l
  double& ell_r = buff[4];        // evaluated function val at alpha_r
  double& ell_guess = buff[5];    // evaluated function val at alpha_guess

  double& dell_dalpha_l = buff[6];  // derivative at alpha_l w.r.t. alpha
  double& dell_dalpha_r = buff[7];  // derivative at alpha_r w.r.t. alpha
  double& dell_dalpha_guess =
      buff[8];  // derivative at alpha_guess w.r.t. alpha
  double& d2ell_dalpha2_guess =
      buff[9];  // 2nd derivative at alpha_guess w.r.t. alpha

  double& prev_alpha = buff[10];   // previous alpha record
  double& dx_negative = buff[11];  // x_negative in the current iteration
  double& flag = buff[12];         // loop flag
  double& first_iter = buff[13];   // first iteration flag

  // scratch space variable initialization
  if (thread_idx == 0) {
    alpha_l = 0.0;
    alpha_r = alpha_max;
    alpha_guess = (alpha_l + alpha_r) / 2.0;
    flag = 1.0;
    first_iter = 1.0;
    for (int i = 0; i < data->NumVelocities(); i++) {
      v_guess_prev(i, 0) = data->v_guess()(i, 0);
    }
  }

  __syncwarp();

  // evaluate the cost function at l_alpha and r_alpha
  // TODO: Update G and gamma
  SAPLineSearchEvalCost(data, alpha_l, ell_l, sums, v_alpha, v_guess_prev);
  MMultiply(1.0, data->J(), data->chol_x(), delta_v_c, sums);
  MMultiply(1.0, data->dynamics_matrix(), data->chol_x(), delta_p, sums);
  SAPLineSearchEvalDer(data, alpha_l, dell_dalpha_l, sums, v_alpha, delta_v_c,
                       delta_p);

  // TODO: Update G and gamma
  SAPLineSearchEvalCost(data, alpha_r, ell_r, sums, v_alpha, v_guess_prev);
  MMultiply(1.0, data->J(), data->chol_x(), delta_v_c, sums);
  MMultiply(1.0, data->dynamics_matrix(), data->chol_x(), delta_p, sums);
  SAPLineSearchEvalDer(data, alpha_r, dell_dalpha_r, sums, v_alpha, delta_v_c,
                       delta_p);

  // return logic: early accept
  if (threadIdx.x == 0) {
    if (dell_dalpha_r <= 0.0) {
      alpha_guess = alpha_r;
      flag = 0.0;
      // printf("TERMINATE - early accept, no root in bracket \n");
    }

    if (-data->dl_dalpha0()(0, 0) <
        cost_abs_tolerance + cost_rel_tolerance * ell_r) {
      alpha_guess = 1.0;
      flag = 0.0;
      // printf("TERMINATE - early accept, der too small \n");
    }
  }

  // we evaluate fguess the last as cache will be left in the global memory
  // TODO: Update G and gamma
  SAPLineSearchEvalCost(data, alpha_guess, ell_guess, sums, v_alpha,
                        v_guess_prev);
  MMultiply(1.0, data->J(), data->chol_x(), delta_v_c, sums);
  MMultiply(1.0, data->dynamics_matrix(), data->chol_x(), delta_p, sums);
  SAPLineSearchEvalDer(data, alpha_guess, dell_dalpha_guess, sums, v_alpha,
                       delta_v_c, delta_p);
  // evaluate the second derivative of mid_alpha
  SAPLineSearchEval2Der(data, alpha_guess, d2ell_dalpha2_guess, sums, v_alpha,
                        delta_v_c, delta_p);

  // TODO: replace this while loop to a for loop
  // SAP line search loop
  // TODO: set max_iteration somewhere else
  for (int i = 0; i < max_iteration; i++) {
    if (flag == 1.0) {
      if (threadIdx.x == 0) {
        bool newton_is_slow = false;

        if (first_iter == 0.0) {
          newton_is_slow = 2.0 * abs(dell_dalpha_guess) >
                           abs(d2ell_dalpha2_guess * dx_negative);
        } else {
          newton_is_slow = false;
          first_iter = 0.0;
        }

        // TODO: remove this debugging output
        // printf("l_alpha: %f, r_alpha: %f, guess: %f, d_guess: %f \n",
        // l_alpha,
        //        r_alpha, guess_alpha, d_guess);
        if (newton_is_slow) {
          // printf("do bisec, newton is slow\n");
          dx_negative = 0.5 * (alpha_l - alpha_r);
          alpha_guess = alpha_l - dx_negative;
        } else {
          // printf("do newton\n");
          dx_negative = dell_dalpha_guess / d2ell_dalpha2_guess;
          alpha_guess = alpha_guess - dx_negative;
          // newton_is_out_of_range
          if (alpha_guess < alpha_l || alpha_guess > alpha_r) {
            // printf("newton oob do bisec!\n");
            dx_negative = 0.5 * (alpha_l - alpha_r);
            alpha_guess = alpha_l - dx_negative;
          }
        }
      }

      __syncwarp();

      // we evaluate fguess the last as cache will be left in the global
      // memory
      // TODO: Update G and gamma
      SAPLineSearchEvalCost(data, alpha_guess, ell_guess, sums, v_alpha,
                            v_guess_prev);
      MMultiply(1.0, data->J(), data->chol_x(), delta_v_c, sums);
      SAPLineSearchEvalDer(data, alpha_guess, dell_dalpha_guess, sums, v_alpha,
                           delta_v_c, delta_p);
      // evaluate the second derivative of guess_alpha
      SAPLineSearchEval2Der(data, alpha_guess, d2ell_dalpha2_guess, sums,
                            v_alpha, delta_v_c, delta_p);

      __syncwarp();

      // based on eval, shrink the interval
      // update l_alpha and r_alpha to update intervals
      if (threadIdx.x == 0) {
        if (dell_dalpha_l * dell_dalpha_guess > 0.0) {
          alpha_l = alpha_guess;
          ell_l = ell_guess;
          dell_dalpha_l = dell_dalpha_guess;
        } else {
          alpha_r = alpha_guess;
          ell_r = ell_guess;
          dell_dalpha_r = dell_dalpha_guess;
        }
      }

      if (threadIdx.x == 0) {
        // return logic
        if (first_iter == 0.0) {
          if (abs(dx_negative) <= f_tolerance * alpha_guess) {
            flag = 0.0;
            // printf("TERMINATE - bracket within tolerance \n");
          }
        }

        if (abs(dell_dalpha_guess) <= f_tolerance) {
          flag = 0.0;
          // printf("TERMINATE - root within tolerance \n");
        }
      }

      __syncwarp();
    }
  }

  return alpha_guess;
}

// device function to evaluate the 2-norm of the neg_grad for each iteration
// this is used to termination logic: ||delta_l_p|| < epsilon_alpha +
// epsilon_r
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

// device function to evaluate outer loop stopping criteria residual
__device__ void CalcStoppingCriteriaResidual(SAPGPUData* data,
                                             double& momentum_residue,
                                             double& momentum_scale) {
  // calculate p_tilde^2 term
  double p_tilde_2 = 0.0;
  for (int i = threadIdx.x; i < data->NumVelocities(); i += blockDim.x) {
    p_tilde_2 +=
        ((data->dynamics_matrix()(i, i) / sqrt(data->dynamics_matrix()(i, i))) *
         data->v_guess()(i, 0)) *
        ((data->dynamics_matrix()(i, i) / sqrt(data->dynamics_matrix()(i, i))) *
         data->v_guess()(i, 0));
  }

  p_tilde_2 += __shfl_down_sync(0xFFFFFFFF, p_tilde_2, 16);
  p_tilde_2 += __shfl_down_sync(0xFFFFFFFF, p_tilde_2, 8);
  p_tilde_2 += __shfl_down_sync(0xFFFFFFFF, p_tilde_2, 4);
  p_tilde_2 += __shfl_down_sync(0xFFFFFFFF, p_tilde_2, 2);
  p_tilde_2 += __shfl_down_sync(0xFFFFFFFF, p_tilde_2, 1);

  __syncwarp();

  // calculate j_tilde^2 term
  double j_tilde_2 = 0.0;
  for (int i = threadIdx.x; i < data->NumVelocities(); i += blockDim.x) {
    double temp = 0.0;
    for (int j = 0; j < 3 * data->NumContacts(); j++) {
      temp += data->J()(j, i) * data->gamma_full()(j);
    }
    j_tilde_2 += (1 / sqrt(data->dynamics_matrix()(i, i))) *
                 (1 / sqrt(data->dynamics_matrix()(i, i))) * temp * temp;
  }

  j_tilde_2 += __shfl_down_sync(0xFFFFFFFF, j_tilde_2, 16);
  j_tilde_2 += __shfl_down_sync(0xFFFFFFFF, j_tilde_2, 8);
  j_tilde_2 += __shfl_down_sync(0xFFFFFFFF, j_tilde_2, 4);
  j_tilde_2 += __shfl_down_sync(0xFFFFFFFF, j_tilde_2, 2);
  j_tilde_2 += __shfl_down_sync(0xFFFFFFFF, j_tilde_2, 1);

  __syncwarp();

  // calculate ell_grad_tilde^2 term
  double ell_grad_tilde_2 = 0.0;
  for (int i = threadIdx.x; i < data->NumVelocities(); i += blockDim.x) {
    ell_grad_tilde_2 +=
        (-data->neg_grad()(i, 0) * (1 / sqrt(data->dynamics_matrix()(i, i)))) *
        (-data->neg_grad()(i, 0) * (1 / sqrt(data->dynamics_matrix()(i, i))));
  }

  ell_grad_tilde_2 += __shfl_down_sync(0xFFFFFFFF, ell_grad_tilde_2, 16);
  ell_grad_tilde_2 += __shfl_down_sync(0xFFFFFFFF, ell_grad_tilde_2, 8);
  ell_grad_tilde_2 += __shfl_down_sync(0xFFFFFFFF, ell_grad_tilde_2, 4);
  ell_grad_tilde_2 += __shfl_down_sync(0xFFFFFFFF, ell_grad_tilde_2, 2);
  ell_grad_tilde_2 += __shfl_down_sync(0xFFFFFFFF, ell_grad_tilde_2, 1);

  __syncwarp();

  // calculate momentum residue
  if (threadIdx.x == 0) {
    momentum_residue = sqrt(ell_grad_tilde_2);
    momentum_scale = max(sqrt(p_tilde_2), sqrt(j_tilde_2));
  }

  __syncwarp();
}

// ========================================================================
// Kernels
// ========================================================================

__global__ void SolveSearchDirectionKernel(SAPGPUData* data) {
  extern __shared__ double sums[];

  CalcSearchDirection(data, sums);

  __syncwarp();
}

__global__ void SolveWithGuessKernel(SAPGPUData* data) {
  extern __shared__ double sums[];

  // SAP Iteration flag
  double& flag = sums[0];
  double& first_iter = sums[1];
  double& momentum_residue = sums[2];
  double& momentum_scale = sums[3];
  double& ell_previous = sums[4];

  Eigen::Map<Eigen::MatrixXd> prev_v_guess(
      sums + 5, data->NumVelocities(),
      1);  // previous v_guess, used for termination logic

  if (threadIdx.x == 0) {
    flag = 1.0;  // thread 0 initializes the flag
    first_iter = 1.0;
  }

  __syncwarp();

  // TODO: might need replace this while loop to a for loop
  // SAP Iteration loop
  for (int iter = 0; iter < max_iteration; iter++) {
    if (flag == 0.0) break;

    if (flag == 1.0) {
      // calculate search direction
      // we add offset to shared memory to avoid SoveWithGuessImplKernel
      // __shared__ varibales being overwritten
      CalcSearchDirection(data, sums + 5 + data->NumVelocities());

      __syncwarp();

      // perform line search
      // we add offset to shared memory to avoid SoveWithGuessImplKernel
      // __shared__ varibales being overwritten
      double alpha = SAPLineSearch(data, sums + 5 + data->NumVelocities());

      __syncwarp();

      // calculate momentum residule and momentum scale
      CalcStoppingCriteriaResidual(data, momentum_residue, momentum_scale);

      __syncwarp();

      // Thread 0 registers first results or check residual if the current
      // iteration is not 0, if necessary, continue
      if (threadIdx.x == 0) {
        if (first_iter == 1.0) {
          first_iter = 0.0;
        } else {
          if (momentum_residue <=
              abs_tolerance + rel_tolerance * momentum_scale) {
            flag = 0.0;
            // printf("OUTER LOOP TERMINATE - momentum residue\n");
          }

          // cost criteria check
          double ell = data->momentum_cost()(0, 0) +
                       data->constraint_cost()(0, 0);  // current cost
          double ell_scale = 0.5 * (abs(ell) + abs(ell_previous));
          double ell_decrement = std::abs(ell_previous - ell);
          // printf("ell_decrement %.30f, rhs: %.30f\n", ell_decrement,
          //        cost_abs_tolerance + cost_rel_tolerance * ell_scale);
          if (ell_decrement <
                  cost_abs_tolerance + cost_rel_tolerance * ell_scale &&
              alpha > 0.5) {
            flag = 0.0;
            // printf("OUTER LOOP TERMINATE - cost criteria\n");
          }
        }

        ell_previous =
            data->momentum_cost()(0, 0) + data->constraint_cost()(0, 0);

        // assign previous
        for (int i = 0; i < data->NumVelocities(); i++) {
          prev_v_guess(i, 0) = data->v_guess()(i, 0);
        }

        // printf("this is a step of cholsolve\n");
        // printf("alpha at this step is %f\n", alpha);
      }

      __syncwarp();
    }
  }
  // TODO: remove this debugging output
  // if (threadIdx.x == 0) printf("SAP Converged!\n");
  __syncwarp();
}

// ==========================================================================
// Driver Functions
// This function is used in the unit test to perform a complete SAP solve,
// including search direction calculation and line search
void SAPGPUData::TestOneStepSapGPU() {
  std::cout << "TestOneStepSapGPU with GPU called with " << num_problems
            << " problems" << std::endl;

  // We create two GPU data struct instances for gtest unit test purpose

  int threadsPerBlock = 32;

  // unit test: check for complete solve without constraint
  // the expected result shall converge to free motion velocity v_star
  SolveWithGuessKernel<<<num_problems, threadsPerBlock,
                         4096 * sizeof(double)>>>(d_sap_gpu_data_solve);

  HANDLE_ERROR(cudaDeviceSynchronize());

  std::cout << "finished call" << std::endl;
}

// // This function is used in the unit test to confirm the cost evaluation and
// // the cholesky solve are correct
// void TestCostEvalAndSolveSapGPU(std::vector<SAPCPUData>& sap_cpu_data,
//                                 std::vector<double>& momentum_cost,
//                                 std::vector<double>& constraint_cost,
//                                 std::vector<Eigen::MatrixXd>& hessian,
//                                 std::vector<Eigen::MatrixXd>& neg_grad,
//                                 std::vector<Eigen::MatrixXd>& chol_x,
//                                 int num_velocities, int num_contacts,
//                                 int num_problems) {
//   SAPGPUData sap_gpu_data_dir;  // GPU data struct instance for validation of
//   // calculation results of one step

//   sap_gpu_data_dir.MakeSAPGPUData(sap_cpu_data);

//   SAPGPUData* d_sap_gpu_data_dir;
//   HANDLE_ERROR(cudaMalloc(&d_sap_gpu_data_dir, sizeof(SAPGPUData)));
//   HANDLE_ERROR(cudaMemcpy(d_sap_gpu_data_dir, &sap_gpu_data_dir,
//                           sizeof(SAPGPUData), cudaMemcpyHostToDevice));

//   // unit test: check for search direction solved by cholesky solve
//   // Hx = -grad

//   int threadsPerBlock = 32;

//   SolveSearchDirectionKernel<<<num_problems, threadsPerBlock,
//                                4096 * sizeof(double)>>>(d_sap_gpu_data_dir);

//   HANDLE_ERROR(cudaDeviceSynchronize());

//   // Transfer back to CPU for gtest validation
//   sap_gpu_data_dir.RetriveMomentumCostToCPU(momentum_cost);
//   sap_gpu_data_dir.RetriveConstraintCostToCPU(constraint_cost);
//   sap_gpu_data_dir.RetriveHessianToCPU(hessian);
//   sap_gpu_data_dir.RetriveNegGradToCPU(neg_grad);
//   sap_gpu_data_dir.RetriveCholXToCPU(chol_x);

//   sap_gpu_data_dir.DestroySAPGPUData();
// }

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