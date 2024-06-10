
#include <iostream>

#include "cuda_matmul.cuh"
#include "cuda_onestepsap.cuh"
#include "cuda_reduce.cuh"

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

void TestOneStepSapGPU(std::vector<Eigen::MatrixXd>& v_guess,
                       std::vector<SAPCPUData>& v_sap_data,
                       std::vector<double>& v_lambda_m,
                       std::vector<double>& v_lambda_r, int num_rbodies,
                       int num_contacts, int num_equations) {
  std::cout << "TestOneStepSapGPU with GPU called with " << num_equations
            << " equations" << std::endl;
  SAPGPUData sap_gpu_data;

  sap_gpu_data.MakeSAPGPUData(v_sap_data);

  int threadsPerBlock = 32;
  // allocate GPU memory to calculate v_guess - v_star
  double* d_v_guess;
  double* d_delta_v;

  size_t size_v_guess = num_equations * num_rbodies * 3 * sizeof(double);
  size_t size_delta_v = num_equations * num_rbodies * 3 * sizeof(double);

  HANDLE_ERROR(cudaMalloc((void**)&d_v_guess, size_v_guess));
  HANDLE_ERROR(cudaMalloc((void**)&d_delta_v, size_delta_v));

  // Copy data to device
  for (int i = 0; i < num_equations; i++) {
    HANDLE_ERROR(cudaMemcpy(d_v_guess + i * num_rbodies * 3, v_guess[i].data(),
                            num_rbodies * 3 * sizeof(double),
                            cudaMemcpyHostToDevice));
  }

  // envoke substraction kernel
  int num_strides = (3 * num_rbodies + threadsPerBlock - 1) / threadsPerBlock;

  // Launch kernel
  // Note: d_v_guess, d_v_star, and d_delta_v are all 3 * num_rbodies * 1
  // MatrixLinOp32thdKernel<<<num_equations, threadsPerBlock>>>(
  //     d_v_guess, sap_gpu_data.v_star(), d_delta_v, 3 * num_rbodies, 1,
  //     LinOpType::SUB, num_strides, num_equations);
  // cudaDeviceSynchronize();
}

void TestOneStepSap(std::vector<Eigen::MatrixXd>& v_guess,
                    std::vector<SAPCPUData>& v_sap_data,
                    std::vector<double>& v_lambda_m,
                    std::vector<double>& v_lambda_r, int num_rbodies,
                    int num_contacts, int num_equations) {
  std::cout << "TestOneStepSap with CPU called with " << num_equations
            << " equations" << std::endl;

  // calculation of lambda_m -> momentum cost
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
  MatrixLinOp32thdKernel<<<num_equations, threadsPerBlock>>>(
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
  MatrixMultiply32thdKernel<<<num_equations, 32>>>(
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
  MatrixMultiply32thdKernel<<<num_equations, 32>>>(d_delta_v, d_delta_P,
                                                   d_lambda, 1, 3 * num_rbodies,
                                                   1, stride, num_equations);
  cudaDeviceSynchronize();

  // copy results back
  HANDLE_ERROR(cudaMemcpy(v_lambda_m.data(), d_lambda, size_lambda,
                          cudaMemcpyDeviceToHost));

  // multiply the GPU computed results by 0.5 to finish the computation of
  // 0.5 * delta_v.transpose() * delta_p
  for (int i = 0; i < num_equations; i++) {
    v_lambda_m[i] *= 0.5;
  }

  // calculation of lambda_r -> impulse cost
  // reserve space for lambda_r on GPU, size is num_equations

  // copy impulse and R to GPU
  double* d_impulse;
  double* d_R;

  size_t size_impulse = num_equations * num_contacts * 3 * sizeof(double);
  size_t size_R = num_equations * num_contacts * 3 * 3 * sizeof(double);

  HANDLE_ERROR(cudaMalloc((void**)&d_impulse, size_impulse));
  HANDLE_ERROR(cudaMalloc((void**)&d_R, size_R));

  for (int i = 0; i < num_equations; i++) {
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
  size_t size_intermediate = num_equations * num_contacts * 3 * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_intermediate, size_intermediate));

  // calculate lambda_r = 0.5 * impulse.transpose() * R * impulse
  stride = 1;  // d_impulse is a column vector, so tranpose is a row vector
  MatrixMultiply32thdKernel<<<num_equations * num_contacts, 32>>>(
      d_impulse, d_R, d_intermediate, 1, 3, 3, stride,
      num_equations * num_contacts);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // allocate GPU space for final results for lambda_r
  double* d_lambda_r;
  size_t size_lambda_r = num_equations * num_contacts * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_lambda_r, size_lambda_r));

  stride = 1;  // intermediate is 1 by 3, and impulse vector is 3 x 1.
  MatrixMultiply32thdKernel<<<num_equations * num_contacts, 32>>>(
      d_intermediate, d_impulse, d_lambda_r, 1, 3, 1, stride,
      num_equations * num_contacts);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // allocate post reduce results for lambda_r
  double* d_lambda_r_reduced;
  size_t size_lambda_r_reduced = num_equations * sizeof(double);
  HANDLE_ERROR(cudaMalloc((void**)&d_lambda_r_reduced, size_lambda_r_reduced));

  // call reduce kernel, to sum up the lambda_r for each equation

  ReduceByProblemKernel<<<num_equations, 32>>>(d_lambda_r, d_lambda_r_reduced,
                                               num_equations, num_contacts);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // copy d_lambda_reduced back to v_lambda_r
  HANDLE_ERROR(cudaMemcpy(v_lambda_r.data(), d_lambda_r_reduced,
                          size_lambda_r_reduced, cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_equations; i++) {
    v_lambda_r[i] *= 0.5;
  }
}