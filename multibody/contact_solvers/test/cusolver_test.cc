#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <gtest/gtest.h>

#include "drake/multibody/contact_solvers/cusolver_utils.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

GTEST_TEST(CusolverTest, BasicTest) {
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;

  using data_type = double;

  const int64_t m = 3;
  const int64_t lda = m;
  const int64_t ldb = m;
  /*       | 1 2 3  |
   *   A = | 4 5 6  |
   *       | 7 8 10 |
   *
   * without pivoting: A = L*U
   *       | 1 0 0 |      | 1  2  3 |
   *   L = | 4 1 0 |, U = | 0 -3 -6 |
   *       | 7 2 1 |      | 0  0  1 |
   *
   * with pivoting: P*A = L*U
   *       | 0 0 1 |
   *   P = | 1 0 0 |
   *       | 0 1 0 |
   *
   *       | 1       0     0 |      | 7  8       10     |
   *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
   *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
   */

  const std::vector<data_type> A = {1.0, 4.0, 7.0, 2.0, 5.0,
                                    8.0, 3.0, 6.0, 10.0};
  const std::vector<data_type> B = {1.0, 2.0, 3.0};
  std::vector<data_type> X(m, 0);
  std::vector<data_type> LU(lda * m, 0);
  std::vector<int64_t> Ipiv(m, 0);
  int info = 0;

  data_type* d_A = nullptr;  /* device copy of A */
  data_type* d_B = nullptr;  /* device copy of B */
  int64_t* d_Ipiv = nullptr; /* pivoting sequence */
  int* d_info = nullptr;     /* error info */

  size_t d_lwork = 0;     /* size of workspace */
  void* d_work = nullptr; /* device workspace for getrf */
  size_t h_lwork = 0;     /* size of workspace */
  void* h_work = nullptr; /* host workspace for getrf */

  const int pivot_on = 1;
  const int algo = 0;

  if (pivot_on) {
    std::printf("pivot is on : compute P*A = L*U \n");
  } else {
    std::printf("pivot is off: compute A = L*U (not numerically stable)\n");
  }

  std::printf("A = (matlab base-1)\n");
  print_matrix(m, m, A.data(), lda);
  std::printf("=====\n");

  std::printf("B = (matlab base-1)\n");
  print_matrix(m, 1, B.data(), ldb);
  std::printf("=====\n");

  /* step 1: create cusolver handle, bind a stream */
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

  /* Create advanced params */
  cusolverDnParams_t params;
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));
  if (algo == 0) {
    std::printf("Using New Algo\n");
    CUSOLVER_CHECK(
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
  } else {
    std::printf("Using Legacy Algo\n");
    CUSOLVER_CHECK(
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1));
  }

  /* step 2: copy A to device */
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Ipiv),
                        sizeof(int64_t) * Ipiv.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(),
                             cudaMemcpyHostToDevice, stream));

  /* step 3: query working space of getrf */
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(
      cusolverH, params, m, m, traits<data_type>::cuda_data_type, d_A, lda,
      traits<data_type>::cuda_data_type, &d_lwork, &h_lwork));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work),
                        sizeof(data_type) * d_lwork));

  /* step 4: LU factorization */
  if (pivot_on) {
    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, m,
                                    traits<data_type>::cuda_data_type, d_A, lda,
                                    d_Ipiv, traits<data_type>::cuda_data_type,
                                    d_work, d_lwork, h_work, h_lwork, d_info));
  } else {
    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, m,
                                    traits<data_type>::cuda_data_type, d_A, lda,
                                    nullptr, traits<data_type>::cuda_data_type,
                                    d_work, d_lwork, h_work, h_lwork, d_info));
  }

  if (pivot_on) {
    CUDA_CHECK(cudaMemcpyAsync(Ipiv.data(), d_Ipiv,
                               sizeof(int64_t) * Ipiv.size(),
                               cudaMemcpyDeviceToHost, stream));
  }
  CUDA_CHECK(cudaMemcpyAsync(LU.data(), d_A, sizeof(data_type) * A.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost,
                             stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::printf("after Xgetrf: info = %d\n", info);
  if (0 > info) {
    std::printf("%d-th parameter is wrong \n", -info);
    exit(1);
  }
  if (pivot_on) {
    std::printf("pivoting sequence, matlab base-1\n");
    for (int j = 0; j < m; j++) {
      std::printf("Ipiv(%d) = %lu\n", j + 1, Ipiv[j]);
    }
  }
  std::printf("L and U = (matlab base-1)\n");
  print_matrix(m, m, LU.data(), lda);
  std::printf("=====\n");

  /*
   * step 5: solve A*X = B
   *       | 1 |       | -0.3333 |
   *   B = | 2 |,  X = |  0.6667 |
   *       | 3 |       |  0      |
   *
   */
  if (pivot_on) {
    CUSOLVER_CHECK(
        cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, m, 1, /* nrhs */
                         traits<data_type>::cuda_data_type, d_A, lda, d_Ipiv,
                         traits<data_type>::cuda_data_type, d_B, ldb, d_info));
  } else {
    CUSOLVER_CHECK(
        cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, m, 1, /* nrhs */
                         traits<data_type>::cuda_data_type, d_A, lda, nullptr,
                         traits<data_type>::cuda_data_type, d_B, ldb, d_info));
  }

  CUDA_CHECK(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * X.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::printf("X = (matlab base-1)\n");
  print_matrix(m, 1, X.data(), ldb);
  std::printf("=====\n");

  /* free resources */
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_Ipiv));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_work));

  CUSOLVER_CHECK(cusolverDnDestroyParams(params));

  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());
}
}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake