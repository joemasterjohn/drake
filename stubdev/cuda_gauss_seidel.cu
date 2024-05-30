#include <stdio.h>

#include <iostream>

#include "cuda_gauss_seidel.h"

static void HandleError(cudaError_t err, const char* file, int line) {
  // CUDA error handeling from the "CUDA by example" book
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void cu_dot(Eigen::Vector3d* v1, Eigen::Vector3d* v2, double* out,
                       size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = v1[idx].dot(v2[idx]);
  }
  return;
}

__global__ void cu_dot_vector_x(double* v1, double* v2, double* out, size_t N) {
  Eigen::Map<Eigen::VectorXd> v1_m(v1, N, 1);
  Eigen::Map<Eigen::VectorXd> v2_m(v1, N, 1);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = v1_m.dot(v2_m);
  }

  return;
}

__host__ __device__ void print(Eigen::Ref<const Eigen::MatrixXd> M) {
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j < M.cols(); ++j) {
      printf("%lf ", M(i, j));
    }
    printf("\n");
  }
  printf("\n");
}

__host__ __device__ void MultiplyStrictlyUpperInPlace(
    const Eigen::Ref<Eigen::MatrixXd> M, Eigen::Ref<Eigen::VectorXd> x) {
  for (int i = 0; i < M.rows(); ++i) {
    double v = 0;
    for (int j = i + 1; j < M.cols(); ++j) {
      v += M(i, j) * x(j);
    }
    x(i) = v;
  }
}

__host__ __device__ void SolveLowerTriangularInPlace(
    const Eigen::Ref<Eigen::MatrixXd> M, Eigen::Ref<Eigen::VectorXd> x) {
  for (int i = 0; i < M.rows(); ++i) {
    double v = x(i);
    for (int j = 0; j < i; ++j) {
      v -= M(i, j) * x(j);
    }
    x(i) = v / M(i, i);
  }
}

// Example of a function that may live in Drake that takes Eigen Refs.
// Must be decorated with __host__ __device__ for it to work on both GPU and
// CPU.
__host__ __device__ void gauss_seidel(const Eigen::Ref<Eigen::MatrixXd> M,
                                      const Eigen::Ref<Eigen::VectorXd> b,
                                      Eigen::Ref<Eigen::VectorXd> x,
                                      const int num_iterations) {
  // printf("d_M:\n");
  // print(M);
  // printf("d_b:\n");
  // print(b);

  x.setOnes();

  // printf("d_X:\n");
  // print(x);

  for (int i = 0; i < num_iterations; ++i) {
    // x = M.triangularView<Eigen::Lower>().solve(b -
    // M.triangularView<Eigen::StrictlyUpper>()*x);
    MultiplyStrictlyUpperInPlace(M, x);
    // printf("Iteration(%d) Step 1 d_X: U*x\n", i);
    // print(x);
    x = b - x;
    // printf("Iteration(%d) Step 2 d_X: b - U*x\n", i);
    // print(x);
    SolveLowerTriangularInPlace(M, x);
    // printf("Iteration(%d) Step 3 d_X: L\\b - U*x\n", i);
    // print(x);
    // printf("||M*x - b||: %lf\n", (M*x - b).norm());
  }
}

__global__ void cu_matrix_solve(double* M, double* b, double* x,
                                size_t num_equations, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_equations) {
    Eigen::Map<Eigen::MatrixXd> d_M(M + idx * n * n, n, n);
    Eigen::Map<Eigen::VectorXd> d_b(b + idx * n, n, 1);
    Eigen::Map<Eigen::VectorXd> d_x(x + idx * n, n, 1);

    // d_x = d_M.fullPivLu().solve(d_b);
    // d_x = d_M.partialPivLu().solve(d_b);
    // d_x = d_M.householderQr().solve(d_b);
    // d_x = d_M.colPivHouseholderQr().solve(d_b);
    // d_x = d_M.fullPivHouseholderQr().solve(d_b);
    // d_x = d_M.completeOrthogonalDecomposition().solve(d_b);
    // d_x = d_M.llt().solve(d_b);
    // d_x = d_M.ldlt().solve(d_b);
    // d_x = d_M.bdcSvd().solve(d_b);
    // d_x = d_M.jacobiSvd().solve(d_b);

    gauss_seidel(d_M, d_b, d_x, 100 /* num_iterations */);
  }
}

// The wrapper for the calling of the actual kernel
double dot(const std::vector<Eigen::Vector3d>& v1,
           const std::vector<Eigen::Vector3d>& v2) {
  int n = v1.size();
  double* ret = new double[n];

  // Allocate device arrays
  Eigen::Vector3d *dev_v1, *dev_v2;
  HANDLE_ERROR(cudaMalloc((void**)&dev_v1, sizeof(Eigen::Vector3d) * n));
  HANDLE_ERROR(cudaMalloc((void**)&dev_v2, sizeof(Eigen::Vector3d) * n));
  double* dev_ret;
  HANDLE_ERROR(cudaMalloc((void**)&dev_ret, sizeof(double) * n));

  // Copy to device
  HANDLE_ERROR(cudaMemcpy(dev_v1, v1.data(), sizeof(Eigen::Vector3d) * n,
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_v2, v2.data(), sizeof(Eigen::Vector3d) * n,
                          cudaMemcpyHostToDevice));

  // Dot product
  cu_dot<<<(n + 1023) / 1024, 1024>>>(dev_v1, dev_v2, dev_ret, n);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Copy to host
  HANDLE_ERROR(
      cudaMemcpy(ret, dev_ret, sizeof(double) * n, cudaMemcpyDeviceToHost));

  // Reduction of the array
  for (int i = 1; i < n; ++i) {
    ret[0] += ret[i];
  }

  // Return
  return ret[0];
}

// The wrapper for the calling of the actual kernel
double dot_vector_x(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
  int n = v1.size();
  double* ret = new double[n];

  // Allocate device arrays
  double *dev_v1, *dev_v2;
  HANDLE_ERROR(cudaMalloc((void**)&dev_v1, sizeof(double) * n));
  HANDLE_ERROR(cudaMalloc((void**)&dev_v2, sizeof(double) * n));
  double* dev_ret;
  HANDLE_ERROR(cudaMalloc((void**)&dev_ret, sizeof(double) * n));

  // Copy to device
  HANDLE_ERROR(cudaMemcpy(dev_v1, v1.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_v2, v2.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));

  // Dot product
  cu_dot_vector_x<<<(n + 1023) / 1024, 1024>>>(dev_v1, dev_v2, dev_ret, n);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Copy to host
  HANDLE_ERROR(
      cudaMemcpy(ret, dev_ret, sizeof(double) * n, cudaMemcpyDeviceToHost));

  // Reduction of the array
  for (int i = 1; i < n; ++i) {
    ret[0] += ret[i];
  }

  // Return
  return ret[0];
}

// The wrapper for the calling of the actual kernel
double matrix_solve(const std::vector<Eigen::MatrixXd>& M,
                    const std::vector<Eigen::VectorXd>& b,
                    const std::vector<Eigen::VectorXd>& x) {
  const int num_equations = M.size();
  const int n = b[0].size();

  double* x_result = new double[num_equations * n];

  // Allocate device arrays
  double *d_M, *d_b, *d_x;
  HANDLE_ERROR(
      cudaMalloc((void**)&d_M, sizeof(double) * num_equations * n * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(double) * num_equations * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_x, sizeof(double) * num_equations * n));

  // Copy to device
  for (int i = 0; i < num_equations; ++i) {
    HANDLE_ERROR(cudaMemcpy(d_M + i * n * n, M[i].data(),
                            sizeof(double) * n * n, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b + i * n, b[i].data(), sizeof(double) * n,
                            cudaMemcpyHostToDevice));
  }

  // Matrix solve
  cu_matrix_solve<<<(num_equations + 1023) / 1024, 1024>>>(d_M, d_b, d_x,
                                                           num_equations, n);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Copy to host
  HANDLE_ERROR(cudaMemcpy(x_result, d_x, sizeof(double) * num_equations * n,
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_equations; ++i) {
    Eigen::Map<Eigen::VectorXd> x_result_i(x_result + i * n, n, 1);
    std::cout << "||M*x - b||: " << (M[i] * x_result_i - b[i]).norm()
              << std::endl;
  }

  // Return
  return 0;
}
