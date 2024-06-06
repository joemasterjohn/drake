
#include "stubdev/cuda_onestepsap.h"

void test_onestep_sap(std::vector<Eigen::MatrixXd>& v_A,
                      std::vector<Eigen::MatrixXd>& v_J,
                      std::vector<Eigen::MatrixXd>& v_gamma,
                      std::vector<Eigen::MatrixXd>& v_m,
                      std::vector<Eigen::MatrixXd>& v_dv, int num_equations) {
  int nc3 = v_J[0].rows();
  int num_rbodies3 = v_A[0].rows();

  // allocate GPU memory
  double *d_A, *d_J, *d_gamma, *d_m, *d_dv;
  cudaMalloc(&d_A,
             num_rbodies3 * num_rbodies3 * num_equations * sizeof(double));
  cudaMalloc(&d_J, nc3 * num_rbodies3 * num_equations * sizeof(double));
  cudaMalloc(&d_gamma, nc3 * num_equations * sizeof(double));
  cudaMalloc(&d_m, num_rbodies3 * num_equations * sizeof(double));
  cudaMalloc(&d_dv, num_rbodies3 * num_equations * sizeof(double));
}