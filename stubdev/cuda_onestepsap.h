#include "stubdev/cuda_cholesky.h"
#include "stubdev/cuda_gauss_seidel.h"
#include "stubdev/cuda_gpu_collision.h"
#include "stubdev/cuda_matmul.h"
#include <eigen3/Eigen/Dense>

void test_onestep_sap(std::vector<Eigen::MatrixXd>& v_A,
                      std::vector<Eigen::MatrixXd>& v_J,
                      std::vector<Eigen::MatrixXd>& v_gamma,
                      std::vector<Eigen::MatrixXd>& v_m,
                      std::vector<Eigen::MatrixXd>& v_dv, int num_equations);