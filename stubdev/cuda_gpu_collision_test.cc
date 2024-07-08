#include "stubdev/cuda_gpu_collision.h"

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, GPU_Collision) {
  const int numSpheres = 10;
  const int numProblems = 1;

  // initialize the problem input spheres_vec, within a box of size 4x4x4, all
  // radius 0.5
  Sphere h_spheres[numProblems * numSpheres];
  for (int i = 0; i < numProblems; ++i) {
    for (int j = 0; j < numSpheres; ++j) {
      h_spheres[i * numSpheres + j].center = Eigen::Vector3d::Random();
      h_spheres[i * numSpheres + j].center(0) =
          3.0 * (h_spheres[i * numSpheres + j].center(0) + 1.0) - 1.5;
      h_spheres[i * numSpheres + j].center(1) =
          3.0 * (h_spheres[i * numSpheres + j].center(1) + 1.0) - 1.5;
      h_spheres[i * numSpheres + j].center(2) =
          3.0 * (h_spheres[i * numSpheres + j].center(2) + 1.0) - 1.5;
      h_spheres[i * numSpheres + j].radius = 0.5;

      // initialize velocity randomly for each sphere
      // the x,y, and z components of the velocity are in the range of [-1, 1]
      h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d::Random();

      // initialize material properties
      h_spheres[i * numSpheres + j].stiffness = 10.0;
      h_spheres[i * numSpheres + j].damping = 0.1;
      h_spheres[i * numSpheres + j].mass = 0.05;
    }
  }

  // Allocate memory for results on host
  CollisionData h_collisionMatrixSpheres[numProblems * numSpheres * numSpheres];
  double* h_jacobian = new double[numProblems * (numSpheres * 3) *
                                  (numSpheres * numSpheres * 3)];
  double* h_gamma = new double[numProblems * (numSpheres * numSpheres * 3)];
  int* h_num_collisions = new int[numProblems];

  double* h_dynamic_matrix =
      new double[numProblems * numSpheres * 3 * numSpheres * 3];
  double* h_velocity_vector = new double[numProblems * numSpheres * 3];

  // Run the GPU collision engine
  CollisionEngine(h_spheres, numProblems, numSpheres, h_collisionMatrixSpheres,
                  h_jacobian, h_gamma, h_num_collisions, h_dynamic_matrix,
                  h_velocity_vector);

  // Print out the results
  for (int i = 0; i < numProblems; i++) {
    std::cout << "Problem " << i << ":" << std::endl;
    std::cout << "Number of valid collisions: " << h_num_collisions[i]
              << std::endl;

    // print out dynamic matrix and velcity vector
    Eigen::Map<Eigen::MatrixXd> dynamic_matrix(
        h_dynamic_matrix + i * numSpheres * 3 * numSpheres * 3, numSpheres * 3,
        numSpheres * 3);
    Eigen::Map<Eigen::VectorXd> velocity_vector(
        h_velocity_vector + i * numSpheres * 3, numSpheres * 3);

    std::cout << "Dynamic Matrix: " << std::endl;
    for (int j = 0; j < numSpheres * 3; j++) {
      for (int k = 0; k < numSpheres * 3; k++) {
        std::cout << dynamic_matrix(j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Velocity Vector: " << std::endl;
    for (int j = 0; j < numSpheres * 3; j++) {
      std::cout << velocity_vector(j) << " ";
    }

    std::cout << std::endl;

    if (h_num_collisions[i] > 0) {
      // map jacobians and gamma to eigen::map
      // note that here we do dynamic mapping, to clip the jacobian and gamma
      // only to the point of valid collision number
      // note that the memory block reserved for jacobian and gamma is assuming
      // maximum number of possible collisions
      Eigen::Map<Eigen::MatrixXd> jacobian(
          h_jacobian + i * (numSpheres * 3) * (numSpheres * numSpheres * 3),
          numSpheres * 3, h_num_collisions[i] * 3);
      Eigen::Map<Eigen::VectorXd> gamma(
          h_gamma + i * (numSpheres * numSpheres * 3), h_num_collisions[i] * 3);

      // print out jacobians
      std::cout << "Jacobian: " << std::endl;
      for (int j = 0; j < numSpheres * 3; j++) {
        for (int k = 0; k < h_num_collisions[i] * 3; k++) {
          std::cout << std::setw(10) << std::setprecision(4) << jacobian(j, k)
                    << " ";
        }
        std::cout << std::endl;
      }

      // print out gamma
      std::cout << "Gamma: " << std::endl;
      for (int j = 0; j < h_num_collisions[i] * 3; j++) {
        std::cout << gamma(j) << " ";
      }
      std::cout << std::endl;
    }

    for (int j = 0; j < numSpheres; j++) {
      for (int k = 0; k < numSpheres; k++) {
        if (h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                     j * numSpheres + k]
                .isColliding) {
          std::cout << "Collision between Sphere " << j << " and Sphere " << k
                    << std::endl;
          std::cout << "Sphere " << j << " center: ("
                    << h_spheres[i * numSpheres + j].center(0) << ", "
                    << h_spheres[i * numSpheres + j].center(1) << ", "
                    << h_spheres[i * numSpheres + j].center(2) << ")"
                    << std::endl;
          std::cout << "Sphere " << k << " center: ("
                    << h_spheres[i * numSpheres + k].center(0) << ", "
                    << h_spheres[i * numSpheres + k].center(1) << ", "
                    << h_spheres[i * numSpheres + k].center(2) << ")"
                    << std::endl;
          std::cout << "Collision Point: ("
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .p_WC(0)
                    << ", "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .p_WC(1)
                    << ", "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .p_WC(2)
                    << ")" << std::endl;
          std::cout << "Collision Normal: ("
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .nhat_BA_W(0)
                    << ", "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .nhat_BA_W(1)
                    << ", "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .nhat_BA_W(2)
                    << ")" << std::endl;
          std::cout << "Overlap Distance: "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .phi0
                    << std::endl;
          std::cout << "Rotation Matrix: " << std::endl;
          for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
              std::cout
                  << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                              j * numSpheres + k]
                         .R(a, b)
                  << " ";
            }
            std::cout << std::endl;
          }

          std::cout << "Sphere " << j << " velocity: ("
                    << h_spheres[i * numSpheres + j].velocity(0) << ", "
                    << h_spheres[i * numSpheres + j].velocity(1) << ", "
                    << h_spheres[i * numSpheres + j].velocity(2) << ")"
                    << std::endl;

          std::cout << "Sphere " << k << " velocity: ("
                    << h_spheres[i * numSpheres + k].velocity(0) << ", "
                    << h_spheres[i * numSpheres + k].velocity(1) << ", "
                    << h_spheres[i * numSpheres + k].velocity(2) << ")"
                    << std::endl;

          std::cout << "Normal Relative Velocity: "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .vn
                    << std::endl;

          std::cout << "Local Gamma: ("
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .gamma(0)
                    << ", "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .gamma(1)
                    << ", "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .gamma(2)
                    << ")" << std::endl;
          std::cout << "Global Gamma: ("
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .gamma_W(0)
                    << ", "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .gamma_W(1)
                    << ", "
                    << h_collisionMatrixSpheres[i * numSpheres * numSpheres +
                                                j * numSpheres + k]
                           .gamma_W(2)
                    << ")" << std::endl;

          std::cout << "====================================" << std::endl;
        }
      }
    }
  }

  std::cout << "GPU monosphere collision check ended" << std::endl;
}

}  // namespace
}  // namespace drake
