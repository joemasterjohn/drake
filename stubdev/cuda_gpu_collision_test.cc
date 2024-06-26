#include "stubdev/cuda_gpu_collision.h"

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, GPU_Collision) {
  const int numSpheres = 10;
  const int numProblems = 100;

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
    }
  }

  // Allocate memory for results on host
  CollisionData h_collisionMatrixSpheres[numProblems * numSpheres * numSpheres];

  // Run the GPU collision engine
  CollisionEngine(h_spheres, numProblems, numSpheres, h_collisionMatrixSpheres);

  // Print out the results
  for (int i = 0; i < numProblems; i++) {
    std::cout << "Problem " << i << ":" << std::endl;
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

          std::cout << "====================================" << std::endl;
        }
      }
    }
  }

  std::cout << "GPU monosphere collision check ended" << std::endl;
}

}  // namespace
}  // namespace drake
