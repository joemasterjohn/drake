#include "stubdev/cuda_gpu_collision.h"

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, GPU_Collision) {
  const int numSpheres = 6;

  Sphere h_spheres[numSpheres] = {// test case 1
                                  {{0.0, 0.0, 0.0}, 0.5},
                                  {{0.5, 0.5, 0.5}, 0.5},
                                  // test case 2
                                  {{20.0, 20.0, 20.0}, 0.5},
                                  {{21.0, 20.0, 20.0}, 0.5},
                                  // test case 3
                                  {{25.0, 25.0, 25.0}, 0.5},
                                  {{25.0, 25.0, 24.2}, 0.5}};

  // Allocate memory for results on host
  CollisionData h_collisionMatrixSpheres[numSpheres * numSpheres];

  // Run the GPU collision engine
  collision_engine(h_spheres, numSpheres, h_collisionMatrixSpheres);

  std::cout << "Sphere-Sphere Collisions:" << std::endl;
  for (int i = 0; i < numSpheres; ++i) {
    for (int j = 0; j < numSpheres; ++j) {
      if (h_collisionMatrixSpheres[i * numSpheres + j].isColliding) {
        std::cout << "Collision between Sphere " << i << " and Sphere " << j
                  << std::endl;
        std::cout << "Collision Point: ("
                  << h_collisionMatrixSpheres[i * numSpheres + j].p_WC(0)
                  << ", "
                  << h_collisionMatrixSpheres[i * numSpheres + j].p_WC(1)
                  << ", "
                  << h_collisionMatrixSpheres[i * numSpheres + j].p_WC(2) << ")"
                  << std::endl;
        std::cout << "Collision Normal: ("
                  << h_collisionMatrixSpheres[i * numSpheres + j].nhat_BA_W(0)
                  << ", "
                  << h_collisionMatrixSpheres[i * numSpheres + j].nhat_BA_W(1)
                  << ", "
                  << h_collisionMatrixSpheres[i * numSpheres + j].nhat_BA_W(2)
                  << ")" << std::endl;
      }
    }
  }

  std::cout << "GPU monosphere collision check ended" << std::endl;
}

}  // namespace
}  // namespace drake
