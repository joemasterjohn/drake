#include "cuda_fullsolve.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, FullSolveTest) {
  int numSpheres = 22;
  int numProblems = 1;
  int numContacts = numSpheres * numSpheres;

  // initialize the problem input spheres_vec, within a box of size 4x4x4, all
  // radius 0.5
  Sphere h_spheres[numProblems * numSpheres];
  for (int i = 0; i < numProblems; i++) {
    for (int j = 0; j < numSpheres; j++) {
      Eigen::Vector3d p;
      if (j == 0) {
        p << 0.0, 0.0, 0.0;
      } else if (j == 1) {
        p << -0.5, 0.866025403784439, 0.0;
      } else if (j == 2) {
        p << 0.5, 0.866025403784439, 0.0;
      } else if (j == 3) {
        p << -1.0, 1.73205080756888, 0.0;
      } else if (j == 4) {
        p << 0.0, 1.73205080756888, 0.0;
      } else if (j == 5) {
        p << 1.0, 1.73205080756888, 0.0;
      } else if (j == 6) {
        p << -1.5, 2.59807621135332, 0.0;
      } else if (j == 7) {
        p << -0.5, 2.59807621135332, 0.0;
      } else if (j == 8) {
        p << 0.5, 2.59807621135332, 0.0;
      } else if (j == 9) {
        p << 1.5, 2.59807621135332, 0.0;
      } else if (j == 10) {
        p << -2.0, 3.46410161513775, 0.0;
      } else if (j == 11) {
        p << -1.0, 3.46410161513775, 0.0;
      } else if (j == 12) {
        p << 0.0, 3.46410161513775, 0.0;
      } else if (j == 13) {
        p << 1.0, 3.46410161513775, 0.0;
      } else if (j == 14) {
        p << 2.0, 3.46410161513775, 0.0;
      } else if (j == 15) {
        p << -2.5, 4.33012701892219, 0.0;
      } else if (j == 16) {
        p << -1.5, 4.33012701892219, 0.0;
      } else if (j == 17) {
        p << -0.5, 4.33012701892219, 0.0;
      } else if (j == 18) {
        p << 0.5, 4.33012701892219, 0.0;
      } else if (j == 19) {
        p << 1.5, 4.33012701892219, 0.0;
      } else if (j == 20) {
        p << 2.5, 4.33012701892219, 0.0;
      } else if (j == 21) {
        // TODO: randomize the last sphere
        // x between -1.5 to 1.5
        // y between -1.0 and -1.5
        // cur ball position
        // int col = i % 20;
        // p << -2.4 + static_cast<double>(col) * (4.8 / 20.0), -1.7, 0.0;

        // double random_angle =
        //     static_cast<double>(rand()) / RAND_MAX * 2.0 * M_PI;
        // p << 0.0 + 4.5 * cos(random_angle), 3.4641 + 4.5 * sin(random_angle),
        //     0.0;

        p << 0.0, -2.0, 0.0;
      }

      h_spheres[i * numSpheres + j].center = p;

      h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d::Zero();

      if (j == 21) [[unlikely]] {
        // a random aiming point, from (0,0.25) to (0.0,3.5)
        // Eigen::Vector3d random_target(
        //     0.0, 0.25 + static_cast<double>(rand()) / RAND_MAX * 3.25, 0.0);
        // Eigen::Vector3d direction = random_target - p;
        // direction.normalize();
        // // scale up the velocity to 8.0 to 20.0, random
        // h_spheres[i * numSpheres + j].velocity =
        //     direction * 8.0 +
        //     static_cast<double>(rand()) / RAND_MAX * 12.0 * direction;
        h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d(0.0, 5.0, 0.0);
      }

      h_spheres[i * numSpheres + j].radius = 0.5;

      // initialize material properties
      h_spheres[i * numSpheres + j].stiffness = 40000.0;
      h_spheres[i * numSpheres + j].damping = 0.5;
      h_spheres[i * numSpheres + j].mass = 0.05;
    }
  }

  FullSolveSapCPUEntry(h_spheres, numProblems, numSpheres, numContacts);
}

// ===================================================
// END OF ACTUAL SAP SOLVER FUNCTION CALLS
// ===================================================

}  // namespace
}  // namespace drake
