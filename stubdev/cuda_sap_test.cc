// cuda_sap_solver testing function
// this unit test runs the sap solver with spheres constrained within a box
// formed by 4 half-planes

// By running this code in default, it runs with number of spheres:{4, 7, 10,
// 11, 12, 13, 14, 15, 22}

// and with batch size:{1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
// 1000, 2000, 5000, 10000, 15000}

// The timing results will be printed out to the terminal

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "cuda_sap_cpu_wrapper.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

// mode identifies if it is
// write out mode (mode = 0),
// or benchmark mode (mode = 1)
double run(int numSpheres, int numProblems, int mode = 0) {
  int numPlanes = 4;
  // int numProblems = 10000;
  int numContacts = numSpheres * numSpheres;

  // initialize the problem input spheres_vec
  Sphere* h_spheres = new Sphere[numProblems * numSpheres];
  for (int i = 0; i < numProblems; i++) {
    for (int j = 0; j < numSpheres; j++) {
      Eigen::Vector3d p;
      // Case 1 - 22 Spheres in the environment
      if (j == 1) {
        p << 0.0, 0.0, 0.0;
      } else if (j == 2) {
        p << -0.035001, 0.0606313723625153, 0.0;
      } else if (j == 3) {
        p << 0.035001, 0.0606313723625153, 0.0;
      } else if (j == 4) {
        p << -0.070002, 0.1212627447250306, 0.0;
      } else if (j == 5) {
        p << 0.0, 0.1212627447250306, 0.0;
      } else if (j == 6) {
        p << 0.070002, 0.1212627447250306, 0.0;
      } else if (j == 7) {
        p << -0.105003, 0.1818941170875459, 0.0;
      } else if (j == 8) {
        p << -0.035001, 0.1818941170875459, 0.0;
      } else if (j == 9) {
        p << 0.035001, 0.1818941170875459, 0.0;
      } else if (j == 10) {
        p << 0.105003, 0.1818941170875459, 0.0;
      } else if (j == 11) {
        p << -0.140004, 0.2425254894500612, 0.0;
      } else if (j == 12) {
        p << -0.070002, 0.2425254894500612, 0.0;
      } else if (j == 13) {
        p << 0.0, 0.2425254894500612, 0.0;
      } else if (j == 14) {
        p << 0.070002, 0.2425254894500612, 0.0;
      } else if (j == 15) {
        p << 0.140004, 0.2425254894500612, 0.0;
      } else if (j == 16) {
        p << -0.175005, 0.3031568618125765, 0.0;
      } else if (j == 17) {
        p << -0.105003, 0.3031568618125765, 0.0;
      } else if (j == 18) {
        p << -0.035001, 0.3031568618125765, 0.0;
      } else if (j == 19) {
        p << 0.035001, 0.3031568618125765, 0.0;
      } else if (j == 20) {
        p << 0.105003, 0.3031568618125765, 0.0;
      } else if (j == 21) {
        p << 0.175005, 0.3031568618125765, 0.0;
      }

      if (j == 0) {
        // TODO: randomize the last sphere
        // x between -1.5 to 1.5
        // y between -1.0 and -1.5
        // cur ball position
        // int col = i % 20;
        // p << -2.4 + static_cast<double>(col) * (4.8 / 20.0), -1.7, 0.0;

        double random_angle =
            static_cast<double>(rand()) / RAND_MAX * 2.0 * M_PI;
        p << 0.0 + 0.15 * cos(random_angle), -0.12 + 0.03 * sin(random_angle),
            0.0;

        // p << 0.12, -0.1, 0.0;
      }

      h_spheres[i * numSpheres + j].center = p;

      h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d::Zero();

      h_spheres[i * numSpheres + j].mass = 0.17;

      if (j == 0) [[unlikely]] {
        // a random aiming point, from (0,0.25) to (0.0,3.5)
        Eigen::Vector3d random_target(
            0.0, 0.03 + static_cast<double>(rand()) / RAND_MAX * 0.15, 0.0);
        Eigen::Vector3d direction = random_target - p;
        direction.normalize();
        // scale up the velocity to 8.0 to 20.0, random
        // h_spheres[i * numSpheres + j].velocity =
        //     direction * 1.2 +

        //     static_cast<double>(rand()) / RAND_MAX * 0.5 * direction;
        h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d(0.0, 1.2, 0.0);

        h_spheres[i * numSpheres + j].mass = 0.17;
      }

      h_spheres[i * numSpheres + j].radius = 0.03;

      // initialize material properties
      h_spheres[i * numSpheres + j].stiffness = 10000.0;
      h_spheres[i * numSpheres + j].damping = 1e-10;
    }
  }

  Plane* h_planes = new Plane[numProblems * numPlanes];
  for (int i = 0; i < numProblems; i++) {
    for (int j = 0; j < numPlanes; j++) {
      if (j == 0) {
        h_planes[i * numPlanes + j].p1 << -0.25, 1.0, 0.0;  // -0.25 | -0.1
        h_planes[i * numPlanes + j].p2 << -0.25, 0.0, 0.0;
        h_planes[i * numPlanes + j].n << 1.0, 0.0, 0.0;
      } else if (j == 1) {
        h_planes[i * numPlanes + j].p1 << 0.0, 0.35, 0.0;
        h_planes[i * numPlanes + j].p2 << 1.0, 0.35, 0.0;  // 0.35 | 0.09
        h_planes[i * numPlanes + j].n << 0.0, -1.0, 0.0;
      } else if (j == 2) {
        h_planes[i * numPlanes + j].p1 << 0.25, 1.0, 0.0;
        h_planes[i * numPlanes + j].p2 << 0.25, 1.0, 0.0;
        h_planes[i * numPlanes + j].n << -1.0, 0.0, 0.0;
      } else if (j == 3) {
        h_planes[i * numPlanes + j].p1 << 0.0, -0.2, 0.0;
        h_planes[i * numPlanes + j].p2 << 1.0, -0.2, 0.0;
        h_planes[i * numPlanes + j].n << 0.0, 1.0, 0.0;
      }

      h_planes[i * numPlanes + j].stiffness = 10000.0;
      h_planes[i * numPlanes + j].damping = 1e-10;
    }
  }

  CudaSapCpuWrapper solver;

  if (mode == 0) {
    solver.init(h_spheres, h_planes, numProblems, numSpheres, numPlanes,
                numContacts, true);
  } else if (mode == 1) {
    solver.init(h_spheres, h_planes, numProblems, numSpheres, numPlanes,
                numContacts, false);
  }

  // Record the start time
  auto start = std::chrono::high_resolution_clock::now();

  if (mode == 0) {
    for (int i = 0; i < 800; i++) {
      solver.step(1);
    }
  } else if (mode == 1) {
    for (int i = 0; i < 1; i++) {
      solver.step(800);
    }
  }

  // Record the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<double, std::milli> duration = end - start;

  solver.destroy();

  return duration.count();
}

GTEST_TEST(KernelTest, OneStepSAP_GPU) {
  std::vector<int> numSpheres_vec = {4, 7, 10, 11, 12, 13, 14, 15, 22};
  std::vector<int> batchSize_vec = {1,    10,   50,   100,   200,  300,
                                    400,  500,  600,  700,   800,  900,
                                    1000, 2000, 5000, 10000, 15000};

  // Output Simulation Results
  // This is write out section
  // This section shall not be run with "bazel run"
  // If output is needed, uncomment this section, build and run in bazel-bin
  // for (int numSpheres : numSpheres_vec) {
  //   run(numSpheres, 1, 0);
  // }

  for (int numSpheres : numSpheres_vec) {
    for (int batchSize : batchSize_vec) {
      double sum = 0.0;
      double min_val = std::numeric_limits<double>::max();
      double max_val = std::numeric_limits<double>::min();

      for (int i = 0; i < 10; i++) {
        double timing = run(numSpheres, batchSize, 1);
        sum += timing;
        min_val = std::min(timing, min_val);
        max_val = std::max(timing, max_val);
      }
      std::cout << "NumSpheres: " << numSpheres << "  BatchSize: " << batchSize
                << "  Average: " << sum / 10.0 << "  min: " << min_val
                << "  max: " << max_val << " [ms]" << std::endl;
    }
  }
}

}  // namespace
}  // namespace drake

// ===================================================
// END OF ACTUAL SAP SOLVER FUNCTION CALLS
// ===================================================
