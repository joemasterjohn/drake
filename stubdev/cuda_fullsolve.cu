#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "cuda_fullsolve.h"
#include "cuda_gpu_collision.cuh"
#include "cuda_gpu_collision.h"
#include "cuda_onestepsap.cuh"
#include "cuda_onestepsap.h"

#if defined(_WIN32)
#include <direct.h>
#define mkdir _mkdir
#else
#include <sys/types.h>
#include <unistd.h>
#endif
#include <iomanip>

bool write_out = true;

bool create_directory(const std::string& path) {
  int result;
#if defined(_WIN32)
  result = mkdir(path.c_str(), 0);
#else
  result = mkdir(path.c_str(), 0755);
#endif
  if (result != 0) {
    if (errno == EEXIST) {
      std::cerr << "Directory already exists: " << path << std::endl;
    } else {
      std::cerr << "Failed to create directory " << path << ": "
                << strerror(errno) << std::endl;
    }
    return false;
  }
  return true;
}

void FullSolveSapCPUEntry(Sphere* h_spheres, const int numProblems,
                          const int numSpheres, const int numContacts) {
  // Get the current working directory
  std::string base_foldername = "/home/jsonzhou/Desktop/drake/output";

  if (write_out) {
    create_directory(base_foldername);
    for (int i = 0; i < numProblems; i++) {
      std::string problem_foldername =
          base_foldername + "/problem_" + std::to_string(i);
      create_directory(problem_foldername);
    }

    std::cout << "Output directories created at: " << base_foldername
              << std::endl;
  }

  CollisionGPUData gpu_collision_data;
  gpu_collision_data.Initialize(h_spheres, numProblems, numSpheres);

  SAPGPUData sap_gpu_data;
  sap_gpu_data.Initialize(numContacts, numSpheres * 3, numProblems,
                          &gpu_collision_data);

  for (int iter = 0; iter < 400; iter++) {
    gpu_collision_data.Update();

    // // Run the GPU collision engine
    gpu_collision_data.CollisionEngine(numProblems, numSpheres);

    // ===================================================
    // END OF GEOMETRY ENGINE FUNCTION CALLS
    // ===================================================

    // ===================================================
    // START OF ACTUAL SAP SOLVER FUNCTION CALLS
    // ===================================================

    int num_contacts = numSpheres * numSpheres;
    int num_velocities = numSpheres * 3;
    int num_problems = numProblems;
    sap_gpu_data.Update();
    sap_gpu_data.TestOneStepSapGPU();
    gpu_collision_data.IntegrateExplicitEuler(numProblems, numSpheres);
    std::cout << "time: " << (iter + 1) * dt << std::endl;

    for (int i = 0; i < num_problems; i++) {
      if (write_out) {
        gpu_collision_data.RetieveSphereDataToCPU(h_spheres);

        if (true) {
          // Create and open the file
          std::ostringstream iterStream;
          iterStream << "output_" << std::setw(4) << std::setfill('0') << iter;
          std::string filename = base_foldername + "/problem_" +
                                 std::to_string(i) + "/" + iterStream.str() +
                                 ".csv";

          std::ofstream file(filename);
          if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
          }

          // Write column titles to the file
          file << "pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,vel_magnitude"
               << std::endl;

          // Write data to the file
          for (int j = 0; j < numSpheres; j++) {
            double vel_magnitude = std::sqrt(
                std::pow(h_spheres[i * numSpheres + j].velocity(0), 2) +
                std::pow(h_spheres[i * numSpheres + j].velocity(1), 2) +
                std::pow(h_spheres[i * numSpheres + j].velocity(2), 2));

            file << h_spheres[i * numSpheres + j].center(0) << ","
                 << h_spheres[i * numSpheres + j].center(1) << ","
                 << h_spheres[i * numSpheres + j].center(2) << ","
                 << h_spheres[i * numSpheres + j].velocity(0) << ","
                 << h_spheres[i * numSpheres + j].velocity(1) << ","
                 << h_spheres[i * numSpheres + j].velocity(2) << ","
                 << vel_magnitude << std::endl;
          }

          file.close();
        }
      }
    }
  }
}
