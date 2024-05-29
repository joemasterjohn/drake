// Class definition for CudaSapCpuWrapper
// CudaSapCPUWrapper contains a CollisionGPUData and a SAPGPUData
// This completes a complete solve step for the simulation

#pragma once

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "cuda_gpu_collision.cuh"
#include "cuda_sap_solver.cuh"
#include "cuda_sap_solver.h"

#if defined(_WIN32)
#include <direct.h>
#define mkdir _mkdir
#else
#include <sys/types.h>
#include <unistd.h>
#endif
#include <iomanip>

class CudaSapCpuWrapper {
 public:
  CudaSapCpuWrapper() {
    // each CudaSapCPUWrapper object contains a CollisionGPUData and a
    // SAPGPUData
    gpu_collision_data = new CollisionGPUData();
    sap_gpu_data = new SAPGPUData();
  }

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

  void init(Sphere* h_spheres_in, Plane* h_plane_in, int numProblems_in,
            int numSpheres_in, int numPlanes_in, int numContacts_in,
            bool writeout_in);
  void step(int num_steps);
  void destroy();

 private:
  CollisionGPUData* gpu_collision_data;
  SAPGPUData* sap_gpu_data;

  Sphere* h_spheres;
  Plane* h_planes;
  int numProblems;
  int numSpheres;
  int numPlanes;
  int numContacts;
  bool writeout;
  int iter = 0;

  std::string base_foldername = "output";
};