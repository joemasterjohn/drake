#pragma once

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

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

class FullSolveSAP {
 public:
  FullSolveSAP() {
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

  void init(Sphere* h_spheres_in, int numProblems_in, int numSpheres_in,
            int numContacts_in, bool writeout_in);
  void step();
  void destroy();

 private:
  CollisionGPUData* gpu_collision_data;
  SAPGPUData* sap_gpu_data;

  Sphere* h_spheres;
  int numProblems;
  int numSpheres;
  int numContacts;
  bool writeout;
  int iter = 0;

  std::string base_foldername = "/home/jsonzhou/Desktop/drake/output";
};