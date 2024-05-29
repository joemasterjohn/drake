// A CPU entry point for cuda_sap_solver

#include "cuda_sap_cpu_wrapper.h"

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

// Initialized data structure
void CudaSapCpuWrapper::init(Sphere* h_spheres_in, Plane* h_plane_in,
                             int numProblems_in, int numSpheres_in,
                             int numPlanes_in, int numContacts_in,
                             bool writeout_in) {
  this->h_spheres = h_spheres_in;
  this->h_planes = h_plane_in;
  this->numProblems = numProblems_in;
  this->numSpheres = numSpheres_in;
  this->numPlanes = numPlanes_in;
  this->numContacts = numContacts_in;
  this->gpu_collision_data->Initialize(this->h_spheres, this->numProblems,
                                       this->numSpheres);
  this->gpu_collision_data->InitializePlane(this->h_planes, this->numPlanes);
  this->gpu_collision_data->CopyStructToGPU();
  this->sap_gpu_data->Initialize(this->numContacts, this->numSpheres * 3,
                                 this->numProblems, this->gpu_collision_data);
  this->writeout = writeout_in;

  if (writeout) {
    base_foldername = base_foldername + std::to_string(this->numSpheres);
    create_directory(base_foldername);
    for (int i = 0; i < numProblems; i++) {
      std::string problem_foldername =
          base_foldername + "/problem_" + std::to_string(i);
      create_directory(problem_foldername);
    }

    std::cout << "Output directories created at: " << base_foldername
              << std::endl;
  }
}

// Perform num_steps with this function call
void CudaSapCpuWrapper::step(int num_steps) {
  if (iter == 0) {
    if (writeout) {
      gpu_collision_data->RetieveSphereDataToCPU(h_spheres);
      for (int i = 0; i < numProblems; i++) {
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
          double vel_magnitude =
              std::sqrt(std::pow(h_spheres[i * numSpheres + j].velocity(0), 2) +
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

    iter++;
  }

  // Run the SAP
  sap_gpu_data->TestOneStepSapGPU(num_steps);

  if (writeout) {
    gpu_collision_data->RetieveSphereDataToCPU(h_spheres);
    for (int i = 0; i < numProblems; i++) {
      // Create and open the file
      std::ostringstream iterStream;
      iterStream << "output_" << std::setw(4) << std::setfill('0') << iter;
      std::string filename = base_foldername + "/problem_" + std::to_string(i) +
                             "/" + iterStream.str() + ".csv";

      std::ofstream file(filename);
      if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
      }

      // Write column titles to the file
      file << "pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,vel_magnitude" << std::endl;

      // Write data to the file
      for (int j = 0; j < numSpheres; j++) {
        double vel_magnitude =
            std::sqrt(std::pow(h_spheres[i * numSpheres + j].velocity(0), 2) +
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

  // print out number of active contacts
  // std::vector<int> h_active_contacts;
  // sap_gpu_data->RetriveNumActiveContactToCPU(h_active_contacts);
  // int avg = 0;

  // for (int i = 0; i < numProblems; i++) {
  //   avg += h_active_contacts[i];
  // }

  // std::cout << "Average number of active contacts: " << avg / numProblems
  //           << std::endl;

  iter++;
}

// Destroy CudaSapCpuWrapper data structure
void CudaSapCpuWrapper::destroy() {
  gpu_collision_data->Destroy();
  sap_gpu_data->Destroy();
}