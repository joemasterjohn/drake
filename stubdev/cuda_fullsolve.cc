#include "cuda_fullsolve.h"

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

void FullSolveSAP::init(Sphere* h_spheres_in, int numProblems_in,
                        int numSpheres_in, int numContacts_in,
                        bool writeout_in) {
  this->h_spheres = h_spheres_in;
  this->numProblems = numProblems_in;
  this->numSpheres = numSpheres_in;
  this->numContacts = numContacts_in;
  this->gpu_collision_data->Initialize(this->h_spheres, this->numProblems,
                                       this->numSpheres);
  this->sap_gpu_data->Initialize(this->numContacts, this->numSpheres * 3,
                                 this->numProblems, this->gpu_collision_data);
  this->writeout = writeout_in;

  if (writeout) {
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
void FullSolveSAP::step() {
  gpu_collision_data->Update();

  // Run the GPU collision engine
  gpu_collision_data->CollisionEngine(numProblems, numSpheres);
  // Run the SAP
  sap_gpu_data->Update();
  sap_gpu_data->TestOneStepSapGPU();
  gpu_collision_data->IntegrateExplicitEuler(numProblems, numSpheres);

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
  iter++;
}

void FullSolveSAP::destroy() {
  gpu_collision_data->Destroy();
  sap_gpu_data->Destroy();
}