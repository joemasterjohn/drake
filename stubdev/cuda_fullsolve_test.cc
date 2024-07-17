#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "stubdev/cuda_cholesky.h"
#include "stubdev/cuda_gpu_collision.h"
#include "stubdev/cuda_onestepsap.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

bool create_directory(const std::string& path) {
  std::error_code ec;
  if (std::filesystem::create_directories(path, ec)) {
    return true;
  } else {
    std::cerr << "Failed to create directory " << path << ": " << ec.message()
              << std::endl;
    return false;
  }
}

GTEST_TEST(KernelTest, FullSolveTest) {
  int numSpheres = 22;

  // i HATE THIS
  int numProblems = 200;

  // Get the current working directory
  std::string base_foldername = "/home/jsonzhou/Desktop/drake/output";
  create_directory(base_foldername);
  for (int i = 0; i < numProblems; i++) {
    std::string problem_foldername =
        base_foldername + "/problem_" + std::to_string(i);
    create_directory(problem_foldername);
  }

  std::cout << "Output directories created at: " << base_foldername
            << std::endl;

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

        double random_angle =
            static_cast<double>(rand()) / RAND_MAX * 2.0 * M_PI;
        p << 0.0 + 4.5 * cos(random_angle), 3.4641 + 4.5 * sin(random_angle),
            0.0;
      }

      h_spheres[i * numSpheres + j].center = p;

      h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d::Zero();

      if (j == 21) [[unlikely]] {
        // a random aiming point, from (0,0.25) to (0.0,3.5)
        Eigen::Vector3d random_target(
            0.0, 0.25 + static_cast<double>(rand()) / RAND_MAX * 3.25, 0.0);
        Eigen::Vector3d direction = random_target - p;
        direction.normalize();
        // scale up the velocity to 8.0 to 20.0, random
        h_spheres[i * numSpheres + j].velocity =
            direction * 8.0 +
            static_cast<double>(rand()) / RAND_MAX * 12.0 * direction;
      }

      h_spheres[i * numSpheres + j].radius = 0.5;

      // initialize material properties
      h_spheres[i * numSpheres + j].stiffness = 40000.0;
      h_spheres[i * numSpheres + j].damping = 0.5;
      h_spheres[i * numSpheres + j].mass = 0.05;
    }
  }

  for (int iter = 0; iter < 300; iter++) {
    // Allocate memory for results on host
    CollisionData* h_collisionMatrixSpheres =
        new CollisionData[numProblems * numSpheres * numSpheres];
    double* h_jacobian = new double[numProblems * (numSpheres * 3) *
                                    (numSpheres * numSpheres * 3)];
    int* h_num_collisions = new int[numProblems];

    double* h_dynamic_matrix =
        new double[numProblems * numSpheres * 3 * numSpheres * 3];
    double* h_velocity_vector = new double[numProblems * numSpheres *
                                           3];  // this shall serve as v_guess

    double* h_phi0 = new double[numProblems * numSpheres * numSpheres];
    double* h_v_star = new double[numProblems * numSpheres * 3];

    double* h_contact_stiffness =
        new double[numProblems * numSpheres *
                   numSpheres];  // vector to store harmonic mean of stiffness
                                 // for each contact
    double* h_contact_damping =
        new double[numProblems * numSpheres * numSpheres];

    // // Run the GPU collision engine
    CollisionEngine(h_spheres, numProblems, numSpheres,
                    h_collisionMatrixSpheres, h_jacobian, h_num_collisions,
                    h_dynamic_matrix, h_velocity_vector, h_v_star, h_phi0,
                    h_contact_stiffness, h_contact_damping);

    std::vector<Eigen::MatrixXd> dynamic_matrix_vec;
    std::vector<Eigen::MatrixXd> J_vec;
    std::vector<Eigen::MatrixXd> v_guess_vec;
    std::vector<Eigen::MatrixXd> v_star_vec;
    std::vector<int> num_collision_vec;
    std::vector<Eigen::VectorXd> phi0_vec;
    std::vector<Eigen::VectorXd> stiffness_vec;
    std::vector<Eigen::VectorXd> damping_vec;

    // Print out the results
    for (int i = 0; i < numProblems; i++) {
      // print out dynamic matrix and velcity vector
      Eigen::Map<Eigen::MatrixXd> dynamic_matrix(
          h_dynamic_matrix + i * numSpheres * 3 * numSpheres * 3,
          numSpheres * 3, numSpheres * 3);
      Eigen::Map<Eigen::VectorXd> velocity_vector(
          h_velocity_vector + i * numSpheres * 3, numSpheres * 3);
      Eigen::Map<Eigen::VectorXd> v_star(h_v_star + i * numSpheres * 3,
                                         numSpheres * 3);
      Eigen::Map<Eigen::VectorXd> phi0(h_phi0 + i * numSpheres * numSpheres,
                                       numSpheres * numSpheres);
      Eigen::Map<Eigen::MatrixXd> jacobian(
          h_jacobian + i * (numSpheres * 3) * (numSpheres * numSpheres * 3),
          numSpheres * numSpheres * 3, numSpheres * 3);
      Eigen::Map<Eigen::VectorXd> contact_stiffness(
          h_contact_stiffness + i * (numSpheres * numSpheres),
          numSpheres * numSpheres);
      Eigen::Map<Eigen::VectorXd> contact_damping(
          h_contact_damping + i * (numSpheres * numSpheres),
          numSpheres * numSpheres);

      // experimental data structure
      dynamic_matrix_vec.push_back(dynamic_matrix);
      J_vec.push_back(jacobian);
      v_guess_vec.push_back(velocity_vector);
      v_star_vec.push_back(v_star);
      num_collision_vec.push_back(h_num_collisions[i]);
      phi0_vec.push_back(phi0);
      stiffness_vec.push_back(contact_stiffness);
      damping_vec.push_back(contact_damping);
    }

    // ===================================================
    // END OF GEOMETRY ENGINE FUNCTION CALLS
    // ===================================================

    // ===================================================
    // START OF ACTUAL SAP SOLVER FUNCTION CALLS
    // ===================================================

    int num_contacts = numSpheres * numSpheres;
    int num_velocities = numSpheres * 3;
    int num_problems = numProblems;

    std::vector<SAPCPUData> sap_cpu_data;

    for (int i = 0; i < numProblems; i++) {
      SAPCPUData sap_data;

      sap_data.num_contacts = num_contacts;
      sap_data.num_velocities = num_velocities;
      sap_data.num_problems = num_problems;

      sap_data.dynamics_matrix = dynamic_matrix_vec[i];
      sap_data.v_star = v_star_vec[i];
      sap_data.v_guess = v_guess_vec[i];
      sap_data.constraint_data.J = J_vec[i];

      sap_data.constraint_data.phi0 = phi0_vec[i];
      sap_data.constraint_data.contact_stiffness = stiffness_vec[i];
      sap_data.constraint_data.contact_damping = damping_vec[i];

      sap_data.constraint_data.num_active_contacts = num_collision_vec[i];

      // initialize impulse data vector and regularization matrices
      for (int j = 0; j < num_contacts; j++) {
        sap_data.R.push_back(Eigen::Vector3d::Ones(3, 1));
      }

      sap_cpu_data.push_back(sap_data);
    }

    std::vector<Eigen::MatrixXd> v_solved;

    TestOneStepSapGPU(sap_cpu_data, v_solved, num_velocities, num_contacts,
                      num_problems);

    for (int i = 0; i < num_problems; i++) {
      std::cout << "Solved velocity: " << std::endl;
      for (int j = 0; j < num_velocities; j++) {
        std::cout << v_solved[i](j, 0) << " ";
      }
      std::cout << std::endl;

      std::cout << "time: " << iter * dt << " Kinetic Energy of the system:";
      double kinetic_energy = 0.0;
      for (int j = 0; j < numSpheres; j++) {
        kinetic_energy +=
            0.5 * h_spheres[i * numSpheres + j].mass *
            (v_solved[i](j * 3, 0) * v_solved[i](j * 3, 0) +
             v_solved[i](j * 3 + 1, 0) * v_solved[i](j * 3 + 1, 0) +
             v_solved[i](j * 3 + 2, 0) * v_solved[i](j * 3 + 2, 0));
      }
      std::cout << kinetic_energy << std::endl;

      // update positions and
      // copy the new velocity back to the sphere

      for (int j = 0; j < numSpheres; j++) {
        // integrate velocity to positions
        h_spheres[i * numSpheres + j].center(0) +=
            h_spheres[i * numSpheres + j].velocity(0) * dt;
        h_spheres[i * numSpheres + j].center(1) +=
            h_spheres[i * numSpheres + j].velocity(1) * dt;
        h_spheres[i * numSpheres + j].center(2) +=
            h_spheres[i * numSpheres + j].velocity(2) * dt;

        h_spheres[i * numSpheres + j].velocity(0) = v_solved[i](j * 3, 0);
        h_spheres[i * numSpheres + j].velocity(1) = v_solved[i](j * 3 + 1, 0);
        h_spheres[i * numSpheres + j].velocity(2) = v_solved[i](j * 3 + 2, 0);
      }

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

      delete[] h_jacobian;
      delete[] h_num_collisions;
      delete[] h_dynamic_matrix;
      delete[] h_velocity_vector;
      delete[] h_phi0;
      delete[] h_v_star;
      delete[] h_contact_stiffness;
      delete[] h_contact_damping;

      h_jacobian = NULL;
      h_num_collisions = NULL;
      h_dynamic_matrix = NULL;
      h_velocity_vector = NULL;
      h_phi0 = NULL;
      h_v_star = NULL;
      h_contact_stiffness = NULL;
      h_contact_damping = NULL;
    }
  }
}

// ===================================================
// END OF ACTUAL SAP SOLVER FUNCTION CALLS
// ===================================================

}  // namespace
}  // namespace drake
