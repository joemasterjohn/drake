#define PRINTOUT

#include <iostream>
#include <vector>

#include "stubdev/cuda_cholesky.h"
#include "stubdev/cuda_gpu_collision.h"
#include "stubdev/cuda_onestepsap.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, FullSolveTest) {
  const int numSpheres = 10;
  const int numProblems = 1;

  std::vector<Eigen::MatrixXd> dynamic_matrix_vec;
  std::vector<Eigen::MatrixXd> J_vec;
  std::vector<Eigen::MatrixXd> v_guess_vec;
  std::vector<Eigen::MatrixXd> v_star_vec;
  std::vector<int> num_collision_vec;
  std::vector<Eigen::VectorXd> phi0_vec;
  std::vector<Eigen::VectorXd> stiffness_vec;
  std::vector<Eigen::VectorXd> damping_vec;

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

      // initialize material properties
      h_spheres[i * numSpheres + j].stiffness = 10.0;
      h_spheres[i * numSpheres + j].damping = 0.1;
      h_spheres[i * numSpheres + j].mass = 0.05;
    }
  }

  // Allocate memory for results on host
  CollisionData h_collisionMatrixSpheres[numProblems * numSpheres * numSpheres];
  double* h_jacobian = new double[numProblems * (numSpheres * 3) *
                                  (numSpheres * numSpheres * 3)];
  int* h_num_collisions = new int[numProblems];

  double* h_dynamic_matrix =
      new double[numProblems * numSpheres * 3 * numSpheres * 3];
  double* h_velocity_vector =
      new double[numProblems * numSpheres * 3];  // this shall serve as v_guess

  double* h_phi0 = new double[numProblems * numSpheres * numSpheres];
  double* h_v_star = new double[numProblems * numSpheres * 3];

  double* h_contact_stiffness =
      new double[numProblems * numSpheres *
                 numSpheres];  // vector to store harmonic mean of stiffness for
                               // each contact
  double* h_contact_damping = new double[numProblems * numSpheres * numSpheres];

  // Run the GPU collision engine
  CollisionEngine(h_spheres, numProblems, numSpheres, h_collisionMatrixSpheres,
                  h_jacobian, h_num_collisions, h_dynamic_matrix,
                  h_velocity_vector, h_v_star, h_phi0, h_contact_stiffness,
                  h_contact_damping);

  // Print out the results
  for (int i = 0; i < numProblems; i++) {
    // print out dynamic matrix and velcity vector
    Eigen::Map<Eigen::MatrixXd> dynamic_matrix(
        h_dynamic_matrix + i * numSpheres * 3 * numSpheres * 3, numSpheres * 3,
        numSpheres * 3);
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

#ifdef PRINTOUT
  for (int i = 0; i < numProblems; i++) {
    std::cout << "Problem " << i << ":" << std::endl;
    std::cout << "Number of valid collisions: " << h_num_collisions[i]
              << std::endl;

    std::cout << "Dynamic Matrix: " << std::endl;
    for (int j = 0; j < numSpheres * 3; j++) {
      for (int k = 0; k < numSpheres * 3; k++) {
        std::cout << dynamic_matrix_vec[i](j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    // we use the orginal velocity as guess velocity
    std::cout << "v vector: " << std::endl;
    for (int j = 0; j < numSpheres * 3; j++) {
      std::cout << v_guess_vec[i](j) << " ";
    }

    std::cout << std::endl;

    std::cout << "Free Motion Velocity Vector: " << std::endl;
    for (int j = 0; j < numSpheres * 3; j++) {
      std::cout << v_star_vec[i](j) << " ";
    }

    std::cout << std::endl;

    if (h_num_collisions[i] > 0) {
      // map jacobians and gamma to eigen::map
      // note that here we do dynamic mapping, to clip the jacobian and gamma
      // only to the point of valid collision number
      // note that the memory block reserved for jacobian and gamma is
      // assuming maximum number of possible collisions

      std::cout << "Phi0: " << std::endl;
      for (int j = 0; j < h_num_collisions[i]; j++) {
        std::cout << phi0_vec[i](j) << " ";
      }
      std::cout << std::endl;

      // print out contact stiffness and damping
      std::cout << "Contact Stiffness: " << std::endl;
      for (int j = 0; j < h_num_collisions[i]; j++) {
        std::cout << stiffness_vec[i](j) << " ";
      }

      std::cout << std::endl;

      std::cout << "Contact Damping: " << std::endl;
      for (int j = 0; j < h_num_collisions[i]; j++) {
        std::cout << damping_vec[i](j) << " ";
      }
    }
  }

  for (int i = 0; i < numProblems; i++) {
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

#endif

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

    // print sap_data.constraint_data.phi0
    std::cout << "Phi0 after move: " << std::endl;
    for (int j = 0; j < num_contacts; j++) {
      std::cout << sap_data.constraint_data.phi0(j) << " ";
    }
    std::cout << std::endl;

    // print out sap_data.contraint_data.J
    std::cout << "Jacobian after move:" << std::endl;
    for (int j = 0; j < h_num_collisions[i] * 3; j++) {
      for (int k = 0; k < numSpheres * 3; k++) {
        std::cout << sap_data.constraint_data.J(j, k) << " ";
      }
      std::cout << std::endl;
    }

    // initialize impulse data vector and regularization matrices
    for (int j = 0; j < num_contacts; j++) {
      sap_data.R.push_back(Eigen::Vector3d::Ones(3, 1));
    }

    sap_cpu_data.push_back(sap_data);
  }

  // Debugging section, checking the solved search direction
  //   std::vector<double> momentum_cost;
  //   std::vector<double> regularizer_cost;
  //   std::vector<Eigen::MatrixXd> hessian;
  //   std::vector<Eigen::MatrixXd> neg_grad;
  //   std::vector<Eigen::MatrixXd> chol_x;

  // invoke test sap call
  //   TestCostEvalAndSolveSapGPU(sap_cpu_data, momentum_cost, regularizer_cost,
  //                              hessian, neg_grad, chol_x, num_velocities,
  //                              num_contacts, num_problems);

  //   std::cout << "chol_x: " << std::endl;
  //   // print out results
  //   for (int i = 0; i < num_problems; i++) {
  //     // print out chol_x
  //     for (int j = 0; j < num_velocities; j++) {
  //       std::cout << chol_x[i](j, 0) << "  ";
  //     }
  //   }
  //   std::cout << std::endl;

  std::vector<Eigen::MatrixXd> v_solved;

  TestOneStepSapGPU(sap_cpu_data, v_solved, num_velocities, num_contacts,
                    num_problems);

  for (int i = 0; i < num_problems; i++) {
    std::cout << "Solved velocity: " << std::endl;
    for (int j = 0; j < num_velocities; j++) {
      std::cout << v_solved[i](j, 0) << " ";
    }
    std::cout << std::endl;
  }
}

// ===================================================
// END OF ACTUAL SAP SOLVER FUNCTION CALLS
// ===================================================

}  // namespace
}  // namespace drake
