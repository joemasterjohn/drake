/**
A benchmark to compare Schur complement solvers.
*/

#include <chrono>
#include <iostream>

#include <gflags/gflags.h>

#include "drake/common/unused.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/fem/block_sparse_cholesky_solver.h"
#include "drake/multibody/fem/cholmod_sparse_matrix.h"
#include "drake/multibody/fem/petsc_symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector4i;
using Eigen::VectorXd;
using geometry::GeometryId;
using geometry::GeometryInstance;
using geometry::SceneGraph;
using geometry::SceneGraphInspector;
using geometry::SourceId;
using geometry::Sphere;
using geometry::VolumeMesh;
using internal::BlockSparseCholeskySolver;
using internal::BuildAdjacencyGraph;
using internal::CalcEliminationOrdering;
using internal::CalcSparsityPattern;
using internal::CholmodSparseMatrix;
using internal::GetFillInGraph;
using internal::PetscSymmetricBlockSparseMatrix;
using internal::RestrictOrdering;
using internal::SchurComplement;
using internal::SymmetricBlockSparseMatrix;
using math::RigidTransformd;
using std::get;
using std::make_unique;
using std::unique_ptr;
using std::vector;

using Clock = std::chrono::high_resolution_clock;
using MatrixBlock = std::tuple<int, int, Matrix3d>;

DEFINE_int32(block_cholesky_solve_iterations, 1,
             "The number of times to run profiling solves.");
DEFINE_int32(cholmod_solve_iterations, 1,
             "The number of times to run profiling solves.");
DEFINE_int32(petsc_solve_iterations, 1,
             "The number of times to run profiling solves.");
DEFINE_double(
    resolution_hint, 1.0,
    "Controls the mesh resolution of a unit sphere and thus the problem size.");
DEFINE_double(percentage_in_contact, 0.2,
              "Controls the percentage of vertices in contact.");

unique_ptr<GeometryInstance> make_sphere_instance(double radius = 1.0) {
  return make_unique<GeometryInstance>(RigidTransformd::Identity(),
                                       make_unique<Sphere>(radius), "sphere");
}

/* Makes an arbitrary SPD element matrix sized 12x12. */
Eigen::Matrix<double, 12, 12> dummy_matrix12x12() {
  Eigen::Matrix<double, 12, 12> A;
  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 12; ++j) {
      A(i, j) = 3.14 * i + 2.7 * j;
    }
  }
  Eigen::Matrix<double, 12, 12> I =
      5 * Eigen::Matrix<double, 12, 12>::Identity();
  return A * A.transpose() + I;
}

/* Builds a Petsc tangent matrix from elements information. */
std::unique_ptr<PetscSymmetricBlockSparseMatrix> MakePetscMatrix(
    const vector<Vector4i>& elements, int num_nodes) {
  const int num_elements = elements.size();
  std::vector<std::set<int>> index_to_row(num_nodes);
  for (int e = 0; e < num_elements; ++e) {
    const Vector4i& element = elements[e];
    for (int a = 0; a < 4; ++a) {
      const int block_row = element(a);
      for (int b = 0; b < 4; ++b) {
        const int block_col = element(b);
        if (block_row >= block_col) index_to_row[block_col].insert(block_row);
      }
    }
  }
  std::vector<int> num_upper_triangular_blocks_per_row(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    num_upper_triangular_blocks_per_row[i] = index_to_row[i].size();
  }

  auto result = std::make_unique<PetscSymmetricBlockSparseMatrix>(
      3 * num_nodes, 3, num_upper_triangular_blocks_per_row);
  for (int e = 0; e < num_elements; ++e) {
    const Vector4i& element = elements[e];
    result->AddToBlock(element, dummy_matrix12x12());
  }
  return result;
}

/* Builds a 3x3 block matrix from elements information. */
std::vector<std::vector<MatrixBlock>> MakeMatrixBlocks(
    const vector<Vector4i>& elements, int num_nodes) {
  const int num_elements = elements.size();
  std::vector<std::set<int>> sorted_block_cols(num_nodes);
  for (int e = 0; e < num_elements; ++e) {
    const Vector4i& element = elements[e];
    for (int a = 0; a < 4; ++a) {
      const int block_row = element(a);
      for (int b = 0; b < 4; ++b) {
        const int block_col = element(b);
        sorted_block_cols[block_row].insert(block_col);
      }
    }
  }

  /* Map col to the index in the array */
  std::vector<std::unordered_map<int, int>> block_col_to_flat_index(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    int index = 0;
    for (int block_col : sorted_block_cols[i]) {
      block_col_to_flat_index[i][block_col] = index;
      ++index;
    }
  }

  /* Initialize all blocks to zero. */
  std::vector<std::vector<MatrixBlock>> result(num_nodes);
  for (int block_row = 0; block_row < num_nodes; ++block_row) {
    for (int block_col : sorted_block_cols[block_row]) {
      result[block_row].emplace_back(block_row, block_col, Matrix3d::Zero());
    }
  }

  /* Add in contribution from each element. */
  const MatrixXd element_matrix = dummy_matrix12x12();
  for (int e = 0; e < num_elements; ++e) {
    const Vector4i& element = elements[e];
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        const int block_row = element(a);
        const int block_col = element(b);
        const int index = block_col_to_flat_index[block_row][block_col];
        get<2>(result[block_row][index]) +=
            element_matrix.block<3, 3>(3 * a, 3 * b);
      }
    }
  }
  return result;
}

/* Makes Eigen::SparseMatrix from 3x3 block matrix. */
Eigen::SparseMatrix<double> MakeEigenSparseMatrix(
    const std::vector<std::vector<MatrixBlock>>& matrix_blocks) {
  std::vector<Eigen::Triplet<double>> triplets;
  for (int i = 0; i < static_cast<int>(matrix_blocks.size()); ++i) {
    for (const auto& block : matrix_blocks[i]) {
      int block_row = get<0>(block);
      int block_col = get<1>(block);
      const Matrix3d& m = get<2>(block);
      DRAKE_DEMAND(i == block_row);
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          triplets.emplace_back(3 * block_row + r, 3 * block_col + c, m(r, c));
        }
      }
    }
  }
  Eigen::SparseMatrix<double> result(matrix_blocks.size() * 3,
                                     matrix_blocks.size() * 3);
  result.setFromTriplets(triplets.begin(), triplets.end());
  result.makeCompressed();
  return result;
}

/* Build connectivity from a sphere mesh. */
int BuildElements(vector<Vector4i>* elements) {
  SceneGraph<double> scene_graph;
  SourceId s_id = scene_graph.RegisterSource();
  const double kRezHint = FLAGS_resolution_hint;
  unique_ptr<GeometryInstance> geometry_instance = make_sphere_instance();
  GeometryId g_id = scene_graph.RegisterDeformableGeometry(
      s_id, scene_graph.world_frame_id(), std::move(geometry_instance),
      kRezHint);
  const SceneGraphInspector<double>& inspector = scene_graph.model_inspector();
  const VolumeMesh<double>* reference_mesh = inspector.GetReferenceMesh(g_id);
  DRAKE_DEMAND(reference_mesh != nullptr);
  for (const auto element : reference_mesh->tetrahedra()) {
    elements->emplace_back(element.vertex(0), element.vertex(1),
                           element.vertex(2), element.vertex(3));
  }
  std::cout << "#V = " << reference_mesh->num_vertices() << std::endl;
  std::cout << "#E = " << reference_mesh->num_elements() << std::endl;
  return reference_mesh->num_vertices();
}

void MakeIndices(int num_nodes, double percentage_in_contact,
                 vector<int>* A_indices, vector<int>* D_indices) {
  const int threshold = static_cast<int>(num_nodes * percentage_in_contact);
  for (int i = 0; i < threshold; ++i) {
    A_indices->emplace_back(i);
  }
  for (int i = threshold; i < num_nodes; ++i) {
    D_indices->emplace_back(i);
  }
}

void CalcBlockCholeskySchurComplement(
    const std::vector<Vector4i>& elements, const std::vector<int>& D_indices,
    const std::vector<int>& initial_ordering,
    const std::vector<std::set<int>>& adjacency_graph,
    bool respect_participation) {
  std::vector<int> ordering;

  if (respect_participation) {
    ordering = RestrictOrdering(initial_ordering, D_indices);
  } else {
    ordering = initial_ordering;
  }

  const int N = ordering.size();
  vector<int> old_to_new(N);
  for (int i = 0; i < N; ++i) {
    old_to_new[ordering[i]] = i;
  }

  vector<Vector4i> new_elements;
  for (const Vector4i& element : elements) {
    Vector4i permuted_element;
    for (int i = 0; i < 4; ++i) {
      permuted_element(i) = old_to_new[element(i)];
    }
    new_elements.emplace_back(permuted_element);
  }

  auto solver = std::make_unique<BlockSparseCholeskySolver>(
      CalcSparsityPattern(adjacency_graph, ordering));
  SymmetricBlockSparseMatrix<double>& M = solver->GetMutableMatrix();
  const Eigen::Matrix<double, 12, 12> element_matrix = dummy_matrix12x12();
  /* Add in element matrices */
  for (const Vector4i& permuted_element : new_elements) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        if (permuted_element(a) >= permuted_element(b)) {
          M.AddToBlock(permuted_element(a), permuted_element(b),
                       element_matrix.block<3, 3>(3 * a, 3 * b));
        }
      }
    }
  }
  solver->CalcSchurComplement(D_indices.size());
}

SchurComplement<double> CalcCholmodSchurComplement(
    const std::vector<std::vector<MatrixBlock>>& matrix_blocks,
    const std::vector<int>& D_indices, const std::vector<int>& A_indices) {
  Eigen::SparseMatrix<double> D(3 * D_indices.size(), 3 * D_indices.size());
  std::vector<Eigen::Triplet<double>> triplets;
  MatrixXd A = MatrixXd::Zero(3 * A_indices.size(), 3 * A_indices.size());
  MatrixXd B = MatrixXd::Zero(3 * D_indices.size(), 3 * A_indices.size());
  std::unordered_set<int> A_set;
  for (int i : A_indices) {
    A_set.insert(i);
  }

  std::vector<int> global_to_local(matrix_blocks.size());
  int A_index = 0;
  int D_index = 0;
  for (int i = 0; i < static_cast<int>(global_to_local.size()); ++i) {
    if (A_set.find(i) == A_set.end()) {
      global_to_local[i] = D_index++;
    } else {
      global_to_local[i] = A_index++;
    }
  }

  /* Here we want A - BD⁻¹Bᵀ. */
  /* Build the eigen sparse matrix D. */
  for (int br = 0; br < static_cast<int>(matrix_blocks.size()); ++br) {
    const auto& block_row = matrix_blocks[br];
    /* There shouldn't be any empty rows. */
    DRAKE_DEMAND(block_row.size() > 0);
    const int local_br = global_to_local[br];
    if (A_set.find(br) == A_set.end()) {
      /* The row belongs to D. */
      for (const auto& t : block_row) {
        const int bc = get<1>(t);
        const int local_bc = global_to_local[bc];
        const Matrix3d& m = get<2>(t);
        if (A_set.find(bc) == A_set.end()) {
          /* The col belongs to D too. */
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              triplets.emplace_back(3 * local_br + i, 3 * local_bc + j,
                                    m(i, j));
            }
          }
        } else {
          /* Otherwise, row belongs to A but col belongs to D, so the entry is
           in B. */
          B.block<3, 3>(3 * local_br, 3 * local_bc) = m;
        }
      }
    } else {
      /* The row belongs to A. */
      for (const auto& t : block_row) {
        const int bc = get<1>(t);
        if (A_set.find(bc) != A_set.end()) {
          /* The col belongs to A too. */
          const int local_bc = global_to_local[bc];
          const Matrix3d& m = get<2>(t);
          A.block<3, 3>(3 * local_br, 3 * local_bc) = m;
        }
        /* Otherwise, row belongs to D but col belongs to A, so the entry is in
         Bᵀ and we ignore it. */
      }
    }
  }
  D.setFromTriplets(triplets.begin(), triplets.end());
  D.makeCompressed();
  CholmodSparseMatrix cholmod_matrix(D);
  cholmod_matrix.Factor();
  return cholmod_matrix.CalcSchurComplement(B, A);
}

constexpr double kEps = 1e-10;
void BenchmarkPerformance() {
  vector<Vector4i> elements;
  const int num_nodes = BuildElements(&elements);
  const auto matrix_blocks = MakeMatrixBlocks(elements, num_nodes);
  std::unique_ptr<PetscSymmetricBlockSparseMatrix> petsc_matrix =
      MakePetscMatrix(elements, num_nodes);
  std::vector<int> A_indices;
  std::vector<int> D_indices;
  MakeIndices(num_nodes, FLAGS_percentage_in_contact, &A_indices, &D_indices);

  auto starting_time = Clock::now();
  for (int i = 0; i < FLAGS_petsc_solve_iterations; ++i) {
    petsc_matrix->CalcSchurComplement(D_indices, A_indices);
  }
  auto ending_time = Clock::now();
  if (FLAGS_petsc_solve_iterations > 0) {
    int petsc_run_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             ending_time - starting_time)
                             .count() /
                         FLAGS_petsc_solve_iterations;
    std::cout << "Petsc Schur complement takes on average " << petsc_run_time
              << " microseconds on average." << std::endl;
  }

  starting_time = Clock::now();
  for (int i = 0; i < FLAGS_cholmod_solve_iterations; ++i) {
    CalcCholmodSchurComplement(matrix_blocks, D_indices, A_indices);
  }
  ending_time = Clock::now();
  if (FLAGS_cholmod_solve_iterations > 0) {
    int cholmod_run_time =
        std::chrono::duration_cast<std::chrono::microseconds>(ending_time -
                                                              starting_time)
            .count() /
        FLAGS_cholmod_solve_iterations;
    std::cout << "Cholmod Schur complement takes on average "
              << cholmod_run_time << " microseconds on average." << std::endl;
  }

  const std::vector<std::set<int>> adj =
      BuildAdjacencyGraph(num_nodes, elements);
  const std::vector<int> ordering = CalcEliminationOrdering(adj);

  starting_time = Clock::now();
  for (int i = 0; i < FLAGS_block_cholesky_solve_iterations; ++i) {
    /* Test case, permutes the ordering that Cholmod thinks is best to get schur
     complement. */
    CalcBlockCholeskySchurComplement(elements, D_indices,
                                     ordering, adj, true);
  }
  ending_time = Clock::now();
  if (FLAGS_block_cholesky_solve_iterations > 0) {
    int block_cholesky_run_time =
        std::chrono::duration_cast<std::chrono::microseconds>(ending_time -
                                                              starting_time)
            .count() /
        FLAGS_block_cholesky_solve_iterations;
    std::cout << "Block Cholesky Schur complement with cholmod ordering that "
                 "respects partition takes on average "
              << block_cholesky_run_time << " microseconds on average."
              << std::endl;
  }

  starting_time = Clock::now();
  for (int i = 0; i < FLAGS_block_cholesky_solve_iterations; ++i) {
    /* Unlikely best case scenario, the cholmod ordering so happens to
     eliminate participating vertices first. */
    CalcBlockCholeskySchurComplement(elements, D_indices, ordering, adj, false);
  }
  ending_time = Clock::now();
  if (FLAGS_block_cholesky_solve_iterations > 0) {
    int block_cholesky_run_time =
        std::chrono::duration_cast<std::chrono::microseconds>(ending_time -
                                                              starting_time)
            .count() /
        FLAGS_block_cholesky_solve_iterations;
    std::cout << "Block Cholesky Schur complement with cholmod ordering that "
                 "DOES NOT respect partition takes on average "
              << block_cholesky_run_time << " microseconds on average."
              << std::endl;
  }

  starting_time = Clock::now();
  for (int i = 0; i < FLAGS_block_cholesky_solve_iterations; ++i) {
    /* Unlikely best case scenario, the cholmod ordering so happens to
     eliminate participating vertices first. */
    const std::vector<int> alternative_ordering =
        CalcEliminationOrdering(adj, D_indices);
    CalcBlockCholeskySchurComplement(elements, D_indices, alternative_ordering,
                                     adj, false);
  }
  ending_time = Clock::now();
  if (FLAGS_block_cholesky_solve_iterations > 0) {
    int block_cholesky_run_time =
        std::chrono::duration_cast<std::chrono::microseconds>(ending_time -
                                                              starting_time)
            .count() /
        FLAGS_block_cholesky_solve_iterations;
    std::cout << "Block Cholesky Schur complement using alternative ordering "
              << block_cholesky_run_time << " microseconds on average."
              << std::endl;
  }

#if 0
  const MatrixXd cholmod_S =
      CalcCholmodSchurComplement(matrix_blocks, D_indices, A_indices)
          .get_D_complement();
  const MatrixXd petsc_S =
      petsc_matrix->CalcSchurComplement(D_indices, A_indices)
          .get_D_complement();
  std::cout << "Schur complement error = " << (cholmod_S - petsc_S).norm()
            << std::endl;
#endif
}

int DoMain() {
  BenchmarkPerformance();
  return 0;
}
}  // namespace fem
}  // namespace multibody
}  // namespace drake

int main(int argc, char** argv) {
  gflags::SetUsageMessage("Matrix assembly comparison.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::fem::DoMain();
}
