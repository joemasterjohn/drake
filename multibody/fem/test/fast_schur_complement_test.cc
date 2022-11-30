#include "drake/multibody/fem/fast_schur_complement.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/multibody/fem/petsc_symmetric_block_sparse_matrix.h"
#include "drake/multibody/fem/symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::unique_ptr;
using std::vector;

constexpr double kEps = 2e-13;
// Size of the matrix in this test.
constexpr int kDofs = 9;

// clang-format off
const Matrix3d A00 =
    (Eigen::Matrix3d() << 11, 2, 3,
                          2, 50, 6,
                          3, 6, 190).finished();
const Matrix3d A11 =
    (Eigen::Matrix3d() << 110, 12, 13,
                          12, 150, 16,
                          13, 16, 190).finished();
const Matrix3d A02 =
    (Eigen::Matrix3d() << 1.2, 12, 13,
                          14, 21, 16,
                          17, 18, 29).finished();
const Matrix3d A22 =
    (Eigen::Matrix3d() << 210, 22, 23,
                          22, 205, 26,
                          23, 26, 290).finished();
// clang-format on

/* Makes a PETSc block sparse matrix
   A =   A00 |  0  | A02
        -----------------
          0  | A11 |  0
        -----------------
         A20 |  0  | A22
where A20 = A02.transpose(). */
unique_ptr<PetscSymmetricBlockSparseMatrix> MakePetscMatrix() {
  /* Number of nonzero blocks (on upper triangular portion + diagonal) per block
   row. */
  const vector<int> num_upper_triangular_blocks_per_row = {2, 1, 1};
  auto A = std::make_unique<PetscSymmetricBlockSparseMatrix>(
      9, 3, num_upper_triangular_blocks_per_row);
  VectorX<int> block_indices;
  MatrixXd block;

  block_indices.resize(1);
  block_indices(0) = 1;
  block = A11;
  A->AddToBlock(block_indices, block);

  block_indices.resize(2);
  block_indices(0) = 0;
  block_indices(1) = 2;
  block.resize(6, 6);
  block.topLeftCorner<3, 3>() = A00;
  block.topRightCorner<3, 3>() = A02;
  block.bottomLeftCorner<3, 3>() = A02.transpose();
  block.bottomRightCorner<3, 3>() = A22;
  A->AddToBlock(block_indices, block);
  return A;
}

unique_ptr<SymmetricBlockSparseMatrix<double>> MakeBlockSparseMatrix() {
  std::vector<std::vector<int>> sparsity_pattern;
  sparsity_pattern.emplace_back(vector<int>{0, 2});
  sparsity_pattern.emplace_back(vector<int>{1});
  sparsity_pattern.emplace_back(vector<int>{2});
  auto A = std::make_unique<SymmetricBlockSparseMatrix<double>>(
      std::move(sparsity_pattern));
  A->SetBlock(0, 0, A00);
  A->SetBlock(2, 0, A02.transpose());
  A->SetBlock(1, 1, A11);
  A->SetBlock(2, 2, A22);
  return A;
}

GTEST_TEST(FastSchurComplementTest, SchurComplementMatrix) {
  unique_ptr<PetscSymmetricBlockSparseMatrix> petsc_matrix = MakePetscMatrix();
  unique_ptr<SymmetricBlockSparseMatrix<double>> block_sparse_matrix =
      MakeBlockSparseMatrix();
  const vector<int> D_block_indexes = {1};
  const vector<int> A_block_indexes = {0, 2};

  const SchurComplement<double> schur_complement =
      petsc_matrix->CalcSchurComplement(D_block_indexes, A_block_indexes);
  const MatrixXd petsc_schur_complement_matrix =
      schur_complement.get_D_complement();
  const FastSchurComplement<double> fast_schur_complement(
      *block_sparse_matrix, D_block_indexes, A_block_indexes);
  const MatrixXd fast_schur_complement_matrix =
      fast_schur_complement.get_D_complement();
  EXPECT_TRUE(CompareMatrices(petsc_schur_complement_matrix,
                              fast_schur_complement_matrix, kEps));

  /* Set arbitrary solution for x in the system
    Ax  + By = a
    Bᵀx + Dy = 0
   Verify that y is solved to be -D⁻¹Bᵀx.

   For FastSchurComplement the roles of x and y are reversed and the system is
    Dx  + By = a
    Bᵀx + Ay = 0 . */
  VectorXd x(6);
  x << 1, 2, 3, 4, 5, 6;
  const VectorXd expected = schur_complement.SolveForY(x);
  const VectorXd calculated = fast_schur_complement.SolveForX(x);
  EXPECT_TRUE(CompareMatrices(calculated, expected, kEps));
}

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
