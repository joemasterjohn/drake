#include <tuple>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/unused.h"
#include "drake/multibody/contact_solvers/block_sparse_matrix.h"
#include "drake/multibody/contact_solvers/block_sparse_supernodal_solver.h"
#include "drake/multibody/contact_solvers/sap/dense_supernodal_solver.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::get;
using std::vector;

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

// Type used to represent a block diagonal matrix in two formats: 1) a dense
// matrix and 2) a std::vector of block diagonal entries.
typedef std::pair<MatrixXd, std::vector<MatrixXd>> DenseBlockDiagonalPair;

// Makes a block diagonal SPD matrix with three diagonal SPD blocks of size
// 2x2.
DenseBlockDiagonalPair Make6x6SpdBlockDiagonalMatrixOf2x2SpdMatrices() {
  MatrixXd A(6, 6);
  // clang-format off
  A << 1, 1, 0, 0, 0, 0,
       1, 5, 0, 0, 0, 0,
       0, 0, 4, 1, 0, 0,
       0, 0, 1, 4, 0, 0,
       0, 0, 0, 0, 4, 2,
       0, 0, 0, 0, 2, 5;
  // clang-format on
  std::vector<MatrixXd> blocks(3);
  blocks.at(0) = A.block<2, 2>(0, 0);
  blocks.at(1) = A.block<2, 2>(2, 2);
  blocks.at(2) = A.block<2, 2>(4, 4);
  return std::make_pair(A, blocks);
}

// Makes a block diagonal SPD matrix with three diagonal SPD blocks of size
// 3x3.
DenseBlockDiagonalPair Make9x9SpdBlockDiagonalMatrixOf3x3SpdMatrices() {
  MatrixXd A(9, 9);
  // clang-format off
  A << 1, 2, 2, 0, 0, 0, 0, 0, 0,
       2, 5, 3, 0, 0, 0, 0, 0, 0,
       2, 3, 8, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 4, 1, 1, 0, 0, 0,
       0, 0, 0, 1, 4, 2, 0, 0, 0,
       0, 0, 0, 1, 2, 5, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 4, 1, 1,
       0, 0, 0, 0, 0, 0, 1, 4, 2,
       0, 0, 0, 0, 0, 0, 1, 2, 5;
  // clang-format on
  std::vector<MatrixXd> blocks(3);
  blocks.at(0) = A.block<3, 3>(0, 0);
  blocks.at(1) = A.block<3, 3>(3, 3);
  blocks.at(2) = A.block<3, 3>(6, 6);
  return std::make_pair(A, blocks);
}

// Makes a block diagonal SPD matrix with four diagonal SPD blocks of size
// 3x3.
DenseBlockDiagonalPair Make12x12SpdBlockDiagonalMatrixOf3x3SpdMatrices() {
  MatrixXd A(12, 12);
  // clang-format off
  A << 5, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 9, 2, 2, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 2, 5, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 2, 3, 8, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 4, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 4, 2, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 2, 5, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 7;
  // clang-format on
  std::vector<MatrixXd> blocks(4);
  blocks.at(0) = A.block<3, 3>(0, 0);
  blocks.at(1) = A.block<3, 3>(3, 3);
  blocks.at(2) = A.block<3, 3>(6, 6);
  blocks.at(3) = A.block<3, 3>(9, 9);
  return std::make_pair(A, blocks);
}

// Helper to make a BlockSparseMatrix for an input dense matrix A.
// block_positions[b] corresponds to the (bi, bj) block indexes of the non-zero
// b-th block.
// dense_positions[b] corresponds to the (i, j) indexes of the top-left corner
// of the b-th block in the dense matrix A.
// block_sizes[b] corresponds to the number of rows (first) and columns (second)
// for the b-th block.
BlockSparseMatrix<double> MakeBlockSparseMatrix(
    const MatrixXd& A, const std::vector<std::pair<int, int>>& block_positions,
    const std::vector<std::pair<int, int>>& dense_positions,
    const std::vector<std::pair<int, int>>& block_sizes) {
  const int num_blocks = block_positions.size();
  int block_rows = 0;
  int block_cols = 0;
  for (int b = 0; b < num_blocks; ++b) {
    block_rows = std::max(block_rows, block_positions[b].first);
    block_cols = std::max(block_cols, block_positions[b].second);
  }
  block_rows += 1;
  block_cols += 1;

  BlockSparseMatrixBuilder<double> builder(block_rows, block_cols, num_blocks);
  for (int b = 0; b < num_blocks; ++b) {
    builder.PushBlock(block_positions[b].first, block_positions[b].second,
                      MatrixBlock<double>(A.block(
                          dense_positions[b].first, dense_positions[b].second,
                          block_sizes[b].first, block_sizes[b].second)));
  }
  return builder.Build();
}

template <class SolverType>
SolverType MakeSolver(const std::vector<MatrixXd>& A,
                      const BlockSparseMatrix<double>& J) {
  if constexpr (std::is_same_v<SolverType, BlockSparseSuperNodalSolver>) {
    return SolverType(A, J);
  } else if constexpr (std::is_same_v<SolverType, DenseSuperNodalSolver>) {
    return SolverType(&A, &J);
  } else {
    DRAKE_UNREACHABLE();
  }
}

template <typename ConcreteSolver>
class SuperNodalSolverTest : public ::testing::Test {};
using Implementations =
    ::testing::Types<BlockSparseSuperNodalSolver, DenseSuperNodalSolver>;
TYPED_TEST_SUITE(SuperNodalSolverTest, Implementations);

// In this test the partition of the columns of J doesn't refine the partition
// induced by M, nor the other way around. We partition the columns of J as {{0,
// 1, 2, 3}, {4}, {5}}. However, we partition M as {{0, 1}, {2, 3}, {4, 5}}.
TYPED_TEST(SuperNodalSolverTest, IncompatibleJacobianAndMass) {
  const auto [M, blocks_of_M] = Make6x6SpdBlockDiagonalMatrixOf2x2SpdMatrices();
  unused(M);

  // Build Jacobian matrix J.
  MatrixXd J(9, 6);
  // clang-format off
  J << 1, 2, 1, 2, 0, 0,
       0, 1, 0, 1, 0, 0,
       1, 3, 1, 3, 0, 0,
       0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 2, 0,
       0, 0, 0, 0, 2, 0,
       0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 3;
  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      {{0, 0}, {1, 1}, {2, 2}},
      {{0, 0}, {3, 4}, {6, 5}},
      {{3, 4}, {3, 1}, {3, 1}});
  // clang-format on

  if constexpr (std::is_same_v<TypeParam, BlockSparseSuperNodalSolver>) {
    DRAKE_EXPECT_THROWS_MESSAGE(MakeSolver<TypeParam>(blocks_of_M, Jblock),
                                ".*Mass.*Jacobians.*incompatible.*");
  } else if constexpr (std::is_same_v<TypeParam, DenseSuperNodalSolver>) {
    // Partition information is lost when we use dense algebra, so for as long
    // as sizes are consistent, the dense implementation is okay.
    EXPECT_NO_THROW(MakeSolver<TypeParam>(blocks_of_M, Jblock));
  } else {
    DRAKE_UNREACHABLE();
  }
}

// Basic test of SuperNodalSolver's public APIs.
// In this case the columns's partition of J exactly matches the partition
// induced by M.
TYPED_TEST(SuperNodalSolverTest, InterfaceTest) {
  const auto [M, blocks_of_M] = Make6x6SpdBlockDiagonalMatrixOf2x2SpdMatrices();

  MatrixXd J(9, 6);
  // clang-format off
  J << 0, 0, 0, 0, 1, 2,
       0, 0, 0, 0, 2, 1,
       0, 0, 0, 0, 2, 3,
       1, 2, 0, 0, 2, 4,
       0, 1, 0, 0, 1, 3,
       1, 3, 0, 0, 2, 4,
       0, 0, 1, 1, 0, 0,
       0, 0, 2, 1, 0, 0,
       0, 0, 3, 3, 0, 0;
  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      {{0, 2}, {1, 0}, {1, 2}, {2, 1}},
      {{0, 4}, {3, 0}, {3, 4}, {6, 2}},
      {{3, 2}, {3, 2}, {3, 2}, {3, 2}});
  // clang-format on

  auto [G, blocks_of_G] = Make9x9SpdBlockDiagonalMatrixOf3x3SpdMatrices();

  TypeParam solver = MakeSolver<TypeParam>(blocks_of_M, Jblock);
  solver.SetWeightMatrix(blocks_of_G);
  const MatrixXd full_matrix_ref = M + J.transpose() * G * J;
  EXPECT_NEAR((solver.MakeFullMatrix() - full_matrix_ref).norm(), 0, 1e-15);

  // Make the block sizes of G incompatible with J and verify an exception is
  // thrown.
  blocks_of_G.at(0) = MatrixXd::Ones(4, 4);
  DRAKE_EXPECT_THROWS_MESSAGE(solver.SetWeightMatrix(blocks_of_G),
                              "Weight matrix incompatible with Jacobian.");
}

// SuperNodalSolver assumes at most two blocks per row. We verify the solver
// throws an exception if more than two blocks per row are supplied.
TYPED_TEST(SuperNodalSolverTest, MoreThanTwoBlocksPerRowInTheJacobian) {
  const auto [M, blocks_of_M] = Make6x6SpdBlockDiagonalMatrixOf2x2SpdMatrices();
  unused(M);

  // Build Jacobian matrix J.
  MatrixXd J(9, 6);
  // clang-format off
  J << 1, 2, 1, 3, 2, 4,
       0, 1, 1, 3, 1, 3,
       1, 3, 1, 2, 2, 4,
       0, 0, 0, 0, 1, 2,
       0, 0, 0, 0, 2, 1,
       0, 0, 0, 0, 2, 3,
       0, 0, 1, 1, 0, 0,
       0, 0, 2, 1, 0, 0,
       0, 0, 3, 3, 0, 0;
  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      {{0, 0}, {0, 1}, {0, 2}, {1, 2}, {2, 1}},
      {{0, 0}, {0, 2}, {0, 4}, {3, 4}, {6, 2}},
      {{3, 2}, {3, 2}, {3, 2}, {3, 2}, {3, 2}});
  // clang-format on

  if constexpr (std::is_same_v<TypeParam, BlockSparseSuperNodalSolver>) {
    DRAKE_EXPECT_THROWS_MESSAGE(
        MakeSolver<TypeParam>(blocks_of_M, Jblock),
        "Jacobian can only be nonzero on at most two column blocks.");
  } else if constexpr (std::is_same_v<TypeParam, DenseSuperNodalSolver>) {
    // Partition information is lost when we use dense algebra, so for as long
    // as sizes are consistent, the dense implementation is okay.
    EXPECT_NO_THROW(MakeSolver<TypeParam>(blocks_of_M, Jblock));
  } else {
    DRAKE_UNREACHABLE();
  }
}

// In this test the partition of the columns of J refines the partition induced
// by M. We partition the columns of J as {{0, 1}, {2, 3}, {4}, {5}}. However,
// we partition M as {{0, 1}, {2, 3}, {4, 5}}.
TYPED_TEST(SuperNodalSolverTest,
           ColumnPartitionOfJacobianRefinesMassMatrixPartition) {
  const auto [M, blocks_of_M] = Make6x6SpdBlockDiagonalMatrixOf2x2SpdMatrices();
  unused(M);

  // Build Jacobian matrix J.
  MatrixXd J(9, 6);
  // clang-format off
  J << 1, 2, 0, 0, 0, 4,
       0, 1, 0, 0, 0, 3,
       1, 3, 0, 0, 0, 4,
       0, 0, 0, 0, 1, 2,
       0, 0, 0, 0, 2, 1,
       0, 0, 0, 0, 2, 3,
       0, 0, 1, 1, 0, 0,
       0, 0, 2, 1, 0, 0,
       0, 0, 3, 3, 0, 0;
  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      {{0, 0}, {0, 3}, {1, 2}, {1, 3}, {2, 1}},
      {{0, 0}, {0, 5}, {3, 4}, {3, 5}, {6, 2}},
      {{3, 2}, {3, 1}, {3, 1}, {3, 1}, {3, 2}});
  // clang-format on

  if constexpr (std::is_same_v<TypeParam, BlockSparseSuperNodalSolver>) {
    DRAKE_EXPECT_THROWS_MESSAGE(MakeSolver<TypeParam>(blocks_of_M, Jblock),
                                ".*Mass.*Jacobians.*incompatible.*");
  } else if constexpr (std::is_same_v<TypeParam, DenseSuperNodalSolver>) {
    // Partition information is lost when we use dense algebra, so for as long
    // as sizes are consistent, the dense implementation is okay.
    EXPECT_NO_THROW(MakeSolver<TypeParam>(blocks_of_M, Jblock));
  } else {
    DRAKE_UNREACHABLE();
  }
}

// In this test the partition induced by M refines the partition of the columns
// of J.
// We partition the columns of J as {{0, 1}, {2, 3}, {4, 5}}.
// However, we partition M as {{0, 1}, {2, 3}, {4}, {5}}.
TYPED_TEST(SuperNodalSolverTest,
           PartitionOfMassMatrixRefinesJacobianColumnsPartition) {
  // Build mass matrix M.
  MatrixXd M(6, 6);
  // clang-format off
  M << 1, 1, 0, 0, 0, 0,
       1, 5, 0, 0, 0, 0,
       0, 0, 4, 1, 0, 0,
       0, 0, 1, 4, 0, 0,
       0, 0, 0, 0, 4, 0,
       0, 0, 0, 0, 0, 5;
  // clang-format on
  std::vector<MatrixXd> blocks_of_M(4);
  blocks_of_M.at(0) = M.block<2, 2>(0, 0);
  blocks_of_M.at(1) = M.block<2, 2>(2, 2);
  blocks_of_M.at(2) = M.block<1, 1>(4, 4);
  blocks_of_M.at(3) = M.block<1, 1>(5, 5);

  // Build Jacobian matrix J.
  MatrixXd J(9, 6);

  // clang-format off
  J << 1, 2, 0, 0, 2, 4,
       0, 1, 0, 0, 1, 3,
       1, 3, 0, 0, 2, 4,
       0, 0, 0, 0, 1, 2,
       0, 0, 0, 0, 2, 1,
       0, 0, 0, 0, 2, 3,
       0, 0, 1, 1, 0, 0,
       0, 0, 2, 1, 0, 0,
       0, 0, 3, 3, 0, 0;
  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      {{0, 0}, {0, 2}, {1, 2}, {2, 1}},
      {{0, 0}, {0, 4}, {3, 4}, {6, 2}},
      {{3, 2}, {3, 2}, {3, 2}, {3, 2}});
  // clang-format on

  const auto [G, blocks_of_G] = Make9x9SpdBlockDiagonalMatrixOf3x3SpdMatrices();

  if constexpr (std::is_same_v<TypeParam, BlockSparseSuperNodalSolver>) {
    DRAKE_EXPECT_THROWS_MESSAGE(MakeSolver<TypeParam>(blocks_of_M, Jblock),
                                ".*Mass.*Jacobians.*incompatible.*");
  } else if constexpr (std::is_same_v<TypeParam, DenseSuperNodalSolver>) {
    // Partition information is lost when we use dense algebra, so for as long
    // as sizes are consistent, the dense implementation is okay.
    EXPECT_NO_THROW(MakeSolver<TypeParam>(blocks_of_M, Jblock));
  } else {
    DRAKE_UNREACHABLE();
  }
}

// Test the condition when J blocks might have different number of rows. Of
// course blocks in the same block-row must have the same number of rows, though
// they don't if in different block-rows. In this test:
//   - J00 and J02 are 6x2 matrices (both must have the same number of rows).
//   - J12 is a 3x2 matrix.
//   - J21 is a 3x2 matrix. NOTE: In this test J00 and J02 happen to have the
//     same number of columns, but it is not a requirement and in general will
//     not be true.
TYPED_TEST(SuperNodalSolverTest, SeveralPointsPerPatch) {
  MatrixXd J(12, 6);

  // Here we repeat the first three rows to emulate a duplicated contact point,
  // something a contact solver should recover from gracefully.
  // clang-format off
  J << 1, 2, 0, 0, 2, 4,
       0, 1, 0, 0, 1, 3,
       1, 3, 0, 0, 2, 4,
       1, 2, 0, 0, 2, 4,
       0, 1, 0, 0, 1, 3,
       1, 3, 0, 0, 2, 4,
       0, 0, 0, 0, 1, 2,
       0, 0, 0, 0, 2, 1,
       0, 0, 0, 0, 2, 3,
       0, 0, 1, 1, 0, 0,
       0, 0, 2, 1, 0, 0,
       0, 0, 3, 3, 0, 0;
  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      {{0, 0}, {0, 2}, {1, 2}, {2, 1}},
      {{0, 0}, {0, 4}, {6, 4}, {9, 2}},
      {{6, 2}, {6, 2}, {3, 2}, {3, 2}});
  // clang-format on

  const auto [G, blocks_of_G] =
      Make12x12SpdBlockDiagonalMatrixOf3x3SpdMatrices();
  const auto [M, blocks_of_M] = Make6x6SpdBlockDiagonalMatrixOf2x2SpdMatrices();

  TypeParam solver = MakeSolver<TypeParam>(blocks_of_M, Jblock);
  solver.SetWeightMatrix(blocks_of_G);
  MatrixXd full_matrix_ref = M + J.transpose() * G * J;
  EXPECT_NEAR((solver.MakeFullMatrix() - full_matrix_ref).norm(), 0, 1e-15);
}

// In this test we provided Jacobian triplets in arbitrary order. This verifies
// there are no implicit sorting assumptions on the input.
TYPED_TEST(SuperNodalSolverTest, JacobianTripletsNotSortedByColumn) {
  MatrixXd J(12, 6);
  // clang-format off
  J << 1, 2, 0, 0, 2, 4,
       0, 1, 0, 0, 1, 3,
       1, 3, 0, 0, 2, 4,
       1, 2, 0, 0, 2, 4,
       0, 1, 0, 0, 1, 3,
       1, 3, 0, 0, 2, 4,
       1, 1, 0, 0, 1, 1,
       1, 1, 0, 0, 1, 2,
       1, 1, 0, 0, 1, 1,
       0, 0, 1, 1, 0, 0,
       0, 0, 2, 1, 0, 0,
       0, 0, 3, 3, 0, 0;
  // We place unsorted inputs: column 2 appears before column 0.
  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      {{0, 0}, {0, 2}, {1, 0}, {1, 2}, {2, 1}},
      {{0, 0}, {0, 4}, {6, 0}, {6, 4}, {9, 2}},
      {{6, 2}, {6, 2}, {3, 2}, {3, 2}, {3, 2}});
  // clang-format on

  const auto [M, blocks_of_M] = Make6x6SpdBlockDiagonalMatrixOf2x2SpdMatrices();
  const auto [G, blocks_of_G] =
      Make12x12SpdBlockDiagonalMatrixOf3x3SpdMatrices();

  TypeParam solver = MakeSolver<TypeParam>(blocks_of_M, Jblock);
  solver.SetWeightMatrix(blocks_of_G);
  MatrixXd full_matrix_ref = M + J.transpose() * G * J;
  EXPECT_NEAR((solver.MakeFullMatrix() - full_matrix_ref).norm(), 0, 1e-15);
}

// In this example there are three contact patches of one contact point each.
// There are three trees. The first two have two dofs and the third one has
// three dofs. The purpose of this test is to verify correctness when trees
// might have different number of dofs.
TYPED_TEST(SuperNodalSolverTest, DifferentTreeSizes) {
  // In this example, the number of patches happens to equal the number of
  // contact points since each patch has only a single point.
  MatrixXd J(9, 7);

  // clang-format off
  J << 1, 2, 0, 0, 2, 4, 4,
       0, 1, 0, 0, 1, 3, 3,
       1, 3, 0, 0, 2, 4, 4,
       0, 0, 0, 0, 1, 2, 2,
       0, 0, 0, 0, 2, 1, 1,
       0, 0, 0, 0, 2, 3, 3,
       0, 0, 1, 1, 0, 0, 0,
       0, 0, 2, 1, 0, 0, 0,
       0, 0, 3, 3, 0, 0, 0;
  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      {{0, 0}, {0, 2}, {1, 2}, {2, 1}},
      {{0, 0}, {0, 4}, {3, 4}, {6, 2}},
      {{3, 2}, {3, 3}, {3, 3}, {3, 2}});
  // clang-format on

  const auto [G, blocks_of_G] = Make9x9SpdBlockDiagonalMatrixOf3x3SpdMatrices();

  MatrixXd M(7, 7);
  // clang-format off
  M << 1, 1, 0, 0, 0, 0, 0,
       1, 5, 0, 0, 0, 0, 0,
       0, 0, 4, 1, 0, 0, 0,
       0, 0, 1, 4, 0, 0, 0,
       0, 0, 0, 0, 4, 2, 0,
       0, 0, 0, 0, 2, 5, 0,
       0, 0, 0, 0, 0, 0, 1;
  // clang-format on

  std::vector<MatrixXd> blocks_of_M(3);
  blocks_of_M.at(0) = M.block(0, 0, 2, 2);
  blocks_of_M.at(1) = M.block(2, 2, 2, 2);
  blocks_of_M.at(2) = M.block(4, 4, 3, 3);

  TypeParam solver = MakeSolver<TypeParam>(blocks_of_M, Jblock);
  solver.SetWeightMatrix(blocks_of_G);
  MatrixXd full_matrix_ref = M + J.transpose() * G * J;
  EXPECT_NEAR((solver.MakeFullMatrix() - full_matrix_ref).norm(), 0, 1e-15);
}

// Unit test for the sparsity pattern occurring on a problem with four stacks of
// two objects each. Patches and trees are provided in some arbitrary
// permutation.
TYPED_TEST(SuperNodalSolverTest, FourStacks) {
  // For this problem each patch has a single contact point. Therefore there'll
  // be num_patches blocks of W.
  const int num_patches = 8;
  const int num_trees = 8;
  // A typical block of J.
  MatrixXd J3x6(3, 6);
  // clang-format off
  J3x6 << 1, 2, 3, 4, 5, 6,
          1, 2, 3, 4, 5, 6,
          1, 2, 3, 4, 5, 6;
  // clang-format on
  const MatrixXd Z3x6 = MatrixXd::Zero(3, 6);
  // There are 8 patches of size 3x6 located at:
  //     (p,t) = (0,6)
  //     (p,t) = (0,7)
  //     (p,t) = (1,4)
  //     (p,t) = (1,5)
  //     (p,t) = (2,6)
  //     (p,t) = (3,0)
  //     (p,t) = (4,4)
  //     (p,t) = (5,2)
  //     (p,t) = (6,0)
  //     (p,t) = (6,1)
  //     (p,t) = (7,2)
  //     (p,t) = (7,3)
  //
  MatrixXd J(24, 48);
  // clang-format off
  J << Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, J3x6, J3x6,
       Z3x6, Z3x6, Z3x6, Z3x6, J3x6, J3x6, Z3x6, Z3x6,
       Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, J3x6, Z3x6,
       J3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6,
       Z3x6, Z3x6, Z3x6, Z3x6, J3x6, Z3x6, Z3x6, Z3x6,
       Z3x6, Z3x6, J3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6,
       J3x6, J3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6, Z3x6,
       Z3x6, Z3x6, J3x6, J3x6, Z3x6, Z3x6, Z3x6, Z3x6;

  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      /* Block positions */
      {{0, 6}, {0, 7}, {1, 4}, {1, 5}, {2, 6}, {3, 0}, {4, 4}, {5, 2}, {6, 0},
       {6, 1}, {7, 2}, {7, 3}},
      /* Dense positions */
      {{0, 36}, {0, 42}, {3, 24}, {3, 30}, {6, 36}, {9, 0}, {12, 24}, {15, 12},
       {18, 0}, {18, 6}, {21, 12}, {21, 18}},
      /* Block sizes */
      {{3, 6}, {3, 6}, {3, 6}, {3, 6}, {3, 6}, {3, 6}, {3, 6}, {3, 6}, {3, 6},
       {3, 6}, {3, 6}, {3, 6}});
  // clang-format on

  MatrixXd Gb(3, 3);
  // clang-format off
  Gb << 1, 2, 2,
        2, 5, 3,
        2, 3, 4;
  // clang-format on
  MatrixXd G = MatrixXd::Zero(3 * num_patches, 3 * num_patches);
  std::vector<MatrixXd> blocks_of_G(num_patches);
  for (int i = 0; i < num_patches; ++i) {
    G.block(3 * i, 3 * i, 3, 3) = Gb;
    blocks_of_G.at(i) = Gb;
  }
  const MatrixXd Mt = VectorXd::LinSpaced(6, 1.0, 6.0).asDiagonal();
  MatrixXd M = MatrixXd::Zero(6 * num_trees, 6 * num_trees);
  std::vector<MatrixXd> blocks_of_M(num_trees);
  for (int i = 0; i < num_trees; ++i) {
    M.block(6 * i, 6 * i, 6, 6) = Mt;
    blocks_of_M.at(i) = Mt;
  }
  TypeParam solver = MakeSolver<TypeParam>(blocks_of_M, Jblock);
  solver.SetWeightMatrix(blocks_of_G);
  MatrixXd full_matrix_ref = M + J.transpose() * G * J;
  EXPECT_NEAR((solver.MakeFullMatrix() - full_matrix_ref).norm(), 0, 1e-12);

  // Construct arbitrary reference solution.
  VectorXd x_ref;
  x_ref.setLinSpaced(M.rows(), -1, 1);
  solver.Factor();
  EXPECT_NEAR((solver.Solve(full_matrix_ref * x_ref) - x_ref).norm(), 0, 1e-12);

  VectorXd b = full_matrix_ref * x_ref;
  solver.SolveInPlace(&b);
  EXPECT_NEAR((b - x_ref).norm(), 0, 1e-8);
}

// Provide input with varying column sizes.  Verifies there
// are no implicit assumptions about a constant Jacobian
// block-size.
TYPED_TEST(SuperNodalSolverTest, ColumnSizesDifferent) {
  MatrixXd J(13, 6);
  // Here we repeat the first three rows to emulate a duplicated contact point,
  // something a contact solver should recover from gracefully.
  // clang-format off
  J << 1, 2, 0,  2, 4, 0,
       0, 1, 0,  1, 3, 0,
       1, 3, 0,  2, 4, 0,
       1, 2, 0,  2, 4, 0,
       0, 1, 0,  1, 3, 0,
       1, 3, 0,  2, 4, 0,

       1, 1, 0,  2, 2, 0,
       1, 1, 0,  0, 2, 0,
       1, 1, 0,  1, 2, 0,

       0, 0, 1,  0, 0, 0,
       0, 0, 2,  0, 0, 0,
       0, 0, 3,  0, 0, 0,

       1, 1, 0,  0, 0, 1;
  const BlockSparseMatrix<double> Jblock = MakeBlockSparseMatrix(J,
      {{0, 0}, {0, 2}, {1, 0}, {1, 2}, {2, 1}, {3, 3}, {3, 0}},
      {{0, 0}, {0, 3}, {6, 0}, {6, 3}, {9, 2}, {12, 5}, {12, 0}},
      {{6, 2}, {6, 2}, {3, 2}, {3, 2}, {3, 1}, {1, 1}, {1, 2}});
  // clang-format on

  MatrixXd G(13, 13);
  // clang-format off
  G << 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 9, 2, 2, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 2, 5, 3, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 2, 3, 8, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 4, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 4, 2, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 2, 5, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 2, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 7, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7;
  // clang-format on

  std::vector<MatrixXd> blocks_of_G(6);
  blocks_of_G.at(0) = G.block(0, 0, 2, 2);
  blocks_of_G.at(1) = G.block(2, 2, 1, 1);
  blocks_of_G.at(2) = G.block(3, 3, 3, 3);
  blocks_of_G.at(3) = G.block(6, 6, 3, 3);
  blocks_of_G.at(4) = G.block(9, 9, 3, 3);
  blocks_of_G.at(5) = G.block(12, 12, 1, 1);

  MatrixXd M(6, 6);
  // clang-format off
  M << 1, 1, 0, 0, 0, 0,
       1, 5, 0, 0, 0, 0,
       0, 0, 4, 0, 0, 0,
       0, 0, 0, 4, 2, 0,
       0, 0, 0, 2, 5, 0,
       0, 0, 0, 0, 0, 4;
  // clang-format on

  std::vector<MatrixXd> blocks_of_M(4);
  blocks_of_M.at(0) = M.block<2, 2>(0, 0);
  blocks_of_M.at(1) = M.block<1, 1>(2, 2);
  blocks_of_M.at(2) = M.block<2, 2>(3, 3);
  blocks_of_M.at(3) = M.block<1, 1>(5, 5);

  TypeParam solver = MakeSolver<TypeParam>(blocks_of_M, Jblock);
  solver.SetWeightMatrix(blocks_of_G);
  MatrixXd full_matrix_ref = M + J.transpose() * G * J;
  EXPECT_NEAR((solver.MakeFullMatrix() - full_matrix_ref).norm(), 0, 1e-15);
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
