#include "drake/multibody/fem/symmetric_block_sparse_matrix.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

constexpr double kEps = 4.0 * std::numeric_limits<double>::epsilon();
// clang-format off
const Matrix3d A00 = (Eigen::Matrix3d() << 1, 2, 3,
                                           2, 5, 6,
                                           3, 6, 9).finished();
const Matrix3d A11 = (Eigen::Matrix3d() << 11, 12, 13,
                                           12, 15, 16,
                                           13, 16, 19).finished();
const Matrix3d A12 = (Eigen::Matrix3d() << 11, 12, 13,
                                           14, 15, 16,
                                           17, 18, 19).finished();
const Matrix3d A22 = (Eigen::Matrix3d() << 21, 22, 23,
                                           22, 25, 26,
                                           23, 26, 29).finished();
// clang-format on

/* Makes a dense matrix
   A =   A00 |  0  |  0
        -----------------
          0  | A11 | A12
        -----------------
          0  | A21 | A22
where A21 = A12.transpose(). */
MatrixXd MakeDenseMatrix() {
  MatrixXd A(9, 9);
  A.block<3, 3>(0, 0) = A00;
  A.block<3, 3>(3, 3) = A11;
  A.block<3, 3>(3, 6) = A12;
  A.block<3, 3>(6, 3) = A12.transpose();
  A.block<3, 3>(6, 6) = A22;
  return A;
}

/* Makes a block sparse matrix
   A =   A00 |  0  |  0
        -----------------
          0  | A11 | A12
        -----------------
          0  | A21 | A22
where A21 = A12.transpose(). */
SymmetricBlockSparseMatrix<double> MakeBlockSparseMatrix() {
  vector<vector<int>> row_blocks;
  row_blocks.push_back({{0}});
  row_blocks.push_back({{1, 2}});
  row_blocks.push_back({{2}});
  SymmetricBlockSparseMatrix<double> A_blocks(std::move(row_blocks));
  A_blocks.AddToBlock(0, 0, A00);
  A_blocks.AddToBlock(1, 1, A11);
  A_blocks.AddToBlock(2, 1, A12.transpose());
  A_blocks.AddToBlock(2, 2, A22);
  return A_blocks;
}

VectorXd MakeVector() {
  return (Eigen::VectorXd(9) << 21, 22, 23, 22, 25, 26, 23, 26, 29).finished();
}

GTEST_TEST(SymmetricBlockSparseMatrixTest, Construction) {
  const MatrixXd A = MakeDenseMatrix();
  const SymmetricBlockSparseMatrix<double> A_blocks = MakeBlockSparseMatrix();
  EXPECT_TRUE(CompareMatrices(A_blocks.MakeDenseMatrix(), A));
  EXPECT_TRUE(CompareMatrices(MatrixXd(A_blocks.MakeEigenSparseMatrix()), A));
}

GTEST_TEST(SymmetricBlockSparseMatrixTest, Multiply) {
  const MatrixXd A = MakeDenseMatrix();
  const SymmetricBlockSparseMatrix<double> A_blocks = MakeBlockSparseMatrix();
  const VectorXd x = MakeVector();
  const VectorXd b_expected = A * x;
  VectorXd b(9);
  A_blocks.Multiply(x, &b);
  EXPECT_TRUE(CompareMatrices(b, b_expected, kEps));
}

GTEST_TEST(SymmetricBlockSparseMatrixTest, SetZero) {
  SymmetricBlockSparseMatrix<double> A_blocks = MakeBlockSparseMatrix();
  A_blocks.SetZero();
  EXPECT_TRUE(
      CompareMatrices(A_blocks.MakeDenseMatrix(), MatrixXd::Zero(9, 9)));
}

GTEST_TEST(SymmetricBlockSparseMatrixTest, CalcAdjacencyGrpah) {
  const SymmetricBlockSparseMatrix<double> A_blocks = MakeBlockSparseMatrix();
  const std::vector<std::set<int>> adj = A_blocks.CalcAdjacencyGraph();
  ASSERT_EQ(adj.size(), 3);

  EXPECT_EQ(adj[0].size(), 1);
  EXPECT_EQ(adj[0].count(0), 1);

  EXPECT_EQ(adj[1].size(), 2);
  EXPECT_EQ(adj[1].count(1), 1);
  EXPECT_EQ(adj[1].count(2), 1);

  EXPECT_EQ(adj[2].size(), 1);
  EXPECT_EQ(adj[2].count(2), 1);
}

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
