#include "drake/multibody/fem/block_sparse_cholesky_solver.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/multibody/fem/schur_complement.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace {

using Eigen::MatrixXd;
using Eigen::Vector4i;
using Eigen::VectorXd;
using std::make_unique;
using std::unique_ptr;
using std::vector;

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

/* Makes a SymmetricBlockSparseMatrix version of the matrix above. */
SymmetricBlockSparseMatrix<double> MakeSparseMatrix() {
  vector<vector<int>> sparsity;
  sparsity.emplace_back(vector<int>{0, 1, 2, 3});
  sparsity.emplace_back(vector<int>{1, 2, 3});
  sparsity.emplace_back(vector<int>{2, 3});
  sparsity.emplace_back(vector<int>{3});

  SymmetricBlockSparseMatrix<double> A(sparsity);
  const Eigen::Matrix<double, 12, 12> dense_A = dummy_matrix12x12();
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b <= a; ++b) {
      A.AddToBlock(a, b, dense_A.block<3, 3>(3 * a, 3 * b));
    }
  }
  return A;
}

GTEST_TEST(BlockSparseCholeskySolverTest, SolveWithBestOrdering) {
  BlockSparseCholeskySolver solver;
  SymmetricBlockSparseMatrix<double> A = MakeSparseMatrix();
  MatrixX<double> dense_A = A.MakeDenseMatrix();
  solver.SetMatrix(A);
  solver.Factor();
  const VectorXd b = VectorXd::LinSpaced(A.cols(), 0.0, 1.0);
  const VectorXd x = solver.Solve(b);
  const VectorXd expected_x = Eigen::LLT<MatrixXd>(dense_A).solve(b);
  EXPECT_TRUE(CompareMatrices(x, expected_x, 1e-13));
}

GTEST_TEST(BlockSparseCholeskySolverTest, SolveWithArbitraryOrdering) {
  BlockSparseCholeskySolver solver;
  SymmetricBlockSparseMatrix<double> A = MakeSparseMatrix();
  MatrixX<double> dense_A = A.MakeDenseMatrix();
  /* The best ordering in the test above is {0, 1, 2, 3}, i.e, the identity,
   because the matrix is fully dense. Here, we choose an ordering that's not
   the identity. */
  std::vector<int> ordering{0, 3, 1, 2};
  solver.SetMatrix(A, ordering);
  solver.Factor();
  const VectorXd b = VectorXd::LinSpaced(A.cols(), 0.0, 1.0);
  const VectorXd x = solver.Solve(b);
  const VectorXd expected_x = Eigen::LLT<MatrixXd>(dense_A).solve(b);
  EXPECT_TRUE(CompareMatrices(x, expected_x, 1e-13));
}

GTEST_TEST(BlockSparseCholeskySolverTest, SchurComplement) {
  // TODO(xuchenhan-tri): This will be tested in the tests for
  // FastSchurComplement.
}

GTEST_TEST(BlockSparseCholeskySolverTest, CalcEliminationOrdering) {
  const int num_verts = 4;
  std::vector<Vector4i> elements;
  elements.emplace_back(0, 1, 2, 3);
  const std::vector<int> p =
      CalcEliminationOrdering(BuildAdjacencyGraph(num_verts, elements));
  /* We expect natural ordering when there's no sparsity to be exploit */
  for (int i = 0; i < num_verts; ++i) {
    EXPECT_EQ(p[i], i);
  }
}

GTEST_TEST(BlockSparseCholeskySolverTest, RestrictOrdering) {
  const std::vector<int> p = {1, 5, 3, 2, 4, 0};
  const std::vector<int> nonparticipating_indices = {0, 1, 3, 4};
  const std::vector<int> expected_permutation = {1, 3, 4, 0, 5, 2};
  const std::vector<int> result = RestrictOrdering(p, nonparticipating_indices);
  ASSERT_EQ(result.size(), expected_permutation.size());
  for (int i = 0; i < static_cast<int>(result.size()); ++i) {
    EXPECT_EQ(result[i], expected_permutation[i]);
  }
}

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
