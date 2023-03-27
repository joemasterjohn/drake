#include "drake/multibody/fem/block_sparse_cholesky_solver.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

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
      A(i, j) = 3 * i + 2 * j;
    }
  }
  Eigen::Matrix<double, 12, 12> I = Eigen::Matrix<double, 12, 12>::Identity();
  return A * A.transpose() + I;
}

/* Makes a TriangularBlockSparseMatrix version of the matrix above. */
TriangularBlockSparseMatrix<double> MakeSparseMatrix() {
  vector<vector<int>> sparsity;
  sparsity.emplace_back(vector<int>{0, 1, 2, 3});
  sparsity.emplace_back(vector<int>{1, 2, 3});
  sparsity.emplace_back(vector<int>{2, 3});
  sparsity.emplace_back(vector<int>{3});
  vector<int> block_sizes = {2, 3, 4, 3};
  BlockSparsityPattern block_pattern = {.diagonals = block_sizes,
                                        .sparsity_pattern = sparsity};

  TriangularBlockSparseMatrix<double> A(block_pattern, true);
  const std::vector<int>& starting_cols = A.starting_cols();
  const Eigen::Matrix<double, 12, 12> dense_A = dummy_matrix12x12();
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b <= a; ++b) {
      A.AddToBlock(a, b,
                   dense_A.block(starting_cols[a], starting_cols[b],
                                 block_sizes[a], block_sizes[b]));
    }
  }
  return A;
}

GTEST_TEST(BlockSparseCholeskySolverTest, SolveWithBestOrdering) {
  BlockSparseCholeskySolver solver;
  TriangularBlockSparseMatrix<double> A = MakeSparseMatrix();
  MatrixX<double> dense_A = A.MakeDenseMatrix();
  solver.SetMatrix(A);
  solver.Factor();
  const VectorXd b = VectorXd::LinSpaced(A.cols(), 0.0, 1.0);
  const VectorXd x = solver.Solve(b);
  const VectorXd expected_x = Eigen::LLT<MatrixXd>(dense_A).solve(b);
  EXPECT_TRUE(CompareMatrices(x, expected_x, 1e-13));
}

// GTEST_TEST(BlockSparseCholeskySolverTest, CalcEliminationOrdering) {
//   const int num_verts = 4;
//   std::vector<Vector4i> elements;
//   elements.emplace_back(0, 1, 2, 3);
//   const std::vector<int> p =
//       CalcEliminationOrdering(BuildAdjacencyGraph(num_verts, elements));
//   /* We expect natural ordering when there's no sparsity to be exploit */
//   for (int i = 0; i < num_verts; ++i) {
//     EXPECT_EQ(p[i], i);
//   }
// }

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
