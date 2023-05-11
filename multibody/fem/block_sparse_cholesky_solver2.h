#pragma once

#include <vector>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"
#include "drake/multibody/fem/triangular_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* Sparse cholesky solver where the blocks are of size 3x3. */
class BlockSparseCholeskySolver2 {
 public:
  /* Constructs a solver. */
  BlockSparseCholeskySolver2() = default;

  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BlockSparseCholeskySolver2);

  /* Sets the matrix to be factored and uses the AMD elimination ordering that
   to reduce fill-ins. */
  void SetMatrix(const TriangularBlockSparseMatrix<double>& A);

  /* Updates the matrix to be factored. This is useful for solving a series of
   matrices with the same sparsity pattern using the same elimination ordering.
   For example, with matrices A, B, and C with the same sparisty pattern. It's
   more efficient to call
     solver.SetMatrix(A);
     solver.UpdateMatrix(B);
     solver.UpdateMatrix(C);
   than to call
     solver.SetMatrix(A);
     solver.SetMatrix(B);
     solver.SetMatrix(C); */
  void UpdateMatrix(const TriangularBlockSparseMatrix<double>& A);

  void Factor();

  void SolveInPlace(VectorX<double>* y) const;

  VectorX<double> Solve(const VectorX<double>& y) const;

 private:
  /* Performs L(j+1:, j+1:) -= L(j+1:,j) * L(j+1:,j).transpose().
   @pre 0 <= j < block_cols_. */
  void RightLookingSymmetricRank1Update(int j);

  int block_cols_{0};
  copyable_unique_ptr<TriangularBlockSparseMatrix<double>> L_;
  std::vector<Eigen::LLT<MatrixX<double>>> L_diag_;
  /* The mapping from the internal indices (i.e, the indices for L_) to
   the indices of the original matrix supplied in SetMatrix(). */
  contact_solvers::internal::PartialPermutation block_index_permutation_;
  contact_solvers::internal::PartialPermutation scalar_index_permutation_;
  bool is_factored_{false};
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
