#pragma once
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/sap/partial_permutation.h"
#include "drake/multibody/fem/block_sparse_cholesky_solver.h"
#include "drake/multibody/fem/symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* Computes the Schur complement of a matrix given its block components.

 Given a linear system of equations Mz = c that can be written in block form
 as:
     Dx  + By = 0     (1)
     Bᵀx + Ay = a     (2)
 where M = [D B; Bᵀ A], zᵀ = [xᵀ yᵀ], cᵀ = [0ᵀ aᵀ], and A(size p-by-p),
 D(size q-by-q) and M(size p+q-by-p+q) are positive definite, one can solve
 the system using Schur complement. Specifically, using equation (1), we get
     x = -D⁻¹By       (3)
 Plugging (3) in (1), we get
    (A - BᵀD⁻¹B)y = a.
 After a solution for y is obtained, we can use (3) to recover the solution for
 x. The matrix S = A - BᵀD⁻¹B is the Schur complement of the block D of the
 matrix M. Since M is positive definite, so is the S.

 @tparam_double_only */
template <typename T>
class FastSchurComplement {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FastSchurComplement);

  /* Constructs an empty FastSchurComplement.*/
  FastSchurComplement() : solver_({}), S_(0, 0) {}

  /* Constructs a FastSchurComplement for the block sparse matrix M of size 3*N.
  `D_indices` and `A_indices` determine which block rows/columns make up the D
  and A diagonal blocks.
  @pre D_indices and A_indices are sorted and disjoint and their union equals
  the set {0, ..., N-1}. */
  FastSchurComplement(const SymmetricBlockSparseMatrix<T>& M,
                      const std::vector<int>& D_indices,
                      const std::vector<int>& A_indices);

  /* Returns the Schur complement for the block D of the matrix M,
   S = A - BD⁻¹Bᵀ. */
  const MatrixX<T>& get_D_complement() const { return S_; }

  /* Given a value of y, solves for x in the equation Dx + By = 0. */
  VectorX<T> SolveForX(const Eigen::Ref<const VectorX<T>>& y) const;

 private:
  int D_size_{0};
  int A_size_{0};
  BlockSparseCholeskySolver solver_;
  MatrixX<T> S_;  // S = A - BᵀD⁻¹B.
  /* Permutation for D block indices. */
  contact_solvers::internal::PartialPermutation D_dof_permutation_;
  /* Permutation for A block indices. */
  contact_solvers::internal::PartialPermutation A_dof_permutation_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
