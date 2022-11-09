#pragma once

#include <memory>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/schur_complement.h"
#include "drake/multibody/fem/symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

class CholmodSparseMatrix {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CholmodSparseMatrix);

  /* Constructs a CHOLMOD sparse matrix from an Eigen::SparseMatrix. */
  explicit CholmodSparseMatrix(const Eigen::SparseMatrix<double>& matrix);

  ~CholmodSparseMatrix();

  /* Performs sparse Cholesky decomposition. */
  void Factor() const;

  VectorX<double> Solve(const VectorX<double>& rhs) const;

  const std::vector<int>& permutation() const;

  /* Given 2-by-2 block matrix
      A  B
      Bᵀ C,
   where A is this matrix. Computes the Schur complement C - BᵀA⁻¹B.
   @pre Factor() has been called. */
  SchurComplement<double> CalcSchurComplement(const MatrixX<double>& B,
                                              const MatrixX<double>& C) const;
  void Print() const;

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
