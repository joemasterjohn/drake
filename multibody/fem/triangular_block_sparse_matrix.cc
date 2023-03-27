#include "drake/multibody/fem/triangular_block_sparse_matrix.h"

#include <algorithm>

#include <fmt/format.h>

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
TriangularBlockSparseMatrix<T>::TriangularBlockSparseMatrix(
    BlockSparsityPattern sparsity_pattern, bool is_symmetric)
    : sparsity_pattern_(std::move(sparsity_pattern)),
      is_symmetric_(is_symmetric),
      block_cols_(sparsity_pattern_.block_sizes().size()),
      starting_cols_(block_cols_, 0),
      blocks_(block_cols_),
      block_row_to_flat_(block_cols_, std::vector<int>(block_cols_, -1)) {
  for (int i = 1; i < block_cols_; ++i) {
    starting_cols_[i] =
        starting_cols_[i - 1] + sparsity_pattern_.block_sizes()[i - 1];
  }
  cols_ = block_cols_ == 0
              ? 0
              : starting_cols_.back() + sparsity_pattern_.block_sizes().back();

  for (int c = 0; c < block_cols_; ++c) {
    blocks_[c].reserve(num_blocks(c));
    for (int index = 0; index < num_blocks(c); ++index) {
      const int r = sparsity_pattern_.neighbors()[c][index];
      DRAKE_DEMAND(r >= c);
      block_row_to_flat_[c][r] = index;

      const int rows = sparsity_pattern_.block_sizes()[r];
      const int cols = sparsity_pattern_.block_sizes()[c];
      blocks_[c].push_back(MatrixX<T>::Zero(rows, cols));
    }
  }
}

template <typename T>
void TriangularBlockSparseMatrix<T>::AddToBlock(
    int i, int j, const Eigen::Ref<const MatrixX<T>>& Aij) {
  DRAKE_DEMAND(0 <= j && j <= i && i < block_cols_);
  const int index = block_row_to_flat_[j][i];
  DRAKE_DEMAND(index >= 0);
  MatrixX<T>& old_value = blocks_[j][index];
  DRAKE_DEMAND(old_value.rows() == Aij.rows());
  DRAKE_DEMAND(old_value.cols() == Aij.cols());
  old_value += Aij;
}

template <typename T>
void TriangularBlockSparseMatrix<T>::SetBlock(int i, int j, MatrixX<T> Aij) {
  DRAKE_DEMAND(0 <= j && j <= i && i < block_cols_);
  const int index = block_row_to_flat_[j][i];
  DRAKE_DEMAND(index >= 0);
  MatrixX<T>& old_value = blocks_[j][index];
  DRAKE_DEMAND(old_value.rows() == Aij.rows());
  DRAKE_DEMAND(old_value.cols() == Aij.cols());
  old_value = std::move(Aij);
}

template <typename T>
void TriangularBlockSparseMatrix<T>::SetZero() {
  for (int c = 0; c < block_cols_; ++c) {
    for (MatrixX<T>& block : blocks_[c]) {
      block.setZero();
    }
  }
}

template <typename T>
MatrixX<T> TriangularBlockSparseMatrix<T>::MakeDenseMatrix() const {
  MatrixX<T> A = MatrixX<T>::Zero(rows(), cols());
  for (int j = 0; j < block_cols_; ++j) {
    for (int index = 0; index < num_blocks(j); ++index) {
      const int i = sparsity_pattern_.neighbors()[j][index];
      const int rows = sparsity_pattern_.block_sizes()[i];
      const int cols = sparsity_pattern_.block_sizes()[j];
      const int starting_row = starting_cols_[i];
      const int starting_col = starting_cols_[j];
      A.template block(starting_row, starting_col, rows, cols) =
          blocks_[j][index];
      if (i != j && is_symmetric_) {
        A.template block(starting_col, starting_row, cols, rows) =
            blocks_[j][index].transpose();
      }
    }
  }
  return A;
}

template class TriangularBlockSparseMatrix<double>;

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
