#pragma once

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* The sparsity pattern of a block sparse matrix.
 Each diagonal block is always non-zero and square, and the number of its rows
 (and columns) is stored in `block_sizes`.
 `neighbors` describes whether off diagonal blocks are nonzero or
 not. `neighbors[c][i]` gives the block row index of the i-th nonzero block in
 the c-th block column. We require that `neighbors[c]` is sorted for each block
 column c and that each entry in `neighbors[c]` is greater than or equal to c.
 In other words, only the lower triangular part of the sparsity pattern is
 specified. As a result, we have `neighbors[c][0] = c` for each c because all
 diagonal blocks are nonzero. */
class BlockSparsityPattern {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BlockSparsityPattern);
  BlockSparsityPattern(std::vector<int> block_sizes,
                       std::vector<std::vector<int>> neighbors)
      : block_sizes_(std::move(block_sizes)), neighbors_(std::move(neighbors)) {
    DRAKE_DEMAND(block_sizes_.size() == neighbors_.size());
    for (int i = 0; i < static_cast<int>(block_sizes_.size()); ++i) {
      DRAKE_DEMAND(neighbors_[i].size() > 0);
      DRAKE_DEMAND(neighbors_[i].back() <
                   static_cast<int>(block_sizes_.size()));
      DRAKE_ASSERT(std::is_sorted(neighbors_[i].begin(), neighbors_[i].end()));
      DRAKE_DEMAND(neighbors_[i][0] == i);
    }
  }

  const std::vector<int>& block_sizes() const { return block_sizes_; }
  const std::vector<std::vector<int>>& neighbors() const { return neighbors_; }

 private:
  std::vector<int> block_sizes_;
  std::vector<std::vector<int>> neighbors_;
};

/* This class provides a representation for sparse matrices with a structure
 consisting of dense blocks. It is similar to
 drake::multibody::contact_solvers::internal::BlockSparseMatrix in that it
 enables efficient algorithms capable of exploiting highly optimized operations
 with dense blocks. It differs from BlockSparseMatrix in a few aspects:
  1. We only store the lower triangular portion of the matrix with a flag to
     make the matrix either lower triangular or symmetric.
  2. It allows modification to the data (but not the sparsity pattern) after
     construction. Therefore, it is suitable for storing matrices with constant
     sparsity pattern and mutable data.
 @tparam_nonsymbolic_scalar */
template <typename T>
class TriangularBlockSparseMatrix {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(TriangularBlockSparseMatrix);

  /* Constructs a TriangularBlockSparseMatrix with the given block sparsity
   pattern.
   @param sparsity_pattern  The block sparsity pattern of lower triangular part
                            of the matrix.
   @param is_symmetric      If true, the lower triangular matrix implicitly
                            represents a symmetric matrix. */
  TriangularBlockSparseMatrix(BlockSparsityPattern sparsity_pattern,
                              bool is_symmetric);

  int rows() const { return cols_; }
  int cols() const { return cols_; }
  int block_rows() const { return block_cols_; }
  int block_cols() const { return block_cols_; }

  /* Adds Aij to the ij-th block of this matrix.
   @pre Aij = Aij.transpose() if i==j and is_symmetric() is true.
   @pre The size of Aij is compatible to the size of ij-th block implied by the
   sparsity pattern at construction. */
  void AddToBlock(int i, int j, const Eigen::Ref<const MatrixX<T>>& Aij);

  /* Similar to AddToBlock, but overwrites instead of accumulates. */
  void SetBlock(int i, int j, MatrixX<T> Aij);

  /* (Advanced) Similar to SetBlock, but uses flat indices instead of block row
   indices. This is slightly faster than SetBlock. */
  void SetBlockFlat(int flat, int j, MatrixX<T> Aij) {
    blocks_[j][flat] = std::move(Aij);
  }

  /* For the ij-th block M, does M -= A * Bᵀ.
   @pre The size of A * Bᵀ is compatible to the size of ij-th block implied by
   the sparsity pattern at construction.
   @pre has_block(i, j) == true */
  void SubtractProductFromBlock(int i, int j, const MatrixX<T>& A,
                                const MatrixX<T>& B) {
    DRAKE_ASSERT(A.cols() == B.cols());
    DRAKE_ASSERT(A.rows() == sparsity_pattern_.block_sizes()[i]);
    DRAKE_ASSERT(B.rows() == sparsity_pattern_.block_sizes()[j]);
    const int index = block_row_to_flat_[j][i];
    blocks_[j][index] -= A * B.transpose();
  }

  /* Sets the numerical values of all nonzero blocks to zero without changing
   the sparsity pattern. */
  void SetZero();

  MatrixX<T> MakeDenseMatrix() const;

  /* Returns true if there exists a ij-th block in this block sparse matrix. */
  bool has_block(int i, int j) const {
    if (i < 0 || i >= block_rows() || j < 0 || j >= block_cols()) {
      return false;
    }
    return block_row_to_flat_[j][i] >= 0;
  }

  /* Returns the ij-th block.
   @pre has_block(i,j) == true. */
  const MatrixX<T>& block(int i, int j) const {
    DRAKE_ASSERT(has_block(i, j));
    return blocks_[j][block_row_to_flat_[j][i]];
  }

  /* Returns the mutable ij-th block.
   @pre has_block(i,j) == true. */
  MatrixX<T>& mutable_block(int i, int j) {
    DRAKE_ASSERT(has_block(i, j));
    return blocks_[j][block_row_to_flat_[j][i]];
  }

  /* Returns the i-th diagonal block. */
  const MatrixX<T>& diagonal_block(int i) const {
    DRAKE_ASSERT(0 <= i && i < block_cols_);
    /* Since block_rows are sorted with in each block column, the first entry is
     necessarily the diagonal. */
    return blocks_[i][0];
  }

  /* (Advanced) Similar to `block`, but returns matrix blocks based on flat
   * indices instead of block row indices. */
  const MatrixX<T>& block_flat(int flat, int j) {
    DRAKE_ASSERT(0 <= j && j < block_cols_);
    return blocks_[j][flat];
  }
  MatrixX<T>& mutable_block_flat(int flat, int j) {
    DRAKE_ASSERT(0 <= j && j < block_cols_);
    return blocks_[j][flat];
  }

  /* Returns the sorted block row indices in the j-th block column. */
  const std::vector<int>& block_row_indices(int j) const {
    DRAKE_DEMAND(0 <= j && j < block_cols_);
    return sparsity_pattern_.neighbors()[j];
  }

  /* Returns the sparsity pattern of the matrix. */
  const BlockSparsityPattern& sparsity_pattern() const {
    return sparsity_pattern_;
  }

  /* Returns the starting (scalar) column of each block column. */
  const std::vector<int>& starting_cols() const { return starting_cols_; }

 private:
  /* The number of nonzero blocks in j-th column. */
  int num_blocks(int j) const { return block_row_indices(j).size(); }

  BlockSparsityPattern sparsity_pattern_;
  bool is_symmetric_{false};
  int block_cols_{};
  int cols_{};
  /* The starting (scalar) column of each block column. */
  std::vector<int> starting_cols_;
  /* Dense blocks stored in a 2d vector. The first index is the block column
   index and the second index is a flat index that can be retrieved from
   block_row_to_flat_ below. */
  std::vector<std::vector<MatrixX<T>>> blocks_;
  /* Mapping from block row index to flat index for each column; i.e.,
   blocks_[c][block_row_to_flat_[c][r]] gives the (r,c) block.
   block_row_to_flat_[c][r] == -1 if the implied block is empty. */
  std::vector<std::vector<int>> block_row_to_flat_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
