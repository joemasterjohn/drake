#pragma once

#include <vector>

#include <Eigen/Sparse>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* This class provides a representation for sparse matrices with a structure
 consisting of dense blocks of size 3x3. It is similar to
 contact_solvers::internal::BlockSparseMatrix in that it enables efficient
 algorithms capable of exploiting highly optimized operations with dense blocks.
 It differs from BlockSparseMatrix in a few aspects:

  1. It is tailored to sparse matrices with a particular 3x3 block structure
  2. It is tailored to symmetric matrices and only stores the lower triangular
     part of the matrix.
  3. It allows modification to the data (but not the sparsity pattern) after
     construction. Therefore, it is suitable for storing matrices with constant
     sparsity pattern and mutable data.

 In particular, these features make SymmetricBlockSparseMatrix suitable for
 storing the stiffness/damping/tangent matrix of an FEM model, where the matrix
 has constant sparsity pattern, has 3x3 block structure, and is symmetric.
 @tparam_nonsymbolic_scalar */
template <typename T>
class SymmetricBlockSparseMatrix {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SymmetricBlockSparseMatrix);

  /* Constructs a SymmetricBlockSparseMatrix with the given sparsity pattern.
   @param[in] sparsity_pattern
     Characterizes the sparsity pattern of the matrix being constructed. The
     number of column blocks is given by `sparsity_pattern.size()`.
     `sparsity_pattern[c][i]` gives the block row index of the i-th block in
     the c-th block column. If `sparsity_pattern[c]` is empty, then the c-th
     block column is empty.
   @pre row_blocks[c][i] >= c. In other words, only the lower triangular part of
   the sparsity pattern should be specified. */
  explicit SymmetricBlockSparseMatrix(
      std::vector<std::vector<int>> sparsity_pattern);

  int rows() const { return cols(); }
  int cols() const { return 3 * num_column_blocks_; }

  /* Adds Aij to the ij-th block of this matrix.
   @pre Aij = Aij.transpose() if i==j.
   @pre sparsity_pattern[j] contains i at constrution. */
  void AddToBlock(int i, int j, const Eigen::Ref<const Matrix3<T>>& Aij);

  /* For the ij-th block M, do M -= A * B.transpose().
   @warning no bound checking. */
  void SubtractProductFromBlock(int i, int j, const Matrix3<T>& A,
                                const Matrix3<T>& B) {
    const int index = block_row_to_flat_[j][i];
    blocks_[j][index] -= A * B.transpose();
  }

  /* Returns the flat-th block in j-th block_column. */
  const Matrix3<T>& get_block_flat(int flat, int j) const {
    return blocks_[j][flat];
  }

  /* Similar to AddToBlock, but overwrites instead of accumulates. */
  void SetBlock(int i, int j, Matrix3<T> Aij);
  void SetBlockFlat(int flat, int j, Matrix3<T> Aij) {
    blocks_[j][flat] = std::move(Aij);
  }

  void SetZero();

  /* Computes *y = A*x.
   @pre y->size() == A.rows().
   @pre x.size() == A.cols(). */
  void Multiply(const VectorX<T>& x, VectorX<T>* y) const;

  MatrixX<T> MakeDenseMatrix() const;
  MatrixX<T> MakeDenseBottomRightCorner(int size) const;
  Eigen::SparseMatrix<T> MakeEigenSparseMatrix() const;

  /* Returns true if there exists a ij-th block in this block sparse matrix.
   @pre 0 <= i < rows()/3.
   @pre 0 <= j < cols()/3. */
  bool has_block(int i, int j) const {
    if (i < 0 || i >= rows() / 3 || j < 0 || j >= cols() / 3) {
      return false;
    }
    return block_row_to_flat_[j][i] >= 0;
  }

  /* Returns the ij-th block.
   @pre has_block(i,j) == true. */
  const Matrix3<T>& get_block(int i, int j) const {
    DRAKE_ASSERT(has_block(i, j));
    return blocks_[j][block_row_to_flat_[j][i]];
  }

  /* Returns the ii-th block.
   @pre has_block(i,i) == true. */
  const Matrix3<T>& get_diagonal_block(int i) const {
    DRAKE_ASSERT(has_block(i, i));
    /* Since block_rows are sorted with in each block column, the first entry is
     necessarily the diagonal. */
    return blocks_[i][0];
  }

  /* Returns the mutable ij-th block.
   @pre has_block(i,j) == true. */
  Matrix3<T>& get_mutable_block(int i, int j) {
    return blocks_[j][block_row_to_flat_[j][i]];
  }

  const std::vector<int>& get_col_blocks(int j) const {
    DRAKE_DEMAND(0 <= j && j < num_column_blocks_);
    return col_blocks_[j];
  }

  int num_blocks() const { return num_blocks_; }

 private:
  friend class BlockSparseCholeskySolver;

  /* sparsity_pattern_[c][i] gives the i-th row block in the c-th column block.
   */
  std::vector<std::vector<int>> sparsity_pattern_;
  int num_column_blocks_;
  /* col_blocks[c][i] gives the block row index of the i-th block in the c-th
   block column. */
  std::vector<std::vector<int>> col_blocks_;
  int num_blocks_;
  /* The 3x3 blocks stored in a 2d vector. The first index is the block column
   index and the second index is a flat index that can be retrieved from
   block_row_to_flat_ below. */
  std::vector<std::vector<Matrix3<T>>> blocks_;
  /* num_blocks_in_col_[c] gives the number of blocks in the c-th column block.
   */
  std::vector<int> num_blocks_in_col_;
  // TODO(xuchenhan-tri): perhaps more efficient to store as
  // std::vector<unordered_map<int, int>> when the size of the matrix gets
  // large.
  /* Mapping from block row index to flat index for each column; i.e.,
   blocks_[c][block_row_to_flat_[c][r]] == r.
   block_row_to_flat_[c][r] == -1 if the implied block is empty. */
  std::vector<std::vector<int>> block_row_to_flat_;
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
