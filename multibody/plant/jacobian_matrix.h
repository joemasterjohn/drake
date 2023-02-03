#pragma once

#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace internal {

/* Block sparse matrix with 3*3 blocks. The size of the matrix is
 3*row_blocks-by-3*col_blocks. The blocks are sorted according to block row
 indices. We use M to denote `this` matrix throughout this class. */
template <class T>
class Matrix3BlockMatrix {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Matrix3BlockMatrix);

  /* block_row, block_col, and a 3x3 dense matrix. */
  using Triplet = std::tuple<int, int, Matrix3<T>>;

  /* Create a matrix with `row_blocks` * `col_blocks` of 3x3 matrices. */
  Matrix3BlockMatrix(int row_blocks, int col_blocks)
      : row_blocks_(row_blocks), col_blocks_(col_blocks) {}

  /* @pre block_row >= all existing block rows. */
  void AddTriplet(int block_row, int block_col, Matrix3<T> m) {
    DRAKE_DEMAND(0 <= block_row && block_row < row_blocks_);
    DRAKE_DEMAND(0 <= block_col && block_col < col_blocks_);
    DRAKE_DEMAND(data_.empty() || std::get<0>(data_.back()) <= block_row);
    data_.emplace_back(block_row, block_col, std::move(m));
  }

  int rows() const { return row_blocks_ * 3; }
  int cols() const { return col_blocks_ * 3; }

  /* Performs *y += M * x. */
  void MultiplyAndAddTo(const Eigen::Ref<const VectorX<T>>& x,
                        EigenPtr<VectorX<T>> y) const;

  /* Performs *y += A * M. */
  void LeftMultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                            EigenPtr<MatrixX<T>> y) const;

  /* Performs *y += Mᵀ * A, where A is dense. */
  void TransposeMultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                                 EigenPtr<MatrixX<T>> y) const;

  /* Performs *y += Mᵀ * A, where A is also 3x3 block sparse. */
  void TransposeMultiplyAndAddTo(const Matrix3BlockMatrix<T>& A,
                                 EigenPtr<MatrixX<T>> y) const;

  /* Returns M * scale.asDiagonal() * M.transpose() */
  MatrixX<T> MultiplyByScaledTranspose(const VectorX<T>& scale) const;

  /* For debugging. */
  MatrixX<T> MakeDenseMatrix() const;

  const std::vector<Triplet>& get_triplets() const { return data_; }

  int num_blocks() const { return data_.size(); }

 private:
  std::vector<Triplet> data_;
  int row_blocks_{};
  int col_blocks_{};
};

/* Data structure to store the jacobian matrix for a particular tree and patch.
 It's eventually stored as a block in the block sparse jacobian matrix and
 thus the name. It can either be a dense Eigen matrix or a block sparse 3x3
 matrix. We use J to denote `this` jacobian block throughout this class. */
template <class T>
class JacobianBlock {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(JacobianBlock);

  /* Default constructs an empty JacobianBlock as a dense 0x0 matrix. */
  JacobianBlock() : JacobianBlock(MatrixX<T>::Zero(0, 0)) {}

  explicit JacobianBlock(Matrix3BlockMatrix<T> data)
      : data_(std::move(data)), is_dense_(false) {}

  explicit JacobianBlock(MatrixX<T> data)
      : data_(std::move(data)), is_dense_(true) {}

  /* We need the static_cast here because Eigen's rows() and cols() are long. */
  int rows() const {
    return std::visit([](auto&& arg) { return static_cast<int>(arg.rows()); },
                      data_);
  }

  int cols() const {
    return std::visit([](auto&& arg) { return static_cast<int>(arg.cols()); },
                      data_);
  }

  /* Performs *y += A * J. */
  void LeftMultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                            EigenPtr<MatrixX<T>> y) const;

  /* Performs *y += J * x. */
  void MultiplyAndAddTo(const Eigen::Ref<const VectorX<T>>& x,
                        EigenPtr<VectorX<T>> y) const;

  /* Performs *y += Jᵀ * A. */
  void TransposeMultiplyAndAddTo(const Eigen::Ref<const MatrixX<T>>& A,
                                 EigenPtr<MatrixX<T>> y) const;

  /* Performs *y += Jᵀ * A. */
  void TransposeMultiplyAndAddTo(const JacobianBlock<T>& A,
                                 EigenPtr<MatrixX<T>> y) const;

  /* Computes G*J where G is a block diagonal matrix with the diagonal blocks
   specified as a vector of dense matrices. In particular, the diagonal blocks
   are the [G_start, G_end] entries in the given `G`. */
  JacobianBlock<T> LeftMultiplyByBlockDiagonal(const std::vector<MatrixX<T>>& G,
                                               int G_start, int G_end) const;

  /* Returns J * scale.asDiagonal() * J.transpose() */
  MatrixX<T> MultiplyByScaledTranspose(const VectorX<T>& scale) const;

  bool is_dense() const { return is_dense_; }

  /* Testing and debugging utilities. */
  MatrixX<T> MakeDenseMatrix() const;
  bool operator==(const JacobianBlock<T>& other) const {
    if constexpr (!std::is_same_v<T, symbolic::Expression>) {
      return this->MakeDenseMatrix() == other.MakeDenseMatrix();
    }
    return true;
  }

 private:
  template <class U>
  friend JacobianBlock<U> StackJacobianBlocks(
      const std::vector<JacobianBlock<U>>& blocks);
  std::variant<MatrixX<T>, Matrix3BlockMatrix<T>> data_;
  bool is_dense_{};
};

template <typename T>
JacobianBlock<T> StackJacobianBlocks(
    const std::vector<JacobianBlock<T>>& blocks);

}  // namespace internal
}  // namespace multibody
}  // namespace drake
