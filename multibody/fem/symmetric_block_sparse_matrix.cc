#include "drake/multibody/fem/symmetric_block_sparse_matrix.h"

#include <algorithm>
#include <iostream>

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
SymmetricBlockSparseMatrix<T>::SymmetricBlockSparseMatrix(
    std::vector<std::vector<int>> sparsity_pattern)
    : sparsity_pattern_(std::move(sparsity_pattern)),
      num_column_blocks_(sparsity_pattern_.size()),
      col_blocks_(num_column_blocks_),
      blocks_(num_column_blocks_),
      num_blocks_in_col_(num_column_blocks_),
      block_row_to_flat_(num_column_blocks_,
                         std::vector<int>(num_column_blocks_, -1)) {
  /* Ensure block rows are sorted with in a block column. */
  for (auto& block_row_indices : sparsity_pattern_) {
    std::sort(block_row_indices.begin(), block_row_indices.end());
  }
  num_blocks_ = 0;
  for (int c = 0; c < num_column_blocks_; ++c) {
    num_blocks_in_col_[c] = sparsity_pattern_[c].size();
    for (int index = 0; index < num_blocks_in_col_[c]; ++index) {
      const int r = sparsity_pattern_[c][index];
      DRAKE_DEMAND(r >= c);
      block_row_to_flat_[c][r] = index;
      col_blocks_[c].emplace_back(r);
      /* Add two blocks if the block is not on the diagonal (due to symmetry).
       */
      num_blocks_ += (r == c) ? 1 : 2;
    }
  }
  std::cout << "NumBlocks = " << num_blocks_ << std::endl;
  for (int c = 0; c < num_column_blocks_; ++c) {
    blocks_[c].resize(num_blocks_in_col_[c], Matrix3<T>::Zero());
  }
}

template <typename T>
void SymmetricBlockSparseMatrix<T>::AddToBlock(
    int i, int j, const Eigen::Ref<const Matrix3<T>>& Aij) {
  DRAKE_DEMAND(0 <= j && j <= i && i < num_column_blocks_);
  const int index = block_row_to_flat_[j][i];
  DRAKE_DEMAND(index >= 0);
  blocks_[j][index] += Aij;
}

template <typename T>
void SymmetricBlockSparseMatrix<T>::SetBlock(
    int i, int j, Matrix3<T> Aij) {
  DRAKE_DEMAND(0 <= j && j <= i && i < num_column_blocks_);
  const int index = block_row_to_flat_[j][i];
  DRAKE_DEMAND(index >= 0);
  blocks_[j][index] = std::move(Aij);
}

template <typename T>
void SymmetricBlockSparseMatrix<T>::SetZero() {
  for (int c = 0; c < num_column_blocks_; ++c) {
    for (auto& block : blocks_[c]) {
      block.setZero();
    }
  }
}

template <typename T>
void SymmetricBlockSparseMatrix<T>::Multiply(const VectorX<T>& x,
                                             VectorX<T>* y) const {
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(x.size() == cols());
  DRAKE_DEMAND(y->size() == rows());
  y->setZero();
  for (int c = 0; c < num_column_blocks_; ++c) {
    for (int index = 0; index < num_blocks_in_col_[c]; ++index) {
      const int r = sparsity_pattern_[c][index];
      y->template segment<3>(3 * r) +=
          blocks_[c][index] * x.template segment<3>(3 * c);
      if (r != c) {
        y->template segment<3>(3 * c) +=
            blocks_[c][index].transpose() * x.template segment<3>(3 * r);
      }
    }
  }
}

template <typename T>
MatrixX<T> SymmetricBlockSparseMatrix<T>::MakeDenseMatrix() const {
  MatrixX<T> A = MatrixX<T>::Zero(rows(), cols());
  for (int c = 0; c < num_column_blocks_; ++c) {
    for (int index = 0; index < num_blocks_in_col_[c]; ++index) {
      const int r = sparsity_pattern_[c][index];
      A.template block<3, 3>(3 * r, 3 * c) = blocks_[c][index];
      if (r != c) {
        A.template block<3, 3>(3 * c, 3 * r) = blocks_[c][index].transpose();
      }
    }
  }
  return A;
}

template <typename T>
MatrixX<T> SymmetricBlockSparseMatrix<T>::MakeDenseBottomRightCorner(
    int block_columns_in_corner) const {
  MatrixX<T> A = MatrixX<T>::Zero(3 * block_columns_in_corner,
                                  3 * block_columns_in_corner);
  const int offset = num_column_blocks_ - block_columns_in_corner;
  for (int c = offset; c < num_column_blocks_; ++c) {
    for (int index = 0; index < num_blocks_in_col_[c]; ++index) {
      const int r = sparsity_pattern_[c][index];
      const int new_r = r - offset;
      const int new_c = c - offset;
      A.template block<3, 3>(3 * new_r, 3 * new_c) = blocks_[c][index];
      if (r != c) {
        A.template block<3, 3>(3 * new_c, 3 * new_r) =
            blocks_[c][index].transpose();
      }
    }
  }
  return A;
}

template <typename T>
Eigen::SparseMatrix<T> SymmetricBlockSparseMatrix<T>::MakeEigenSparseMatrix()
    const {
  Eigen::SparseMatrix<T> A(rows(), cols());
  std::vector<Eigen::Triplet<T>> triplets;
  triplets.reserve(9 * num_blocks_);

  auto add_block_to_triplets = [&triplets](
                                   const Eigen::Ref<const Matrix3<T>>& block,
                                   int block_row, int block_column) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        triplets.emplace_back(3 * block_row + i, 3 * block_column + j,
                              block(i, j));
      }
    }
  };
  for (int c = 0; c < num_column_blocks_; ++c) {
    for (int index = 0; index < num_blocks_in_col_[c]; ++index) {
      const int r = sparsity_pattern_[c][index];
      add_block_to_triplets(blocks_[c][index], r, c);
      if (r != c) {
        add_block_to_triplets(blocks_[c][index].transpose(), c, r);
      }
    }
  }
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  return A;
}

template <typename T>
std::vector<std::set<int>> SymmetricBlockSparseMatrix<T>::CalcAdjacencyGraph()
    const {
  std::vector<std::set<int>> result;
  result.reserve(num_column_blocks_);
  for (const std::vector<int>& neighbors : sparsity_pattern_) {
    result.emplace_back(std::set<int>(neighbors.begin(), neighbors.end()));
  }
  return result;
}

template class SymmetricBlockSparseMatrix<double>;

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
