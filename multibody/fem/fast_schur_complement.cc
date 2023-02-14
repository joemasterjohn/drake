#include "drake/multibody/fem/fast_schur_complement.h"

#include <set>
#include <unordered_set>
#include <utility>

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <class T>
FastSchurComplement<T>::FastSchurComplement(
    const SymmetricBlockSparseMatrix<T>& M, const std::vector<int>& D_indices,
    const std::vector<int>& A_indices)
    : D_indices_(D_indices), A_indices_(A_indices) {
  const int block_cols = A_indices.size() + D_indices.size();
  DRAKE_DEMAND(block_cols * 3 == M.cols());
  S_ = solver_.CalcSchurComplementAndFactor(M, D_indices);
}

template <class T>
VectorX<T> FastSchurComplement<T>::SolveForX(
    const Eigen::Ref<const VectorX<T>>& a) const {
  DRAKE_DEMAND(static_cast<int>(a.size()) ==
               3 * static_cast<int>(A_indices_.size()));
  if (D_indices_.size() == 0) {
    return VectorX<T>::Zero(0);
  }
  if (A_indices_.size() == 0) {
    return VectorX<T>::Zero(D_indices_.size());
  }
  const int block_cols = A_indices_.size() + D_indices_.size();
  VectorX<T> rhs(VectorX<T>::Zero(3 * block_cols));
  for (int i = 0; i < static_cast<int>(A_indices_.size()); ++i) {
    for (int d = 0; d < 3; ++d) {
      rhs(3 * A_indices_[i] + d) = a(3 * i + d);
    }
  }
  solver_.SolveInPlace(&rhs);
  VectorX<T> x(3 * D_indices_.size());
  for (int i = 0; i < static_cast<int>(D_indices_.size()); ++i) {
    for (int d = 0; d < 3; ++d) {
      x(3 * i + d) = rhs(3 * D_indices_[i] + d);
    }
  }
  return x;
}

template class FastSchurComplement<double>;

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
