#include "drake/multibody/fem/fast_schur_complement.h"

#include <set>
#include <unordered_set>
#include <utility>

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
using drake::multibody::contact_solvers::internal::PartialPermutation;

namespace {

/* Given an input matrix M, and a permutation mapping e, sets the resulting
 matrix M̃ such that M̃(i, j) = M(e(i), e(j)). */
template <typename T>
void PermuteSymmetricBlockSparseMatrix(
    const SymmetricBlockSparseMatrix<T>& input,
    const std::vector<int>& permutation,
    SymmetricBlockSparseMatrix<T>* result) {
  DRAKE_DEMAND(result != nullptr);
  const int N = permutation.size();
  DRAKE_DEMAND(3 * N == input.cols());
  DRAKE_DEMAND(3 * N == result->cols());
  result->SetZero();

  /* Construct the inverse mapping of e, f. */
  std::vector<int> inverse_permutation(permutation.size());
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    inverse_permutation[permutation[i]] = i;
  }
  /* M̃(i, j) = M(e(i), e(j)) is equivalent to M̃(f(i), f(j)) = M(i, j). */
  for (int j = 0; j < N; ++j) {
    const std::vector<int>& row_indices = input.get_col_blocks(j);
    for (int i : row_indices) {
      const Matrix3<T>& block = input.get_block(i, j);
      const int fi = inverse_permutation[i];
      const int fj = inverse_permutation[j];
      if (fi >= fj) {
        result->SetBlock(fi, fj, block);
      } else {
        result->SetBlock(fj, fi, block.transpose());
      }
    }
  }
}

/* Given an input matrix M, and a permutation mapping e, sets the resulting
 matrix M̃ such that M̃(e(i), e(j)) = M(i, j). */
template <typename T>
void InversePermuteDenseMatrix(const MatrixX<T>& input,
                               const std::vector<int>& permutation,
                               MatrixX<T>* result) {
  DRAKE_DEMAND(result != nullptr);
  const int N = permutation.size();
  DRAKE_DEMAND(3 * N == input.cols());
  DRAKE_DEMAND(3 * N == result->cols());
  result->setZero();

  /* Construct the inverse mapping of e, f. */
  std::vector<int> inverse_permutation(permutation.size());
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    inverse_permutation[permutation[i]] = i;
  }
  /* Expand the permutation into permutation on dofs instead of vertices. */
  VectorX<int> p(3 * permutation.size());
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    for (int d = 0; d < 3; ++d) {
      p(3 * i + d) = 3 * inverse_permutation[i] + d;
    }
  }

  Eigen::PermutationMatrix<Eigen::Dynamic> P(p);
  *result = P.inverse() * input * P;
}

}  // namespace

template <class T>
FastSchurComplement<T>::FastSchurComplement(
    const SymmetricBlockSparseMatrix<T>& M, const std::vector<int>& D_indices,
    const std::vector<int>& A_indices)
    : D_size_(D_indices.size()), A_size_(A_indices.size()), solver_({}) {
  DRAKE_DEMAND(static_cast<int>(D_indices.size() + A_indices.size()) ==
               M.cols() / 3);
  const std::vector<std::set<int>> adjacency_graph = M.CalcAdjacencyGraph();
  /* Mapping from new indices to old indices. */
  std::vector<int> elimination_order =
      RestrictOrdering(CalcEliminationOrdering(adjacency_graph), D_indices);

  solver_ = BlockSparseCholeskySolver(
      CalcSparsityPattern(adjacency_graph, elimination_order));

  /* Set the matrix M̃ in `solver_` using a permutation of M such that
   M̃(i, j) = M(e(i), e(j)). */
  PermuteSymmetricBlockSparseMatrix(M, elimination_order,
                                    &solver_.GetMutableMatrix());

  std::unordered_set<int> D_set(D_indices.begin(), D_indices.end());
  /* Maps the global index of a vertex to its implied index within a group (A or
   D). For example, suppose D_indices are {0, 2, 5}, and A_indices are {1, 3, 4,
   6}, then the implied indices for D vertices are 0 -> 0, 2 -> 1, 5 -> 2, and
   the implied indices for A vertices are 1 -> 0, 3 -> 1, 4 -> 2, 6 -> 3, and
   the global_to_local mapping is {0, 0, 1, 1, 2, 2, 3}. */
  int N = A_size_ + D_size_;
  std::vector<int> global_to_local(N);
  int local_D_index = 0;
  int local_A_index = 0;

  for (int i = 0; i < N; ++i) {
    if (D_set.count(i) > 0) {
      global_to_local[i] = local_D_index++;
    } else {
      global_to_local[i] = local_A_index++;
    }
  }

  /* The permutation of A/D indices induced by the elimination ordering. Note
   that D_vertex_permutation[new_index] = old_index. */
  std::vector<int> D_vertex_permutation;
  std::vector<int> A_vertex_permutation;
  D_vertex_permutation.reserve(D_size_);
  A_vertex_permutation.reserve(A_size_);
  for (int i = 0; i < static_cast<int>(elimination_order.size()); ++i) {
    const int old_index = elimination_order[i];
    if (D_set.count(old_index) > 0) {
      D_vertex_permutation.emplace_back(global_to_local[old_index]);
    } else {
      A_vertex_permutation.emplace_back(global_to_local[old_index]);
    }
  }

  /* Calculate the Schur complement in the elimination ordering the then permute
   it back to the original ordering. */
  MatrixX<T> permuted_S = solver_.CalcSchurComplementAndFactor(D_size_);
  S_.resizeLike(permuted_S);
  InversePermuteDenseMatrix(permuted_S, A_vertex_permutation, &S_);

  std::vector<int> D_dof_permutation(3 * D_size_);
  std::vector<int> A_dof_permutation(3 * A_size_);
  for (int i = 0; i < D_size_; ++i) {
    for (int d = 0; d < 3; ++d) {
      D_dof_permutation[3 * i + d] = 3 * D_vertex_permutation[i] + d;
    }
  }
  for (int i = 0; i < A_size_; ++i) {
    for (int d = 0; d < 3; ++d) {
      A_dof_permutation[3 * i + d] = 3 * A_vertex_permutation[i] + d;
    }
  }

  D_dof_permutation_ = PartialPermutation(std::move(D_dof_permutation));
  A_dof_permutation_ = PartialPermutation(std::move(A_dof_permutation));
}

template <class T>
VectorX<T> FastSchurComplement<T>::SolveForX(
    const Eigen::Ref<const VectorX<T>>& a) const {
  DRAKE_DEMAND(a.size() == 3 * A_size_);
  VectorX<T> rhs(VectorX<T>::Zero(3 * (A_size_ + D_size_)));
  VectorX<T> permuted_a(a.size());
  A_dof_permutation_.ApplyInverse(a, &permuted_a);
  rhs.tail(3 * A_size_) = permuted_a;
  solver_.SolveInPlace(&rhs);
  VectorX<T> permuted_x = rhs.head(3 * D_size_);
  VectorX<T> x(3 * D_size_);
  D_dof_permutation_.Apply(permuted_x, &x);
  return x;
}

template class FastSchurComplement<double>;

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
