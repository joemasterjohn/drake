#pragma once

#include <set>
#include <unordered_set>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/symmetric_block_sparse_matrix.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

/* result[j] is the set of i>=j such that the i,j block of the matrix is
 nonzero.*/
std::vector<std::set<int>> BuildAdjacencyGraph(
    int num_verts, const std::vector<Vector4<int>>& elements);

/* Computes the elimination of the matrix with the given `adjacency_graph` that
 CHOLMOD thinks is the best. For example if the result is [1, 3, 0, 2], it means
 that we should first eliminate vertex 1, then 3, 0, and 2. In other words, this
 is a permutation mapping from new vertex indices to old indices. */
std::vector<int> CalcEliminationOrdering(
    const std::vector<std::set<int>>& adjacency_graph);

/* Computes the elimination ordering for the subgraph of `adjacency_graph`
 involving vertices in `D_indices` only as well as the subgraph that includes
 the complement of `D_indices` only and then returns a concatenation of those
 two orderings. */
std::vector<int> CalcEliminationOrdering(
    const std::vector<std::set<int>>& adjacency_graph,
    const std::vector<int>& D_indices);

/* Computes an elimination ordering consistent with the given `ordering` that
 puts vertices in the set `D` first. More specifically, let P: V->V be the given
 `ordering` and D ⊂ V. This function computes a new elimination ordering Q:V->V
 such that
 1. Q(d) < Q(a) if d ∈ D and a ∉ D,
 2. if d₁, d₂ ∈ D, Q(d₁) < Q(d₂) iff P(d₁) < P(d₂), and
 3. if a₁, a₂ ∉ D, Q(a₁) < Q(a₂) iff P(a₁) < P(a₂). */
std::vector<int> RestrictOrdering(const std::vector<int>& ordering,
                                  const std::vector<int>& D);

/* Returns the column-wise sparsity pattern of L given the adjacency graph of A
 and the elimination ordering.
 Note that the elimination ordering is a mapping from new index to old index. */
std::vector<std::vector<int>> CalcSparsityPattern(
    const std::vector<std::set<int>>& adjacency_graph,
    const std::vector<int>& elimination_ordering);

std::vector<std::vector<int>> GetFillInGraph(
    int num_verts, const std::vector<Vector4<int>>& cliques);

/* Sparse cholesky solver where the blocks are of size 3x3. */
class BlockSparseCholeskySolver {
 public:
  /* @param sparsity_pattern Specifies the sparsity pattern of the matrix. */
  explicit BlockSparseCholeskySolver(
      std::vector<std::vector<int>> sparsity_pattern);

  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BlockSparseCholeskySolver);

  SymmetricBlockSparseMatrix<double>& GetMutableMatrix() { return L_; }

  /* If the SchurComplement is S = A - BᵀD⁻¹B, `num_eliminated_blocks` is the
   block columns in D. */
  MatrixX<double> CalcSchurComplement(int num_eliminated_blocks);
  MatrixX<double> CalcSchurComplementAndFactor(int num_eliminated_blocks);

  void Factor() {
    DRAKE_DEMAND(!is_factored_);
    FactorImpl(0, block_cols_);
  }

  int size() const { return block_cols_ * 3; }

  void SolveInPlace(VectorX<double>* y) const;

  VectorX<double> Solve(const VectorX<double>& y) const;

 private:
  void FactorImpl(int starting_col_block, int ending_col_block);

  /* Performs L(j+1:, j+1:) -= L(j+1:,j) * L(j+1:,j).transpose().
   @pre 0 <= j < block_cols_. */
  void RightLookingSymmetricRank1Update(int j);

  int block_cols_{0};
  SymmetricBlockSparseMatrix<double> L_;
  std::vector<Matrix3<double>> L_diag_;
  bool is_factored_{false};
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
