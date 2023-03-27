#include "drake/multibody/fem/block_sparse_cholesky_solver.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace {

using std::set;
using Vector4i = Vector4<int>;
using contact_solvers::internal::PartialPermutation;
using std::vector;

/* Computes a union b. */
std::vector<int> Union(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> result;
  result.reserve(a.size() + b.size());
  std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                 std::back_inserter(result));
  return result;
}

/* Computes a\b. */
std::vector<int> SetDifference(const std::vector<int>& a,
                               const std::vector<int>& b) {
  std::vector<int> result;
  result.reserve(a.size());
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::back_inserter(result));
  return result;
}

void RemoveValueFromSortedVector(int value, std::vector<int>* sorted_vector) {
  auto it =
      std::lower_bound(sorted_vector->begin(), sorted_vector->end(), value);
  /* Check if the value was found and remove it. */
  if (it != sorted_vector->end() && *it == value) {
    sorted_vector->erase(it);
  }
}

void InsertValueInSortedVector(int value, std::vector<int>* sorted_vector) {
  auto it =
      std::lower_bound(sorted_vector->begin(), sorted_vector->end(), value);
  /* Check if the value doesn't already exist and insert it. */
  if (it == sorted_vector->end() || *it != value) {
    sorted_vector->insert(it, value);
  }
}

/* Notation taken from Algorithm 1 in An Approximate Minimum Degree Ordering
 Algorithm. */
struct Node {
  void UpdateExternalDegree(const std::vector<Node>& nodes) {
    degree = 0;
    for (int a : A) {
      degree += nodes[a].size;
    }
    std::vector<int> L_union;
    for (int e : E) {
      L_union = Union(L_union, nodes[e].L);
    }
    RemoveValueFromSortedVector(index, &L_union);
    for (int l : L_union) {
      degree += nodes[l].size;
    }
  }

  int degree{0};
  int size{0};
  int index{-1};
  /* A, E, L are sorted. */
  std::vector<int> A;
  std::vector<int> E;
  std::vector<int> L;
};

/* A simplified version of Node. Only contains index of the node and the degree.
 */
struct IndexDegree {
  int degree;
  int index;
};

bool operator>(const IndexDegree& a, const IndexDegree& b) {
  if (a.degree != b.degree)
    return a.degree > b.degree;
  else
    return a.index > b.index;
}

/* Computes the elimination of the matrix with the given `adjacency_graph` that
 CHOLMOD thinks is the best. For example if the result is [1, 3, 0, 2], it means
 that we should first eliminate vertex 1, then 3, 0, and 2. In other words, this
 is a permutation mapping from new vertex indices to old indices. */
std::vector<int> CalcEliminationOrdering(
    const BlockSparsityPattern& block_sparsity_pattern) {
  /* Intialize for AMD algorithm. */
  const std::vector<int>& block_sizes = block_sparsity_pattern.block_sizes();
  int num_nodes = block_sizes.size();
  std::vector<Node> nodes(num_nodes);
  for (int n = 0; n < num_nodes; ++n) {
    nodes[n].index = n;
    nodes[n].size = block_sizes[n];
  }
  for (int n = 0; n < num_nodes; ++n) {
    Node& node = nodes[n];
    std::vector<int> neighbors = block_sparsity_pattern.neighbors()[n];
    for (int neighbor : neighbors) {
      if (n != neighbor) {
        // We modify both `node` and `neighbor` here because for each ij-pair
        // only one of (i, j) and (j, i) is recorded in `sparsity pattern` but
        // we want j be part of Ai and i to be part of Aj.
        node.A.emplace_back(neighbor);
        node.degree += nodes[neighbor].size;
        nodes[neighbor].A.emplace_back(n);
        nodes[neighbor].degree += node.size;
      }
    }
  }
  /* Maintain the invariance that A, E, L vectors in a node are all sorted. */
  for (auto& node : nodes) {
    std::vector<int>& A = node.A;
    /* Only A is non-empty at this point. */
    std::sort(A.begin(), A.end());
  }

  std::priority_queue<IndexDegree, std::vector<IndexDegree>,
                      std::greater<IndexDegree>>
      queue;
  // Put all nodes in the priority queue.
  for (int n = 0; n < num_nodes; ++n) {
    IndexDegree node = {.degree = nodes[n].degree, .index = n};
    queue.emplace(node);
  }

  std::vector<int> result(num_nodes);
  /* Begin elimination. */
  for (int k = 0; k < num_nodes; ++k) {
    IndexDegree index_degree = queue.top();
    /* Pop stale priority queue elements because std::priority_queue can't
     replace nodes. */
    while (index_degree.degree != nodes[index_degree.index].degree) {
      queue.pop();
      index_degree = queue.top();
    }
    queue.pop();
    int p = index_degree.index;
    result[k] = p;
    Node& node_p = nodes[p];
    /* Form Lp */
    node_p.L = node_p.A;
    for (int e : node_p.E) {
      node_p.L = Union(node_p.L, nodes[e].L);
    }
    RemoveValueFromSortedVector(p, &(node_p.L));
    for (int i : node_p.L) {
      Node& node_i = nodes[i];
      // Remove redundant entries.
      node_i.A = SetDifference(node_i.A, node_p.L);
      RemoveValueFromSortedVector(p, &(node_i.A));
      // Element absorption.
      node_i.E = SetDifference(node_i.E, node_p.E);
      InsertValueInSortedVector(p, &(node_i.E));
      // Compute external degree.
      node_i.UpdateExternalDegree(nodes);

      IndexDegree new_node = {.degree = node_i.degree, .index = node_i.index};
      queue.emplace(new_node);
    }
    // Convert node_p to element p.
    node_p.A.clear();
    node_p.E.clear();
    node_p.degree = -1;
  }

  return result;
}

/* Returns the column-wise sparsity pattern of L given the adjacency graph of A
 and the elimination ordering.
 Note that the elimination ordering is a mapping from new index to old index. */
std::vector<std::vector<int>> CalcSparsityPattern(
    const std::vector<std::vector<int>>& sparsity_pattern,
    const std::vector<int>& elimination_ordering) {
  int N = elimination_ordering.size();
  DRAKE_DEMAND(static_cast<int>(sparsity_pattern.size()) == N);
  std::vector<int> old_to_new(N);
  for (int i = 0; i < N; ++i) {
    old_to_new[elimination_ordering[i]] = i;
  }

  /* Computes the adjacency graph for the new ordering. */
  std::vector<std::set<int>> new_graph(N);
  for (int i = 0; i < N; ++i) {
    for (int v : sparsity_pattern[i]) {
      int a = old_to_new[i];
      int b = old_to_new[v];
      if (a >= b) {
        new_graph[b].insert(a);
      } else {
        new_graph[a].insert(b);
      }
    }
  }

  /* children[p] is a vector of sort children of p. */
  std::vector<std::vector<int>> children(N);
  std::vector<std::vector<int>> result(N);
  for (int i = 0; i < N; ++i) {
    /* Turn set into vector. */
    const std::set<int>& neighbor_i = new_graph[i];
    result[i].reserve(N);
    for (int n : neighbor_i) result[i].emplace_back(n);

    /* Merge the neighbors of i and all neighbors of children of i. */
    const auto& children_i = children[i];
    std::vector<int> result_i;  // Temp variable to hold result[i] as we
                                // accumulate all values.
    result_i.reserve(N);
    for (int c : children_i) {
      const auto& neighbor_c = result[c];
      std::set_union(result[i].begin(), result[i].end(), neighbor_c.begin() + 2,
                     neighbor_c.end(), std::back_inserter(result_i));
      result_i.swap(result[i]);
      result_i.clear();
    }
    /* Record the parent of i if i isn't already the root. */
    if (result[i].size() > 1) {
      const int p = result[i][1];
      children[p].emplace_back(i);
    }
  }
  return result;
}

/* Given an input matrix M, and a permutation mapping e, sets the resulting
 matrix M̃ such that M̃(i, j) = M(e(i), e(j)). */
void PermuteTriangularBlockSparseMatrix(
    const TriangularBlockSparseMatrix<double>& input,
    const std::vector<int>& permutation,
    TriangularBlockSparseMatrix<double>* result) {
  DRAKE_DEMAND(result != nullptr);
  const int N = permutation.size();
  DRAKE_DEMAND(N == input.block_cols());
  result->SetZero();

  /* Construct the inverse mapping of e, f. */
  std::vector<int> inverse_permutation(permutation.size());
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    inverse_permutation[permutation[i]] = i;
  }
  /* M̃(i, j) = M(e(i), e(j)) is equivalent to M̃(f(i), f(j)) = M(i, j). */
  for (int j = 0; j < N; ++j) {
    const std::vector<int>& row_indices = input.block_row_indices(j);
    for (int i : row_indices) {
      const MatrixX<double>& block = input.block(i, j);
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
}  // namespace

void BlockSparseCholeskySolver::SetMatrix(
    const TriangularBlockSparseMatrix<double>& A) {
  const BlockSparsityPattern& A_block_pattern = A.sparsity_pattern();
  const std::vector<int> elimination_ordering =
      CalcEliminationOrdering(A_block_pattern);
  block_cols_ = elimination_ordering.size();
  vector<int> scalar_permutation(A.cols());
  const std::vector<int>& A_block_sizes = A_block_pattern.block_sizes();
  const std::vector<int>& starting_indices = A.starting_cols();
  int new_scalar_index = 0;
  for (int i = 0; i < static_cast<int>(elimination_ordering.size()); ++i) {
    const int block = elimination_ordering[i];
    const int start = starting_indices[block];
    const int size = A_block_sizes[block];
    for (int s = start; s < start + size; ++s) {
      scalar_permutation[new_scalar_index++] = s;
    }
  }
  block_index_permutation_ = PartialPermutation(elimination_ordering);
  scalar_index_permutation_ = PartialPermutation(move(scalar_permutation));

  L_diag_.resize(block_cols_);

  std::vector<std::vector<int>> L_sparsity_pattern =
      CalcSparsityPattern(A_block_pattern.neighbors(), elimination_ordering);
  std::vector<int> L_block_sizes(A.block_cols());
  block_index_permutation_.ApplyInverse(A_block_sizes, &L_block_sizes);
  BlockSparsityPattern L_block_pattern(std::move(L_block_sizes),
                                       std::move(L_sparsity_pattern));
  L_ = std::make_unique<TriangularBlockSparseMatrix<double>>(
      std::move(L_block_pattern), false);
  UpdateMatrix(A);
}

void BlockSparseCholeskySolver::UpdateMatrix(
    const TriangularBlockSparseMatrix<double>& A) {
  PermuteTriangularBlockSparseMatrix(A, block_index_permutation_.permutation(),
                                     L_.get_mutable());
  is_factored_ = false;
}

void BlockSparseCholeskySolver::SolveInPlace(VectorX<double>* y) const {
  DRAKE_DEMAND(is_factored_);
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(y->size() == L_->cols());
  VectorX<double> permuted_y(*y);
  scalar_index_permutation_.ApplyInverse(*y, &permuted_y);

  const BlockSparsityPattern& block_sparsity_pattern = L_->sparsity_pattern();
  const std::vector<int>& block_sizes = block_sparsity_pattern.block_sizes();
  const std::vector<int>& starting_cols = L_->starting_cols();

  /* Solve Lz = y in place. */
  for (int j = 0; j < block_cols_; ++j) {
    const int block_size = block_sizes[j];
    const int offset = starting_cols[j];
    /* Solve for the j-th block entry. */
    permuted_y.segment(offset, block_size) =
        L_diag_[j].matrixL().solve(permuted_y.segment(offset, block_size));
    const auto& yj = permuted_y.segment(offset, block_size);
    /* Eliminate for the j-th block entry from the system. */
    const auto& blocks_in_col_j = L_->block_row_indices(j);
    for (int flat = 1; flat < static_cast<int>(blocks_in_col_j.size());
         ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_y.segment(starting_cols[i], block_sizes[i]).noalias() -=
          L_->block(i, j) * yj;
    }
  }

  VectorX<double>& permuted_z = permuted_y;
  /* Solve Lᵀx = z in place. */
  for (int j = block_cols_ - 1; j >= 0; --j) {
    /* Eliminate all solved variables. */
    const auto& blocks_in_col_j = L_->block_row_indices(j);
    for (int flat = 1; flat < static_cast<int>(blocks_in_col_j.size());
         ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_z.segment(starting_cols[j], block_sizes[j]).noalias() -=
          L_->block(i, j).transpose() *
          permuted_z.segment(starting_cols[i], block_sizes[i]);
    }
    /* Solve for the j-th block entry. */
    permuted_z.segment(starting_cols[j], block_sizes[j]) =
        L_diag_[j].matrixU().solve(
            permuted_z.segment(starting_cols[j], block_sizes[j]));
  }
  scalar_index_permutation_.Apply(permuted_z, y);
}

VectorX<double> BlockSparseCholeskySolver::Solve(
    const VectorX<double>& y) const {
  VectorX<double> x(y);
  SolveInPlace(&x);
  return x;
}

void BlockSparseCholeskySolver::Factor() {
  for (int j = 0; j < L_->block_cols(); ++j) {
    /* Update diagonal. */
    const MatrixX<double>& Ajj = L_->diagonal_block(j);
    L_diag_[j].compute(Ajj);
    DRAKE_DEMAND(L_diag_[j].info() == Eigen::Success);
    /* Technically, we should have
      L_->SetBlockFlat(0, j, L_diag_[j]);
    for a complete L_ matrix, but we omit it here since we don't ever touch the
    diagonal block of L_. Instead, we directly invoke L_diag_. */

    /* Update column.
     | a₁₁  *  | = | λ₁₁  0 | * | λ₁₁ᵀ L₂₁ᵀ |
     | a₂₁ a₂₂ |   | L₂₁ L₂₂|   |  0   L₂₂ᵀ |
     So we have
      L₂₁λ₁₁ᵀ = a₂₁, and thus
      λ₁₁L₂₁ᵀ = a₂₁ᵀ */
    const std::vector<int>& row_blocks = L_->block_row_indices(j);
    for (int a = 0; a < static_cast<int>(row_blocks.size()) - 1; ++a) {
      const int flat = a + 1;
      const MatrixX<double>& Aij = L_->block_flat(flat, j);
      MatrixX<double>& Lij = L_->mutable_block_flat(flat, j);
      Lij.noalias() = L_diag_[j].matrixL().solve(Aij.transpose()).transpose();
    }
    RightLookingSymmetricRank1Update(j);
  }
  is_factored_ = true;
}

void BlockSparseCholeskySolver::RightLookingSymmetricRank1Update(int j) {
  const std::vector<int>& blocks_in_col_j = L_->block_row_indices(j);
  const int N = blocks_in_col_j.size();
  /* The following omp parallel for loop is equivalent to this easier to read
   non-openmp compliant for loop. */
  /*
   // We start from f1 = 1 here to skip the j,j entry.
   for (int f1 = 1; f1 < N; ++f1) {
     const int col = blocks_in_col_j[f1];
     const Matrix3<double>& B = L_->block_flat(f1, j);
     for (int f2 = f1; f2 < N; ++f2) {
       const int row = blocks_in_col_j[f2];
       const Matrix3<double>& A = L_->block_flat(f2, j);
       L_->SubtractProductFromBlock(row, col, A, B);
     }
   }
  */
#if defined(_OPENMP)
#pragma omp parallel for num_threads(12)
#endif
  for (int a = 0; a < N - 1; ++a) {
    const int f1 = a + 1;
    const int col = blocks_in_col_j[f1];
    const MatrixX<double>& B = L_->block_flat(f1, j);
    for (int f2 = f1; f2 < N; ++f2) {
      const int row = blocks_in_col_j[f2];
      const MatrixX<double>& A = L_->block_flat(f2, j);
      L_->SubtractProductFromBlock(row, col, A, B);
    }
  }
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
