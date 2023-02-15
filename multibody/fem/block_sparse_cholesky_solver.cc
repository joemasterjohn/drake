#include "drake/multibody/fem/block_sparse_cholesky_solver.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <cholmod.h>

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

using std::set;
using std::unordered_set;
using Vector4i = Vector4<int>;
using contact_solvers::internal::PartialPermutation;
using std::vector;

namespace {

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

vector<set<int>> BuildAdjacencyGraph(int num_verts,
                                     const vector<Vector4i>& elements) {
  vector<set<int>> adj(num_verts);
  for (const Vector4i& e : elements) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (e(i) >= e(j)) {
          adj[e(j)].insert(e(i));
        }
      }
    }
  }
  return adj;
}

vector<int> CalcEliminationOrdering(const vector<set<int>>& adjacency_graph) {
  /* Size of the matrix. */
  const int N = adjacency_graph.size();
  int nnz = 0;
  for (const auto& col : adjacency_graph) {
    nnz += col.size();
  }
  /* Build a matrix A for symbolic analysis. */
  cholmod_common cm;
  cholmod_start(&cm);
  auto A = std::unique_ptr<cholmod_sparse>(cholmod_allocate_sparse(
      N, N,
      /* max number of nonzeros */
      nnz,
      /* sorted */ true,
      /* packed */ true,
      /* ignore top right corner */ -1, CHOLMOD_REAL, &cm));
  /* Fill out A->i (inner index), A->p (outer index), and A->x (values). See
   CHOLMOD guide for definitions. */
  int value_index = 0; /* index for nonzero values */
  int* Ap = static_cast<int*>(A->p);
  int* Ai = static_cast<int*>(A->i);
  int* Ax = static_cast<int*>(A->x);
  Ap[0] = 0;
  for (int col = 0; col < static_cast<int>(adjacency_graph.size()); ++col) {
    Ap[col + 1] = Ap[col] + adjacency_graph[col].size();
    for (const int r : adjacency_graph[col]) {
      Ai[value_index] = r;
      /* We set the value to an arbitrary dummy value because we only need to
       perform symbolic analysis. */
      Ax[value_index] = 1.0;
      ++value_index;
    }
  }
  auto L = std::unique_ptr<cholmod_factor>(cholmod_analyze(A.get(), &cm));
  std::vector<int> permutation(N);
  memcpy(permutation.data(), L->Perm,
         permutation.size() * sizeof(permutation[0]));
  /* Clean up memory */
  auto* A_ptr = A.release();
  cholmod_free_sparse(&A_ptr, &cm);
  auto* L_ptr = L.release();
  cholmod_free_factor(&L_ptr, &cm);
  cholmod_finish(&cm);

  return permutation;
}

vector<int> RestrictOrdering(const vector<int>& perm,
                             const vector<int>& D_indices) {
  const int N = perm.size();
  /* perm maps new index into old index. */
  unordered_set<int> D_set;
  for (int d : D_indices) {
    D_set.insert(d);
  }

  /* The new index for non-participating vertices. */
  int pd = 0;
  /* The new index participating. */
  int pa = D_indices.size();
  /* The resulting permutation that respects participation mapping from old to
   new. */
  vector<int> old_to_new(N);
  for (int i = 0; i < N; ++i) {
    int old_index = perm[i];
    if (D_set.count(old_index) > 0) {
      old_to_new[old_index] = pd++;
    } else {
      old_to_new[old_index] = pa++;
    }
  }
  /* We want to return the permutation mapping from new to old*/
  vector<int> result(N);
  for (int i = 0; i < N; ++i) {
    result[old_to_new[i]] = i;
  }
  return result;
}

std::vector<int> CalcEliminationOrdering(
    const std::vector<std::set<int>>& adjacency_graph,
    const vector<int>& D_indices) {
  unordered_set<int> D_set;
  for (int d : D_indices) {
    D_set.insert(d);
  }
  /* Size of A, D matrices and upper bound on the number of nonzeros. */
  const int N = adjacency_graph.size();
  const int ND = D_indices.size();
  const int NA = N - ND;
  /* Build the index mapping from the global adjacency graph to local A and D
   graphs as well as the inverse mapping. */
  int A_index = 0;
  int D_index = 0;
  std::vector<int> global_to_local(N);
  std::vector<int> local_to_global_A(NA);
  std::vector<int> local_to_global_D(ND);
  for (int i = 0; i < N; ++i) {
    if (D_set.count(i) > 0) {
      global_to_local[i] = D_index;
      local_to_global_D[D_index] = i;
      ++D_index;
    } else {
      global_to_local[i] = A_index;
      local_to_global_A[A_index] = i;
      ++A_index;
    }
  }

  /* Build the adjacency graph of A and D. */
  std::vector<std::set<int>> adjacency_graph_A(NA);
  std::vector<std::set<int>> adjacency_graph_D(ND);
  for (int i = 0; i < N; ++i) {
    if (D_set.count(i) > 0) {
      for (int j : adjacency_graph[i]) {
        if (D_set.count(j) > 0) {
          adjacency_graph_D[global_to_local[i]].insert(global_to_local[j]);
        }
      }
    } else {
      for (int j : adjacency_graph[i]) {
        if (D_set.count(j) == 0) {
          adjacency_graph_A[global_to_local[i]].insert(global_to_local[j]);
        }
      }
    }
  }

  const std::vector<int> D_ordering =
      CalcEliminationOrdering(adjacency_graph_D);
  const std::vector<int> A_ordering =
      CalcEliminationOrdering(adjacency_graph_A);
  std::vector<int> result;
  result.reserve(N);
  /* D comes before A. */
  for (int d : D_ordering) {
    result.emplace_back(local_to_global_D[d]);
  }
  for (int a : A_ordering) {
    result.emplace_back(local_to_global_A[a]);
  }
  return result;
}

std::vector<std::vector<int>> CalcSparsityPattern(
    const std::vector<std::set<int>>& adjacency_graph,
    const std::vector<int>& elimination_ordering) {
  int N = elimination_ordering.size();
  DRAKE_DEMAND(static_cast<int>(adjacency_graph.size()) == N);
  std::vector<int> old_to_new(N);
  for (int i = 0; i < N; ++i) {
    old_to_new[elimination_ordering[i]] = i;
  }

  /* Computes the adjacency graph for the new ordering. */
  std::vector<std::set<int>> new_graph(N);
  for (int i = 0; i < N; ++i) {
    for (int v : adjacency_graph[i]) {
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

std::vector<std::vector<int>> GetFillInGraph(
    int num_verts, const std::vector<Vector4<int>>& cliques) {
  std::vector<std::unordered_set<int>> fill_in_graph(num_verts);
  const auto original_graph = BuildAdjacencyGraph(num_verts, cliques);
  int i = 0;
  for (const auto& neighbors : original_graph) {
    fill_in_graph[i].reserve(num_verts);
    for (const int& n : neighbors) {
      if (i <= n) fill_in_graph[i].insert(n);
    }
    ++i;
  }

  // Add fill-ins.
  for (int v = 0; v < num_verts; ++v) {
    const auto& one_ring = fill_in_graph[v];
    for (int n1 : one_ring) {
      if (n1 <= v) continue;
      for (int n2 : one_ring) {
        if (n1 > v && n2 > v) {
          fill_in_graph[std::min(n1, n2)].insert(std::max(n1, n2));
        }
      }
    }
  }

  // Turn set into vector.
  std::vector<std::vector<int>> results(num_verts);
  for (int v = 0; v < num_verts; ++v) {
    for (int n : fill_in_graph[v]) {
      if (n >= v) results[v].push_back(n);
    }
  }
  return results;
}

/* Given an input matrix M, and a permutation mapping e, sets the resulting
 matrix M̃ such that M̃(i, j) = M(e(i), e(j)). */
void PermuteSymmetricBlockSparseMatrix(
    const SymmetricBlockSparseMatrix<double>& input,
    const std::vector<int>& permutation,
    SymmetricBlockSparseMatrix<double>* result) {
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
      const Matrix3<double>& block = input.get_block(i, j);
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

void BlockSparseCholeskySolver::SetMatrix(
    const SymmetricBlockSparseMatrix<double>& A) {
  const std::vector<std::set<int>> adj = A.CalcAdjacencyGraph();
  std::vector<int> elimination_ordering = CalcEliminationOrdering(adj);
  SetMatrixImpl(A, adj, elimination_ordering);
}

void BlockSparseCholeskySolver::SetMatrix(
    const SymmetricBlockSparseMatrix<double>& A,
    const std::vector<int>& elimination_ordering) {
  const std::vector<std::set<int>> adj = A.CalcAdjacencyGraph();
  SetMatrixImpl(A, adj, elimination_ordering);
}

void BlockSparseCholeskySolver::UpdateMatrix(
    const SymmetricBlockSparseMatrix<double>& A) {
  PermuteSymmetricBlockSparseMatrix(A, internal_to_original_.permutation(),
                                    &L_);
  is_factored_ = false;
}

MatrixX<double> BlockSparseCholeskySolver::CalcSchurComplementAndFactor(
    const SymmetricBlockSparseMatrix<double>& M,
    const std::vector<int>& eliminated_block_indices) {
  const int D_size = eliminated_block_indices.size();
  // if (D_size == 0) return M.MakeDenseMatrix();
  /* The number of block columns in M. */
  const int N = M.cols() / 3;
  const int A_size = N - D_size;
  DRAKE_DEMAND(A_size >= 0);
  // if (A_size == 0) return MatrixX<double>::Zero(0, 0);

  const std::vector<std::set<int>> adjacency_graph = M.CalcAdjacencyGraph();
  /* Ensure that the eliminated blocks appear first in the elimination order. */
  const std::vector<int> elimination_order = RestrictOrdering(
      CalcEliminationOrdering(adjacency_graph), eliminated_block_indices);
  SetMatrixImpl(M, adjacency_graph, elimination_order);

  std::unordered_set<int> D_set(eliminated_block_indices.begin(),
                                eliminated_block_indices.end());
  /* Maps the global index of a vertex to its implied index within a group (A or
   D). For example, suppose D_indices are {0, 2, 5}, and A_indices are {1, 3, 4,
   6}, then the implied indices for D vertices are 0 -> 0, 2 -> 1, 5 -> 2, and
   the implied indices for A vertices are 1 -> 0, 3 -> 1, 4 -> 2, 6 -> 3, and
   the global_to_local mapping is {0, 0, 1, 1, 2, 2, 3}. */
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

  /* The permutation of A indices induced by the elimination ordering. Note
   that A_vertex_permutation[new_index] = old_index. */
  std::vector<int> A_vertex_permutation;
  A_vertex_permutation.reserve(A_size);
  for (int i = 0; i < static_cast<int>(elimination_order.size()); ++i) {
    const int old_index = elimination_order[i];
    if (D_set.count(old_index) == 0) {
      A_vertex_permutation.emplace_back(global_to_local[old_index]);
    }
  }

  const int num_eliminated_blocks = D_size;
  FactorImpl(0, num_eliminated_blocks);
  const MatrixX<double> permuted_S =
      L_.MakeDenseBottomRightCorner(block_cols_ - num_eliminated_blocks);
  FactorImpl(num_eliminated_blocks, block_cols_);
  MatrixX<double> S;
  S.resizeLike(permuted_S);
  InversePermuteDenseMatrix(permuted_S, A_vertex_permutation, &S);
  return S;
}

void BlockSparseCholeskySolver::SolveInPlace(VectorX<double>* y) const {
  DRAKE_DEMAND(is_factored_);
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(y->size() == size());
  VectorX<double> permuted_y(*y);
  internal_to_original_dof_.ApplyInverse(*y, &permuted_y);

  /* Solve Lz = y in place. */
  for (int j = 0; j < block_cols_; ++j) {
    /* Solve for the j-th block entry. */
    permuted_y.segment<3>(3 * j) =
        L_diag_[j].triangularView<Eigen::Lower>().solve(
            permuted_y.segment<3>(3 * j));
    const auto& yj = permuted_y.segment<3>(3 * j);
    /* Eliminate for the j-th block entry from the system. */
    const auto& blocks_in_col_j = L_.get_col_blocks(j);
    for (int flat = 1; flat < static_cast<int>(blocks_in_col_j.size());
         ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_y.segment<3>(3 * i) -= L_.get_block(i, j) * yj;
    }
  }

  VectorX<double>& permuted_z = permuted_y;
  /* Solve Lᵀx = z in place. */
  for (int j = block_cols_ - 1; j >= 0; --j) {
    /* Eliminate all solved variables. */
    const auto& blocks_in_col_j = L_.get_col_blocks(j);
    for (int flat = 1; flat < static_cast<int>(blocks_in_col_j.size());
         ++flat) {
      const int i = blocks_in_col_j[flat];
      permuted_z.segment<3>(3 * j) -=
          L_.get_block(i, j).transpose() * permuted_z.segment<3>(3 * i);
    }
    /* Solve for the j-th block entry. */
    permuted_z.segment<3>(3 * j) =
        L_diag_[j].transpose().triangularView<Eigen::Upper>().solve(
            permuted_z.segment<3>(3 * j));
  }
  internal_to_original_dof_.Apply(permuted_z, y);
}

VectorX<double> BlockSparseCholeskySolver::Solve(
    const VectorX<double>& y) const {
  VectorX<double> x(y);
  SolveInPlace(&x);
  return x;
}

void BlockSparseCholeskySolver::SetMatrixImpl(
    const SymmetricBlockSparseMatrix<double>& A,
    const std::vector<std::set<int>>& adjacency_graph,
    const std::vector<int>& elimination_ordering) {
  DRAKE_DEMAND(elimination_ordering.size() == adjacency_graph.size());
  block_cols_ = elimination_ordering.size();
  vector<int> dof_permutation(elimination_ordering.size() * 3);
  for (int i = 0; i < static_cast<int>(elimination_ordering.size()); ++i) {
    for (int d = 0; d < 3; ++d) {
      dof_permutation[3 * i + d] = 3 * elimination_ordering[i] + d;
    }
  }
  internal_to_original_ = PartialPermutation(move(elimination_ordering));
  internal_to_original_dof_ = PartialPermutation(move(dof_permutation));

  L_diag_.resize(block_cols_);

  std::vector<std::vector<int>> sparsitiy_pattern =
      CalcSparsityPattern(adjacency_graph, internal_to_original_.permutation());
  L_ = SymmetricBlockSparseMatrix<double>(std::move(sparsitiy_pattern));
  UpdateMatrix(A);
}

void BlockSparseCholeskySolver::FactorImpl(int starting_col_block,
                                           int ending_col_block) {
  for (int j = starting_col_block; j < ending_col_block; ++j) {
    /* Update diagonal. */
    const Matrix3<double>& Ajj = L_.get_diagonal_block(j);
    const auto llt = Eigen::LLT<Matrix3<double>>(Ajj);
    DRAKE_DEMAND(llt.info() == Eigen::Success);
    L_diag_[j] = llt.matrixL();
    /* Technically, there's no need to spell out the diagonal block of the L
     matrix, but we do it here for easy debugging. */
    L_.SetBlockFlat(0, j, L_diag_[j]);

    /* Update column.
     | a₁₁  *  | = | λ₁₁  0 | * | λ₁₁ᵀ L₂₁ᵀ |
     | a₂₁ a₂₂ |   | L₂₁ L₂₂|   |  0   L₂₂ᵀ |
     So we have
      L₂₁λ₁₁ᵀ = a₂₁, and thus
      λ₁₁L₂₁ᵀ = a₂₁ᵀ */
    const std::vector<int>& blocks_in_col_j = L_.get_col_blocks(j);
    for (int a = 0; a < static_cast<int>(blocks_in_col_j.size()) - 1; ++a) {
      const int flat = a + 1;
      const auto& L_diag_j = L_diag_[j].triangularView<Eigen::Lower>();
      const Matrix3<double>& Aij = L_.get_block_flat(flat, j);
      Matrix3<double> Lij = L_diag_j.solve(Aij.transpose()).transpose();
      L_.SetBlockFlat(flat, j, std::move(Lij));
    }
    RightLookingSymmetricRank1Update(j);
  }
  is_factored_ = true;
}

void BlockSparseCholeskySolver::RightLookingSymmetricRank1Update(int j) {
  const std::vector<int>& blocks_in_col_j = L_.get_col_blocks(j);
  const int N = blocks_in_col_j.size();
  /* The following omp parallel for loop is equivalent to this easier to read
   non-openmp compliant for loop. */
  /*
   // We start from f1 = 1 here to skip the j,j entry.
   for (int f1 = 1; f1 < N; ++f1) {
     const int col = blocks_in_col_j[f1];
     const Matrix3<double>& B = L_.get_block_flat(f1, j);
     for (int f2 = f1; f2 < N; ++f2) {
       const int row = blocks_in_col_j[f2];
       const Matrix3<double>& A = L_.get_block_flat(f2, j);
       L_.SubtractProductFromBlock(row, col, A, B);
     }
   }
  */
#if defined(_OPENMP)
#pragma omp parallel for num_threads(12)
#endif
  for (int a = 0; a < N - 1; ++a) {
    const int f1 = a + 1;
    const int col = blocks_in_col_j[f1];
    const Matrix3<double>& B = L_.get_block_flat(f1, j);
    for (int f2 = f1; f2 < N; ++f2) {
      const int row = blocks_in_col_j[f2];
      const Matrix3<double>& A = L_.get_block_flat(f2, j);
      L_.SubtractProductFromBlock(row, col, A, B);
    }
  }
  /*
    std::vector<std::array<int, 2>> index_map;
    index_map.reserve(N * N / 2);
    for (int f1 = 1; f1 < N; ++f1) {
      for (int f2 = f1; f2 < N; ++f2) {
        index_map.push_back({f1, f2});
      }
    }
  #if defined(_OPENMP)
  #pragma omp parallel for num_threads(12)
  #endif
    for (int i = 0; i < static_cast<int>(index_map.size()); ++i) {
      const int f1 = index_map[i][0];
      const int f2 = index_map[i][1];
      const int col = blocks_in_col_j[f1];
      const int row = blocks_in_col_j[f2];
      const Matrix3<double>& B = L_.get_block_flat(f1, j);
      const Matrix3<double>& A = L_.get_block_flat(f2, j);
      L_.SubtractProductFromBlock(row, col, A, B);
    }
  */
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
