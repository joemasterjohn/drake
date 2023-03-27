#include "drake/multibody/contact_solvers/supernodal_solver2.h"

#include <algorithm>
#include <set>
#include <utility>

using Eigen::MatrixXd;
using std::vector;
using MatrixBlock = std::pair<Eigen::MatrixXd, std::vector<int>>;
using MatrixBlocks = std::vector<MatrixBlock>;
using drake::multibody::fem::internal::BlockSparseCholeskySolver;
using drake::multibody::fem::internal::BlockSparsityPattern;
using drake::multibody::fem::internal::TriangularBlockSparseMatrix;

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

// Returns a vector<vector<int>> y where y[i] contains the indices of the block
// Jacobians on row i organized by column.
vector<std::vector<int>> GetRowToTripletMapping(
    int num_row_blocks,
    const std::vector<BlockMatrixTriplet>& jacobian_blocks) {
  vector<vector<int>> y(num_row_blocks);
  vector<vector<int>> column(num_row_blocks);

  // Sorts row data by column using our assumption that at most two columns
  // are non-zero per row.
  auto sort_row_data_by_column = [](vector<int>* col_index,
                                    vector<int>* triplet_position) {
    if ((*col_index)[0] > (*col_index)[1]) {
      std::swap((*col_index)[0], (*col_index)[1]);
      std::swap((*triplet_position)[0], (*triplet_position)[1]);
    }
  };

  int cnt = 0;
  for (const auto& j : jacobian_blocks) {
    int index = std::get<0>(j);
    y[index].emplace_back(cnt);
    column[index].emplace_back(std::get<1>(j));
    if (column[index].size() == 2) {
      sort_row_data_by_column(&column[index], &y[index]);
    }
    if (column[index].size() > 2) {
      throw std::runtime_error(
          "Jacobian can only be nonzero on at most two column blocks.");
    }
    ++cnt;
  }
  return y;
}

void VerifyMassMatrixPartitionRefinesJacobianPartition(
    const std::vector<int>& jacobian_column_block_size,
    const vector<MatrixXd>& mass_matrices) {
  int size = 0;
  size_t block_column = 0;
  for (size_t i = 0; i < mass_matrices.size(); i++) {
    // Check if too many mass matrices
    if (block_column >= jacobian_column_block_size.size()) {
      throw std::runtime_error(
          "The dimensions of the mass matrix and Jacobian are incompatible. "
          "The mass matrix has more scalar columns than the Jacobian.");
    }

    size += mass_matrices[i].cols();
    // Check if mass matrix overlaps two jacobian blocks.
    if (size > jacobian_column_block_size[block_column]) {
      throw std::runtime_error(
          "Column partition induced by mass matrix must refine the partition "
          "induced by the Jacobian.");
    }

    // Advance to next block if refinement of current block found.
    if (size == jacobian_column_block_size[block_column]) {
      ++block_column;
      size = 0;
    }
  }

  // Verify all jacobian blocks are refined.
  if (block_column < jacobian_column_block_size.size()) {
    throw std::runtime_error(
        "The dimensions of the mass matrix and Jacobian are incompatible. The "
        "Jacobian has more scalar columns than the mass matrix.");
  }
}

std::vector<int> GetJacobianBlockSizesVerifyTriplets(
    const std::vector<BlockMatrixTriplet>& jacobian_blocks) {
  int max_index = -1;
  for (const auto& j : jacobian_blocks) {
    int col_index = std::get<1>(j);
    if (col_index > max_index) {
      max_index = col_index;
    }
  }

  std::vector<int> block_column_size(max_index + 1, -1);

  for (const auto& j : jacobian_blocks) {
    int col_index = std::get<1>(j);

    if ((std::get<2>(j).cols() == 0) || (std::get<2>(j).rows() == 0)) {
      throw std::runtime_error(
          "Invalid Jacobian triplets: a triplet contains an empty matrix");
    }

    if (block_column_size[col_index] == -1) {
      block_column_size[col_index] = std::get<2>(j).cols();
    } else {
      if (block_column_size[col_index] != std::get<2>(j).cols()) {
        throw std::runtime_error(
            "Invalid Jacobian triplets: conflicting block sizes specified for "
            "column " +
            std::to_string(col_index) + ".");
      }
    }
  }

  for (int i = 0; i < max_index; i++) {
    // Verify that initial value of -1 has been overwritten. (A block column
    // size of zero will be caught by the empty matrix check.)
    if (block_column_size[i] < 0) {
      throw std::runtime_error(
          "Invalid Jacobian triplets: no triplet provided for column " +
          std::to_string(i) + ".");
    }
  }

  return block_column_size;
}

}  // namespace

SuperNodalSolver2::~SuperNodalSolver2() {}

SuperNodalSolver2::SuperNodalSolver2(
    int num_jacobian_row_blocks,
    const std::vector<BlockMatrixTriplet>& jacobian_blocks,
    const std::vector<Eigen::MatrixXd>& mass_matrices)
    : mass_matrices_(mass_matrices), jacobian_blocks_(jacobian_blocks) {
  solver_ = std::make_unique<BlockSparseCholeskySolver>();
  std::vector<int> jacobian_column_block_size =
      GetJacobianBlockSizesVerifyTriplets(jacobian_blocks);
  // Will throw an exception if verification fails.
  VerifyMassMatrixPartitionRefinesJacobianPartition(jacobian_column_block_size,
                                                    mass_matrices);
  row_to_triplet_list_ =
      GetRowToTripletMapping(num_jacobian_row_blocks, jacobian_blocks);

  const int num_nodes = mass_matrices.size();
  std::vector<int> block_sizes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    block_sizes[i] = mass_matrices[i].rows();
  }
  // Build diagonal entry in sparsity pattern.
  std::vector<std::set<int>> sparsity(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    sparsity[i].emplace(i);
  }
  // Build off-diagonal entry in sparsity pattern.
  const int num_constraints = row_to_triplet_list_.size();
  for (int r = 0; r < num_constraints; ++r) {
    const std::vector<int>& triplets = row_to_triplet_list_[r];
    DRAKE_DEMAND(triplets.size() <= 2);
    if (triplets.size() == 2) {
      const int j = std::get<1>(jacobian_blocks[triplets[0]]);
      const int i = std::get<1>(jacobian_blocks[triplets[1]]);
      DRAKE_DEMAND(j < i);
      sparsity[j].emplace(i);
    }
  }
  // Turn set into vector.
  std::vector<std::vector<int>> sparsity2(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    sparsity2[i] = std::vector<int>(sparsity[i].begin(), sparsity[i].end());
  }
  BlockSparsityPattern block_sparsity_pattern = {.diagonals = block_sizes,
                                                 .sparsity_pattern = sparsity2};
  A_ = std::make_unique<TriangularBlockSparseMatrix<double>>(
      std::move(block_sparsity_pattern), true);
  solver_->SetMatrix(*A_);
}

void SuperNodalSolver2::SetWeightMatrix(
    const std::vector<Eigen::MatrixXd>& weight_matrix) {
  A_->SetZero();
  // Add mass matrices.
  const int block_cols = mass_matrices_.size();
  for (int i = 0; i < block_cols; ++i) {
    A_->SetBlock(i, i, mass_matrices_[i]);
  }
  // J^T*G*J
  const int num_constraints = row_to_triplet_list_.size();
  DRAKE_DEMAND(static_cast<int>(weight_matrix.size()) >= num_constraints);
  int weight_start = 0;
  int weight_end = 0;
  for (int k = 0; k < num_constraints; ++k) {
    const std::vector<int>& triplets = row_to_triplet_list_[k];
    const int num_constraint_equations =
        std::get<2>(jacobian_blocks_[triplets[0]]).rows();
    int G_rows = 0;
    while (G_rows < num_constraint_equations) {
      G_rows += weight_matrix[weight_end++].rows();
    }
    DRAKE_DEMAND(G_rows == num_constraint_equations);

    if (triplets.size() == 1) {
      const MatrixBlock<double>& J = std::get<2>(jacobian_blocks_[triplets[0]]);
      MatrixBlock<double> GJ = J.LeftMultiplyByBlockDiagonal(
          weight_matrix, weight_start, weight_end - 1);
      MatrixXd JTGJ = MatrixXd::Zero(J.cols(), J.cols());
      J.TransposeAndMultiplyAndAddTo(GJ, &JTGJ);
      int c = std::get<1>(jacobian_blocks_[triplets[0]]);
      A_->AddToBlock(c, c, std::move(JTGJ));
    } else {
      DRAKE_DEMAND(triplets.size() == 2);
      const int j = std::get<1>(jacobian_blocks_[triplets[0]]);
      const int i = std::get<1>(jacobian_blocks_[triplets[1]]);
      DRAKE_DEMAND(j < i);
      const MatrixBlock<double>& Jj =
          std::get<2>(jacobian_blocks_[triplets[0]]);
      const MatrixBlock<double>& Ji =
          std::get<2>(jacobian_blocks_[triplets[1]]);

      MatrixBlock<double> GJj = Jj.LeftMultiplyByBlockDiagonal(
          weight_matrix, weight_start, weight_end - 1);
      MatrixBlock<double> GJi = Ji.LeftMultiplyByBlockDiagonal(
          weight_matrix, weight_start, weight_end - 1);

      MatrixXd JiTGJi = MatrixXd::Zero(Ji.cols(), Ji.cols());
      MatrixXd JiTGJj = MatrixXd::Zero(Ji.cols(), Jj.cols());
      MatrixXd JjTGJj = MatrixXd::Zero(Jj.cols(), Jj.cols());

      Ji.TransposeAndMultiplyAndAddTo(GJi, &JiTGJi);
      Ji.TransposeAndMultiplyAndAddTo(GJj, &JiTGJj);
      Jj.TransposeAndMultiplyAndAddTo(GJj, &JjTGJj);

      A_->AddToBlock(i, i, JiTGJi);
      A_->AddToBlock(i, j, JiTGJj);
      A_->AddToBlock(j, j, JjTGJj);
    }
    weight_start = weight_end;
  }
  solver_->UpdateMatrix(*A_);
  factorization_ready_ = false;
  matrix_ready_ = true;
}

bool SuperNodalSolver2::Factor() {
  if (!matrix_ready_) {
    throw std::runtime_error("Call to Factor() failed: weight matrix not set.");
  }
  solver_->Factor();
  factorization_ready_ = true;
  matrix_ready_ = false;
  return true;
}

Eigen::VectorXd SuperNodalSolver2::Solve(const Eigen::VectorXd& b) const {
  if (!factorization_ready_) {
    throw std::runtime_error(
        "Call to Solve() failed: factorization not ready.");
  }
  Eigen::VectorXd y = b;
  solver_->SolveInPlace(&y);
  return y;
}

void SuperNodalSolver2::SolveInPlace(Eigen::VectorXd* b) const {
  if (!factorization_ready_) {
    throw std::runtime_error(
        "Call to Solve() failed: factorization not ready.");
  }
  solver_->SolveInPlace(b);
}

Eigen::MatrixXd SuperNodalSolver2::MakeFullMatrix() const {
  if (!matrix_ready_) {
    throw std::runtime_error(
        "Call to MakeFullMatrix() failed: weight matrix not set or matrix has "
        "been factored in place.");
  }
  return A_->MakeDenseMatrix();
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
