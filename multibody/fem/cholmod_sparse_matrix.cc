#include "drake/multibody/fem/cholmod_sparse_matrix.h"

#include <iostream>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <cholmod.h>

#include "drake/common/drake_throw.h"
#include "drake/multibody/fem/schur_complement.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

class CholmodSparseMatrix::Impl {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Impl);

  explicit Impl(const Eigen::SparseMatrix<double>& matrix) {
    cholmod_start(&cm_);
    A_ = std::unique_ptr<cholmod_sparse>(cholmod_allocate_sparse(
        matrix.rows(), matrix.cols(), /* max number of nonzeros */
        matrix.nonZeros(),
        /* sorted */ true, /* packed */ true,
        /* ignore top right corner */ -1, CHOLMOD_REAL, &cm_));

    const int rows = static_cast<int>(matrix.rows());

    ja_.resize(matrix.nonZeros());
    memcpy(ja_.data(), matrix.innerIndexPtr(),
           matrix.nonZeros() * sizeof(matrix.innerIndexPtr()[0]));

    ia_.resize(rows + 1);
    memcpy(ia_.data(), matrix.outerIndexPtr(),
           (rows + 1) * sizeof(matrix.outerIndexPtr()[0]));

    a_.resize(matrix.nonZeros());
    memcpy(a_.data(), matrix.valuePtr(),
           matrix.nonZeros() * sizeof(matrix.valuePtr()[0]));

    /* Hold on to the memory allocated by CHOLMOD. */
    Ai_ = A_->i;
    Ap_ = A_->p;
    Ax_ = A_->x;
    /* Point to the data from Eigen. */
    A_->i = ja_.data();
    A_->p = ia_.data();
    A_->x = a_.data();

    rows_ = rows;
  }

  ~Impl() {
    if (A_ != nullptr) {
      /* Make sure the memory allocated by CHOLMOD gets properly freed. */
      A_->i = Ai_;
      A_->p = Ap_;
      A_->x = Ax_;
      auto* A_ptr = A_.release();
      cholmod_free_sparse(&A_ptr, &cm_);
    }

    if (L_ != nullptr) {
      auto* L_ptr = L_.release();
      cholmod_free_factor(&L_ptr, &cm_);
    }

    cholmod_finish(&cm_);
  }

  int rows() const { return rows_; }

  void Factor() {
    /* Forbid repeated factorization. */
    DRAKE_THROW_UNLESS(L_ == nullptr);
    L_ = std::unique_ptr<cholmod_factor>(cholmod_analyze(A_.get(), &cm_));
    std::cout << "NNZ = " << L_->xsize << std::endl;
    permutation_.resize(L_->n);
    memcpy(permutation_.data(), L_->Perm,
           permutation_.size() * sizeof(permutation_[0]));
    cholmod_factorize(A_.get(), L_.get(), &cm_);
    /* Throw if factorization fails. */
    DRAKE_THROW_UNLESS(cm_.status != CHOLMOD_NOT_POSDEF);
  }

  /* @pre Factor() has been called. */
  const std::vector<int>& permutation() const { return permutation_; }

  VectorXd Solve(const VectorXd& rhs) const {
    DRAKE_DEMAND(rhs.size() == rows());
    cholmod_dense* b;
    b = cholmod_allocate_dense(rows(), 1, rows(), CHOLMOD_REAL, &cm_);
    /* Hold on to memory */
    void* bx = b->x;
    /* We const_cast away here to satisfy compiler, but we don't really modify
     the data here. */
    b->x = static_cast<void*>(const_cast<double*>(rhs.data()));
    cholmod_dense* x;
    x = cholmod_solve(CHOLMOD_A, L_.get(), b, &cm_);
    VectorXd result(rhs.size());
    memcpy(result.data(), x->x, result.size() * sizeof(result[0]));
    b->x = bx;
    cholmod_free_dense(&b, &cm_);
    cholmod_free_dense(&x, &cm_);
    return result;
  }

  SchurComplement<double> CalcSchurComplement(const MatrixX<double>& B,
                                              const MatrixX<double>& C) const {
    DRAKE_DEMAND(L_ != nullptr);
    DRAKE_DEMAND(B.rows() > 0);
    DRAKE_DEMAND(B.cols() > 0);
    DRAKE_DEMAND(B.rows() == rows());
    DRAKE_DEMAND(B.cols() == C.cols());

    MatrixXd AinvB(B.rows(), B.cols());
    for (int i = 0; i < B.cols(); ++i) {
      /* Columns of B. */
      cholmod_dense* b;
      /* Columns of A⁻¹B. */
      cholmod_dense* x;
      b = cholmod_allocate_dense(rows(), 1, rows(), CHOLMOD_REAL, &cm_);
      /* Hold on to memory */
      void* bx = b->x;
      /* We const_cast away here to satisfy compiler, but we don't really modify
       the data here. */
      b->x = static_cast<void*>(const_cast<double*>(B.col(i).data()));
      x = cholmod_solve(CHOLMOD_A, L_.get(), b, &cm_);
      memcpy(AinvB.col(i).data(), x->x, AinvB.rows() * sizeof(AinvB(0, 0)));
      /* Clean up. */
      b->x = bx;
      cholmod_free_dense(&b, &cm_);
      cholmod_free_dense(&x, &cm_);
    }
    MatrixXd neg_AinvB = -AinvB;
    Eigen::SparseMatrix<double> B_sparse = B.sparseView();
    auto neg_BAinvB = neg_AinvB.transpose() * B_sparse;
    MatrixXd complement = C + neg_BAinvB;
    return SchurComplement<double>(std::move(complement), std::move(neg_AinvB));
  }

  void Print() const { cholmod_print_sparse(A_.get(), "A", &cm_); }

 private:
  mutable cholmod_common cm_;
  std::unique_ptr<cholmod_sparse> A_;
  std::unique_ptr<cholmod_factor> L_;
  std::vector<int> permutation_;
  /* Temporary pointers to memory that CHOLMOD allocated that we won't use. */
  void* Ax_;
  void* Ap_;
  void* Ai_;

  VectorXd a_;
  VectorXi ia_;
  VectorXi ja_;
  int rows_{0};
};

CholmodSparseMatrix::CholmodSparseMatrix(
    const Eigen::SparseMatrix<double>& matrix) {
  pimpl_ = std::make_unique<Impl>(matrix);
}

CholmodSparseMatrix::~CholmodSparseMatrix() = default;

void CholmodSparseMatrix::Factor() const { pimpl_->Factor(); }

const std::vector<int>& CholmodSparseMatrix::permutation() const {
  return pimpl_->permutation();
}

VectorX<double> CholmodSparseMatrix::Solve(const VectorX<double>& rhs) const {
  return pimpl_->Solve(rhs);
}

SchurComplement<double> CholmodSparseMatrix::CalcSchurComplement(
    const MatrixX<double>& B, const MatrixX<double>& C) const {
  return pimpl_->CalcSchurComplement(B, C);
}

void CholmodSparseMatrix::Print() const { pimpl_->Print(); }

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
