#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>

#include "drake/solvers/decision_variable.h"
#include "drake/solvers/evaluator_base.h"
#include "drake/solvers/sparse_and_dense_matrix.h"

namespace drake {
namespace solvers {

/**
 * Provides an abstract base for all costs.
 *
 * @ingroup solver_evaluators
 */
class Cost : public EvaluatorBase {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Cost);

 protected:
  /**
   * Constructs a cost evaluator.
   * @param num_vars Number of input variables.
   * @param description Human-friendly description.
   */
  explicit Cost(int num_vars, const std::string& description = "")
      : EvaluatorBase(1, num_vars, description) {}
};

/**
 * Implements a cost of the form @f[ a'x + b @f].
 *
 * @ingroup solver_evaluators
 */
class LinearCost : public Cost {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LinearCost);

  /**
   * Construct a linear cost of the form @f[ a'x + b @f].
   * @param a Linear term.
   * @param b (optional) Constant term.
   */
  // NOLINTNEXTLINE(runtime/explicit) This conversion is desirable.
  LinearCost(const Eigen::Ref<const Eigen::VectorXd>& a, double b = 0.)
      : Cost(a.rows()), a_(a), b_(b) {
    set_is_thread_safe(true);
  }

  ~LinearCost() override;

  Eigen::SparseMatrix<double> GetSparseMatrix() const {
    // TODO(eric.cousineau): Consider storing or caching sparse matrix, such
    // that we can return a const lvalue reference.
    return a_.sparseView();
  }

  const Eigen::VectorXd& a() const { return a_; }

  double b() const { return b_; }

  /**
   * Updates the coefficients of the cost.
   * Note that the number of variables (size of a) cannot change.
   * @param new_a New linear term.
   * @param new_b (optional) New constant term.
   */
  void UpdateCoefficients(const Eigen::Ref<const Eigen::VectorXd>& new_a,
                          double new_b = 0.);

  /**
   * Updates one entry in the coefficient of the cost.
   * a[i] = val.
   * @param i The index of the coefficient to be updated.
   * @param val The value of that updated entry.
   */
  void update_coefficient_entry(int i, double val);

  /**
   * Updates the constant term in the cost to `new_b`.
   */
  void update_constant_term(double new_b);

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override;

  std::ostream& DoDisplay(std::ostream&,
                          const VectorX<symbolic::Variable>&) const override;

  std::string DoToLatex(const VectorX<symbolic::Variable>&, int) const override;

 private:
  template <typename DerivedX, typename U>
  void DoEvalGeneric(const Eigen::MatrixBase<DerivedX>& x, VectorX<U>* y) const;

  Eigen::VectorXd a_;
  double b_{};
};

/**
 * Implements a cost of the form @f[ .5 x'Qx + b'x + c @f].
 *
 * @ingroup solver_evaluators
 */
class QuadraticCost : public Cost {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuadraticCost);

  /**
   * Constructs a cost of the form @f[ .5 x'Qx + b'x + c @f].
   * @param Q Quadratic term.
   * @param b Linear term.
   * @param c (optional) Constant term.
   * @param is_hessian_psd (optional) Indicates if the Hessian matrix Q is
   * positive semidefinite (psd) or not. If set to true, then the user
   * guarantees that Q is psd; if set to false, then the user guarantees that Q
   * is not psd. If set to std::nullopt, then the constructor will check if Q is
   * psd or not. The default is std::nullopt. To speed up the constructor, set
   * is_hessian_psd to either true or false.
   */
  template <typename DerivedQ, typename Derivedb>
  QuadraticCost(const Eigen::MatrixBase<DerivedQ>& Q,
                const Eigen::MatrixBase<Derivedb>& b, double c = 0.,
                std::optional<bool> is_hessian_psd = std::nullopt)
      : Cost(Q.rows()), Q_((Q + Q.transpose()) / 2), b_(b), c_(c) {
    set_is_thread_safe(true);
    DRAKE_THROW_UNLESS(Q_.rows() == Q_.cols());
    DRAKE_THROW_UNLESS(Q_.cols() == b_.rows());
    if (is_hessian_psd.has_value()) {
      is_convex_ = is_hessian_psd.value();
    } else {
      is_convex_ = CheckHessianPsd();
    }
  }

  ~QuadraticCost() override;

  /// Returns the symmetric matrix Q, as the Hessian of the cost.
  const Eigen::MatrixXd& Q() const { return Q_; }

  const Eigen::VectorXd& b() const { return b_; }

  double c() const { return c_; }

  /**
   * Returns true if this cost is convex. A quadratic cost if convex if and only
   * if its Hessian matrix Q is positive semidefinite.
   */
  bool is_convex() const { return is_convex_; }

  /**
   * Updates the quadratic and linear term of the constraint. The new
   * matrices need to have the same dimension as before.
   * @param new_Q New quadratic term.
   * @param new_b New linear term.
   * @param new_c (optional) New constant term.
   * @param is_hessian_psd (optional) Indicates if the Hessian matrix Q is
   * positive semidefinite (psd) or not. If set to true, then the user
   * guarantees that Q is psd; if set to false, then the user guarantees that Q
   * is not psd. If set to std::nullopt, then this function will check if Q is
   * psd or not. The default is std::nullopt. To speed up the computation, set
   * is_hessian_psd to either true or false.
   */
  template <typename DerivedQ, typename DerivedB>
  void UpdateCoefficients(const Eigen::MatrixBase<DerivedQ>& new_Q,
                          const Eigen::MatrixBase<DerivedB>& new_b,
                          double new_c = 0.,
                          std::optional<bool> is_hessian_psd = std::nullopt) {
    if (new_Q.rows() != new_Q.cols() || new_Q.rows() != new_b.rows() ||
        new_b.cols() != 1) {
      throw std::runtime_error("New constraints have invalid dimensions");
    }

    if (new_b.rows() != b_.rows()) {
      throw std::runtime_error("Can't change the number of decision variables");
    }

    Q_ = (new_Q + new_Q.transpose()) / 2;
    b_ = new_b;
    c_ = new_c;
    if (is_hessian_psd.has_value()) {
      is_convex_ = is_hessian_psd.value();
    } else {
      is_convex_ = CheckHessianPsd();
    }
  }

  /**
   * Updates both Q(i, j) and Q(j, i) to val
   * @param is_hessian_psd If this is `nullopt`, the new Hessian is
   * checked (possibly expensively) for PSD-ness.  If this is
   * set true/false, the cost's convexity is updated to that
   * value without checking (it is the user's responsibility to make sure the
   * flag is set correctly).
   * @note If you have multiple entries in the Hessian matrix to update, and you
   * don't specify is_hessian_psd, then it is much faster to call
   * UpdateCoefficients(new_A, new_b) where new_A contains all the updated
   * entries.
   */
  void UpdateHessianEntry(int i, int j, double val,
                          std::optional<bool> is_hessian_psd = std::nullopt);

  /**
   * Updates b(i)=val.
   */
  void update_linear_coefficient_entry(int i, double val);

  /**
   * Updates the constant term to `new_c`.
   */
  void update_constant_term(double new_c);

 private:
  template <typename DerivedX, typename U>
  void DoEvalGeneric(const Eigen::MatrixBase<DerivedX>& x, VectorX<U>* y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override;

  std::ostream& DoDisplay(std::ostream&,
                          const VectorX<symbolic::Variable>&) const override;

  std::string DoToLatex(const VectorX<symbolic::Variable>&, int) const override;

  bool CheckHessianPsd();

  Eigen::MatrixXd Q_;
  Eigen::VectorXd b_;
  double c_{};
  bool is_convex_{};
};

/**
 * Creates a cost term of the form (x-x_desired)'*Q*(x-x_desired).
 *
 * @ingroup solver_evaluators
 */
std::shared_ptr<QuadraticCost> MakeQuadraticErrorCost(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& x_desired);

/**
 * Creates a quadratic cost of the form |Ax-b|²=(Ax-b)ᵀ(Ax-b)
 *
 * @ingroup solver_evaluators
 */
std::shared_ptr<QuadraticCost> Make2NormSquaredCost(
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& b);

/**
 * Implements a cost of the form ‖Ax + b‖₁. Note that this cost is
 * non-differentiable when any element of Ax + b equals zero.
 *
 * @ingroup solver_evaluators
 */
class L1NormCost : public Cost {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(L1NormCost);

  /**
   * Construct a cost of the form ‖Ax + b‖₁.
   * @param A Linear term.
   * @param b Constant term.
   * @throws std::exception if the size of A and b don't match.
   */
  L1NormCost(const Eigen::Ref<const Eigen::MatrixXd>& A,
             const Eigen::Ref<const Eigen::VectorXd>& b);

  ~L1NormCost() override;

  const Eigen::MatrixXd& A() const { return A_; }

  const Eigen::VectorXd& b() const { return b_; }

  /**
   * Updates the coefficients of the cost.
   * Note that the number of variables (columns of A) cannot change.
   * @param new_A New linear term.
   * @param new_b New constant term.
   */
  void UpdateCoefficients(const Eigen::Ref<const Eigen::MatrixXd>& new_A,
                          const Eigen::Ref<const Eigen::VectorXd>& new_b);

  /** Updates A(i, j) = val.
   * @throws if i or j are invalid indices.
   */
  void update_A_entry(int i, int j, double val);

  /** Updates b(i) = val.
   * @throws if i is an invalid index.
   */
  void update_b_entry(int i, double val);

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override;

  std::ostream& DoDisplay(std::ostream&,
                          const VectorX<symbolic::Variable>&) const override;

  std::string DoToLatex(const VectorX<symbolic::Variable>&, int) const override;

 private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
};

/**
 * Implements a cost of the form ‖Ax + b‖₂.
 *
 * @ingroup solver_evaluators
 */
class L2NormCost : public Cost {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(L2NormCost);

  // TODO(russt): Add an option to select an implementation that smooths the
  // gradient discontinuity at the origin.
  /**
   * Construct a cost of the form ‖Ax + b‖₂.
   * @param A Linear term.
   * @param b Constant term.
   * @throws std::exception if the size of A and b don't match.
   * @pydrake_mkdoc_identifier{dense_A}
   */
  L2NormCost(const Eigen::Ref<const Eigen::MatrixXd>& A,
             const Eigen::Ref<const Eigen::VectorXd>& b);

  /**
   * Overloads constructor with a sparse A matrix.
   * @pydrake_mkdoc_identifier{sparse_A}
   */
  L2NormCost(const Eigen::SparseMatrix<double>& A,
             const Eigen::Ref<const Eigen::VectorXd>& b);

  ~L2NormCost() override;

  const Eigen::SparseMatrix<double>& get_sparse_A() const {
    return A_.get_as_sparse();
  }

  const Eigen::MatrixXd& GetDenseA() const { return A_.GetAsDense(); }

  const Eigen::VectorXd& b() const { return b_; }

  /**
   * Updates the coefficients of the cost.
   * Note that the number of variables (columns of A) cannot change.
   * @param new_A New linear term.
   * @param new_b New constant term.
   * @pydrake_mkdoc_identifier{dense_A}
   */
  void UpdateCoefficients(const Eigen::Ref<const Eigen::MatrixXd>& new_A,
                          const Eigen::Ref<const Eigen::VectorXd>& new_b);

  /**
   * Overloads UpdateCoefficients but with a sparse A matrix.
   * @pydrake_mkdoc_identifier{sparse_A}
   */
  void UpdateCoefficients(const Eigen::SparseMatrix<double>& new_A,
                          const Eigen::Ref<const Eigen::VectorXd>& new_b);

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override;

  std::ostream& DoDisplay(std::ostream&,
                          const VectorX<symbolic::Variable>&) const override;

  std::string DoToLatex(const VectorX<symbolic::Variable>&, int) const override;

 private:
  internal::SparseAndDenseMatrix A_;
  Eigen::VectorXd b_;
};

/**
 * Implements a cost of the form ‖Ax + b‖∞. Note that this cost is
 * non-differentiable when any two or more elements of Ax + b are equal.
 *
 * @ingroup solver_evaluators
 */
class LInfNormCost : public Cost {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LInfNormCost);

  /**
   * Construct a cost of the form ‖Ax + b‖∞.
   * @param A Linear term.
   * @param b Constant term.
   * @throws std::exception if the size of A and b don't match.
   */
  LInfNormCost(const Eigen::Ref<const Eigen::MatrixXd>& A,
               const Eigen::Ref<const Eigen::VectorXd>& b);

  ~LInfNormCost() override;

  const Eigen::MatrixXd& A() const { return A_; }

  const Eigen::VectorXd& b() const { return b_; }

  /**
   * Updates the coefficients of the cost.
   * Note that the number of variables (columns of A) cannot change.
   * @param new_A New linear term.
   * @param new_b New constant term.
   */
  void UpdateCoefficients(const Eigen::Ref<const Eigen::MatrixXd>& new_A,
                          const Eigen::Ref<const Eigen::VectorXd>& new_b);

  /** Updates A(i, j) = val.
   * @throws if i or j are invalid indices.
   */
  void update_A_entry(int i, int j, double val);

  /** Updates b(i) = val.
   * @throws if i is an invalid index.
   */
  void update_b_entry(int i, double val);

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override;

  std::ostream& DoDisplay(std::ostream&,
                          const VectorX<symbolic::Variable>&) const override;

  std::string DoToLatex(const VectorX<symbolic::Variable>&, int) const override;

 private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
};

/**
 * If \f$ z = Ax + b,\f$ implements a cost of the form:
 * @f[
 * (z_1^2 + z_2^2 + ... + z_{n-1}^2) / z_0.
 * @f]
 * Note that this cost is convex when we additionally constrain z_0 > 0. It is
 * treated as a generic nonlinear objective by most solvers.
 *
 * Costs of this form are sometimes referred to as "quadratic over linear".
 *
 * @ingroup solver_evaluators
 */
class PerspectiveQuadraticCost : public Cost {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PerspectiveQuadraticCost);

  /**
   * Construct a cost of the form (z_1^2 + z_2^2 + ... + z_{n-1}^2) / z_0 where
   * z = Ax + b.
   * @param A Linear term.
   * @param b Constant term.
   */
  PerspectiveQuadraticCost(const Eigen::Ref<const Eigen::MatrixXd>& A,
                           const Eigen::Ref<const Eigen::VectorXd>& b);

  ~PerspectiveQuadraticCost() override;

  const Eigen::MatrixXd& A() const { return A_; }

  const Eigen::VectorXd& b() const { return b_; }

  /**
   * Updates the coefficients of the cost.
   * Note that the number of variables (columns of A) cannot change.
   * @param new_A New linear term.
   * @param new_b New constant term.
   */
  void UpdateCoefficients(const Eigen::Ref<const Eigen::MatrixXd>& new_A,
                          const Eigen::Ref<const Eigen::VectorXd>& new_b);

  /** Updates A(i, j) = val.
   * @throws if i or j are invalid indices.
   */
  void update_A_entry(int i, int j, double val);

  /** Updates b(i) = val.
   * @throws if i is an invalid index.
   */
  void update_b_entry(int i, double val);

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override;

  std::ostream& DoDisplay(std::ostream&,
                          const VectorX<symbolic::Variable>&) const override;

  std::string DoToLatex(const VectorX<symbolic::Variable>&, int) const override;

 private:
  template <typename DerivedX, typename U>
  void DoEvalGeneric(const Eigen::MatrixBase<DerivedX>& x, VectorX<U>* y) const;

  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
};

/**
 * A cost that may be specified using another (potentially nonlinear)
 * evaluator.
 * @tparam EvaluatorType The nested evaluator.
 *
 * @ingroup solver_evaluators
 */
// TODO(hongkai.dai):
// MathematicalProgram::AddCost(EvaluatorCost<LinearConstraint>(...)) should
// recognize that the compounded cost is linear.
template <typename EvaluatorType = EvaluatorBase>
class EvaluatorCost : public Cost {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EvaluatorCost);

  explicit EvaluatorCost(const std::shared_ptr<EvaluatorType>& evaluator)
      : Cost(evaluator->num_vars()),
        evaluator_{evaluator},
        a_{std::nullopt},
        b_{0} {
    set_is_thread_safe(evaluator->is_thread_safe());
    DRAKE_THROW_UNLESS(evaluator->num_outputs() == 1);
  }

  /**
   * This cost computes a.dot(evaluator(x)) + b
   * @pre a.rows() == evaluator->num_outputs()
   */
  EvaluatorCost(const std::shared_ptr<EvaluatorType>& evaluator,
                const Eigen::Ref<const Eigen::VectorXd>& a, double b = 0)
      : Cost(evaluator->num_vars()), evaluator_(evaluator), a_{a}, b_{b} {
    set_is_thread_safe(true);
    DRAKE_THROW_UNLESS(evaluator->num_outputs() == a_->rows());
  }

 protected:
  const EvaluatorType& evaluator() const { return *evaluator_; }

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override {
    this->DoEvalGeneric<double, double>(x, y);
  }
  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override {
    this->DoEvalGeneric<AutoDiffXd, AutoDiffXd>(x, y);
  }

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override {
    this->DoEvalGeneric<symbolic::Variable, symbolic::Expression>(x, y);
  }

 private:
  template <typename T, typename S>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x,
                     VectorX<S>* y) const {
    if (a_.has_value()) {
      VectorX<S> y_inner;
      evaluator_->Eval(x, &y_inner);
      y->resize(1);
      (*y)(0) = a_->dot(y_inner) + b_;
    } else {
      evaluator_->Eval(x, y);
    }
  }

  std::shared_ptr<EvaluatorType> evaluator_;
  std::optional<Eigen::VectorXd> a_;
  double b_{};
};

/**
 * Implements a cost of the form P(x, y...) where P is a multivariate
 * polynomial in x, y, ...
 *
 * The Polynomial class uses a different variable naming scheme; thus the
 * caller must provide a list of Polynomial::VarType variables that correspond
 * to the members of the Binding<> (the individual scalar elements of the
 * given VariableList).
 *
 * @ingroup solver_evaluators
 */
class PolynomialCost : public EvaluatorCost<PolynomialEvaluator> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PolynomialCost);

  /**
   * Constructs a polynomial cost
   * @param polynomials Polynomial vector, a 1 x 1 vector.
   * @param poly_vars Polynomial variables, a `num_vars` x 1 vector.
   */
  PolynomialCost(const VectorXPoly& polynomials,
                 const std::vector<Polynomiald::VarType>& poly_vars)
      : EvaluatorCost(
            std::make_shared<PolynomialEvaluator>(polynomials, poly_vars)) {}

  const VectorXPoly& polynomials() const { return evaluator().polynomials(); }

  const std::vector<Polynomiald::VarType>& poly_vars() const {
    return evaluator().poly_vars();
  }
};

/**
 * Impose a generic (potentially nonlinear) cost represented as a symbolic
 * Expression.  Expression::Evaluate is called on every constraint evaluation.
 *
 * Uses symbolic::Jacobian to provide the gradients to the AutoDiff method.
 *
 * @ingroup solver_evaluators
 */
class ExpressionCost : public Cost {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ExpressionCost);

  explicit ExpressionCost(const symbolic::Expression& e);

  /**
   * @return the list of the variables involved in the vector of expressions,
   * in the order that they are expected to be received during DoEval.
   * Any Binding that connects this constraint to decision variables should
   * pass this list of variables to the Binding.
   */
  const VectorXDecisionVariable& vars() const;

  /** @return the symbolic expression. */
  const symbolic::Expression& expression() const;

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
              VectorX<symbolic::Expression>* y) const override;

  std::ostream& DoDisplay(std::ostream&,
                          const VectorX<symbolic::Variable>&) const override;

  std::string DoToLatex(const VectorX<symbolic::Variable>&, int) const override;

 private:
  std::unique_ptr<EvaluatorBase> evaluator_;
};

/**
 * Converts an input of type @p F to a nonlinear cost.
 * @tparam FF The forwarded function type (e.g., `const F&, `F&&`, ...).
 * The class `F` should have functions numInputs(), numOutputs(), and
 * eval(x, y).
 *
 * @ingroup solver_evaluators
 */
template <typename FF>
std::shared_ptr<Cost> MakeFunctionCost(FF&& f) {
  return std::make_shared<EvaluatorCost<>>(
      MakeFunctionEvaluator(std::forward<FF>(f)));
}

}  // namespace solvers
}  // namespace drake
