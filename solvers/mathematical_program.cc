#include "drake/solvers/mathematical_program.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "drake/common/eigen_types.h"
#include "drake/common/fmt_eigen.h"
#include "drake/common/ssize.h"
#include "drake/common/symbolic/decompose.h"
#include "drake/common/symbolic/latex.h"
#include "drake/common/symbolic/monomial_util.h"
#include "drake/common/text_logging.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/binding.h"
#include "drake/solvers/decision_variable.h"
#include "drake/solvers/sos_basis_generator.h"

namespace drake {
namespace solvers {

using std::enable_if;
using std::endl;
using std::find;
using std::is_same;
using std::make_pair;
using std::make_shared;
using std::map;
using std::numeric_limits;
using std::ostringstream;
using std::pair;
using std::runtime_error;
using std::shared_ptr;
using std::string;
using std::unordered_map;
using std::vector;

using symbolic::Expression;
using symbolic::Formula;
using symbolic::Variable;
using symbolic::Variables;

using internal::CreateBinding;

const double kInf = std::numeric_limits<double>::infinity();

MathematicalProgram::MathematicalProgram() = default;

MathematicalProgram::MathematicalProgram(const MathematicalProgram&) = default;

MathematicalProgram::~MathematicalProgram() = default;

std::unique_ptr<MathematicalProgram> MathematicalProgram::Clone() const {
  return std::unique_ptr<MathematicalProgram>(new MathematicalProgram(*this));
}

string MathematicalProgram::to_string() const {
  std::ostringstream os;
  os << *this;
  return os.str();
}

bool MathematicalProgram::IsThreadSafe() const {
  const std::vector<Binding<Cost>> costs = GetAllCosts();
  const std::vector<Binding<Constraint>> constraints = GetAllConstraints();
  return std::all_of(visualization_callbacks_.begin(),
                     visualization_callbacks_.end(),
                     [](const Binding<VisualizationCallback>& c) {
                       return c.evaluator()->is_thread_safe();
                     }) &&
         std::all_of(costs.begin(), costs.end(),
                     [](const Binding<Cost>& c) {
                       return c.evaluator()->is_thread_safe();
                     }) &&
         std::all_of(constraints.begin(), constraints.end(),
                     [](const Binding<Constraint>& c) {
                       return c.evaluator()->is_thread_safe();
                     });
}

std::string MathematicalProgram::ToLatex(int precision) {
  if (num_vars() == 0) {
    return "\\text{This MathematicalProgram has no decision variables.}";
  }

  std::stringstream ss;
  ss << "\\begin{align*}\n";
  if (GetAllCosts().empty()) {
    ss << "\\text{find}_{";
  } else {
    ss << "\\min_{";
  }
  // TODO(russt): summarize vectors and matrices here by name, instead of every
  // element.
  bool first = true;
  for (int i = 0; i < num_vars(); ++i) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << symbolic::ToLatex(
        decision_variables()[i]);  // precision is not needed.
  }
  ss << "} \\quad & ";

  first = true;
  for (const auto& b : GetAllCosts()) {
    if (!first) ss << "\\\\\n &  + ";
    first = false;
    ss << b.ToLatex(precision);
  }
  std::vector<Binding<Constraint>> constraints = GetAllConstraints();
  for (int i = 0; i < ssize(constraints); ++i) {
    if (i == 0) {
      ss << "\\\\\n \\text{subject to}\\quad";
    }
    ss << " & " << constraints[i].ToLatex(precision);
    if (i == ssize(constraints) - 1) {
      ss << ".";
    } else {
      ss << ",";
    }
    ss << "\\\\\n";
  }
  ss << "\\end{align*}\n";
  return ss.str();
}

MatrixXDecisionVariable MathematicalProgram::NewVariables(
    VarType type, int rows, int cols, bool is_symmetric,
    const vector<string>& names) {
  MatrixXDecisionVariable decision_variable_matrix(rows, cols);
  NewVariables_impl(type, names, is_symmetric, decision_variable_matrix);
  return decision_variable_matrix;
}

MatrixXDecisionVariable MathematicalProgram::NewSymmetricContinuousVariables(
    int rows, const string& name) {
  vector<string> names(rows * (rows + 1) / 2);
  int count = 0;
  for (int j = 0; j < static_cast<int>(rows); ++j) {
    for (int i = j; i < static_cast<int>(rows); ++i) {
      names[count] =
          name + "(" + std::to_string(i) + "," + std::to_string(j) + ")";
      ++count;
    }
  }
  return NewVariables(VarType::CONTINUOUS, rows, rows, true, names);
}

namespace {
template <typename T>
VectorX<T> Flatten(const Eigen::Ref<const MatrixX<T>>& mat) {
  if (mat.cols() == 1) {
    return mat;
  } else {
    // Cannot use Eigen::Map to flatten the matrix since mat.outerStride() might
    // not equal to mat.rows(), namely the data in mat is not in contiguous
    // space on memory.
    // TODO(hongkai.dai): figure out a better way that avoids copy and dynamic
    // memory allocation.
    VectorX<T> vec(mat.size());
    for (int j = 0; j < mat.cols(); ++j) {
      vec.segment(j * mat.rows(), mat.rows()) = mat.col(j);
    }
    return vec;
  }
}
}  // namespace

void MathematicalProgram::AddDecisionVariables(
    const Eigen::Ref<const MatrixXDecisionVariable>& decision_variables) {
  int new_var_count = 0;
  for (int i = 0; i < decision_variables.rows(); ++i) {
    for (int j = 0; j < decision_variables.cols(); ++j) {
      const auto& var = decision_variables(i, j);
      if (decision_variable_index_.find(var.get_id()) !=
          decision_variable_index_.end()) {
        continue;
      }
      if (indeterminates_index_.find(var.get_id()) !=
          indeterminates_index_.end()) {
        throw std::runtime_error(
            fmt::format("{} is already an indeterminate.", var));
      }
      CheckVariableType(var.get_type());
      decision_variables_.push_back(var);
      const int var_index = decision_variables_.size() - 1;
      decision_variable_index_.insert(std::make_pair(var.get_id(), var_index));
      ++new_var_count;
    }
  }
  AppendNanToEnd(new_var_count, &x_initial_guess_);
}

symbolic::Polynomial MathematicalProgram::NewFreePolynomialImpl(
    const Variables& indeterminates, const int degree, const string& coeff_name,
    symbolic::internal::DegreeType degree_type) {
  const drake::VectorX<symbolic::Monomial> m =
      symbolic::internal::ComputeMonomialBasis<Eigen::Dynamic>(
          indeterminates, degree, degree_type);
  const VectorXDecisionVariable coeffs{
      this->NewContinuousVariables(m.size(), coeff_name)};
  symbolic::Polynomial::MapType p_map;
  // Since each entry in m is unique, we construct the polynomial using a map
  // with m(i) being the map key.
  for (int i = 0; i < coeffs.rows(); ++i) {
    p_map.emplace(m(i), coeffs(i));
  }
  return symbolic::Polynomial(std::move(p_map));
}

symbolic::Polynomial MathematicalProgram::NewFreePolynomial(
    const Variables& indeterminates, const int degree,
    const string& coeff_name) {
  return NewFreePolynomialImpl(indeterminates, degree, coeff_name,
                               symbolic::internal::DegreeType::kAny);
}

symbolic::Polynomial MathematicalProgram::NewEvenDegreeFreePolynomial(
    const symbolic::Variables& indeterminates, int degree,
    const std::string& coeff_name) {
  return NewFreePolynomialImpl(indeterminates, degree, coeff_name,
                               symbolic::internal::DegreeType::kEven);
}

symbolic::Polynomial MathematicalProgram::NewOddDegreeFreePolynomial(
    const symbolic::Variables& indeterminates, int degree,
    const std::string& coeff_name) {
  return NewFreePolynomialImpl(indeterminates, degree, coeff_name,
                               symbolic::internal::DegreeType::kOdd);
}

// This is the utility function for creating new nonnegative polynomials
// (sos-polynomial, sdsos-polynomial, dsos-polynomial). It creates a
// symmetric matrix Q as decision variables, and return m' * Q * m as the new
// polynomial, where m is the monomial basis.
pair<symbolic::Polynomial, MatrixXDecisionVariable>
MathematicalProgram::NewSosPolynomial(
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    NonnegativePolynomial type, const std::string& gram_name) {
  const MatrixXDecisionVariable Q =
      NewSymmetricContinuousVariables(monomial_basis.size(), gram_name);
  const symbolic::Polynomial p = NewSosPolynomial(Q, monomial_basis, type);
  return std::make_pair(p, Q);
}

namespace {
symbolic::Polynomial ComputePolynomialFromMonomialBasisAndGramMatrix(
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& gram) {
  // TODO(hongkai.dai & soonho.kong): ideally we should compute p in one line as
  // monomial_basis.dot(gramian * monomial_basis). But as explained in #10200,
  // this one line version is too slow, so we use this double for loop to
  // compute the matrix product by hand. I will revert to the one line version
  // when it is fast.
  symbolic::Polynomial p{};
  for (int i = 0; i < gram.rows(); ++i) {
    p.AddProduct(gram(i, i), pow(monomial_basis(i), 2));
    for (int j = i + 1; j < gram.cols(); ++j) {
      p.AddProduct(2 * gram(i, j), monomial_basis(i) * monomial_basis(j));
    }
  }
  return p;
}
}  // namespace

symbolic::Polynomial MathematicalProgram::NewSosPolynomial(
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& gramian,
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    NonnegativePolynomial type) {
  DRAKE_ASSERT(gramian.rows() == gramian.cols());
  DRAKE_ASSERT(gramian.rows() == monomial_basis.rows());
  const symbolic::Polynomial p =
      ComputePolynomialFromMonomialBasisAndGramMatrix(monomial_basis, gramian);
  switch (type) {
    case MathematicalProgram::NonnegativePolynomial::kSos: {
      AddPositiveSemidefiniteConstraint(gramian);
      return p;
    }
    case MathematicalProgram::NonnegativePolynomial::kSdsos: {
      AddScaledDiagonallyDominantMatrixConstraint(gramian);
      return p;
    }
    case MathematicalProgram::NonnegativePolynomial::kDsos: {
      AddPositiveDiagonallyDominantMatrixConstraint(
          gramian.cast<symbolic::Expression>());
      return p;
    }
  }
  throw std::runtime_error(
      "NewSosPolynomial() was passed an invalid NonnegativePolynomial type");
}

pair<symbolic::Polynomial, MatrixXDecisionVariable>
MathematicalProgram::NewSosPolynomial(const symbolic::Variables& indeterminates,
                                      int degree, NonnegativePolynomial type,
                                      const std::string& gram_name) {
  DRAKE_DEMAND(degree >= 0 && degree % 2 == 0);
  if (degree == 0) {
    // The polynomial only has a non-negative constant term.
    const symbolic::Variable poly_constant =
        NewContinuousVariables<1>(gram_name)(0);
    AddBoundingBoxConstraint(0, kInf, poly_constant);
    MatrixXDecisionVariable gram(1, 1);
    gram(0, 0) = poly_constant;
    return std::make_pair(
        symbolic::Polynomial({{symbolic::Monomial(), poly_constant}}), gram);
  } else {
    const drake::VectorX<symbolic::Monomial> x{
        MonomialBasis(indeterminates, degree / 2)};
    return NewSosPolynomial(x, type, gram_name);
  }
}

std::tuple<symbolic::Polynomial, MatrixXDecisionVariable,
           MatrixXDecisionVariable>
MathematicalProgram::NewEvenDegreeNonnegativePolynomial(
    const symbolic::Variables& indeterminates, int degree,
    MathematicalProgram::NonnegativePolynomial type) {
  DRAKE_DEMAND(degree % 2 == 0);
  const VectorX<symbolic::Monomial> m_e =
      EvenDegreeMonomialBasis(indeterminates, degree / 2);
  const VectorX<symbolic::Monomial> m_o =
      OddDegreeMonomialBasis(indeterminates, degree / 2);
  symbolic::Polynomial p1, p2;
  MatrixXDecisionVariable Q_ee, Q_oo;
  std::tie(p1, Q_ee) = NewSosPolynomial(m_e, type);
  std::tie(p2, Q_oo) = NewSosPolynomial(m_o, type);
  const symbolic::Polynomial p = p1 + p2;
  return std::make_tuple(p, Q_oo, Q_ee);
}

std::tuple<symbolic::Polynomial, MatrixXDecisionVariable,
           MatrixXDecisionVariable>
MathematicalProgram::NewEvenDegreeSosPolynomial(
    const symbolic::Variables& indeterminates, int degree) {
  return NewEvenDegreeNonnegativePolynomial(
      indeterminates, degree, MathematicalProgram::NonnegativePolynomial::kSos);
}

std::tuple<symbolic::Polynomial, MatrixXDecisionVariable,
           MatrixXDecisionVariable>
MathematicalProgram::NewEvenDegreeSdsosPolynomial(
    const symbolic::Variables& indeterminates, int degree) {
  return NewEvenDegreeNonnegativePolynomial(
      indeterminates, degree,
      MathematicalProgram::NonnegativePolynomial::kSdsos);
}

std::tuple<symbolic::Polynomial, MatrixXDecisionVariable,
           MatrixXDecisionVariable>
MathematicalProgram::NewEvenDegreeDsosPolynomial(
    const symbolic::Variables& indeterminates, int degree) {
  return NewEvenDegreeNonnegativePolynomial(
      indeterminates, degree,
      MathematicalProgram::NonnegativePolynomial::kDsos);
}

symbolic::Polynomial MathematicalProgram::MakePolynomial(
    const symbolic::Expression& e) const {
  return symbolic::Polynomial{e, symbolic::Variables{indeterminates()}};
}

void MathematicalProgram::Reparse(symbolic::Polynomial* const p) const {
  p->SetIndeterminates(symbolic::Variables{indeterminates()});
}

MatrixXIndeterminate MathematicalProgram::NewIndeterminates(
    int rows, int cols, const vector<string>& names) {
  MatrixXIndeterminate indeterminates_matrix(rows, cols);
  NewIndeterminates_impl(names, indeterminates_matrix);
  return indeterminates_matrix;
}

VectorXIndeterminate MathematicalProgram::NewIndeterminates(
    int rows, const std::vector<std::string>& names) {
  return NewIndeterminates(rows, 1, names);
}

VectorXIndeterminate MathematicalProgram::NewIndeterminates(
    int rows, const string& name) {
  vector<string> names(rows);
  for (int i = 0; i < static_cast<int>(rows); ++i) {
    names[i] = name + "(" + std::to_string(i) + ")";
  }
  return NewIndeterminates(rows, names);
}

MatrixXIndeterminate MathematicalProgram::NewIndeterminates(
    int rows, int cols, const string& name) {
  vector<string> names(rows * cols);
  int count = 0;
  for (int j = 0; j < static_cast<int>(cols); ++j) {
    for (int i = 0; i < static_cast<int>(rows); ++i) {
      names[count] =
          name + "(" + std::to_string(i) + "," + std::to_string(j) + ")";
      ++count;
    }
  }
  return NewIndeterminates(rows, cols, names);
}

int MathematicalProgram::AddIndeterminate(
    const symbolic::Variable& new_indeterminate) {
  if (decision_variable_index_.find(new_indeterminate.get_id()) !=
      decision_variable_index_.end()) {
    throw std::runtime_error(
        fmt::format("{} is a decision variable in the optimization program.",
                    new_indeterminate));
  }
  if (new_indeterminate.get_type() != symbolic::Variable::Type::CONTINUOUS) {
    throw std::runtime_error(
        fmt::format("{} should be of type CONTINUOUS.", new_indeterminate));
  }
  auto it = indeterminates_index_.find(new_indeterminate.get_id());
  if (it == indeterminates_index_.end()) {
    const int var_index = indeterminates_.size();
    indeterminates_index_.insert(
        std::make_pair(new_indeterminate.get_id(), var_index));
    indeterminates_.push_back(new_indeterminate);
    indeterminates_index_.emplace(new_indeterminate.get_id(), var_index);
    return var_index;
  } else {
    return it->second;
  }
}

void MathematicalProgram::AddIndeterminates(
    const Eigen::Ref<const MatrixXDecisionVariable>& new_indeterminates) {
  for (int i = 0; i < new_indeterminates.rows(); ++i) {
    for (int j = 0; j < new_indeterminates.cols(); ++j) {
      const auto& var = new_indeterminates(i, j);
      this->AddIndeterminate(var);
    }
  }
}

void MathematicalProgram::AddIndeterminates(
    const symbolic::Variables& new_indeterminates) {
  for (const auto& var : new_indeterminates) {
    this->AddIndeterminate(var);
  }
}

Binding<VisualizationCallback> MathematicalProgram::AddVisualizationCallback(
    const VisualizationCallback::CallbackFunction& callback,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  visualization_callbacks_.push_back(
      internal::CreateBinding<VisualizationCallback>(
          make_shared<VisualizationCallback>(vars.size(), callback), vars));
  required_capabilities_.insert(ProgramAttribute::kCallback);
  return visualization_callbacks_.back();
}

Binding<Cost> MathematicalProgram::AddCost(const Binding<Cost>& binding) {
  // See AddConstraint(const Binding<Constraint>&) for explanation
  Cost* cost = binding.evaluator().get();
  if (dynamic_cast<QuadraticCost*>(cost)) {
    return AddCost(internal::BindingDynamicCast<QuadraticCost>(binding));
  } else if (dynamic_cast<LinearCost*>(cost)) {
    return AddCost(internal::BindingDynamicCast<LinearCost>(binding));
  } else if (dynamic_cast<L2NormCost*>(cost)) {
    return AddCost(internal::BindingDynamicCast<L2NormCost>(binding));
  } else {
    DRAKE_DEMAND(CheckBinding(binding));
    required_capabilities_.insert(ProgramAttribute::kGenericCost);
    generic_costs_.push_back(binding);
    return generic_costs_.back();
  }
}

Binding<LinearCost> MathematicalProgram::AddCost(
    const Binding<LinearCost>& binding) {
  DRAKE_DEMAND(CheckBinding(binding));
  required_capabilities_.insert(ProgramAttribute::kLinearCost);
  linear_costs_.push_back(binding);
  return linear_costs_.back();
}

Binding<LinearCost> MathematicalProgram::AddLinearCost(const Expression& e) {
  return AddCost(internal::ParseLinearCost(e));
}

Binding<LinearCost> MathematicalProgram::AddLinearCost(
    const Eigen::Ref<const Eigen::VectorXd>& a, double b,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  return AddCost(make_shared<LinearCost>(a, b), vars);
}

Binding<QuadraticCost> MathematicalProgram::AddCost(
    const Binding<QuadraticCost>& binding) {
  DRAKE_DEMAND(CheckBinding(binding));
  required_capabilities_.insert(ProgramAttribute::kQuadraticCost);
  DRAKE_ASSERT(binding.evaluator()->Q().rows() ==
                   static_cast<int>(binding.GetNumElements()) &&
               binding.evaluator()->b().rows() ==
                   static_cast<int>(binding.GetNumElements()));
  quadratic_costs_.push_back(binding);
  return quadratic_costs_.back();
}

Binding<QuadraticCost> MathematicalProgram::AddQuadraticCost(
    const Expression& e, std::optional<bool> is_convex) {
  return AddCost(internal::ParseQuadraticCost(e, is_convex));
}

Binding<QuadraticCost> MathematicalProgram::AddQuadraticErrorCost(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& x_desired,
    const VariableRefList& vars) {
  return AddQuadraticErrorCost(Q, x_desired, ConcatenateVariableRefList(vars));
}

Binding<QuadraticCost> MathematicalProgram::AddQuadraticErrorCost(
    double w, const Eigen::Ref<const Eigen::VectorXd>& x_desired,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  return AddQuadraticErrorCost(
      w * Eigen::MatrixXd::Identity(x_desired.size(), x_desired.size()),
      x_desired, vars);
}

Binding<QuadraticCost> MathematicalProgram::AddQuadraticErrorCost(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& x_desired,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  return AddCost(MakeQuadraticErrorCost(Q, x_desired), vars);
}

Binding<QuadraticCost> MathematicalProgram::AddQuadraticCost(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b, double c,
    const Eigen::Ref<const VectorXDecisionVariable>& vars,
    std::optional<bool> is_convex) {
  return AddCost(make_shared<QuadraticCost>(Q, b, c, is_convex), vars);
}

Binding<QuadraticCost> MathematicalProgram::AddQuadraticCost(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const VectorXDecisionVariable>& vars,
    std::optional<bool> is_convex) {
  return AddQuadraticCost(Q, b, 0., vars, is_convex);
}

Binding<L2NormCost> MathematicalProgram::AddCost(
    const Binding<L2NormCost>& binding) {
  DRAKE_DEMAND(CheckBinding(binding));
  required_capabilities_.insert(ProgramAttribute::kL2NormCost);
  l2norm_costs_.push_back(binding);
  return l2norm_costs_.back();
}

Binding<L2NormCost> MathematicalProgram::AddL2NormCost(
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  return AddCost(std::make_shared<L2NormCost>(A, b), vars);
}

Binding<L2NormCost> MathematicalProgram::AddL2NormCost(
    const symbolic::Expression& e, double psd_tol, double coefficient_tol) {
  return AddCost(internal::ParseL2NormCost(e, psd_tol, coefficient_tol));
}

std::tuple<symbolic::Variable, Binding<LinearCost>,
           Binding<LorentzConeConstraint>>
MathematicalProgram::AddL2NormCostUsingConicConstraint(
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  auto s = this->NewContinuousVariables<1>("slack")(0);
  auto linear_cost =
      this->AddLinearCost(Vector1d(1), 0, Vector1<symbolic::Variable>(s));
  // A_full = [1 0]
  //          [0 A]
  // b_full = [0 b]
  // A_full * [s ; vars] + b_full = [s, A*vars+b]
  Eigen::MatrixXd A_full(A.rows() + 1, A.cols() + 1);
  A_full.setZero();
  A_full(0, 0) = 1;
  A_full.bottomRightCorner(A.rows(), A.cols()) = A;
  Eigen::VectorXd b_full(b.rows() + 1);
  b_full(0) = 0;
  b_full.bottomRows(b.rows()) = b;
  auto lorentz_cone_constraint = this->AddLorentzConeConstraint(
      A_full, b_full, {Vector1<symbolic::Variable>(s), vars});
  return std::make_tuple(s, linear_cost, lorentz_cone_constraint);
}

Binding<PolynomialCost> MathematicalProgram::AddPolynomialCost(
    const Expression& e) {
  auto binding = AddCost(internal::ParsePolynomialCost(e));
  return internal::BindingDynamicCast<PolynomialCost>(binding);
}

Binding<Cost> MathematicalProgram::AddCost(const Expression& e) {
  return AddCost(internal::ParseCost(e));
}

namespace {
void CreateLogDetermiant(
    MathematicalProgram* prog,
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& X,
    VectorX<symbolic::Variable>* t, MatrixX<symbolic::Expression>* Z) {
  DRAKE_DEMAND(X.rows() == X.cols());
  const int X_rows = X.rows();
  auto Z_lower = prog->NewContinuousVariables(X_rows * (X_rows + 1) / 2);
  Z->resize(X_rows, X_rows);
  Z->setZero();
  // diag_Z is the diagonal matrix that only contains the diagonal entries of Z.
  MatrixX<symbolic::Expression> diag_Z(X_rows, X_rows);
  diag_Z.setZero();
  int Z_lower_index = 0;
  for (int j = 0; j < X_rows; ++j) {
    for (int i = j; i < X_rows; ++i) {
      (*Z)(i, j) = Z_lower(Z_lower_index++);
    }
    diag_Z(j, j) = (*Z)(j, j);
  }

  MatrixX<symbolic::Expression> psd_mat(2 * X_rows, 2 * X_rows);
  // clang-format off
  psd_mat << X,             *Z,
             Z->transpose(), diag_Z;
  // clang-format on
  prog->AddLinearMatrixInequalityConstraint(psd_mat);
  // Now introduce the slack variable t.
  *t = prog->NewContinuousVariables(X_rows);
  // Introduce the constraint log(Z(i, i)) >= t(i).
  for (int i = 0; i < X_rows; ++i) {
    prog->AddExponentialConeConstraint(
        Vector3<symbolic::Expression>((*Z)(i, i), 1, (*t)(i)));
  }
}
}  // namespace
std::tuple<Binding<LinearCost>, VectorX<symbolic::Variable>,
           MatrixX<symbolic::Expression>>
MathematicalProgram::AddMaximizeLogDeterminantCost(
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& X) {
  VectorX<symbolic::Variable> t;
  MatrixX<symbolic::Expression> Z;
  CreateLogDetermiant(this, X, &t, &Z);

  const auto cost = AddLinearCost(-Eigen::VectorXd::Ones(t.rows()), t);
  return std::make_tuple(cost, std::move(t), std::move(Z));
}

std::tuple<Binding<LinearConstraint>, VectorX<symbolic::Variable>,
           MatrixX<symbolic::Expression>>
MathematicalProgram::AddLogDeterminantLowerBoundConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& X, double lower) {
  VectorX<symbolic::Variable> t;
  MatrixX<symbolic::Expression> Z;
  CreateLogDetermiant(this, X, &t, &Z);
  const auto constraint =
      AddLinearConstraint(Eigen::RowVectorXd::Ones(t.rows()), lower, kInf, t);
  return std::make_tuple(constraint, std::move(t), std::move(Z));
}

Binding<LinearCost> MathematicalProgram::AddMaximizeGeometricMeanCost(
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x) {
  if (A.rows() != b.rows() || A.cols() != x.rows()) {
    throw std::invalid_argument(
        "MathematicalProgram::AddMaximizeGeometricMeanCost: the argument A, b "
        "and x don't have consistent size.");
  }
  if (A.rows() <= 1) {
    throw std::runtime_error(
        "MathematicalProgram::AddMaximizeGeometricMeanCost: the size of A*x+b "
        "should be at least 2.");
  }
  // We will impose the constraint w(i)² ≤ (A.row(2i) * x + b(2i)) *
  // (A.row(2i+1) * x + b(2i+1)). This could be reformulated as the vector
  // C * [x;w(i)] + d is in the rotated Lorentz cone, where
  // C = [A.row(2i)   0]
  //     [A.row(2i+1) 0]
  //     [0, 0, ...,0 1]
  // d = [b(2i)  ]
  //     [b(2i+1)]
  //     [0      ]
  // The special case is that if A.rows() is an odd number, then for the last
  // entry of w, we will impose (w((A.rows() - 1)/2)² ≤ A.row(A.rows() - 1) * x
  // + b(b.rows() - 1)
  auto w = NewContinuousVariables((A.rows() + 1) / 2);
  DRAKE_ASSERT(w.rows() >= 1);

  VectorX<symbolic::Variable> xw(x.rows() + 1);
  xw.head(x.rows()) = x;
  Eigen::Matrix3Xd C(3, x.rows() + 1);
  for (int i = 0; i < w.size(); ++i) {
    C.setZero();
    C.row(0) << A.row(2 * i), 0;
    Eigen::Vector3d d;
    d(0) = b(2 * i);
    if (2 * i + 1 == A.rows()) {
      // The special case, C.row(1) * x + d(1) = 1.
      C.row(1).setZero();
      d(1) = 1;
    } else {
      // The normal case, C.row(1) * x + d(1) = A.row(2i+1) * x + b(2i+1)
      C.row(1) << A.row(2 * i + 1), 0;
      d(1) = b(2 * i + 1);
    }
    C.row(2).setZero();
    C(2, C.cols() - 1) = 1;
    d(2) = 0;
    xw(x.rows()) = w(i);
    AddRotatedLorentzConeConstraint(C, d, xw);
  }
  if (w.rows() == 1) {
    return AddLinearCost(-w(0));
  }
  return AddMaximizeGeometricMeanCost(w, 1);
}

Binding<LinearCost> MathematicalProgram::AddMaximizeGeometricMeanCost(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x, double c) {
  if (c <= 0) {
    throw std::invalid_argument(
        "MathematicalProgram::AddMaximizeGeometricMeanCost(): c should be "
        "positive.");
  }
  // We maximize the geometric mean through a recursive procedure. If we assume
  // that the size of x is 2ᵏ, then in each iteration, we introduce new slack
  // variables w of size 2ᵏ⁻¹, with the constraint
  // w(i)² ≤ x(2i) * x(2i+1)
  // we then call AddMaximizeGeometricMeanCost(w). This recursion ends until
  // w.size() == 2. We then add the constraint z(0)² ≤ w(0) * w(1), and maximize
  // the cost z(0).
  if (x.rows() <= 1) {
    throw std::invalid_argument(
        "MathematicalProgram::AddMaximizeGeometricMeanCost(): x should have "
        "more than one entry.");
  }
  // We will impose the constraint w(i)² ≤ x(2i) * x(2i+1). Namely the vector
  // [x(2i); x(2i+1); w(i)] is in the rotated Lorentz cone.
  // The special case is when x.rows() = 2n+1, then for the last
  // entry of w, we impose the constraint w(n)² ≤ x(2n), namely the vector
  // [x(2n); 1; w(n)] is in the rotated Lorentz cone.
  auto w = NewContinuousVariables((x.rows() + 1) / 2);
  DRAKE_ASSERT(w.rows() >= 1);
  for (int i = 0; i < w.rows() - 1; ++i) {
    AddRotatedLorentzConeConstraint(
        Vector3<symbolic::Variable>(x(2 * i), x(2 * i + 1), w(i)));
  }
  if (2 * w.rows() == x.rows()) {
    // x has even number of rows.
    AddRotatedLorentzConeConstraint(Vector3<symbolic::Variable>(
        x(x.rows() - 2), x(x.rows() - 1), w(w.rows() - 1)));
  } else {
    // x has odd number of rows.
    // C * xw + d = [x(2n); 1; w(n)], where xw = [x(2n); w(n)].
    Eigen::Matrix<double, 3, 2> C;
    C << 1, 0, 0, 0, 0, 1;
    const Eigen::Vector3d d(0, 1, 0);
    AddRotatedLorentzConeConstraint(
        C, d, Vector2<symbolic::Variable>(x(x.rows() - 1), w(w.rows() - 1)));
  }
  if (x.rows() == 2) {
    return AddLinearCost(-c * w(0));
  }
  return AddMaximizeGeometricMeanCost(w);
}

Binding<Constraint> MathematicalProgram::AddConstraint(
    const Binding<Constraint>& binding) {
  // TODO(eric.cousineau): Use alternative to RTTI.
  // Move kGenericConstraint, etc. to Constraint. Dispatch based on this
  // information. As it is, this causes extra work when we explicitly want a
  // generic constraint.

  // If we get here, then this was possibly a dynamically-simplified
  // constraint. Determine correct container. As last resort, add to generic
  // constraints.
  Constraint* constraint = binding.evaluator().get();
  // Check constraints types in reverse order, such that classes that inherit
  // from other classes will not be prematurely added to less specific (or
  // incorrect) container.
  if (dynamic_cast<LinearMatrixInequalityConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<LinearMatrixInequalityConstraint>(
            binding));
  } else if (dynamic_cast<PositiveSemidefiniteConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<PositiveSemidefiniteConstraint>(binding));
  } else if (dynamic_cast<RotatedLorentzConeConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<RotatedLorentzConeConstraint>(binding));
  } else if (dynamic_cast<LorentzConeConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<LorentzConeConstraint>(binding));
  } else if (dynamic_cast<QuadraticConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<QuadraticConstraint>(binding));
  } else if (dynamic_cast<LinearConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<LinearConstraint>(binding));
  } else {
    if (!CheckBinding(binding)) {
      return binding;
    }
    required_capabilities_.insert(ProgramAttribute::kGenericConstraint);
    generic_constraints_.push_back(binding);
    return generic_constraints_.back();
  }
}

Binding<Constraint> MathematicalProgram::AddConstraint(const Expression& e,
                                                       const double lb,
                                                       const double ub) {
  return AddConstraint(internal::ParseConstraint(e, lb, ub));
}

Binding<Constraint> MathematicalProgram::AddConstraint(
    const Eigen::Ref<const MatrixX<Expression>>& v,
    const Eigen::Ref<const Eigen::MatrixXd>& lb,
    const Eigen::Ref<const Eigen::MatrixXd>& ub) {
  DRAKE_DEMAND(v.rows() == lb.rows());
  DRAKE_DEMAND(v.rows() == ub.rows());
  DRAKE_DEMAND(v.cols() == lb.cols());
  DRAKE_DEMAND(v.cols() == ub.cols());
  return AddConstraint(
      internal::ParseConstraint(Flatten(v), Flatten(lb), Flatten(ub)));
}

Binding<Constraint> MathematicalProgram::AddConstraint(const Formula& f) {
  return AddConstraint(internal::ParseConstraint(f));
}

Binding<LinearConstraint> MathematicalProgram::AddLinearConstraint(
    const Expression& e, const double lb, const double ub) {
  Binding<Constraint> binding = internal::ParseConstraint(e, lb, ub);
  Constraint* constraint = binding.evaluator().get();
  if (dynamic_cast<LinearConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<LinearConstraint>(binding));
  } else {
    std::stringstream oss;
    oss << "Expression " << e << " is non-linear.";
    throw std::runtime_error(oss.str());
  }
}

Binding<LinearConstraint> MathematicalProgram::AddLinearConstraint(
    const Eigen::Ref<const MatrixX<Expression>>& v,
    const Eigen::Ref<const Eigen::MatrixXd>& lb,
    const Eigen::Ref<const Eigen::MatrixXd>& ub) {
  DRAKE_DEMAND(v.rows() == lb.rows());
  DRAKE_DEMAND(v.rows() == ub.rows());
  DRAKE_DEMAND(v.cols() == lb.cols());
  DRAKE_DEMAND(v.cols() == ub.cols());
  Binding<Constraint> binding =
      internal::ParseConstraint(Flatten(v), Flatten(lb), Flatten(ub));
  Constraint* constraint = binding.evaluator().get();
  if (dynamic_cast<LinearConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<LinearConstraint>(binding));
  } else {
    throw std::runtime_error(
        fmt::format("Expression {} is non-linear.", fmt_eigen(v)));
  }
}

Binding<LinearConstraint> MathematicalProgram::AddLinearConstraint(
    const Formula& f) {
  Binding<Constraint> binding = internal::ParseConstraint(f);
  Constraint* constraint = binding.evaluator().get();
  if (dynamic_cast<LinearConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<LinearConstraint>(binding));
  } else {
    std::stringstream oss;
    oss << "Formula " << f << " is non-linear.";
    throw std::runtime_error(oss.str());
  }
}

Binding<LinearConstraint> MathematicalProgram::AddLinearConstraint(
    const Eigen::Ref<const Eigen::Array<symbolic::Formula, Eigen::Dynamic,
                                        Eigen::Dynamic>>& formulas) {
  Binding<Constraint> binding = internal::ParseConstraint(formulas);
  Constraint* constraint = binding.evaluator().get();
  if (dynamic_cast<LinearConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<LinearConstraint>(binding));
  } else {
    std::stringstream oss;
    oss << "Formulas are non-linear.";
    throw std::runtime_error(
        "AddLinearConstraint called but formulas are non-linear");
  }
}

Binding<LinearConstraint> MathematicalProgram::AddConstraint(
    const Binding<LinearConstraint>& binding) {
  // Because the ParseConstraint methods can return instances of
  // LinearEqualityConstraint or BoundingBoxConstraint, do a dynamic check
  // here.
  LinearConstraint* constraint = binding.evaluator().get();
  if (dynamic_cast<BoundingBoxConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<BoundingBoxConstraint>(binding));
  } else if (dynamic_cast<LinearEqualityConstraint*>(constraint)) {
    return AddConstraint(
        internal::BindingDynamicCast<LinearEqualityConstraint>(binding));
  } else {
    // TODO(eric.cousineau): This is a good assertion... But seems out of place,
    // possibly redundant w.r.t. the binding infrastructure.
    DRAKE_ASSERT(binding.evaluator()->get_sparse_A().cols() ==
                 static_cast<int>(binding.GetNumElements()));
    if (!CheckBinding(binding)) {
      return binding;
    }
    required_capabilities_.insert(ProgramAttribute::kLinearConstraint);
    linear_constraints_.push_back(binding);
    return linear_constraints_.back();
  }
}

Binding<LinearConstraint> MathematicalProgram::AddLinearConstraint(
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& lb,
    const Eigen::Ref<const Eigen::VectorXd>& ub,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  return AddConstraint(make_shared<LinearConstraint>(A, lb, ub), vars);
}

Binding<LinearConstraint> MathematicalProgram::AddLinearConstraint(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::Ref<const Eigen::VectorXd>& lb,
    const Eigen::Ref<const Eigen::VectorXd>& ub,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  return AddConstraint(make_shared<LinearConstraint>(A, lb, ub), vars);
}

Binding<LinearEqualityConstraint> MathematicalProgram::AddConstraint(
    const Binding<LinearEqualityConstraint>& binding) {
  DRAKE_ASSERT(binding.evaluator()->get_sparse_A().cols() ==
               static_cast<int>(binding.GetNumElements()));
  if (!CheckBinding(binding)) {
    return binding;
  }
  required_capabilities_.insert(ProgramAttribute::kLinearEqualityConstraint);
  linear_equality_constraints_.push_back(binding);
  return linear_equality_constraints_.back();
}

Binding<LinearEqualityConstraint>
MathematicalProgram::AddLinearEqualityConstraint(const Expression& e,
                                                 double b) {
  return AddConstraint(internal::ParseLinearEqualityConstraint(e, b));
}

Binding<LinearEqualityConstraint>
MathematicalProgram::AddLinearEqualityConstraint(const Formula& f) {
  return AddConstraint(internal::ParseLinearEqualityConstraint(f));
}

Binding<LinearEqualityConstraint>
MathematicalProgram::AddLinearEqualityConstraint(
    const Eigen::Ref<const Eigen::Array<Formula, Eigen::Dynamic,
                                        Eigen::Dynamic>>& formulas) {
  std::set<Formula> formula_set;
  for (int i = 0; i < formulas.rows(); ++i) {
    for (int j = 0; j < formulas.cols(); ++j) {
      if (is_conjunction(formulas(i, j))) {
        for (const Formula& operand : get_operands(formulas(i, j))) {
          formula_set.insert(operand);
        }
      } else {
        formula_set.insert(formulas(i, j));
      }
    }
  }
  return AddConstraint(internal::ParseLinearEqualityConstraint(formula_set));
}

Binding<LinearEqualityConstraint>
MathematicalProgram::AddLinearEqualityConstraint(
    const Eigen::Ref<const Eigen::MatrixXd>& Aeq,
    const Eigen::Ref<const Eigen::VectorXd>& beq,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  return AddConstraint(make_shared<LinearEqualityConstraint>(Aeq, beq), vars);
}

Binding<LinearEqualityConstraint>
MathematicalProgram::AddLinearEqualityConstraint(
    const Eigen::SparseMatrix<double>& Aeq,
    const Eigen::Ref<const Eigen::VectorXd>& beq,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  return AddConstraint(make_shared<LinearEqualityConstraint>(Aeq, beq), vars);
}

Binding<BoundingBoxConstraint> MathematicalProgram::AddConstraint(
    const Binding<BoundingBoxConstraint>& binding) {
  if (!CheckBinding(binding)) {
    return binding;
  }
  DRAKE_ASSERT(binding.evaluator()->num_outputs() ==
               static_cast<int>(binding.GetNumElements()));
  required_capabilities_.insert(ProgramAttribute::kLinearConstraint);
  bbox_constraints_.push_back(binding);
  return bbox_constraints_.back();
}

Binding<LorentzConeConstraint> MathematicalProgram::AddConstraint(
    const Binding<LorentzConeConstraint>& binding) {
  DRAKE_DEMAND(CheckBinding(binding));
  required_capabilities_.insert(ProgramAttribute::kLorentzConeConstraint);
  lorentz_cone_constraint_.push_back(binding);
  return lorentz_cone_constraint_.back();
}

Binding<LorentzConeConstraint> MathematicalProgram::AddLorentzConeConstraint(
    const symbolic::Formula& f, LorentzConeConstraint::EvalType eval_type,
    double psd_tol, double coefficient_tol) {
  return AddConstraint(internal::ParseLorentzConeConstraint(
      f, eval_type, psd_tol, coefficient_tol));
}

Binding<LorentzConeConstraint> MathematicalProgram::AddLorentzConeConstraint(
    const Eigen::Ref<const VectorX<Expression>>& v,
    LorentzConeConstraint::EvalType eval_type) {
  return AddConstraint(internal::ParseLorentzConeConstraint(v, eval_type));
}

Binding<LorentzConeConstraint> MathematicalProgram::AddLorentzConeConstraint(
    const Expression& linear_expression, const Expression& quadratic_expression,
    double tol, LorentzConeConstraint::EvalType eval_type) {
  return AddConstraint(internal::ParseLorentzConeConstraint(
      linear_expression, quadratic_expression, tol, eval_type));
}

Binding<LorentzConeConstraint> MathematicalProgram::AddLorentzConeConstraint(
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const VectorXDecisionVariable>& vars,
    LorentzConeConstraint::EvalType eval_type) {
  shared_ptr<LorentzConeConstraint> constraint =
      make_shared<LorentzConeConstraint>(A, b, eval_type);
  return AddConstraint(Binding<LorentzConeConstraint>(constraint, vars));
}

Binding<RotatedLorentzConeConstraint> MathematicalProgram::AddConstraint(
    const Binding<RotatedLorentzConeConstraint>& binding) {
  DRAKE_DEMAND(CheckBinding(binding));
  required_capabilities_.insert(
      ProgramAttribute::kRotatedLorentzConeConstraint);
  rotated_lorentz_cone_constraint_.push_back(binding);
  return rotated_lorentz_cone_constraint_.back();
}

Binding<RotatedLorentzConeConstraint>
MathematicalProgram::AddRotatedLorentzConeConstraint(
    const symbolic::Expression& linear_expression1,
    const symbolic::Expression& linear_expression2,
    const symbolic::Expression& quadratic_expression, double tol) {
  auto binding = internal::ParseRotatedLorentzConeConstraint(
      linear_expression1, linear_expression2, quadratic_expression, tol);
  AddConstraint(binding);
  return binding;
}

Binding<RotatedLorentzConeConstraint>
MathematicalProgram::AddRotatedLorentzConeConstraint(
    const Eigen::Ref<const VectorX<Expression>>& v) {
  auto binding = internal::ParseRotatedLorentzConeConstraint(v);
  AddConstraint(binding);
  return binding;
}

Binding<RotatedLorentzConeConstraint>
MathematicalProgram::AddRotatedLorentzConeConstraint(
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  shared_ptr<RotatedLorentzConeConstraint> constraint =
      make_shared<RotatedLorentzConeConstraint>(A, b);
  return AddConstraint(constraint, vars);
}

Binding<RotatedLorentzConeConstraint>
MathematicalProgram::AddQuadraticAsRotatedLorentzConeConstraint(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b, double c,
    const Eigen::Ref<const VectorXDecisionVariable>& vars, double psd_tol) {
  auto constraint =
      internal::ParseQuadraticAsRotatedLorentzConeConstraint(Q, b, c, psd_tol);
  return AddConstraint(constraint, vars);
}

Binding<BoundingBoxConstraint> MathematicalProgram::AddBoundingBoxConstraint(
    const Eigen::Ref<const Eigen::MatrixXd>& lb,
    const Eigen::Ref<const Eigen::MatrixXd>& ub,
    const Eigen::Ref<const MatrixXDecisionVariable>& vars) {
  DRAKE_DEMAND(lb.rows() == ub.rows());
  DRAKE_DEMAND(lb.rows() == vars.rows());
  DRAKE_DEMAND(lb.cols() == ub.cols());
  DRAKE_DEMAND(lb.cols() == vars.cols());
  shared_ptr<BoundingBoxConstraint> constraint =
      make_shared<BoundingBoxConstraint>(Flatten(lb), Flatten(ub));
  return AddConstraint(
      Binding<BoundingBoxConstraint>(constraint, Flatten(vars)));
}

Binding<QuadraticConstraint> MathematicalProgram::AddConstraint(
    const Binding<QuadraticConstraint>& binding) {
  DRAKE_DEMAND(CheckBinding(binding));
  required_capabilities_.insert(ProgramAttribute::kQuadraticConstraint);
  quadratic_constraints_.push_back(binding);
  return quadratic_constraints_.back();
}

Binding<QuadraticConstraint> MathematicalProgram::AddQuadraticConstraint(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b, double lb, double ub,
    const Eigen::Ref<const VectorXDecisionVariable>& vars,
    std::optional<QuadraticConstraint::HessianType> hessian_type) {
  auto constraint =
      std::make_shared<QuadraticConstraint>(Q, b, lb, ub, hessian_type);

  return AddConstraint(Binding<QuadraticConstraint>(constraint, vars));
}

Binding<QuadraticConstraint> MathematicalProgram::AddQuadraticConstraint(
    const symbolic::Expression& e, double lb, double ub,
    std::optional<QuadraticConstraint::HessianType> hessian_type) {
  return AddConstraint(
      internal::ParseQuadraticConstraint(e, lb, ub, hessian_type));
}

Binding<LinearComplementarityConstraint> MathematicalProgram::AddConstraint(
    const Binding<LinearComplementarityConstraint>& binding) {
  if (!CheckBinding(binding)) {
    return binding;
  }

  required_capabilities_.insert(
      ProgramAttribute::kLinearComplementarityConstraint);

  linear_complementarity_constraints_.push_back(binding);
  return linear_complementarity_constraints_.back();
}

Binding<LinearComplementarityConstraint>
MathematicalProgram::AddLinearComplementarityConstraint(
    const Eigen::Ref<const Eigen::MatrixXd>& M,
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  shared_ptr<LinearComplementarityConstraint> constraint =
      make_shared<LinearComplementarityConstraint>(M, q);
  return AddConstraint(constraint, vars);
}

Binding<Constraint> MathematicalProgram::AddPolynomialConstraint(
    const Eigen::Ref<const MatrixX<Polynomiald>>& polynomials,
    const vector<Polynomiald::VarType>& poly_vars,
    const Eigen::Ref<const Eigen::MatrixXd>& lb,
    const Eigen::Ref<const Eigen::MatrixXd>& ub,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  DRAKE_DEMAND(polynomials.rows() == lb.rows());
  DRAKE_DEMAND(polynomials.rows() == ub.rows());
  DRAKE_DEMAND(polynomials.cols() == lb.cols());
  DRAKE_DEMAND(polynomials.cols() == ub.cols());
  auto constraint = internal::MakePolynomialConstraint(
      Flatten(polynomials), poly_vars, Flatten(lb), Flatten(ub));
  return AddConstraint(constraint, vars);
}

Binding<PositiveSemidefiniteConstraint> MathematicalProgram::AddConstraint(
    const Binding<PositiveSemidefiniteConstraint>& binding) {
  if (!CheckBinding(binding)) {
    return binding;
  }
  DRAKE_ASSERT(math::IsSymmetric(Eigen::Map<const MatrixXDecisionVariable>(
      binding.variables().data(), binding.evaluator()->matrix_rows(),
      binding.evaluator()->matrix_rows())));
  required_capabilities_.insert(
      ProgramAttribute::kPositiveSemidefiniteConstraint);
  positive_semidefinite_constraint_.push_back(binding);
  return positive_semidefinite_constraint_.back();
}

Binding<PositiveSemidefiniteConstraint> MathematicalProgram::AddConstraint(
    shared_ptr<PositiveSemidefiniteConstraint> con,
    const Eigen::Ref<const MatrixXDecisionVariable>& symmetric_matrix_var) {
  DRAKE_ASSERT(math::IsSymmetric(symmetric_matrix_var));
  return AddConstraint(CreateBinding(con, Flatten(symmetric_matrix_var)));
}

Binding<PositiveSemidefiniteConstraint>
MathematicalProgram::AddPositiveSemidefiniteConstraint(
    const Eigen::Ref<const MatrixXDecisionVariable>& symmetric_matrix_var) {
  auto constraint =
      make_shared<PositiveSemidefiniteConstraint>(symmetric_matrix_var.rows());
  return AddConstraint(constraint, symmetric_matrix_var);
}

Binding<PositiveSemidefiniteConstraint>
MathematicalProgram::AddPositiveSemidefiniteConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& e) {
  DRAKE_THROW_UNLESS(e.rows() == e.cols());
  DRAKE_ASSERT(CheckStructuralEquality(e, e.transpose().eval()));
  const MatrixXDecisionVariable M = NewSymmetricContinuousVariables(e.rows());
  // Adds the linear equality constraint that M = e.
  AddLinearEqualityConstraint(e - M, Eigen::MatrixXd::Zero(e.rows(), e.rows()),
                              true);
  return AddPositiveSemidefiniteConstraint(M);
}

Binding<PositiveSemidefiniteConstraint>
MathematicalProgram::AddPrincipalSubmatrixIsPsdConstraint(
    const Eigen::Ref<const MatrixXDecisionVariable>& symmetric_matrix_var,
    const std::set<int>& minor_indices) {
  // This function relies on AddPositiveSemidefiniteConstraint to validate the
  // documented symmetry prerequisite.
  return AddPositiveSemidefiniteConstraint(
      math::ExtractPrincipalSubmatrix(symmetric_matrix_var, minor_indices));
}

Binding<LinearMatrixInequalityConstraint>
MathematicalProgram::AddPrincipalSubmatrixIsPsdConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& e,
    const std::set<int>& minor_indices) {
  // This function relies on AddLinearMatrixInequalityConstraint to validate the
  // documented symmetry prerequisite.
  return AddLinearMatrixInequalityConstraint(
      math::ExtractPrincipalSubmatrix(e, minor_indices));
}

Binding<LinearMatrixInequalityConstraint> MathematicalProgram::AddConstraint(
    const Binding<LinearMatrixInequalityConstraint>& binding) {
  if (!CheckBinding(binding)) {
    return binding;
  }
  DRAKE_ASSERT(static_cast<int>(binding.evaluator()->F().size()) ==
               static_cast<int>(binding.GetNumElements()) + 1);
  required_capabilities_.insert(
      ProgramAttribute::kPositiveSemidefiniteConstraint);
  linear_matrix_inequality_constraint_.push_back(binding);
  return linear_matrix_inequality_constraint_.back();
}

Binding<LinearMatrixInequalityConstraint>
MathematicalProgram::AddLinearMatrixInequalityConstraint(
    vector<Eigen::MatrixXd> F,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  auto constraint = make_shared<LinearMatrixInequalityConstraint>(std::move(F));
  return AddConstraint(constraint, vars);
}

Binding<LinearMatrixInequalityConstraint>
MathematicalProgram::AddLinearMatrixInequalityConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& X) {
  DRAKE_THROW_UNLESS(X.rows() == X.cols());
  DRAKE_ASSERT(CheckStructuralEquality(X, X.transpose().eval()));
  std::vector<symbolic::Variable> vars_vec;
  std::unordered_map<symbolic::Variable::Id, int> map_var_to_index;
  for (int j = 0; j < X.cols(); ++j) {
    for (int i = j; i < X.rows(); ++i) {
      symbolic::ExtractAndAppendVariablesFromExpression(X(i, j), &vars_vec,
                                                        &map_var_to_index);
    }
  }
  std::vector<Eigen::MatrixXd> F(vars_vec.size() + 1,
                                 Eigen::MatrixXd::Zero(X.rows(), X.rows()));
  Eigen::RowVectorXd coeffs(vars_vec.size());
  double constant_term{};
  for (int j = 0; j < X.cols(); ++j) {
    for (int i = j; i < X.rows(); ++i) {
      DecomposeAffineExpression(X(i, j), map_var_to_index, &coeffs,
                                &constant_term);
      F[0](i, j) = constant_term;
      F[0](j, i) = F[0](i, j);
      for (int k = 0; k < ssize(vars_vec); ++k) {
        F[1 + k](i, j) = coeffs(k);
        F[1 + k](j, i) = F[1 + k](i, j);
      }
    }
  }
  return AddLinearMatrixInequalityConstraint(
      F, Eigen::Map<VectorX<symbolic::Variable>>(vars_vec.data(),
                                                 vars_vec.size()));
}

MatrixX<symbolic::Expression>
MathematicalProgram::AddPositiveDiagonallyDominantMatrixConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& X) {
  // First create the slack variables Y with the same size as X, Y being the
  // symmetric matrix representing the absolute value of X.
  const int num_X_rows = X.rows();
  DRAKE_DEMAND(X.cols() == num_X_rows);
  auto Y_upper = NewContinuousVariables((num_X_rows - 1) * num_X_rows / 2,
                                        "diagonally_dominant_slack");
  MatrixX<symbolic::Expression> Y(num_X_rows, num_X_rows);
  int Y_upper_count = 0;
  // Fill in the upper triangle of Y.
  for (int j = 0; j < num_X_rows; ++j) {
    for (int i = 0; i < j; ++i) {
      Y(i, j) = Y_upper(Y_upper_count);
      Y(j, i) = Y(i, j);
      ++Y_upper_count;
    }
    // The diagonal entries of Y.
    Y(j, j) = X(j, j);
  }
  // Add the constraint that Y(i, j) >= |X(i, j) + X(j, i) / 2|
  for (int i = 0; i < num_X_rows; ++i) {
    for (int j = i + 1; j < num_X_rows; ++j) {
      AddLinearConstraint(Y(i, j) >= (X(i, j) + X(j, i)) / 2);
      AddLinearConstraint(Y(i, j) >= -(X(i, j) + X(j, i)) / 2);
    }
  }

  // Add the constraint X(i, i) >= sum_j Y(i, j), j ≠ i
  for (int i = 0; i < num_X_rows; ++i) {
    symbolic::Expression y_sum = 0;
    for (int j = 0; j < num_X_rows; ++j) {
      if (j == i) {
        continue;
      }
      y_sum += Y(i, j);
    }
    AddLinearConstraint(X(i, i) >= y_sum);
  }
  return Y;
}

MatrixX<symbolic::Expression> MathematicalProgram::TightenPsdConstraintToDd(
    const Binding<PositiveSemidefiniteConstraint>& constraint) {
  RemoveConstraint(constraint);
  // Variables are flattened by the Flatten method, which flattens in
  // column-major order. This is the same convention as Eigen, so we can use the
  // map methods.
  const int n = constraint.evaluator()->matrix_rows();
  const MatrixXDecisionVariable mat_vars =
      Eigen::Map<const MatrixXDecisionVariable>(constraint.variables().data(),
                                                n, n);
  return AddPositiveDiagonallyDominantMatrixConstraint(
      mat_vars.cast<Expression>());
}

namespace {

// Constructs the matrices A, lb, ub for the linear constraint lb <= A * X <= ub
// encoding that X is in DD* for a matrix of size n. Returns the tuple
// (A, lb, ub).
std::tuple<Eigen::SparseMatrix<double>, Eigen::VectorXd, Eigen::VectorXd>
ConstructPositiveDiagonallyDominantDualConeConstraintMatricesForN(const int n) {
  // Return the index of Xᵢⱼ in the vector created by stacking the column of X
  // into a vector.
  auto compute_flat_index = [&n](int i, int j) {
    return i + n * j;
  };

  // The DD dual cone constraint is a sparse linear constraint. We instantiate
  // the A matrix using this triplet list.
  std::vector<Eigen::Triplet<double>> A_triplet_list;
  // There are n rows with one non-zero entry in the row, and 2 * (n choose 2)
  // rows with 4 non-zero entries in the row. This requires 4*n*n-3*n non-zero
  // entries.
  A_triplet_list.reserve(4 * n * n - 3 * n);

  const Eigen::VectorXd lb = Eigen::VectorXd::Zero(n * n);
  const Eigen::VectorXd ub = kInf * Eigen::VectorXd::Ones(n * n);

  // vᵢᵀXvᵢ ≥ 0 is equivalent to Xᵢᵢ ≥ 0 when vᵢ is a vector with exactly one
  // entry equal to 1.
  for (int i = 0; i < n; ++i) {
    // Variable Xᵢᵢ is in position i*(n+1)
    A_triplet_list.emplace_back(i, compute_flat_index(i, i), 1);
  }
  // When vᵢ is a vector with two non-zero at entries k and j, we can choose
  // without loss of generality that the jth entry to be 1, and the kth entry be
  // either +1 or -1. This enumerates over all the parities of vᵢ. Under this
  // choice vᵢᵀXvᵢ = Xₖₖ + sign(vᵢ(k))* (Xₖⱼ+ Xⱼₖ) +  Xⱼⱼ
  int row_ctr = n;
  for (int j = 0; j < n; ++j) {
    for (int k = j + 1; k < n; ++k) {
      // X(k, k) + X(k, j) + X(j, k) + X(j, j)
      A_triplet_list.emplace_back(row_ctr, compute_flat_index(k, k), 1);
      A_triplet_list.emplace_back(row_ctr, compute_flat_index(j, j), 1);
      A_triplet_list.emplace_back(row_ctr, compute_flat_index(j, k), 1);
      A_triplet_list.emplace_back(row_ctr, compute_flat_index(k, j), 1);
      ++row_ctr;

      // X(k, k) - X(k, j) - X(j, k) + X(j, j)
      A_triplet_list.emplace_back(row_ctr, compute_flat_index(k, k), 1);
      A_triplet_list.emplace_back(row_ctr, compute_flat_index(j, j), 1);
      A_triplet_list.emplace_back(row_ctr, compute_flat_index(j, k), -1);
      A_triplet_list.emplace_back(row_ctr, compute_flat_index(k, j), -1);
      ++row_ctr;
    }
  }
  DRAKE_ASSERT(row_ctr == n * n);
  DRAKE_ASSERT(ssize(A_triplet_list) == 4 * n * n - 3 * n);
  Eigen::SparseMatrix<double> A(row_ctr, n * n);
  A.setFromTriplets(A_triplet_list.begin(), A_triplet_list.end());
  return std::make_tuple(std::move(A), std::move(lb), std::move(ub));
}
}  // namespace

Binding<LinearConstraint>
MathematicalProgram::AddPositiveDiagonallyDominantDualConeMatrixConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& X) {
  const int n = X.rows();
  DRAKE_DEMAND(X.cols() == n);
  Eigen::MatrixXd A_expr;
  Eigen::VectorXd b_expr;
  VectorX<Variable> variables;
  symbolic::DecomposeAffineExpressions(
      Eigen::Map<const VectorX<symbolic::Expression>>(X.data(), X.size()),
      &A_expr, &b_expr, &variables);
  const std::tuple<Eigen::SparseMatrix<double>, Eigen::VectorXd,
                   Eigen::VectorXd>
      constraint_mats{
          ConstructPositiveDiagonallyDominantDualConeConstraintMatricesForN(n)};
  return AddLinearConstraint(
      (std::get<0>(constraint_mats) * A_expr).sparseView(),  // A * A_expr
      std::get<1>(constraint_mats) -
          std::get<0>(constraint_mats) * b_expr,  // lb - A * b_expr
      std::get<2>(constraint_mats),  // ub - A * b_expr, but since ub is kInf no
                                     // need to do the operations
      variables);
}

Binding<LinearConstraint>
MathematicalProgram::AddPositiveDiagonallyDominantDualConeMatrixConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& X) {
  const int n = X.rows();
  DRAKE_DEMAND(X.cols() == n);
  const std::tuple<Eigen::SparseMatrix<double>, Eigen::VectorXd,
                   Eigen::VectorXd>
      constraint_mats{
          ConstructPositiveDiagonallyDominantDualConeConstraintMatricesForN(n)};
  return AddLinearConstraint(
      std::get<0>(constraint_mats), std::get<1>(constraint_mats),
      std::get<2>(constraint_mats),
      Eigen::Map<const VectorXDecisionVariable>(X.data(), X.size()));
}

Binding<LinearConstraint> MathematicalProgram::RelaxPsdConstraintToDdDualCone(
    const Binding<PositiveSemidefiniteConstraint>& constraint) {
  RemoveConstraint(constraint);
  // Variables are flattened by the Flatten method, which flattens in
  // column-major order. This is the same convention as Eigen, so we can use the
  // map methods.
  const int n = constraint.evaluator()->matrix_rows();
  const MatrixXDecisionVariable mat_vars =
      Eigen::Map<const MatrixXDecisionVariable>(constraint.variables().data(),
                                                n, n);
  return AddPositiveDiagonallyDominantDualConeMatrixConstraint(mat_vars);
}

namespace {
// Add the slack variable for scaled diagonally dominant matrix constraint. In
// AddScaledDiagonallyDominantMatrixConstraint, we should add the constraint
// that the diagonal terms in the sdd matrix should match the summation of
// the diagonally terms in the slack variable, and the upper diagonal corner
// in M[i][j] should satisfy the rotated Lorentz cone constraint.
template <typename T>
void AddSlackVariableForScaledDiagonallyDominantMatrixConstraint(
    const Eigen::Ref<const MatrixX<T>>& X, MathematicalProgram* prog,
    Eigen::Matrix<symbolic::Variable, 2, Eigen::Dynamic>* M_ij_diagonal,
    std::vector<std::vector<Matrix2<T>>>* M) {
  const int n = X.rows();
  DRAKE_DEMAND(X.cols() == n);
  // The diagonal terms of M[i][j] are new variables.
  // M[i][j](0, 0) = M_ij_diagonal(0, k)
  // M[i][j](1, 1) = M_ij_diagonal(1, k)
  // where k = (2n - 1) * i / 2 + j - i - 1, namely k is the index of X(i, j)
  // in the vector X_upper_diagonal, where X_upper_diagonal is obtained by
  // stacking each row of the upper diagonal part (not including the diagonal
  // entries) in X to a row vector.
  *M_ij_diagonal = prog->NewContinuousVariables<2, Eigen::Dynamic>(
      2, (n - 1) * n / 2, "sdd_slack_M");
  int k = 0;
  M->resize(n);
  for (int i = 0; i < n; ++i) {
    (*M)[i].resize(n);
    for (int j = i + 1; j < n; ++j) {
      (*M)[i][j](0, 0) = (*M_ij_diagonal)(0, k);
      (*M)[i][j](1, 1) = (*M_ij_diagonal)(1, k);
      (*M)[i][j](0, 1) = X(i, j);
      (*M)[i][j](1, 0) = X(j, i);
      ++k;
    }
  }
}
}  // namespace

std::vector<std::vector<Matrix2<symbolic::Expression>>>
MathematicalProgram::AddScaledDiagonallyDominantMatrixConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& X) {
  const int n = X.rows();
  std::vector<std::vector<Matrix2<symbolic::Expression>>> M(n);
  Matrix2X<symbolic::Variable> M_ij_diagonal;
  AddSlackVariableForScaledDiagonallyDominantMatrixConstraint<
      symbolic::Expression>(X, this, &M_ij_diagonal, &M);
  for (int i = 0; i < n; ++i) {
    symbolic::Expression diagonal_sum = 0;
    for (int j = 0; j < i; ++j) {
      diagonal_sum += M[j][i](1, 1);
    }
    for (int j = i + 1; j < n; ++j) {
      diagonal_sum += M[i][j](0, 0);
      AddRotatedLorentzConeConstraint(Vector3<symbolic::Expression>(
          M[i][j](0, 0), M[i][j](1, 1), M[i][j](0, 1)));
    }
    AddLinearEqualityConstraint(X(i, i) - diagonal_sum, 0);
  }
  return M;
}

std::vector<std::vector<Matrix2<symbolic::Variable>>>
MathematicalProgram::AddScaledDiagonallyDominantMatrixConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& X) {
  const int n = X.rows();
  std::vector<std::vector<Matrix2<symbolic::Variable>>> M(n);
  Matrix2X<symbolic::Variable> M_ij_diagonal;
  AddSlackVariableForScaledDiagonallyDominantMatrixConstraint<
      symbolic::Variable>(X, this, &M_ij_diagonal, &M);

  // k is the index of X(i, j) in the vector X_upper_diagonal, where
  // X_upper_diagonal is obtained by stacking each row of the upper diagonal
  // part in X to a row vector.
  auto ij_to_k = [&n](int i, int j) {
    return (2 * n - 1 - i) * i / 2 + j - i - 1;
  };
  // diagonal_sum_var = [M_ij_diagonal(:); X(0, 0); X(1, 1); ...; X(n-1, n-1)]
  const int n_square = n * n;
  VectorXDecisionVariable diagonal_sum_var(n_square);
  for (int i = 0; i < (n_square - n) / 2; ++i) {
    diagonal_sum_var.segment<2>(2 * i) = M_ij_diagonal.col(i);
  }
  for (int i = 0; i < n; ++i) {
    diagonal_sum_var(n_square - n + i) = X(i, i);
  }

  // Create a RotatedLorentzConeConstraint
  auto rotated_lorentz_cone_constraint =
      std::make_shared<RotatedLorentzConeConstraint>(
          Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  // A_diagonal_sum.row(i) * diagonal_sum_var = M[0][i](1, 1) + M[1][i](1, 1) +
  // ... + M[i-1][i](1, 1) - X(i, i) + M[i][i+1](0, 0) + M[i][i+2](0, 0) + ... +
  // M[i][n-1](0, 0);
  Eigen::MatrixXd A_diagonal_sum(n, n_square);
  A_diagonal_sum.setZero();
  for (int i = 0; i < n; ++i) {
    // The coefficient for X(i, i)
    A_diagonal_sum(i, n_square - n + i) = -1;
    for (int j = 0; j < i; ++j) {
      // The coefficient for M[j][i](1, 1)
      A_diagonal_sum(i, 2 * ij_to_k(j, i) + 1) = 1;
    }
    for (int j = i + 1; j < n; ++j) {
      // The coefficient for M[i][j](0, 0)
      A_diagonal_sum(i, 2 * ij_to_k(i, j)) = 1;
      // Bind the rotated Lorentz cone constraint to (M[i][j](0, 0); M[i][j](1,
      // 1); M[i][j](0, 1))
      AddConstraint(rotated_lorentz_cone_constraint,
                    Vector3<symbolic::Variable>(M[i][j](0, 0), M[i][j](1, 1),
                                                M[i][j](0, 1)));
    }
  }
  AddLinearEqualityConstraint(A_diagonal_sum, Eigen::VectorXd::Zero(n),
                              diagonal_sum_var);
  return M;
}

std::vector<std::vector<Matrix2<symbolic::Variable>>>
MathematicalProgram::TightenPsdConstraintToSdd(
    const Binding<PositiveSemidefiniteConstraint>& constraint) {
  RemoveConstraint(constraint);
  // Variables are flattened by the Flatten method, which flattens in
  // column-major order. This is the same convention as Eigen, so we can use the
  // map methods.
  const int n = constraint.evaluator()->matrix_rows();
  const MatrixXDecisionVariable mat_vars =
      Eigen::Map<const MatrixXDecisionVariable>(constraint.variables().data(),
                                                n, n);
  return AddScaledDiagonallyDominantMatrixConstraint(mat_vars);
}

std::vector<Binding<RotatedLorentzConeConstraint>>
MathematicalProgram::AddScaledDiagonallyDominantDualConeMatrixConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& X) {
  const int n = X.rows();
  DRAKE_DEMAND(X.cols() == n);
  std::vector<Binding<RotatedLorentzConeConstraint>> ret;
  ret.reserve(n);

  // if i ≥ j
  for (int i = 0; i < n; ++i) {
    // VᵢⱼᵀXVᵢⱼ = [[Xᵢᵢ, Xᵢⱼ],[Xᵢⱼ,Xⱼⱼ]] and this matrix is PSD if an only if
    // Xᵢᵢ ≥ 0, Xⱼⱼ ≥ 0, and XᵢᵢXⱼⱼ - XᵢⱼXᵢⱼ >= 0. Notice that if i == j, this
    // is simply that Xᵢᵢ ≥ 0 and so we don't need to include the Xᵢᵢ ≥ 0 as it
    // is already added when i ≠ j.
    for (int j = i + 1; j < n; ++j) {
      // VᵢⱼᵀXVᵢⱼ = [[Xᵢᵢ, Xᵢⱼ],[Xᵢⱼ,Xⱼⱼ]]. Since we already imposed that Xᵢᵢ ≥
      // 0 and Xⱼⱼ ≥ 0, we only have to impose that XᵢᵢXⱼⱼ - XᵢⱼXᵢⱼ >= 0 which
      // can be
      ret.push_back(
          AddRotatedLorentzConeConstraint(Vector3<symbolic::Expression>(
              X(i, i), X(j, j), 0.5 * (X(i, j) + X(j, i)))));
    }
  }
  return ret;
}

std::vector<Binding<RotatedLorentzConeConstraint>>
MathematicalProgram::AddScaledDiagonallyDominantDualConeMatrixConstraint(
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& X) {
  return AddScaledDiagonallyDominantDualConeMatrixConstraint(
      X.cast<Expression>());
}

std::vector<Binding<RotatedLorentzConeConstraint>>
MathematicalProgram::RelaxPsdConstraintToSddDualCone(
    const Binding<PositiveSemidefiniteConstraint>& constraint) {
  RemoveConstraint(constraint);
  // Variables are flattened by the Flatten method, which flattens in
  // column-major order. This is the same convention as Eigen, so we can use the
  // map methods.
  const int n = constraint.evaluator()->matrix_rows();
  const MatrixXDecisionVariable mat_vars =
      Eigen::Map<const MatrixXDecisionVariable>(constraint.variables().data(),
                                                n, n);
  return AddScaledDiagonallyDominantDualConeMatrixConstraint(mat_vars);
}

Binding<ExponentialConeConstraint> MathematicalProgram::AddConstraint(
    const Binding<ExponentialConeConstraint>& binding) {
  DRAKE_DEMAND(CheckBinding(binding));
  required_capabilities_.insert(ProgramAttribute::kExponentialConeConstraint);
  exponential_cone_constraints_.push_back(binding);
  return exponential_cone_constraints_.back();
}

Binding<ExponentialConeConstraint>
MathematicalProgram::AddExponentialConeConstraint(
    const Eigen::Ref<const Eigen::SparseMatrix<double>>& A,
    const Eigen::Ref<const Eigen::Vector3d>& b,
    const Eigen::Ref<const VectorXDecisionVariable>& vars) {
  auto constraint = std::make_shared<ExponentialConeConstraint>(A, b);
  return AddConstraint(constraint, vars);
}

Binding<ExponentialConeConstraint>
MathematicalProgram::AddExponentialConeConstraint(
    const Eigen::Ref<const Vector3<symbolic::Expression>>& z) {
  Eigen::MatrixXd A{};
  Eigen::VectorXd b(3);
  VectorXDecisionVariable vars{};
  symbolic::DecomposeAffineExpressions(z, &A, &b, &vars);
  return AddExponentialConeConstraint(A.sparseView(), Eigen::Vector3d(b), vars);
}

std::vector<Binding<Cost>> MathematicalProgram::GetAllCosts() const {
  auto costlist = generic_costs_;
  costlist.insert(costlist.end(), linear_costs_.begin(), linear_costs_.end());
  costlist.insert(costlist.end(), quadratic_costs_.begin(),
                  quadratic_costs_.end());
  costlist.insert(costlist.end(), l2norm_costs_.begin(), l2norm_costs_.end());
  return costlist;
}

std::vector<Binding<LinearConstraint>>
MathematicalProgram::GetAllLinearConstraints() const {
  std::vector<Binding<LinearConstraint>> conlist = linear_constraints_;
  conlist.insert(conlist.end(), linear_equality_constraints_.begin(),
                 linear_equality_constraints_.end());
  return conlist;
}

std::vector<Binding<Constraint>> MathematicalProgram::GetAllConstraints()
    const {
  std::vector<Binding<Constraint>> conlist = generic_constraints_;
  auto extend = [&conlist](auto container) {
    conlist.insert(conlist.end(), container.begin(), container.end());
  };
  extend(quadratic_constraints_);
  extend(linear_constraints_);
  extend(linear_equality_constraints_);
  extend(bbox_constraints_);
  extend(lorentz_cone_constraint_);
  extend(rotated_lorentz_cone_constraint_);
  extend(linear_matrix_inequality_constraint_);
  extend(positive_semidefinite_constraint_);
  extend(linear_complementarity_constraints_);
  extend(exponential_cone_constraints_);
  return conlist;
}

int MathematicalProgram::FindDecisionVariableIndex(const Variable& var) const {
  auto it = decision_variable_index_.find(var.get_id());
  if (it == decision_variable_index_.end()) {
    ostringstream oss;
    oss << var
        << " is not a decision variable in the mathematical program, "
           "when calling FindDecisionVariableIndex.\n";
    throw runtime_error(oss.str());
  }
  return it->second;
}

std::vector<int> MathematicalProgram::FindDecisionVariableIndices(
    const Eigen::Ref<const VectorXDecisionVariable>& vars) const {
  std::vector<int> x_indices(vars.rows());
  for (int i = 0; i < vars.rows(); ++i) {
    x_indices[i] = FindDecisionVariableIndex(vars(i));
  }
  return x_indices;
}

size_t MathematicalProgram::FindIndeterminateIndex(const Variable& var) const {
  auto it = indeterminates_index_.find(var.get_id());
  if (it == indeterminates_index_.end()) {
    ostringstream oss;
    oss << var
        << " is not an indeterminate in the mathematical program, "
           "when calling GetSolution.\n";
    throw runtime_error(oss.str());
  }
  return it->second;
}

bool MathematicalProgram::CheckSatisfied(
    const Binding<Constraint>& binding,
    const Eigen::Ref<const Eigen::VectorXd>& prog_var_vals, double tol) const {
  const Eigen::VectorXd vals = GetBindingVariableValues(binding, prog_var_vals);
  return binding.evaluator()->CheckSatisfied(vals, tol);
}

bool MathematicalProgram::CheckSatisfied(
    const std::vector<Binding<Constraint>>& bindings,
    const Eigen::Ref<const Eigen::VectorXd>& prog_var_vals, double tol) const {
  for (const auto& b : bindings) {
    if (!CheckSatisfied(b, prog_var_vals, tol)) {
      return false;
    }
  }
  return true;
}

bool MathematicalProgram::CheckSatisfiedAtInitialGuess(
    const Binding<Constraint>& binding, double tol) const {
  return CheckSatisfied(binding, x_initial_guess_, tol);
}

bool MathematicalProgram::CheckSatisfiedAtInitialGuess(
    const std::vector<Binding<Constraint>>& bindings, double tol) const {
  return CheckSatisfied(bindings, x_initial_guess_, tol);
}

namespace {
// Body of MathematicalProgram::AddSosConstraint(const symbolic::Polynomial&,
// const Eigen::Ref<const VectorX<symbolic::Monomial>>&).
MatrixXDecisionVariable DoAddSosConstraint(
    MathematicalProgram* const prog, const symbolic::Polynomial& p,
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    MathematicalProgram::NonnegativePolynomial type,
    const std::string& gram_name) {
  const auto pair = prog->NewSosPolynomial(monomial_basis, type, gram_name);
  const symbolic::Polynomial& sos_poly{pair.first};
  const MatrixXDecisionVariable& Q{pair.second};

  const symbolic::Polynomial poly_diff = sos_poly - p;

  for (const auto& term : poly_diff.monomial_to_coefficient_map()) {
    prog->AddLinearEqualityConstraint(term.second, 0);
  }
  return Q;
}
// Body of MathematicalProgram::AddSosConstraint(const symbolic::Polynomial&).
pair<MatrixXDecisionVariable, VectorX<symbolic::Monomial>> DoAddSosConstraint(
    MathematicalProgram* const prog, const symbolic::Polynomial& p,
    MathematicalProgram::NonnegativePolynomial type,
    const std::string& gram_name) {
  const symbolic::Polynomial p_expanded = p.Expand();
  const VectorX<symbolic::Monomial> m = ConstructMonomialBasis(p_expanded);
  const MatrixXDecisionVariable Q =
      prog->AddSosConstraint(p_expanded, m, type, gram_name);
  return std::make_pair(Q, m);
}

}  // namespace

MatrixXDecisionVariable MathematicalProgram::AddSosConstraint(
    const symbolic::Polynomial& p,
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    MathematicalProgram::NonnegativePolynomial type,
    const std::string& gram_name) {
  const Variables indeterminates_vars{indeterminates()};
  if (Variables(p.indeterminates()).IsSubsetOf(indeterminates_vars) &&
      intersect(indeterminates_vars, Variables(p.decision_variables()))
          .empty()) {
    return DoAddSosConstraint(this, p, monomial_basis, type, gram_name);
  } else {
    // Need to reparse p, we first make a copy of p and reparse that.
    symbolic::Polynomial p_reparsed{p};
    Reparse(&p_reparsed);
    return DoAddSosConstraint(this, p_reparsed, monomial_basis, type,
                              gram_name);
  }
}

pair<MatrixXDecisionVariable, VectorX<symbolic::Monomial>>
MathematicalProgram::AddSosConstraint(
    const symbolic::Polynomial& p,
    MathematicalProgram::NonnegativePolynomial type,
    const std::string& gram_name) {
  const Variables indeterminates_vars{indeterminates()};
  if (Variables(p.indeterminates()).IsSubsetOf(indeterminates_vars) &&
      intersect(indeterminates_vars, Variables(p.decision_variables()))
          .empty()) {
    return DoAddSosConstraint(this, p, type, gram_name);
  } else {
    // Need to reparse p, we first make a copy of p and reparse that.
    symbolic::Polynomial p_reparsed{p};
    Reparse(&p_reparsed);
    return DoAddSosConstraint(this, p_reparsed, type, gram_name);
  }
}

MatrixXDecisionVariable MathematicalProgram::AddSosConstraint(
    const symbolic::Expression& e,
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    MathematicalProgram::NonnegativePolynomial type,
    const std::string& gram_name) {
  return AddSosConstraint(
      symbolic::Polynomial{e, symbolic::Variables{this->indeterminates()}},
      monomial_basis, type, gram_name);
}

pair<MatrixXDecisionVariable, VectorX<symbolic::Monomial>>
MathematicalProgram::AddSosConstraint(
    const symbolic::Expression& e,
    MathematicalProgram::NonnegativePolynomial type,
    const std::string& gram_name) {
  return AddSosConstraint(
      symbolic::Polynomial{e, symbolic::Variables{this->indeterminates()}},
      type, gram_name);
}

std::vector<Binding<LinearEqualityConstraint>>
MathematicalProgram::AddEqualityConstraintBetweenPolynomials(
    const symbolic::Polynomial& p1, const symbolic::Polynomial& p2) {
  symbolic::Polynomial poly_diff = p1 - p2;
  Reparse(&poly_diff);
  std::vector<Binding<LinearEqualityConstraint>> ret;
  for (const auto& item : poly_diff.monomial_to_coefficient_map()) {
    ret.push_back(AddLinearEqualityConstraint(item.second, 0));
  }
  return ret;
}

double MathematicalProgram::GetInitialGuess(
    const symbolic::Variable& decision_variable) const {
  return x_initial_guess_[FindDecisionVariableIndex(decision_variable)];
}

void MathematicalProgram::SetInitialGuess(
    const symbolic::Variable& decision_variable, double variable_guess_value) {
  x_initial_guess_(FindDecisionVariableIndex(decision_variable)) =
      variable_guess_value;
}

void MathematicalProgram::SetDecisionVariableValueInVector(
    const symbolic::Variable& decision_variable,
    double decision_variable_new_value,
    EigenPtr<Eigen::VectorXd> values) const {
  DRAKE_THROW_UNLESS(values != nullptr);
  DRAKE_THROW_UNLESS(values->size() == num_vars());
  const int index = FindDecisionVariableIndex(decision_variable);
  (*values)(index) = decision_variable_new_value;
}

void MathematicalProgram::SetDecisionVariableValueInVector(
    const Eigen::Ref<const MatrixXDecisionVariable>& decision_variables,
    const Eigen::Ref<const Eigen::MatrixXd>& decision_variables_new_values,
    EigenPtr<Eigen::VectorXd> values) const {
  DRAKE_THROW_UNLESS(values != nullptr);
  DRAKE_THROW_UNLESS(values->size() == num_vars());
  DRAKE_THROW_UNLESS(decision_variables.rows() ==
                     decision_variables_new_values.rows());
  DRAKE_THROW_UNLESS(decision_variables.cols() ==
                     decision_variables_new_values.cols());
  for (int i = 0; i < decision_variables.rows(); ++i) {
    for (int j = 0; j < decision_variables.cols(); ++j) {
      const int index = FindDecisionVariableIndex(decision_variables(i, j));
      (*values)(index) = decision_variables_new_values(i, j);
    }
  }
}

void MathematicalProgram::AppendNanToEnd(int new_var_size, Eigen::VectorXd* v) {
  v->conservativeResize(v->rows() + new_var_size);
  v->tail(new_var_size).fill(std::numeric_limits<double>::quiet_NaN());
}

void MathematicalProgram::EvalVisualizationCallbacks(
    const Eigen::Ref<const Eigen::VectorXd>& prog_var_vals) const {
  if (prog_var_vals.rows() != num_vars()) {
    std::ostringstream oss;
    oss << "The input binding variable is not in the right size. Expects "
        << num_vars() << " rows, but it actually has " << prog_var_vals.rows()
        << " rows.\n";
    throw std::logic_error(oss.str());
  }

  Eigen::VectorXd this_x;

  for (auto const& binding : visualization_callbacks_) {
    auto const& obj = binding.evaluator();

    const int num_v_variables = binding.GetNumElements();
    this_x.resize(num_v_variables);
    for (int j = 0; j < num_v_variables; ++j) {
      this_x(j) =
          prog_var_vals(FindDecisionVariableIndex(binding.variables()(j)));
    }

    obj->EvalCallback(this_x);
  }
}

void MathematicalProgram::SetVariableScaling(const symbolic::Variable& var,
                                             double s) {
  DRAKE_DEMAND(0 < s);
  int idx = FindDecisionVariableIndex(var);
  if (var_scaling_map_.find(idx) != var_scaling_map_.end()) {
    // Update the scaling factor
    var_scaling_map_[idx] = s;
  } else {
    // Add a new scaling factor
    var_scaling_map_.insert(std::pair<int, double>(idx, s));
  }
}

namespace {
template <typename C>
[[nodiscard]] bool IsVariableBound(const symbolic::Variable& var,
                                   const std::vector<Binding<C>>& bindings,
                                   std::string* binding_description) {
  for (const auto& binding : bindings) {
    if (binding.ContainsVariable(var)) {
      *binding_description = binding.to_string();
      return true;
    }
  }
  return false;
}

// Return true if the variable is bound with a cost or constraint (except for a
// bounding box constraint); false otherwise.
[[nodiscard]] bool IsVariableBound(const symbolic::Variable& var,
                                   const MathematicalProgram& prog,
                                   std::string* binding_description) {
  if (IsVariableBound(var, prog.GetAllCosts(), binding_description)) {
    return true;
  }
  if (IsVariableBound(var, prog.GetAllConstraints(), binding_description)) {
    return true;
  }
  if (IsVariableBound(var, prog.visualization_callbacks(),
                      binding_description)) {
    return true;
  }
  return false;
}
}  // namespace

int MathematicalProgram::RemoveDecisionVariable(const symbolic::Variable& var) {
  if (decision_variable_index_.count(var.get_id()) == 0) {
    throw std::invalid_argument(
        fmt::format("RemoveDecisionVariable: {} is not a decision variable of "
                    "this MathematicalProgram.",
                    var.get_name()));
  }
  std::string binding_description;
  if (IsVariableBound(var, *this, &binding_description)) {
    throw std::invalid_argument(
        fmt::format("RemoveDecisionVariable: {} is associated with a {}.",
                    var.get_name(), binding_description));
  }
  const auto var_it = decision_variable_index_.find(var.get_id());
  const int var_index = var_it->second;
  // Update decision_variable_index_.
  decision_variable_index_.erase(var_it);
  for (auto& [variable_id, variable_index] : decision_variable_index_) {
    // Decrement the index of the variable after `var`.
    if (variable_index > var_index) {
      --variable_index;
    }
  }
  // Remove the variable from decision_variables_.
  decision_variables_.erase(decision_variables_.begin() + var_index);
  // Remove from var_scaling_map_.
  std::unordered_map<int, double> new_var_scaling_map;
  for (const auto& [variable_index, scale] : var_scaling_map_) {
    if (variable_index < var_index) {
      new_var_scaling_map.emplace(variable_index, scale);
    } else if (variable_index > var_index) {
      new_var_scaling_map.emplace(variable_index - 1, scale);
    }
  }
  var_scaling_map_ = std::move(new_var_scaling_map);
  // Update x_initial_guess_;
  for (int i = var_index; i < x_initial_guess_.rows() - 1; ++i) {
    x_initial_guess_(i) = x_initial_guess_(i + 1);
  }
  x_initial_guess_.conservativeResize(x_initial_guess_.rows() - 1);
  return var_index;
}

template <typename C>
int MathematicalProgram::RemoveCostOrConstraintImpl(
    const Binding<C>& removal, ProgramAttribute affected_capability,
    std::vector<Binding<C>>* existings) {
  const int num_existing = static_cast<int>(existings->size());
  existings->erase(std::remove(existings->begin(), existings->end(), removal),
                   existings->end());
  UpdateRequiredCapability(affected_capability);
  const int num_removed = num_existing - static_cast<int>(existings->size());
  return num_removed;
}

namespace {
// Update @p program_capabilities. If @p binding is empty, then remove @p
// capability from @p program_capabilities; otherwise add @p capability to @p
// program_capabilities.
template <typename C>
void UpdateRequiredCapabilityImpl(ProgramAttribute capability,
                                  const std::vector<C>& bindings,
                                  ProgramAttributes* program_capabilities) {
  if (bindings.empty()) {
    // erasing a non-existent key doesn't cause an error.
    program_capabilities->erase(capability);
  } else {
    program_capabilities->emplace(capability);
  }
}
}  // namespace

void MathematicalProgram::UpdateRequiredCapability(
    ProgramAttribute query_capability) {
  switch (query_capability) {
    case ProgramAttribute::kLinearCost: {
      UpdateRequiredCapabilityImpl(query_capability, this->linear_costs(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kQuadraticCost: {
      UpdateRequiredCapabilityImpl(query_capability, this->quadratic_costs(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kGenericCost: {
      UpdateRequiredCapabilityImpl(query_capability, this->generic_costs(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kGenericConstraint: {
      UpdateRequiredCapabilityImpl(query_capability,
                                   this->generic_constraints(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kLinearConstraint: {
      if (this->linear_constraints().empty() &&
          this->bounding_box_constraints().empty()) {
        required_capabilities_.erase(ProgramAttribute::kLinearConstraint);
      } else {
        required_capabilities_.emplace(ProgramAttribute::kLinearConstraint);
      }
      break;
    }
    case ProgramAttribute::kLinearEqualityConstraint: {
      UpdateRequiredCapabilityImpl(query_capability,
                                   this->linear_equality_constraints(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kLinearComplementarityConstraint: {
      UpdateRequiredCapabilityImpl(query_capability,
                                   this->linear_complementarity_constraints(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kLorentzConeConstraint: {
      UpdateRequiredCapabilityImpl(query_capability,
                                   this->lorentz_cone_constraints(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kRotatedLorentzConeConstraint: {
      UpdateRequiredCapabilityImpl(query_capability,
                                   this->rotated_lorentz_cone_constraints(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kPositiveSemidefiniteConstraint: {
      if (positive_semidefinite_constraint_.empty() &&
          linear_matrix_inequality_constraint_.empty()) {
        required_capabilities_.erase(
            ProgramAttribute::kPositiveSemidefiniteConstraint);
      } else {
        required_capabilities_.emplace(
            ProgramAttribute::kPositiveSemidefiniteConstraint);
      }
      break;
    }
    case ProgramAttribute::kExponentialConeConstraint: {
      UpdateRequiredCapabilityImpl(query_capability,
                                   this->exponential_cone_constraints(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kQuadraticConstraint: {
      UpdateRequiredCapabilityImpl(query_capability,
                                   this->quadratic_constraints(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kL2NormCost: {
      UpdateRequiredCapabilityImpl(query_capability, this->l2norm_costs(),
                                   &required_capabilities_);
      break;
    }
    case ProgramAttribute::kBinaryVariable: {
      bool has_binary_var = false;
      for (int i = 0; i < num_vars(); ++i) {
        if (decision_variables_[i].get_type() ==
            symbolic::Variable::Type::BINARY) {
          has_binary_var = true;
          break;
        }
      }
      if (has_binary_var) {
        required_capabilities_.emplace(ProgramAttribute::kBinaryVariable);
      } else {
        required_capabilities_.erase(ProgramAttribute::kBinaryVariable);
      }
      break;
    }
    case ProgramAttribute::kCallback: {
      UpdateRequiredCapabilityImpl(query_capability, visualization_callbacks_,
                                   &required_capabilities_);
      break;
    }
  }
}

int MathematicalProgram::RemoveCost(const Binding<Cost>& cost) {
  Cost* cost_evaluator = cost.evaluator().get();
  // TODO(hongkai.dai): Remove the dynamic cast as part of #8349.
  if (dynamic_cast<QuadraticCost*>(cost_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<QuadraticCost>(cost),
        ProgramAttribute::kQuadraticCost, &(this->quadratic_costs_));
  } else if (dynamic_cast<LinearCost*>(cost_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<LinearCost>(cost),
        ProgramAttribute::kLinearCost, &(this->linear_costs_));
  } else if (dynamic_cast<L2NormCost*>(cost_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<L2NormCost>(cost),
        ProgramAttribute::kL2NormCost, &(this->l2norm_costs_));
  } else {
    return RemoveCostOrConstraintImpl(cost, ProgramAttribute::kGenericCost,
                                      &(this->generic_costs_));
  }
  DRAKE_UNREACHABLE();
}

int MathematicalProgram::RemoveConstraint(
    const Binding<Constraint>& constraint) {
  Constraint* constraint_evaluator = constraint.evaluator().get();
  // TODO(hongkai.dai): Remove the dynamic cast as part of #8349.
  // Check constraints types in reverse order, such that classes that inherit
  // from other classes will not be prematurely added to less specific (or
  // incorrect) container.
  if (dynamic_cast<ExponentialConeConstraint*>(constraint_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<ExponentialConeConstraint>(constraint),
        ProgramAttribute::kExponentialConeConstraint,
        &exponential_cone_constraints_);
  } else if (dynamic_cast<LinearMatrixInequalityConstraint*>(
                 constraint_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<LinearMatrixInequalityConstraint>(
            constraint),
        ProgramAttribute::kPositiveSemidefiniteConstraint,
        &linear_matrix_inequality_constraint_);
  } else if (dynamic_cast<PositiveSemidefiniteConstraint*>(
                 constraint_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<PositiveSemidefiniteConstraint>(
            constraint),
        ProgramAttribute::kPositiveSemidefiniteConstraint,
        &positive_semidefinite_constraint_);
  } else if (dynamic_cast<QuadraticConstraint*>(constraint_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<QuadraticConstraint>(constraint),
        ProgramAttribute::kQuadraticConstraint, &quadratic_constraints_);
  } else if (dynamic_cast<RotatedLorentzConeConstraint*>(
                 constraint_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<RotatedLorentzConeConstraint>(constraint),
        ProgramAttribute::kRotatedLorentzConeConstraint,
        &rotated_lorentz_cone_constraint_);
  } else if (dynamic_cast<LorentzConeConstraint*>(constraint_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<LorentzConeConstraint>(constraint),
        ProgramAttribute::kLorentzConeConstraint, &lorentz_cone_constraint_);
  } else if (dynamic_cast<LinearComplementarityConstraint*>(
                 constraint_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<LinearComplementarityConstraint>(
            constraint),
        ProgramAttribute::kLinearComplementarityConstraint,
        &linear_complementarity_constraints_);
  } else if (dynamic_cast<LinearEqualityConstraint*>(constraint_evaluator)) {
    // LinearEqualityConstraint is derived from LinearConstraint. Put this
    // branch before the LinearConstraint branch.
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<LinearEqualityConstraint>(constraint),
        ProgramAttribute::kLinearEqualityConstraint,
        &linear_equality_constraints_);
  } else if (dynamic_cast<BoundingBoxConstraint*>(constraint_evaluator)) {
    // BoundingBoxConstraint is derived from LinearConstraint. Put this branch
    // before the LinearConstraint branch.
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<BoundingBoxConstraint>(constraint),
        ProgramAttribute::kLinearConstraint, &bbox_constraints_);
  } else if (dynamic_cast<LinearConstraint*>(constraint_evaluator)) {
    return RemoveCostOrConstraintImpl(
        internal::BindingDynamicCast<LinearConstraint>(constraint),
        ProgramAttribute::kLinearConstraint, &linear_constraints_);
  } else {
    // All constraints are derived from Constraint class. Put this branch last.
    return RemoveCostOrConstraintImpl(constraint,
                                      ProgramAttribute::kGenericConstraint,
                                      &generic_constraints_);
  }
  DRAKE_UNREACHABLE();
}

int MathematicalProgram::RemoveVisualizationCallback(
    const Binding<VisualizationCallback>& callback) {
  return RemoveCostOrConstraintImpl(callback, ProgramAttribute::kCallback,
                                    &visualization_callbacks_);
}

void MathematicalProgram::CheckVariableType(VarType var_type) {
  switch (var_type) {
    case VarType::CONTINUOUS:
      break;
    case VarType::BINARY:
      required_capabilities_.insert(ProgramAttribute::kBinaryVariable);
      break;
    case VarType::INTEGER:
      throw std::runtime_error(
          "MathematicalProgram does not support integer variables yet.");
    case VarType::BOOLEAN:
      throw std::runtime_error(
          "MathematicalProgram does not support Boolean variables.");
    case VarType::RANDOM_UNIFORM:
      throw std::runtime_error(
          "MathematicalProgram does not support random uniform variables.");
    case VarType::RANDOM_GAUSSIAN:
      throw std::runtime_error(
          "MathematicalProgram does not support random Gaussian variables.");
    case VarType::RANDOM_EXPONENTIAL:
      throw std::runtime_error(
          "MathematicalProgram does not support random exponential "
          "variables.");
  }
}

void MathematicalProgram::CheckIsDecisionVariable(
    const VectorXDecisionVariable& vars) const {
  for (int i = 0; i < vars.rows(); ++i) {
    for (int j = 0; j < vars.cols(); ++j) {
      if (!decision_variable_index_.contains(vars(i, j).get_id())) {
        throw std::logic_error(fmt::format(
            "{} is not a decision variable of the mathematical program.",
            vars(i, j)));
      }
    }
  }
}

template <typename C>
bool MathematicalProgram::CheckBinding(const Binding<C>& binding) const {
  // TODO(eric.cousineau): In addition to identifiers, hash bindings by
  // their constraints and their variables, to prevent duplicates.
  // TODO(eric.cousineau): Once bindings have identifiers (perhaps
  // retrofitting `description`), ensure that they have unique names.
  CheckIsDecisionVariable(binding.variables());
  return (binding.evaluator()->num_outputs() > 0);
}

std::ostream& operator<<(std::ostream& os, const MathematicalProgram& prog) {
  if (prog.num_vars() > 0) {
    os << fmt::format("Decision variables: {}\n\n",
                      fmt_eigen(prog.decision_variables().transpose()));
  } else {
    os << "No decision variables.\n";
  }

  if (prog.num_indeterminates() > 0) {
    os << fmt::format("Indeterminates: {}\n\n",
                      fmt_eigen(prog.indeterminates().transpose()));
  }

  for (const auto& b : prog.GetAllCosts()) {
    os << b << "\n";
  }
  for (const auto& b : prog.GetAllConstraints()) {
    os << b;
  }
  return os;
}

}  // namespace solvers
}  // namespace drake
