/* clang-format off to disable clang-format-includes */
#include "drake/solvers/ipopt_solver.h"
/* clang-format on */

#include <stdexcept>

namespace drake {
namespace solvers {

const char* IpoptSolverDetails::ConvertStatusToString() const {
  throw std::runtime_error(
      "The IPOPT bindings were not compiled.  You'll need to use a different "
      "solver.");
}

IpoptSolver::IpoptSolver()
    : SolverBase(id(), &is_available, &is_enabled,
                 &ProgramAttributesSatisfied) {}

bool IpoptSolver::is_available() {
  return false;
}

void IpoptSolver::DoSolve2(const MathematicalProgram&, const Eigen::VectorXd&,
                           internal::SpecificOptions*,
                           MathematicalProgramResult*) const {
  throw std::runtime_error(
      "The IPOPT bindings were not compiled.  You'll need to use a different "
      "solver.");
}

}  // namespace solvers
}  // namespace drake
