#include "drake/multibody/fem/fem_solver.h"

#include <algorithm>
#include <iostream>

#include "drake/common/text_logging.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
void FemSolverScratchData<T>::Resize(const FemModel<T>& model) {
  if (num_dofs() != model.num_dofs() || petsc_tangent_matrix_ == nullptr ||
      tangent_matrix_ == nullptr) {
    b_.resize(model.num_dofs());
    dz_.resize(model.num_dofs());
    petsc_tangent_matrix_ = model.MakePetscSymmetricBlockSparseTangentMatrix();
    tangent_matrix_ = model.MakeSymmetricBlockSparseTangentMatrix();
    linear_solver_.SetMatrix(*tangent_matrix_);
  }
}

template <typename T>
std::unique_ptr<FemSolverScratchData<T>> FemSolverScratchData<T>::Clone()
    const {
  std::unique_ptr<FemSolverScratchData<T>> clone(new FemSolverScratchData<T>());
  clone->b_ = this->b_;
  clone->dz_ = this->dz_;
  DRAKE_DEMAND(petsc_tangent_matrix_ != nullptr);
  petsc_tangent_matrix_->AssembleIfNecessary();
  clone->petsc_tangent_matrix_ = this->petsc_tangent_matrix_->Clone();
  DRAKE_DEMAND(tangent_matrix_ != nullptr);
  clone->tangent_matrix_ =
      std::make_unique<internal::SymmetricBlockSparseMatrix<T>>(
          *this->tangent_matrix_);
  clone->linear_solver_ = this->linear_solver_;
  return clone;
}

template <typename T>
FemSolver<T>::FemSolver(const FemModel<T>* model,
                        const DiscreteTimeIntegrator<T>* integrator,
                        FemSolverOption option)
    : model_(model), integrator_(integrator), option_(option) {
  DRAKE_DEMAND(model_ != nullptr);
  DRAKE_DEMAND(integrator_ != nullptr);
}

template <typename T>
int FemSolver<T>::AdvanceOneTimeStep(const FemState<T>& prev_state,
                                     FemState<T>* next_state,
                                     FemSolverScratchData<T>* scratch) const {
  DRAKE_DEMAND(next_state != nullptr);
  model_->ThrowIfModelStateIncompatible(__func__, prev_state);
  model_->ThrowIfModelStateIncompatible(__func__, *next_state);
  const VectorX<T>& unknown_variable = integrator_->GetUnknowns(prev_state);
  integrator_->AdvanceOneTimeStep(prev_state, unknown_variable, next_state);
  /* Run Newton-Raphson iterations. */
  return SolveWithInitialGuess(next_state, scratch);
}

template <typename T>
bool FemSolver<T>::solver_converged(const T& residual_norm,
                                    const T& initial_residual_norm) const {
  return residual_norm < std::max(relative_tolerance_ * initial_residual_norm,
                                  absolute_tolerance_);
}

template <typename T>
double FemSolver<T>::linear_solve_tolerance(
    const T& residual_norm, const T& initial_residual_norm) const {
  /* The relative tolerance when solving for A * dz = -b, where A is the tangent
   matrix. We set it to be on the order of the residual norm to achieve local
   second order convergence [Nocedal and Wright, section 7.1]. We also set it to
   be smaller than the relative tolerance to ensure that linear models converge
   in exact one Newton iteration.

   [Nocedal and Wright] Nocedal, J., & Wright, S. (2006). Numerical
   optimization. Springer Science & Business Media. */
  constexpr double kLinearToleranceFactor = 0.2;
  double linear_solve_tolerance =
      std::min(kLinearToleranceFactor * relative_tolerance_,
               ExtractDoubleOrThrow(residual_norm) /
                   std::max(ExtractDoubleOrThrow(initial_residual_norm),
                            absolute_tolerance_));
  return linear_solve_tolerance;
}

template <typename T>
int FemSolver<T>::SolveWithInitialGuess(
    FemState<T>* state, FemSolverScratchData<T>* scratch) const {
  /* Make sure the scratch quantities are of the correct sizes. */
  scratch->Resize(*model_);

  VectorX<T>& b = scratch->mutable_b();
  VectorX<T>& dz = scratch->mutable_dz();
  internal::PetscSymmetricBlockSparseMatrix& petsc_tangent_matrix =
      scratch->mutable_petsc_tangent_matrix();
  internal::SymmetricBlockSparseMatrix<T>& tangent_matrix =
      scratch->mutable_tangent_matrix();
  internal::BlockSparseCholeskySolver& solver = scratch->mutable_linear_solver();

  model_->ApplyBoundaryCondition(state);
  model_->CalcResidual(*state, &b);
  T residual_norm = b.norm();
  const T initial_residual_norm = residual_norm;
  int iter = 0;
  /* Newton-Raphson iterations. We iterate until any of the following is true:
   1. The max number of allowed iterations is reached;
   2. The norm of the residual is smaller than the absolute tolerance.
   3. The relative error (the norm of the residual divided by the norm of the
      initial residual) is smaller than the unitless relative tolerance. */
  while (iter < kMaxIterations_ &&
         /* Equivalent to residual_norm < absolute_tolerance_ on first
            iteration. */
         !solver_converged(residual_norm, initial_residual_norm)) {
    if (option_ == FemSolverOption::kUsePetsc) {
      model_->CalcTangentMatrix(*state, integrator_->GetWeights(),
                                &petsc_tangent_matrix);
      petsc_tangent_matrix.AssembleIfNecessary();
      /* Solve for A * dz = -b, where A is the tangent matrix. */
      petsc_tangent_matrix.set_relative_tolerance(
          linear_solve_tolerance(residual_norm, initial_residual_norm));
      const auto linear_solve_status = petsc_tangent_matrix.Solve(
          internal::PetscSymmetricBlockSparseMatrix::SolverType::kConjugateGradient,
          internal::PetscSymmetricBlockSparseMatrix::PreconditionerType::
              kIncompleteCholesky,
          -b, &dz);
      if (linear_solve_status == PetscSolverStatus::kFailure) {
        drake::log()->warn(
            "Linear solve did not converge in Newton iterations in FemSolver.");
        return -1;
      }
    } else {
      model_->CalcTangentMatrix(*state, integrator_->GetWeights(),
                                &tangent_matrix);
      solver.UpdateMatrix(tangent_matrix);
      solver.Factor();
      dz = solver.Solve(-b);
    }
    integrator_->UpdateStateFromChangeInUnknowns(dz, state);
    model_->CalcResidual(*state, &b);
    residual_norm = b.norm();
    ++iter;
  }
  if (!solver_converged(residual_norm, initial_residual_norm)) {
    return -1;
  }
  return iter;
}

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake

template class drake::multibody::fem::internal::FemSolverScratchData<double>;
template class drake::multibody::fem::internal::FemSolver<double>;
