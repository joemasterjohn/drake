#pragma once

#include <memory>
#include <vector>

#include <Eigen/Sparse>

#include "drake/common/drake_bool.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/drake_throw.h"
#include "drake/common/eigen_types.h"
#include "drake/common/name_value.h"
#include "drake/common/trajectories/trajectory.h"
#include "drake/math/bspline_basis.h"

namespace drake {
namespace trajectories {
/** Represents a B-spline curve using a given `basis` with ordered
`control_points` such that each control point is a matrix in ℝʳᵒʷˢ ˣ ᶜᵒˡˢ.
@see math::BsplineBasis
@tparam_default_scalar
*/
template <typename T>
class BsplineTrajectory final : public trajectories::Trajectory<T> {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BsplineTrajectory);

  BsplineTrajectory() : BsplineTrajectory<T>({}, {}) {}

  /** Constructs a B-spline trajectory with the given `basis` and
  `control_points`.
  @pre control_points.size() == basis.num_basis_functions() */
  BsplineTrajectory(math::BsplineBasis<T> basis,
                    std::vector<MatrixX<T>> control_points);

#ifdef DRAKE_DOXYGEN_CXX
  /** Constructs a T-valued B-spline trajectory from a double-valued `basis` and
  T-valued `control_points`.
  @pre control_points.size() == basis.num_basis_functions() */
  BsplineTrajectory(math::BsplineBasis<double> basis,
                    std::vector<MatrixX<T>> control_points);
#else
  template <typename U = T>
  BsplineTrajectory(math::BsplineBasis<double> basis,
                    std::vector<MatrixX<T>> control_points,
                    typename std::enable_if_t<!std::is_same_v<U, double>>* = {})
      : BsplineTrajectory(math::BsplineBasis<T>(basis), control_points) {}
#endif

  ~BsplineTrajectory() final;

  /** Evaluates the BsplineTrajectory at the given time t.
  @param t The time at which to evaluate the %BsplineTrajectory.
  @return The matrix of evaluated values.
  @pre If T == symbolic::Expression, `t.is_constant()` must be true.
  @warning If t does not lie in the range [start_time(), end_time()], the
           trajectory will silently be evaluated at the closest
           valid value of time to t. For example, `value(-1)` will return
           `value(0)` for a trajectory defined over [0, 1]. */
  MatrixX<T> value(const T& t) const {
    // We shadowed the base class to add documentation, not to change logic.
    return Trajectory<T>::value(t);
  }

  /** Supports writing optimizations using the control points as decision
  variables.  This method returns the matrix, `M`, defining the control points
  of the `order` derivative in the form:
  <pre>
  derivative.control_points() = this.control_points() * M
  </pre>
  See `BezierCurve::AsLinearInControlPoints()` for more details.
  @pre derivative_order >= 0. */
  Eigen::SparseMatrix<T> AsLinearInControlPoints(
      int derivative_order = 1) const;

  /** Returns the vector, M, such that
  @verbatim
  EvalDerivative(t, derivative_order) = control_points() * M
  @endverbatim
  where cols()==1 (so control_points() is a matrix). This is useful for
  writing linear constraints on the control points. Note that if the derivative
  order is greater than or equal to the order of the basis, then the result is
  a zero vector.

  @pre t ≥ start_time()
  @pre t ≤ end_time() */
  VectorX<T> EvaluateLinearInControlPoints(const T& t,
                                           int derivative_order = 0) const;

  /** Returns the number of control points in this curve. */
  int num_control_points() const { return basis_.num_basis_functions(); }

  /** Returns the control points of this curve. */
  const std::vector<MatrixX<T>>& control_points() const {
    return control_points_;
  }

  /** Returns this->value(this->start_time()) */
  MatrixX<T> InitialValue() const;

  /** Returns this->value(this->end_time()) */
  MatrixX<T> FinalValue() const;

  /** Returns the basis of this curve. */
  const math::BsplineBasis<T>& basis() const { return basis_; }

  /** Adds new knots at the specified `additional_knots` without changing the
  behavior of the trajectory. The basis and control points of the trajectory are
  adjusted such that it produces the same value for any valid time before and
  after this method is called. The resulting trajectory is guaranteed to have
  the same level of continuity as the original, even if knot values are
  duplicated. Note that `additional_knots` need not be sorted.
  @pre start_time() <= t <= end_time() for all t in `additional_knots` */
  void InsertKnots(const std::vector<T>& additional_knots);

  /** Returns a new BsplineTrajectory that uses the same basis as `this`, and
  whose control points are the result of calling `select(point)` on each `point`
  in `this->control_points()`.*/
  BsplineTrajectory<T> CopyWithSelector(
      const std::function<MatrixX<T>(const MatrixX<T>&)>& select) const;

  /** Returns a new BsplineTrajectory that uses the same basis as `this`, and
  whose control points are the result of calling
  `point.block(start_row, start_col, block_rows, block_cols)` on each `point`
  in `this->control_points()`.*/
  BsplineTrajectory<T> CopyBlock(int start_row, int start_col, int block_rows,
                                 int block_cols) const;

  /** Returns a new BsplineTrajectory that uses the same basis as `this`, and
  whose control points are the result of calling `point.head(n)` on each `point`
  in `this->control_points()`.
  @pre this->cols() == 1
  @pre control_points()[0].head(n) must be a valid operation. */
  BsplineTrajectory<T> CopyHead(int n) const;

  boolean<T> operator==(const BsplineTrajectory<T>& other) const;

  /** Passes this object to an Archive.
  Refer to @ref yaml_serialization "YAML Serialization" for background.
  This method is only available when T = double. */
  template <typename Archive>
#ifdef DRAKE_DOXYGEN_CXX
  void
#else
  // Restrict this method to T = double only; we must mix "Archive" into the
  // conditional type for SFINAE to work, so we just check it against void.
  std::enable_if_t<std::is_same_v<T, double> && !std::is_void_v<Archive>>
#endif
  Serialize(Archive* a) {
    a->Visit(MakeNameValue("basis", &basis_));
    a->Visit(MakeNameValue("control_points", &control_points_));
    CheckInvariants();
  }

 private:
  // Trajectory overrides.
  std::unique_ptr<trajectories::Trajectory<T>> DoClone() const final;
  MatrixX<T> do_value(const T& t) const final;
  bool do_has_derivative() const final;
  MatrixX<T> DoEvalDerivative(const T& t, int derivative_order) const final;
  std::unique_ptr<trajectories::Trajectory<T>> DoMakeDerivative(
      int derivative_order) const final;
  Eigen::Index do_rows() const final { return control_points()[0].rows(); }
  Eigen::Index do_cols() const final { return control_points()[0].cols(); }
  T do_start_time() const final { return basis_.initial_parameter_value(); }
  T do_end_time() const final { return basis_.final_parameter_value(); }

  void CheckInvariants() const;

  math::BsplineBasis<T> basis_;
  std::vector<MatrixX<T>> control_points_;
};

}  // namespace trajectories
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::trajectories::BsplineTrajectory);
