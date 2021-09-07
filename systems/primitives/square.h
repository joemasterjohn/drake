#pragma once

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {

/// A sine system which outputs `y = a * sin(f * t + p)` and first and second
/// derivatives w.r.t. the time parameter `t`. The block parameters are:
/// `a` the amplitude, `f` the frequency (radians/second), and `p` the phase
/// (radians), all of which are constant vectors provided at construction time.
/// This system has one or zero input ports and three vector valued output ports
/// (`y` and its first two derivatives). The user can specify whether to use
/// simulation time as the source of values for the time variable or an external
/// source. If an external time source is specified, the system is created with
/// an input port for the time source. Otherwise, the system is created with
/// zero input ports.
///
/// @tparam_default_scalar
/// @ingroup primitive_systems
template <typename T>
class Square final : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Square)

  /// Constructs a %Square system where the amplitude, frequency, and phase is
  /// applied to every input.
  ///
  /// @param[in] amplitude
  /// @param[in] pulse_width the sine wave amplitude
  /// @param[in] period the sine wave frequency (radians/second)
  /// @param[in] phase the sine wave phase (radians)
  /// @param[in] size number of elements in the output signal.
  /// @param[in] is_time_based indicates whether to use the simulation time as
  ///            the source for the sine wave time variable, or use an external
  ///            source, in which case an input port of size @p size is created.
  Square(double amplitude, double pulse_width, double period, double phase,
         int size, bool is_time_based = true);

  /// Constructs a %Square system where different amplitudes, frequencies, and
  /// phases can be applied to each sine wave.
  ///
  /// @param[in] amplitudes
  /// @param[in] pulse_widths the sine wave amplitudes
  /// @param[in] periods the sine wave frequencies (radians/second)
  /// @param[in] phases the sine wave phases (radians)
  /// @param[in] is_time_based indicates whether to use the simulation time as
  ///            the source for the sine wave time variable, or use an external
  ///            source, in which case an input port is created.
  explicit Square(const Eigen::VectorXd& amplitudes,
                  const Eigen::VectorXd& pulse_widths,
                  const Eigen::VectorXd& frequencies,
                  const Eigen::VectorXd& phases, bool is_time_based = true);

  /// Scalar-converting copy constructor. See @ref system_scalar_conversion.
  template <typename U>
  explicit Square(const Square<U>&);

  double amplitude() const;

  /// Returns the amplitude constant. This method should only be called if the
  /// amplitude can be represented as a scalar value, i.e., every element in the
  /// amplitude vector is the same. It will abort if the amplitude cannot be
  /// represented as a single scalar value.
  double pulse_width() const;

  /// Returns the frequency constant. This method should only be called if the
  /// frequency can be represented as a scalar value, i.e., every element in the
  /// frequency vector is the same. It will abort if the frequency cannot be
  /// represented as a single scalar value.
  double period() const;

  /// Returns the phase constant. This method should only be called if the phase
  /// can be represented as a scalar value, i.e., every element in the phase
  /// vector is the same. It will abort if the phase cannot be represented as a
  /// single scalar value.
  double phase() const;

  /// Returns a boolean indicting whether to use simulation time as the source
  /// of values for the time variable or an external source. Returns true if the
  /// simulation time is used as the source, and returns false otherwise.
  bool is_time_based() const;

  const Eigen::VectorXd& amplitude_vector() const;

  /// Returns the amplitude vector constant.
  const Eigen::VectorXd& pulse_width_vector() const;

  /// Returns the frequency vector constant.
  const Eigen::VectorXd& period_vector() const;

  /// Returns the phase vector constant.
  const Eigen::VectorXd& phase_vector() const;

 private:
  void CalcValueOutput(const Context<T>& context, BasicVector<T>* output) const;

  const Eigen::VectorXd amplitude_;
  const Eigen::VectorXd pulse_width_;
  const Eigen::VectorXd period_;
  const Eigen::VectorXd phase_;
  const bool is_time_based_;
  bool is_const_amplitude_{false};
  bool is_const_pulse_width_{false};
  bool is_const_period_{false};
  bool is_const_phase_{false};

  int value_output_port_index_{-1};
};

}  // namespace systems
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::Square)
