#include "drake/systems/primitives/square.h"

#include <sstream>

#include "drake/common/drake_throw.h"

namespace drake {
namespace systems {

template <typename T>
Square<T>::Square(double amplitude, double pulse_width, double period,
                  double phase, int size, bool is_time_based)
    : Square(Eigen::VectorXd::Ones(size) * amplitude,
             Eigen::VectorXd::Ones(size) * pulse_width,
             Eigen::VectorXd::Ones(size) * period,
             Eigen::VectorXd::Ones(size) * phase, is_time_based) {}

template <typename T>
Square<T>::Square(const Eigen::VectorXd& amplitudes,
                  const Eigen::VectorXd& pulse_widths,
                  const Eigen::VectorXd& periods, const Eigen::VectorXd& phases,
                  bool is_time_based)
    : LeafSystem<T>(SystemTypeTag<Square>{}),
      amplitude_(amplitudes),
      pulse_width_(pulse_widths),
      period_(periods),
      phase_(phases),
      is_time_based_(is_time_based) {
  // Ensure the incoming vectors are all the same size
  DRAKE_THROW_UNLESS(pulse_widths.size() == amplitudes.size());
  DRAKE_THROW_UNLESS(pulse_widths.size() == periods.size());
  DRAKE_THROW_UNLESS(pulse_widths.size() == phases.size());

  // Check each of the incoming vectors. For each vector, set a flag if every
  // element in that vector is the same.
  is_const_amplitude_ = amplitude_.isConstant(amplitude_[0]);
  is_const_pulse_width_ = pulse_width_.isConstant(pulse_width_[0]);
  is_const_period_ = period_.isConstant(period_[0]);
  is_const_phase_ = phase_.isConstant(phase_[0]);

  // If the Square system is system time based, do not create an input port.
  // System time is used as the time variable in this case. If the Square system
  // is not system time based, create an input port that contains the signal to
  // be used as the time variable.
  if (!is_time_based) {
    this->DeclareInputPort(kUseDefaultName, kVectorValued, pulse_widths.size());
  }
  value_output_port_index_ =
      this->DeclareVectorOutputPort(kUseDefaultName, pulse_widths.size(),
                                    &Square::CalcValueOutput)
          .get_index();
}

template <typename T>
template <typename U>
Square<T>::Square(const Square<U>& other)
    : Square<T>(other.amplitude_vector(), other.pulse_width_vector(),
                other.period_vector(), other.phase_vector(),
                other.is_time_based()) {}

template <typename T>
double Square<T>::amplitude() const {
  if (!is_const_amplitude_) {
    std::stringstream s;
    s << "The amplitude vector, [" << amplitude_ << "], cannot be represented "
      << "as a scalar value. Please use "
      << "drake::systems::Square::amplitude_vector() instead.";
    throw std::logic_error(s.str());
  }
  return amplitude_[0];
}

template <typename T>
double Square<T>::pulse_width() const {
  if (!is_const_pulse_width_) {
    std::stringstream s;
    s << "The pulse_width vector, [" << pulse_width_
      << "], cannot be represented "
      << "as a scalar value. Please use "
      << "drake::systems::Square::pulse_width_vector() instead.";
    throw std::logic_error(s.str());
  }
  return pulse_width_[0];
}

template <typename T>
double Square<T>::period() const {
  if (!is_const_period_) {
    std::stringstream s;
    s << "The period vector, [" << period_ << "], cannot be represented "
      << "as a scalar value. Please use "
      << "drake::systems::Square::period_vector() instead.";
    throw std::logic_error(s.str());
  }
  return period_[0];
}

template <typename T>
double Square<T>::phase() const {
  if (!is_const_phase_) {
    std::stringstream s;
    s << "The phase vector, [" << phase_ << "], cannot be represented as a "
      << "scalar value. Please use "
      << "drake::systems::Square::phase_vector() instead.";
    throw std::logic_error(s.str().c_str());
  }
  return phase_[0];
}

template <typename T>
bool Square<T>::is_time_based() const {
  return is_time_based_;
}

template <typename T>
const Eigen::VectorXd& Square<T>::amplitude_vector() const {
  return amplitude_;
}

template <typename T>
const Eigen::VectorXd& Square<T>::pulse_width_vector() const {
  return pulse_width_;
}

template <typename T>
const Eigen::VectorXd& Square<T>::period_vector() const {
  return period_;
}

template <typename T>
const Eigen::VectorXd& Square<T>::phase_vector() const {
  return phase_;
}

template <typename T>
void Square<T>::CalcValueOutput(const Context<T>& context,
                                BasicVector<T>* output) const {
  Eigen::VectorBlock<VectorX<T>> output_block = output->get_mutable_value();

  const T time = context.get_time();

  for (int i = 0; i < pulse_width_.size(); ++i) {
    T t = time + phase_[i];
    if (!is_time_based_) {
      t = this->get_input_port(0).Eval(context)[i] + phase_[i];
    }
    output_block[i] =
        amplitude_[i] *
        (t - floor(t / period_[i]) * period_[i] < pulse_width_[i] ? 1 : 0);
  }
}

}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::Square)
