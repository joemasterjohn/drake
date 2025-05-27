#include "drake/multibody/plant/speculative_contact_info.h"

namespace drake {
namespace multibody {

template <typename T>
SpeculativeContactInfo<T>::SpeculativeContactInfo(
    BodyIndex bodyA_index, BodyIndex bodyB_index, const Vector3<T>& p_WAp,
    const Vector3<T>& p_WBq, const Vector3<T>& f_Ap_W, const Vector3<T>& f_Bq_W)
    : bodyA_index_(bodyA_index),
      bodyB_index_(bodyB_index),
      p_WAp_(p_WAp),
      p_WBq_(p_WBq),
      f_Ap_W_(f_Ap_W),
      f_Bq_W_(f_Bq_W) {}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::SpeculativeContactInfo);
