#include "drake/multibody/plant/speculative_contact_info.h"

namespace drake {
namespace multibody {

template <typename T>
SpeculativeContactInfo<T>::SpeculativeContactInfo(
    BodyIndex bodyA_index, BodyIndex bodyB_index,
    geometry::GeometryId geometryA_id, geometry::GeometryId geometryB_id,
    const Vector3<T>& p_WAp, const Vector3<T>& p_WBq,
    const Vector3<T>& nhat_BA_W, const Vector3<T>& p_WC,
    const Vector3<T>& f_AC_W)
    : bodyA_index_(bodyA_index),
      bodyB_index_(bodyB_index),
      geometryA_id_(geometryA_id),
      geometryB_id_(geometryB_id),
      p_WAp_(p_WAp),
      p_WBq_(p_WBq),
      nhat_BA_W_(nhat_BA_W),
      p_WC_(p_WC),
      f_AC_W_(f_AC_W) {}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::multibody::SpeculativeContactInfo);
