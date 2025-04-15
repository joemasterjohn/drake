#include "drake/geometry/query_results/speculative_contact.h"

#include "drake/common/ssize.h"

namespace drake {
namespace geometry {
namespace internal {

template <typename T>
SpeculativeContactSurface<T>::SpeculativeContactSurface(
    GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WC,
    std::vector<T> time_of_contact, std::vector<Vector3<T>> zhat_BA_W,
    std::vector<T> coefficient, std::vector<Vector3<T>> nhat_BA_W,
    std::vector<Vector3<T>> grad_eA_W, std::vector<Vector3<T>> grad_eB_W,
    std::vector<ClosestPointResult<T>> closest_points,
    std::vector<std::pair<int, int>> element_pairs)
    : id_A_(id_A),
      id_B_(id_B),
      p_WC_(std::move(p_WC)),
      time_of_contact_(std::move(time_of_contact)),
      zhat_BA_W_(std::move(zhat_BA_W)),
      coefficient_(std::move(coefficient)),
      nhat_BA_W_(std::move(nhat_BA_W)),
      grad_eA_W_(std::move(grad_eA_W)),
      grad_eB_W_(std::move(grad_eB_W)),
      closest_points_(std::move(closest_points)),
      element_pairs_(std::move(element_pairs)) {
  const int num_contact_points = ssize(p_WC_);
  DRAKE_DEMAND(num_contact_points == ssize(time_of_contact_));
  DRAKE_DEMAND(num_contact_points == ssize(zhat_BA_W_));
  DRAKE_DEMAND(num_contact_points == ssize(coefficient_));
  DRAKE_DEMAND(num_contact_points == ssize(nhat_BA_W_));
  DRAKE_DEMAND(num_contact_points == ssize(grad_eA_W_));
  DRAKE_DEMAND(num_contact_points == ssize(grad_eB_W_));
  DRAKE_DEMAND(num_contact_points == ssize(closest_points_));
  //DRAKE_DEMAND(num_contact_points == ssize(element_pairs_));
}

template <typename T>
SpeculativeContactSurface<T>::~SpeculativeContactSurface() = default;

}  // namespace internal
}  // namespace geometry
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::geometry::internal::SpeculativeContactSurface);
