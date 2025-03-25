#pragma once

#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/ccd.h"

namespace drake {
namespace geometry {
namespace internal {

/* A speculative contact surface.
 @tparam_nonsymbolic_scalar */
template <typename T>
class SpeculativeContactSurface {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SpeculativeContactSurface);

  /* Constructs a deformable contact surface with the given data.
   @param[in] id_A
      The GeometryId of the deformable geometry A.
   @param[in] id_B
      The GeometryId of the geometry B that may be deformable.
   @param[in] p_WC
   @param[in] z_hat_BA_W
   @param[in] coefficient
   @param[in] nhat_BA_W
   @param[in] grad_eA_W
   @param[in] grad_eB_W
   @param[in] closest_points
   @param[in] element_pairs
   @pre p_WC.size() == time_of_contact.size()
   @pre p_WC.size() == z_hat_BA_W.size()
   @pre p_WC.size() == coefficient.size()
   @pre p_WC.size() == nhat_BA_W.size()
   @pre p_WC.size() == grad_eA_W.size()
   @pre p_WC.size() == grad_eB_W.size()
   @pre p_WC.size() == closest_points.size()
   @pre p_WC.size() == element_pairs.size()
   */
  SpeculativeContactSurface(
      GeometryId id_A, GeometryId id_B, std::vector<Vector3<T>> p_WC,
      std::vector<T> time_of_contact, std::vector<Vector3<T>> zhat_BA_W,
      std::vector<T> coefficient, std::vector<Vector3<T>> nhat_BA_W,
      std::vector<Vector3<T>> grad_eA_W, std::vector<Vector3<T>> grad_eB_W,
      std::vector<ClosestPointResult<T>> closest_points,
      std::vector<std::pair<int, int>> element_pairs);

  ~SpeculativeContactSurface();

  /* Returns the GeometryId of geometry A. */
  GeometryId id_A() const { return id_A_; }

  /* Returns the GeometryId of geometry B.*/
  GeometryId id_B() const { return id_B_; }

  /* Returns the total number of contact points on this contact surface. */
  int num_contact_points() const { return p_WC_.size(); }

  /* Returns the world frame positions of the contact_points.*/
  const std::vector<Vector3<T>>& p_WC() const { return p_WC_; }

  /* Returns the time of contact for the contact points in p_WC. */
  const std::vector<T>& time_of_contact() const { return time_of_contact_; }

  /* Returns the world frame volume normals pointing from geometry B into
   geometry A. The ordering of volume normals is the same as that in
   `p_WC()`.*/
  const std::vector<Vector3<T>>& zhat_BA_W() const { return zhat_BA_W_; }

  /* Returns the coefficients of the polynomials expressing the volume of
   overlap along the volume normal direction, zhat: V(z) = C * z³ */
  const std::vector<T>& coefficient() const { return coefficient_; }

  /* Returns the contact normals:
       nhat_BA_W = (grad_eA_W - grad_eB_W).normalized() */
  const std::vector<Vector3<T>>& nhat_BA_W() const { return nhat_BA_W_; }

  /* Returns the gradients of the pressure field eA on the tetrahedral element
   of A for each contact. grad_eA is assumed to be constant on an element. */
  const std::vector<Vector3<T>>& grad_eA_W() const { return grad_eA_W_; }

  /* Returns the gradients of the pressure field eB on the tetrahedral element
   of B for each contact. grad_eB is assumed to be constant on an element. */
  const std::vector<Vector3<T>>& grad_eB_W() const { return grad_eB_W_; }

  /* Returns the closest point data for each contact. */
  const std::vector<ClosestPointResult<T>>& closest_points() const {
    return closest_points_;
  }

  /* Returns the pair of elements of A and B for each contact. */
  const std::vector<std::pair<int, int>>& element_pairs() const {
    return element_pairs_;
  }

 private:
  GeometryId id_A_;
  GeometryId id_B_;
  /* per-contact point data. */
  std::vector<Vector3<T>> p_WC_;
  std::vector<T> time_of_contact_;
  std::vector<Vector3<T>> zhat_BA_W_;
  std::vector<T> coefficient_;
  std::vector<Vector3<T>> nhat_BA_W_;
  std::vector<Vector3<T>> grad_eA_W_;
  std::vector<Vector3<T>> grad_eB_W_;
  /* Extra info for debugging. */
  std::vector<ClosestPointResult<T>> closest_points_;
  std::vector<std::pair<int, int>> element_pairs_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::geometry::internal::SpeculativeContactSurface);
