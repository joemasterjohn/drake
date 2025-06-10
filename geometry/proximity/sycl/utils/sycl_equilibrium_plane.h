#pragma once

#include <cmath>

#include <sycl/sycl.hpp>

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

#ifdef __SYCL_DEVICE_ONLY__
#define DRAKE_SYCL_DEVICE_INLINE [[sycl::device]]
#else
#define DRAKE_SYCL_DEVICE_INLINE
#endif

/* Computes the equilibrium plane between two pressure fields.
 * The equilibrium plane is defined as the plane where the pressure
 * values of both fields are equal.
 *
 * @param gradP_A_Wo_x X component of pressure gradient for geometry A in world
 * frame
 * @param gradP_A_Wo_y Y component of pressure gradient for geometry A in world
 * frame
 * @param gradP_A_Wo_z Z component of pressure gradient for geometry A in world
 * frame
 * @param p_A_Wo Pressure at world origin for geometry A
 * @param gradP_B_Wo_x X component of pressure gradient for geometry B in world
 * frame
 * @param gradP_B_Wo_y Y component of pressure gradient for geometry B in world
 * frame
 * @param gradP_B_Wo_z Z component of pressure gradient for geometry B in world
 * frame
 * @param p_B_Wo Pressure at world origin for geometry B
 * @param eq_plane_out Output array to store equilibrium plane data:
 *   [0-2]: normalized normal vector
 *   [3-5]: point on plane
 *   [6]: gM value (dot product of gradient A with normal)
 *   [7]: gN value (negative dot product of gradient B with normal)
 * @return true if a valid equilibrium plane was found, false otherwise
 */
SYCL_EXTERNAL inline bool ComputeEquilibriumPlane(
    double gradP_A_Wo_x, double gradP_A_Wo_y, double gradP_A_Wo_z,
    double p_A_Wo, double gradP_B_Wo_x, double gradP_B_Wo_y,
    double gradP_B_Wo_z, double p_B_Wo, double* eq_plane_out) {
  // Compute n_W = grad_f0_W - grad_f1_W
  const double n_W_x = gradP_A_Wo_x - gradP_B_Wo_x;
  const double n_W_y = gradP_A_Wo_y - gradP_B_Wo_y;
  const double n_W_z = gradP_A_Wo_z - gradP_B_Wo_z;
  const double n_W_norm =
      sycl::sqrt(n_W_x * n_W_x + n_W_y * n_W_y + n_W_z * n_W_z);

  if (n_W_norm <= 0.0) {
    return false;
  }

  const double n_W_x_normalized = n_W_x / n_W_norm;
  const double n_W_y_normalized = n_W_y / n_W_norm;
  const double n_W_z_normalized = n_W_z / n_W_norm;

  // Normalized pressure gradient for A
  const double gradP_A_W_norm =
      sycl::sqrt(gradP_A_Wo_x * gradP_A_Wo_x + gradP_A_Wo_y * gradP_A_Wo_y +
                 gradP_A_Wo_z * gradP_A_Wo_z);
  const double gradP_A_W_normalized_x = gradP_A_Wo_x / gradP_A_W_norm;
  const double gradP_A_W_normalized_y = gradP_A_Wo_y / gradP_A_W_norm;
  const double gradP_A_W_normalized_z = gradP_A_Wo_z / gradP_A_W_norm;
  const double cos_theta_A = n_W_x_normalized * gradP_A_W_normalized_x +
                             n_W_y_normalized * gradP_A_W_normalized_y +
                             n_W_z_normalized * gradP_A_W_normalized_z;

  constexpr double kAlpha = 5. * M_PI / 8.;
  const double kCosAlpha = sycl::cos(kAlpha);

  if (cos_theta_A <= kCosAlpha) {
    return false;
  }

  // Normalized pressure gradient for B
  const double gradP_B_W_norm =
      sycl::sqrt(gradP_B_Wo_x * gradP_B_Wo_x + gradP_B_Wo_y * gradP_B_Wo_y +
                 gradP_B_Wo_z * gradP_B_Wo_z);
  const double gradP_B_W_normalized_x = gradP_B_Wo_x / gradP_B_W_norm;
  const double gradP_B_W_normalized_y = gradP_B_Wo_y / gradP_B_W_norm;
  const double gradP_B_W_normalized_z = gradP_B_Wo_z / gradP_B_W_norm;
  const double cos_theta_B = -n_W_x_normalized * gradP_B_W_normalized_x +
                             -n_W_y_normalized * gradP_B_W_normalized_y +
                             -n_W_z_normalized * gradP_B_W_normalized_z;

  if (cos_theta_B <= kCosAlpha) {
    return false;
  }

  // gM corresponds to the dot product of gradient for object A with the normal
  const double gM = gradP_A_Wo_x * n_W_x_normalized +
                    gradP_A_Wo_y * n_W_y_normalized +
                    gradP_A_Wo_z * n_W_z_normalized;

  // gN corresponds to the negative dot product of gradient for object B with
  // the normal
  const double gN =
      -(gradP_B_Wo_x * n_W_x_normalized + gradP_B_Wo_y * n_W_y_normalized +
        gradP_B_Wo_z * n_W_z_normalized);

  // Plane point
  double p_WQ_x = ((p_B_Wo - p_A_Wo) / n_W_norm) * n_W_x_normalized;
  double p_WQ_y = ((p_B_Wo - p_A_Wo) / n_W_norm) * n_W_y_normalized;
  double p_WQ_z = ((p_B_Wo - p_A_Wo) / n_W_norm) * n_W_z_normalized;
  eq_plane_out[0] = n_W_x_normalized;
  eq_plane_out[1] = n_W_y_normalized;
  eq_plane_out[2] = n_W_z_normalized;
  eq_plane_out[3] = p_WQ_x;
  eq_plane_out[4] = p_WQ_y;
  eq_plane_out[5] = p_WQ_z;
  eq_plane_out[6] = gM;
  eq_plane_out[7] = gN;
  return true;
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake