#include "drake/geometry/query_results/speculative_contact.h"

#include <fstream>
#include <sstream>

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
    std::vector<std::pair<int, int>> element_pairs,
    std::vector<T> effective_radius)
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
      element_pairs_(std::move(element_pairs)),
      effective_radius_(std::move(effective_radius)) {
  const int num_contact_points = ssize(p_WC_);
  DRAKE_DEMAND(num_contact_points == ssize(time_of_contact_));
  DRAKE_DEMAND(num_contact_points == ssize(zhat_BA_W_));
  DRAKE_DEMAND(num_contact_points == ssize(coefficient_));
  DRAKE_DEMAND(num_contact_points == ssize(nhat_BA_W_));
  DRAKE_DEMAND(num_contact_points == ssize(grad_eA_W_));
  DRAKE_DEMAND(num_contact_points == ssize(grad_eB_W_));
  DRAKE_DEMAND(num_contact_points == ssize(closest_points_));
  // DRAKE_DEMAND(num_contact_points == ssize(element_pairs_));
  DRAKE_DEMAND(num_contact_points == ssize(effective_radius_));
}

template <typename T>
SpeculativeContactSurface<T>::~SpeculativeContactSurface() = default;

template <typename T>
std::string SpeculativeContactSurface<T>::ToString() const {
  if constexpr (!std::is_same_v<T, double>) {
    throw std::logic_error(
        "SpeculativeContactSurface<T>::ToString() is only supported for T = "
        "double.");
  } else {
    std::stringstream out;

    for (int i = 0; i < num_contact_points(); ++i) {
      out << fmt::format("{} {} {} ", p_WC()[i](0), p_WC()[i](1), p_WC()[i](2));
      out << fmt::format("{} ", time_of_contact()[i]);
      out << fmt::format("{} {} {} ", zhat_BA_W()[i](0), zhat_BA_W()[i](1),
                         zhat_BA_W()[i](2));
      out << fmt::format("{} ", coefficient()[i]);
      out << fmt::format("{} {} {} ", nhat_BA_W()[i](0), nhat_BA_W()[i](1),
                         nhat_BA_W()[i](2));
      out << fmt::format("{} {} {} ", grad_eA_W()[i](0), grad_eA_W()[i](1),
                         grad_eA_W()[i](2));
      out << fmt::format("{} {} {} ", grad_eB_W()[i](0), grad_eB_W()[i](1),
                         grad_eB_W()[i](2));
      const ClosestPointResult<double>& result = closest_points()[i];
      out << fmt::format("{} {} {} ", result.closest_A.p(0),
                         result.closest_A.p(1), result.closest_A.p(2));
      switch (result.closest_A.type) {
        case ClosestPointType::Vertex:
          out << fmt::format("VERTEX ");
          break;
        case ClosestPointType::Edge:
          out << fmt::format("EDGE ");
          break;
        case ClosestPointType::Face:
          out << fmt::format("FACE ");
          break;
      }
      out << fmt::format("{} {} {} ", result.closest_A.indices[0],
                         result.closest_A.indices[1],
                         result.closest_A.indices[2]);

      out << fmt::format("{} {} {} ", result.closest_B.p(0),
                         result.closest_B.p(1), result.closest_B.p(2));
      switch (result.closest_B.type) {
        case ClosestPointType::Vertex:
          out << fmt::format("VERTEX ");
          break;
        case ClosestPointType::Edge:
          out << fmt::format("EDGE ");
          break;
        case ClosestPointType::Face:
          out << fmt::format("FACE ");
          break;
      }
      out << fmt::format("{} {} {} ", result.closest_B.indices[0],
                         result.closest_B.indices[1],
                         result.closest_B.indices[2]);
      out << fmt::format("{} ", result.squared_dist);
      out << fmt::format("{} {}\n", element_pairs()[i].first,
                         element_pairs()[i].second);
    }

    return out.str();
  }
}

template <typename T>
std::string SpeculativeContactSurface<T>::ToString(int i) const {
  if constexpr (!std::is_same_v<T, double>) {
    throw std::logic_error(
        "SpeculativeContactSurface<T>::ToString() is only supported for T = "
        "double.");
  } else {
    DRAKE_DEMAND(i >= 0 && i < num_contact_points());
    std::stringstream out;
    out << fmt::format("{} {} {} ", p_WC()[i](0), p_WC()[i](1), p_WC()[i](2));
    out << fmt::format("{} ", time_of_contact()[i]);
    out << fmt::format("{} {} {} ", zhat_BA_W()[i](0), zhat_BA_W()[i](1),
                       zhat_BA_W()[i](2));
    out << fmt::format("{} ", coefficient()[i]);
    out << fmt::format("{} {} {} ", nhat_BA_W()[i](0), nhat_BA_W()[i](1),
                       nhat_BA_W()[i](2));
    out << fmt::format("{} {} {} ", grad_eA_W()[i](0), grad_eA_W()[i](1),
                       grad_eA_W()[i](2));
    out << fmt::format("{} {} {} ", grad_eB_W()[i](0), grad_eB_W()[i](1),
                       grad_eB_W()[i](2));
    const ClosestPointResult<double>& result = closest_points()[i];
    out << fmt::format("{} {} {} ", result.closest_A.p(0),
                       result.closest_A.p(1), result.closest_A.p(2));
    switch (result.closest_A.type) {
      case ClosestPointType::Vertex:
        out << fmt::format("VERTEX ");
        break;
      case ClosestPointType::Edge:
        out << fmt::format("EDGE ");
        break;
      case ClosestPointType::Face:
        out << fmt::format("FACE ");
        break;
    }
    out << fmt::format("{} {} {} ", result.closest_A.indices[0],
                       result.closest_A.indices[1],
                       result.closest_A.indices[2]);

    out << fmt::format("{} {} {} ", result.closest_B.p(0),
                       result.closest_B.p(1), result.closest_B.p(2));
    switch (result.closest_B.type) {
      case ClosestPointType::Vertex:
        out << fmt::format("VERTEX ");
        break;
      case ClosestPointType::Edge:
        out << fmt::format("EDGE ");
        break;
      case ClosestPointType::Face:
        out << fmt::format("FACE ");
        break;
    }
    out << fmt::format("{} {} {} ", result.closest_B.indices[0],
                       result.closest_B.indices[1],
                       result.closest_B.indices[2]);
    out << fmt::format("{} ", result.squared_dist);
    out << fmt::format("{} {}\n", element_pairs()[i].first,
                       element_pairs()[i].second);
    return out.str();
  }
}

template <typename T>
void SpeculativeContactSurface<T>::SaveToFile(
    const std::string& filename) const {
  if constexpr (!std::is_same_v<T, double>) {
    throw std::logic_error(
        "SpeculativeContactSurface<T>::SaveToFile() is only supported for T = "
        "double.");
  } else {
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Could not open file");
    }

    for (int i = 0; i < num_contact_points(); ++i) {
      out << fmt::format("{} {} {} ", p_WC()[i](0), p_WC()[i](1), p_WC()[i](2));
      out << fmt::format("{} ", time_of_contact()[i]);
      out << fmt::format("{} {} {} ", zhat_BA_W()[i](0), zhat_BA_W()[i](1),
                         zhat_BA_W()[i](2));
      out << fmt::format("{} ", coefficient()[i]);
      out << fmt::format("{} {} {} ", nhat_BA_W()[i](0), nhat_BA_W()[i](1),
                         nhat_BA_W()[i](2));
      out << fmt::format("{} {} {} ", grad_eA_W()[i](0), grad_eA_W()[i](1),
                         grad_eA_W()[i](2));
      out << fmt::format("{} {} {} ", grad_eB_W()[i](0), grad_eB_W()[i](1),
                         grad_eB_W()[i](2));
      const ClosestPointResult<double>& result = closest_points()[i];
      out << fmt::format("{} {} {} ", result.closest_A.p(0),
                         result.closest_A.p(1), result.closest_A.p(2));
      switch (result.closest_A.type) {
        case ClosestPointType::Vertex:
          out << fmt::format("VERTEX ");
          break;
        case ClosestPointType::Edge:
          out << fmt::format("EDGE ");
          break;
        case ClosestPointType::Face:
          out << fmt::format("FACE ");
          break;
      }
      out << fmt::format("{} {} {} ", result.closest_A.indices[0],
                         result.closest_A.indices[1],
                         result.closest_A.indices[2]);

      out << fmt::format("{} {} {} ", result.closest_B.p(0),
                         result.closest_B.p(1), result.closest_B.p(2));
      switch (result.closest_B.type) {
        case ClosestPointType::Vertex:
          out << fmt::format("VERTEX ");
          break;
        case ClosestPointType::Edge:
          out << fmt::format("EDGE ");
          break;
        case ClosestPointType::Face:
          out << fmt::format("FACE ");
          break;
      }
      out << fmt::format("{} {} {} ", result.closest_B.indices[0],
                         result.closest_B.indices[1],
                         result.closest_B.indices[2]);
      out << fmt::format("{} ", result.squared_dist);
      out << fmt::format("{} {}\n", element_pairs()[i].first,
                         element_pairs()[i].second);
    }

    out << "\n";
  }
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::geometry::internal::SpeculativeContactSurface);
