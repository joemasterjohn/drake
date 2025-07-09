#include "drake/multibody/meshcat/speculative_contact_visualizer.h"

#include <algorithm>
#include <utility>

#include <fmt/format.h>

#include "drake/common/unused.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace multibody {
namespace meshcat {
namespace internal {

using Eigen::Vector3d;
using Eigen::Vector4d;
using drake::geometry::Rgba;
using geometry::Cylinder;
using geometry::Meshcat;
using geometry::MeshcatCone;
using math::RigidTransformd;
using math::RotationMatrixd;

SpeculativeContactVisualizer::SpeculativeContactVisualizer(
    std::shared_ptr<Meshcat> meshcat, ContactVisualizerParams params)
    : meshcat_(std::move(meshcat)), params_(std::move(params)) {
  DRAKE_DEMAND(meshcat_ != nullptr);
}

SpeculativeContactVisualizer::~SpeculativeContactVisualizer() = default;

void SpeculativeContactVisualizer::Delete() {
  meshcat_->Delete(params_.prefix);
  path_visibility_status_.clear();
}

void SpeculativeContactVisualizer::Update(
    double time, const std::vector<SpeculativeContactVisualizerItem>& items) {
  // Set all contacts to be inactive. They will be re-activated as we loop over
  // `items`, below. Anything that is not re-activated will be set to invisible
  // in a final clean-up pass at the end.
  for (auto& [path, status] : path_visibility_status_) {
    unused(path);
    status.active = false;
  }

  // Process the new contacts to find the active ones.
  for (const SpeculativeContactVisualizerItem& item : items) {
    // Find our meshcat state for this contact pair.
    const std::string path =
        fmt::format("{}/{}+{}", params_.prefix, item.body_A, item.body_B);

    VisibilityStatus& status = FindOrAdd(path);

    const int num_points = ssize(item.p_WAp);
    if (num_points > 0) {
      status.active = true;

      Eigen::Matrix3Xd start_P = Eigen::Matrix3Xd::Zero(3, num_points);
      Eigen::Matrix3Xd end_Q = Eigen::Matrix3Xd::Zero(3, num_points);
      Eigen::Matrix3Xd start_normals = Eigen::Matrix3Xd::Zero(3, num_points);
      Eigen::Matrix3Xd end_normals = Eigen::Matrix3Xd::Zero(3, num_points);

      Rgba color_PQ(0.3, 0.6, 0.3, 1.0);
      Rgba color_normal(0.6, 0.2, 0.2, 1.0);

      for (int i = 0; i < num_points; ++i) {
        start_P.col(i) = item.p_WAp[i];
        end_Q.col(i) = item.p_WBq[i];
        start_normals.col(i) = item.p_WC[i];
        end_normals.col(i) = start_normals.col(i) - 0.005 * item.nhat_BA_W[i];
      }

      meshcat_->SetLineSegments(path + "/speculative_PQ", start_P, end_Q, 2.0,
                               color_PQ);
      meshcat_->SetLineSegments(path + "/speculative_normals", start_normals,
                               end_normals, 2.0, color_normal);
    } else {
      status.active = false;
    }
  }

  // Update meshcat visibility to match the active status.
  for (auto& [path, status] : path_visibility_status_) {
    if (status.visible != status.active) {
      meshcat_->SetProperty(path, "visible", status.active, time);
      status.visible = status.active;
    }
  }
}

SpeculativeContactVisualizer::VisibilityStatus&
SpeculativeContactVisualizer::FindOrAdd(const std::string& path) {
  auto iter = path_visibility_status_.find(path);
  if (iter != path_visibility_status_.end()) {
    return iter->second;
  }

  // Start with it being invisible, to prevent flickering at the origin.
  iter = path_visibility_status_.insert({path, {false, false}}).first;
  meshcat_->SetProperty(path, "visible", false, 0);
  return iter->second;
}

}  // namespace internal
}  // namespace meshcat
}  // namespace multibody
}  // namespace drake
