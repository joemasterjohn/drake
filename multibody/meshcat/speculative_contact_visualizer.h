#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/geometry/meshcat.h"
#include "drake/multibody/meshcat/contact_visualizer_params.h"

namespace drake {
namespace multibody {
namespace meshcat {
namespace internal {

/* Like multibody::SpeculativeContactInfo, but only the visualization info. */
struct SpeculativeContactVisualizerItem {
  SpeculativeContactVisualizerItem(std::string body_A_, std::string body_B_)
      : body_A(std::move(body_A_)), body_B(std::move(body_B_)) {}

  std::string body_A;
  std::string body_B;
  std::vector<Eigen::Vector3d> p_WAp;
  std::vector<Eigen::Vector3d> p_WBq;
  std::vector<Eigen::Vector3d> nhat_BA_W;
  std::vector<Eigen::Vector3d> p_WC;
  std::vector<Eigen::Vector3d> f_AC_W;
};

/* SpeculativeContactVisualizer publishes Speculative contact results for
MeshCat. It draws a line segment between the two witness points P and Q as
well as the contact normal at the extrapolated contact point C.
*/
class SpeculativeContactVisualizer {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SpeculativeContactVisualizer);

  /* Creates an instance of SpeculativeContactVisualizer.
  Note that not all fields of `params` are relevant nor used.

  @pre meshcat != nullptr
  */
  SpeculativeContactVisualizer(std::shared_ptr<geometry::Meshcat> meshcat,
                                ContactVisualizerParams params);

  ~SpeculativeContactVisualizer();

  /* Update meshcat to show _only_ the given contacts.
  Any previously-visualized contacts will no longer be visible. */
  void Update(double time,
              const std::vector<SpeculativeContactVisualizerItem>& items);

  /* Calls geometry::Meshcat::Delete(path), with the path set to params.prefix.
  Since this visualizer will only ever add geometry under this prefix, this will
  remove all geometry/transforms added by the visualizer, or by a previous
  instance of this visualizer using the same prefix. */
  void Delete();

 private:
  /* When a contact disappears, we mark it invisible rather than deleting it
  (to improve responsiveness). This struct tracks that state. */
  struct VisibilityStatus {
    /* Whether this path is currently visible in meshcat. */
    bool visible{false};
    /* Whether this contact was active as of the most recent Update(). */
    bool active{false};
  };

  /* Find an entry in path_visibility_status_, or else add one and return it.
  When an entry is added by this function, the arrow geometry is also added to
  meshcat (with visible=false and active=false) as a side-effect. */
  VisibilityStatus& FindOrAdd(const std::string& path);

  const std::shared_ptr<geometry::Meshcat> meshcat_;
  const ContactVisualizerParams params_;

  /* Map of from a contact pair's path to its status. When the map has no key
  for a given path, that indicates no geometry for that pair exists yet. */
  std::unordered_map<std::string, VisibilityStatus> path_visibility_status_;
};

}  // namespace internal
}  // namespace meshcat
}  // namespace multibody
}  // namespace drake
