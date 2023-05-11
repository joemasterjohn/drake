#include "drake/geometry/proximity/deformable_contact_internal.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity/deformable_contact_geometries.h"
#include "drake/geometry/proximity/deformable_field_intersection.h"
#include "drake/geometry/proximity/deformable_mesh_intersection.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"

namespace drake {
namespace geometry {
namespace internal {
namespace deformable {

void Geometries::RemoveGeometry(GeometryId id) {
  deformable_geometries_.erase(id);
  rigid_geometries_.erase(id);
}

void Geometries::MaybeAddRigidGeometry(
    const Shape& shape, GeometryId id, const ProximityProperties& props,
    const math::RigidTransform<double>& X_WG) {
  // TODO(xuchenhan-tri): Right now, rigid geometries participating in
  // deformable contact share the property "kRezHint" with hydroelastics. It's
  // reasonable to use the contact mesh with the same resolution for both hydro
  // and deformable contact. Consider reorganizing the proximity properties to
  // make this sharing more explicit. We should also avoid having two copies of
  // the same rigid geometry for both hydro and deformable contact.
  if (props.HasProperty(kHydroGroup, kRezHint)) {
    ReifyData data{id, props};
    shape.Reify(this, &data);
    UpdateRigidWorldPose(id, X_WG);
  }
}

void Geometries::UpdateRigidWorldPose(
    GeometryId id, const math::RigidTransform<double>& X_WG) {
  if (is_rigid(id)) {
    rigid_geometries_.at(id).set_pose_in_world(X_WG);
  }
}

void Geometries::AddDeformableGeometry(GeometryId id, VolumeMesh<double> mesh) {
  deformable_geometries_.insert({id, DeformableGeometry(std::move(mesh))});
}

void Geometries::UpdateDeformableVertexPositions(
    GeometryId id, const Eigen::Ref<const VectorX<double>>& q_WG) {
  if (is_deformable(id)) {
    deformable_geometries_.at(id).UpdateVertexPositions(q_WG);
  }
}

DeformableContact<double> Geometries::ComputeDeformableContact() const {
  DeformableContact<double> result;
  /* Register all deformable geometries. */
  for (const auto& [deformable_id, deformable_geometry] :
       deformable_geometries_) {
    const VolumeMesh<double>& deformable_mesh =
        deformable_geometry.deformable_mesh().mesh();
    result.RegisterDeformableGeometry(deformable_id,
                                      deformable_mesh.num_vertices());
  }

  std::unordered_map<GeometryId, std::unique_ptr<VolumeMeshFieldLinear<double, double>>>
      field_map;

  for (auto it = deformable_geometries_.begin();
       it != deformable_geometries_.end(); ++it) {
    const GeometryId deformable_id = it->first;
    const DeformableGeometry& deformable_geometry = it->second;
    field_map[deformable_id] = deformable_geometry.GetSignedDistanceField();
  }

  for (auto it = deformable_geometries_.begin();
       it != deformable_geometries_.end(); ++it) {
    const GeometryId deformable_id = it->first;
    const DeformableGeometry& deformable_geometry = it->second;
    /* collect all deformable rigid contact. */
    for (const auto& [rigid_id, rigid_geometry] : rigid_geometries_) {
      const math::RigidTransform<double>& X_WR = rigid_geometry.pose_in_world();
      const auto& rigid_bvh = rigid_geometry.rigid_mesh().bvh();
      const auto& rigid_tri_mesh = rigid_geometry.rigid_mesh().mesh();
      AddDeformableRigidContactSurface(deformable_geometry, deformable_id,
                                       rigid_id, rigid_tri_mesh, rigid_bvh,
                                       X_WR, &result);
    }

    const VolumeMeshFieldLinear<double, double>& field1 =
        *(field_map[deformable_id]);
    const auto& bvh1 = deformable_geometry.deformable_mesh().bvh();
    for (auto it2 = std::next(it, 1); it2 != deformable_geometries_.end();
         ++it2) {
      const GeometryId deformable_id2 = it2->first;
      const DeformableGeometry& deformable_geometry2 = it2->second;
      const VolumeMeshFieldLinear<double, double>& field2 =
          *(field_map[deformable_id2]);
      const auto& bvh2 = deformable_geometry2.deformable_mesh().bvh();
      IntersectDeformableVolumes(
          deformable_id, field1, bvh1, math::RigidTransform<double>::Identity(),
          deformable_id2, field2, bvh2,
          math::RigidTransform<double>::Identity(), &result);
    }
  }
  return result;
}

void Geometries::ImplementGeometry(const Box& box, void* user_data) {
  AddRigidGeometry(box, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Capsule& capsule, void* user_data) {
  AddRigidGeometry(capsule, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Convex& convex, void* user_data) {
  AddRigidGeometry(convex, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Cylinder& cylinder, void* user_data) {
  AddRigidGeometry(cylinder, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const Ellipsoid& ellipsoid,
                                   void* user_data) {
  AddRigidGeometry(ellipsoid, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const HalfSpace&, void*) {
  static const logging::Warn log_once(
      "Rigid (non-deformable) half spaces are not currently supported for "
      "deformable contact; registration is allowed, but no contact data will "
      "be reported.");
}

void Geometries::ImplementGeometry(const Mesh& mesh, void* user_data) {
  AddRigidGeometry(mesh, *static_cast<ReifyData*>(user_data));
}

void Geometries::ImplementGeometry(const MeshcatCone&, void*) {
  static const logging::Warn log_once(
      "Rigid (non-deformable) Meshcat cones are not currently supported for "
      "deformable contact; registration is allowed, but no contact data will "
      "be reported.");
}

void Geometries::ImplementGeometry(const Sphere& sphere, void* user_data) {
  AddRigidGeometry(sphere, *static_cast<ReifyData*>(user_data));
}

template <typename ShapeType>
void Geometries::AddRigidGeometry(const ShapeType& shape,
                                  const ReifyData& data) {
  /* Forward to hydroelastics to construct the geometry. */
  std::optional<internal::hydroelastic::RigidGeometry> hydro_rigid_geometry =
      internal::hydroelastic::MakeRigidRepresentation(shape, data.properties);
  /* Unsupported geometries will be handle through the
   `ThrowUnsupportedGeometry()` code path. */
  DRAKE_DEMAND(hydro_rigid_geometry.has_value());
  rigid_geometries_.insert(
      {data.id, RigidGeometry(hydro_rigid_geometry->release_mesh())});
}

}  // namespace deformable
}  // namespace internal
}  // namespace geometry
}  // namespace drake
