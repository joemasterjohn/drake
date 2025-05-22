#include "drake/geometry/proximity/sycl/sycl_proximity_engine.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"
#include "drake/geometry/proximity/sycl/sycl_proximity_engine.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

namespace {

/*
Creates a simple SoftGeometry with two tets whose faces align and their heights
are in opposite directions, and a simple linear field.
*/
hydroelastic::SoftGeometry MakeSimpleSoftGeometry() {
  // Create mesh
  std::vector<Vector3<double>> p_MV;
  std::vector<VolumeElement> elements;
  p_MV.push_back(Vector3<double>(0, 0, -1));
  p_MV.push_back(Vector3<double>(-1, -1, 0));
  p_MV.push_back(Vector3<double>(1, -1, 0));
  p_MV.push_back(Vector3<double>(0, 1, 0));
  p_MV.push_back(Vector3<double>(0, 0, 1));
  elements.emplace_back(1, 3, 2, 0);
  elements.emplace_back(1, 2, 3, 4);
  auto mesh = std::make_unique<VolumeMesh<double>>(std::move(elements),
                                                   std::move(p_MV));

  // Create field
  std::vector<double> pressure(mesh->num_vertices());
  for (int i = 0; i < mesh->num_vertices(); ++i) {
    pressure[i] = i;
  }
  auto field = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      std::move(pressure), mesh.get());

  // Construct SoftGeometry
  return hydroelastic::SoftGeometry(
      hydroelastic::SoftMesh(std::move(mesh), std::move(field)));
}

GTEST_TEST(SPETest, ZeroMeshes) {
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries;
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries);
  // No collision candidates, no transforms
  engine.UpdateCollisionCandidates({});
  std::unordered_map<GeometryId, math::RigidTransform<double>> X_WGs;
  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Use Attorney class to get the internals of interest
  auto impl = SyclProximityEngineAttorney::get_impl(engine);
  auto vertices_M = SyclProximityEngineAttorney::get_vertices_M(impl);
  auto vertices_W = SyclProximityEngineAttorney::get_vertices_W(impl);
  auto elements = SyclProximityEngineAttorney::get_elements(impl);
  auto pressures = SyclProximityEngineAttorney::get_pressures(impl);
  auto gradient_M_pressure_at_Mo =
      SyclProximityEngineAttorney::get_gradient_M_pressure_at_Mo(impl);
  auto gradient_W_pressure_at_Wo =
      SyclProximityEngineAttorney::get_gradient_W_pressure_at_Wo(impl);
  auto collision_filter =
      SyclProximityEngineAttorney::get_collision_filter(impl);
  auto collision_filter_host_body_index =
      SyclProximityEngineAttorney::get_collision_filter_host_body_index(impl);
}

GTEST_TEST(SPETest, SingleMesh) {
  GeometryId id = GeometryId::get_new_id();
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {id, MakeSimpleSoftGeometry()}};
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries);
  engine.UpdateCollisionCandidates({});
  std::unordered_map<GeometryId, math::RigidTransform<double>> X_WGs{
      {id, math::RigidTransform<double>::Identity()}};
  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);
}

GTEST_TEST(SPETest, TwoMeshesColliding) {
  GeometryId idA = GeometryId::get_new_id();
  GeometryId idB = GeometryId::get_new_id();
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {idA, MakeSimpleSoftGeometry()}, {idB, MakeSimpleSoftGeometry()}};
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries);
  engine.UpdateCollisionCandidates({SortedPair<GeometryId>(idA, idB)});
  // Move meshes along Z so that they just intersect
  std::unordered_map<GeometryId, math::RigidTransform<double>> X_WGs{
      {idA, math::RigidTransform<double>(Vector3<double>{0, 0, 0})},
      {idB, math::RigidTransform<double>(Vector3<double>{0, 0, 0.6})}};
  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);
}

GTEST_TEST(SPETest, ThreeMeshesAllColliding) {
  GeometryId idA = GeometryId::get_new_id();
  GeometryId idB = GeometryId::get_new_id();
  GeometryId idC = GeometryId::get_new_id();
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {idA, MakeSimpleSoftGeometry()},
      {idB, MakeSimpleSoftGeometry()},
      {idC, MakeSimpleSoftGeometry()}};
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries);
  engine.UpdateCollisionCandidates({SortedPair<GeometryId>(idA, idB),
                                    SortedPair<GeometryId>(idA, idC),
                                    SortedPair<GeometryId>(idB, idC)});
  // Move meshes along Z so that they just intersect
  std::unordered_map<GeometryId, math::RigidTransform<double>> X_WGs{
      {idA, math::RigidTransform<double>(Vector3<double>{0, 0, 0})},
      {idB, math::RigidTransform<double>(Vector3<double>{0, 0, 0.6})},
      {idC, math::RigidTransform<double>(Vector3<double>{0, 0, 1.2})}};
  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);
}

}  // namespace
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
