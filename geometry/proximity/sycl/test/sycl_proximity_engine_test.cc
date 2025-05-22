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
  // Should throw when soft_geometries is empty
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries;
  EXPECT_THROW(drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
                   soft_geometries),
               std::runtime_error);
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

  // Get the total checks - this should be 0
  auto impl = SyclProximityEngineAttorney::get_impl(engine);
  EXPECT_EQ(SyclProximityEngineAttorney::get_total_checks(impl), 0);

  // Vertices stored should be same as vertices from mesh
  auto verticies_M_host = SyclProximityEngineAttorney::get_vertices_M(impl);
  auto verticies_W_host = SyclProximityEngineAttorney::get_vertices_W(impl);

  auto verticies_from_mesh = MakeSimpleSoftGeometry().mesh().vertices();
  EXPECT_EQ(verticies_M_host.size(), verticies_from_mesh.size());
  for (size_t i = 0; i < verticies_M_host.size(); ++i) {
    EXPECT_EQ(verticies_M_host[i], verticies_from_mesh[i]);
  }
  // Vertices in world frame should be same as vertices in mesh frame
  EXPECT_EQ(verticies_W_host.size(), verticies_from_mesh.size());
  for (size_t i = 0; i < verticies_W_host.size(); ++i) {
    EXPECT_EQ(verticies_W_host[i], verticies_from_mesh[i]);
  }

  // Elements stored should be same as elements from mesh
  auto elements_host = SyclProximityEngineAttorney::get_elements(impl);
  auto elements_from_mesh =
      MakeSimpleSoftGeometry().mesh().pack_element_vertices();
  EXPECT_EQ(elements_host.size(), elements_from_mesh.size());
  for (size_t i = 0; i < elements_host.size(); ++i) {
    EXPECT_EQ(elements_host[i], elements_from_mesh[i]);
  }
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
