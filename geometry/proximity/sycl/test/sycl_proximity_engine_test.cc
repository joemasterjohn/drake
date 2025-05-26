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

#include <fmt/core.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/make_sphere_field.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"
#include "drake/geometry/proximity/sycl/sycl_proximity_engine.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

namespace {

using Eigen::Vector3d;
using math::RigidTransformd;

/*
Creates a simple SoftGeometry with two tets whose faces align and their heights
are in opposite directions, and a simple linear field.
*/
hydroelastic::SoftGeometry MakeSimpleSoftGeometry() {
  // Create mesh
  std::vector<Vector3d> p_MV;
  std::vector<VolumeElement> elements;
  p_MV.push_back(Vector3d(0, 0, -1));
  p_MV.push_back(Vector3d(-1, -1, 0));
  p_MV.push_back(Vector3d(1, -1, 0));
  p_MV.push_back(Vector3d(0, 1, 0));
  p_MV.push_back(Vector3d(0, 0, 1));
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
  auto geometry = MakeSimpleSoftGeometry();
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {id, geometry}};
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries);
  engine.UpdateCollisionCandidates({});
  std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {id, RigidTransformd::Identity()}};
  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Get the total checks - this should be 0
  auto impl = SyclProximityEngineAttorney::get_impl(engine);
  EXPECT_EQ(SyclProximityEngineAttorney::get_total_checks(impl), 0);

  // Vertices stored should be same as vertices from mesh
  auto verticies_M_host = SyclProximityEngineAttorney::get_vertices_M(impl);
  auto verticies_W_host = SyclProximityEngineAttorney::get_vertices_W(impl);

  auto verticies_from_mesh = geometry.mesh().vertices();
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
  auto elements_from_mesh = geometry.mesh().pack_element_vertices();
  EXPECT_EQ(elements_host.size(), elements_from_mesh.size());
  for (size_t i = 0; i < elements_host.size(); ++i) {
    EXPECT_EQ(elements_host[i], elements_from_mesh[i]);
  }
}

GTEST_TEST(SPETest, TwoMeshesColliding) {
  GeometryId idA = GeometryId::get_new_id();
  GeometryId idB = GeometryId::get_new_id();
  auto geometryA = MakeSimpleSoftGeometry();
  auto geometryB = MakeSimpleSoftGeometry();
  std::unordered_map<GeometryId, hydroelastic::SoftGeometry> soft_geometries{
      {idA, geometryA}, {idB, geometryB}};
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries);
  engine.UpdateCollisionCandidates({SortedPair<GeometryId>(idA, idB)});
  // Move meshes along Z so that they just intersect
  std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {idA, RigidTransformd(Vector3d{0, 0, 0})},
      {idB, RigidTransformd(Vector3d{0, 0, 1.1})}};
  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Get the total checks - This should be 4 + 0 = 4
  // Geometry A has 2 other elements to check its 2 elements against
  // Geometry B has nothing to check due to symmetry
  auto impl = SyclProximityEngineAttorney::get_impl(engine);
  EXPECT_EQ(SyclProximityEngineAttorney::get_total_checks(impl), 4);

  auto verticies_from_meshA = geometryA.mesh().vertices();
  auto verticies_from_meshB = geometryB.mesh().vertices();
  std::vector<Vector3d> vertices_of_both_meshes;
  vertices_of_both_meshes.insert(vertices_of_both_meshes.end(),
                                 verticies_from_meshA.begin(),
                                 verticies_from_meshA.end());
  vertices_of_both_meshes.insert(vertices_of_both_meshes.end(),
                                 verticies_from_meshB.begin(),
                                 verticies_from_meshB.end());

  auto vertices_M_host = SyclProximityEngineAttorney::get_vertices_M(impl);
  auto vertices_W_host = SyclProximityEngineAttorney::get_vertices_W(impl);
  EXPECT_EQ(vertices_M_host.size(), vertices_of_both_meshes.size());
  EXPECT_EQ(vertices_M_host, vertices_of_both_meshes);
  // Elements stored should be same as elements from mesh
  auto elements_host = SyclProximityEngineAttorney::get_elements(impl);
  auto elements_from_meshA = geometryA.mesh().pack_element_vertices();
  auto elements_from_meshB = geometryB.mesh().pack_element_vertices();
  std::vector<std::array<int, 4>> elements_of_both_meshes;
  elements_of_both_meshes.insert(elements_of_both_meshes.end(),
                                 elements_from_meshA.begin(),
                                 elements_from_meshA.end());
  elements_of_both_meshes.insert(elements_of_both_meshes.end(),
                                 elements_from_meshB.begin(),
                                 elements_from_meshB.end());
  EXPECT_EQ(elements_host.size(), elements_of_both_meshes.size());
  for (size_t i = 0; i < elements_host.size(); ++i) {
    EXPECT_EQ(elements_host[i], elements_of_both_meshes[i]);
  }

  // Collision filter should be [0 0 1 0] since element 1 of A is colliding with
  // element 0 of B
  bool* collision_filter =
      SyclProximityEngineAttorney::get_collision_filter(impl);

  std::vector<int> expected_collision_filter{0, 0, 1, 0};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(collision_filter[i], expected_collision_filter[i]);
  }

  // Move geometries closer so that all elements are colliding and check
  // collision filter
  X_WGs[idB] = RigidTransformd(Vector3d{0, 0, 0.3});
  surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);
  // Element 0 of A collides with element 0 of B
  // Element 1 of A collides with element 0 and 1 of B
  expected_collision_filter = {1, 0, 1, 1};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(collision_filter[i], expected_collision_filter[i]);
  }
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
  std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {idA, RigidTransformd(Vector3d{0, 0, 0})},
      {idB, RigidTransformd(Vector3d{0, 0, 1.1})},
      {idC, RigidTransformd(Vector3d{0, 0, 2.2})}};
  auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Get the total checks
  auto impl = SyclProximityEngineAttorney::get_impl(engine);
  // Geom A checks 2 elements against 4 = 8 checks
  // Geom B checks 2 elements against 2 = 4 checks
  // Geom C checks none
  // Total = 12 checks
  EXPECT_EQ(SyclProximityEngineAttorney::get_total_checks(impl), 12);

  // Collision filter check
  bool* collision_filter =
      SyclProximityEngineAttorney::get_collision_filter(impl);

  std::vector<int> expected_collision_filter{0, 0, 0, 0, 1, 0,
                                             0, 0, 0, 0, 1, 0};
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(expected_collision_filter[i], collision_filter[i]);
  }

  // Move meshes closer so all elements collide
  X_WGs[idB] = RigidTransformd(Vector3d{0, 0, 0.3});
  X_WGs[idC] = RigidTransformd(Vector3d{0, 0, 0.6});
  surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // With meshes closer, more elements should be colliding
  expected_collision_filter = {1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1};
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(expected_collision_filter[i], collision_filter[i]);
  }
}

GTEST_TEST(SPETest, TwoSpheresColliding) {
  constexpr double radius = 0.5;
  constexpr double resolution_hint = 0.5 * radius;
  constexpr double hydroelastic_modulus = 1e+7;

  // Sphere A
  const Sphere sphereA(radius);
  auto meshA =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereA, resolution_hint,
          TessellationStrategy::kSingleInteriorVertex));
  auto pressureA = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereA, meshA.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereA(*meshA);

  const hydroelastic::SoftGeometry soft_geometryA(
      hydroelastic::SoftMesh(std::move(meshA), std::move(pressureA)));
  const GeometryId sphereA_id = GeometryId::get_new_id();

  // Sphere B
  const Sphere sphereB(radius);
  auto meshB =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereB, resolution_hint,
          TessellationStrategy::kSingleInteriorVertex));
  auto pressureB = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereB, meshB.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereB(*meshB);
  const hydroelastic::SoftGeometry soft_geometryB(
      hydroelastic::SoftMesh(std::move(meshB), std::move(pressureB)));
  const GeometryId sphereB_id = GeometryId::get_new_id();

  // Compute the candidate tets with the two BVHs
  std::vector<std::pair<int, int>> candidate_tetrahedra;
  const auto callback = [&candidate_tetrahedra](
                            int tet0, int tet1) -> BvttCallbackResult {
    candidate_tetrahedra.emplace_back(tet0, tet1);
    return BvttCallbackResult::Continue;
  };

  const RigidTransformd X_WA =
      RigidTransformd(Vector3d{0.2 * radius, 0.1 * radius, 0.3 * radius});
  const RigidTransformd X_WB =
      RigidTransformd(Vector3d{0.1 * radius, 0.2 * radius, 0.3 * radius});
  const RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);

  bvhSphereA.Collide(bvhSphereB, X_AB, callback);

  // Convert cadidate tets to collision_filter_ that can be compared to one from
  // sycl_proximity_engine
  std::vector<bool> expected_filter(soft_geometryA.mesh().num_elements() *
                                        soft_geometryB.mesh().num_elements(),
                                    false);
  for (auto [eA, eB] : candidate_tetrahedra) {
    const int i = eA * soft_geometryB.mesh().num_elements() + eB;
    expected_filter[i] = true;
  }

  // Create soft geometries
  const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>
      soft_geometries{{sphereA_id, soft_geometryA},
                      {sphereB_id, soft_geometryB}};

  // Instantiate SyclProximityEngine to obtain collision filter
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries);

  // Update collision candidates
  engine.UpdateCollisionCandidates(
      {SortedPair<GeometryId>(sphereA_id, sphereB_id)});

  // Move spheres closer so that they collide
  const std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {sphereA_id, X_WA}, {sphereB_id, X_WB}};
  const auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Get the total checks
  const auto impl = SyclProximityEngineAttorney::get_impl(engine);

  // Collision filter check
  const bool* collision_filter =
      SyclProximityEngineAttorney::get_collision_filter(impl);

  const int total_checks = SyclProximityEngineAttorney::get_total_checks(impl);

  ASSERT_EQ(total_checks, ssize(expected_filter));

  for (int i = 0; i < ssize(expected_filter); ++i) {
    EXPECT_EQ(collision_filter[i], expected_filter[i]);
  }
}

}  // namespace
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
