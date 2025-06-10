#include "drake/geometry/proximity/sycl/sycl_proximity_engine.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/core.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include "drake/common/text_logging.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/field_intersection.h"
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

  // Compare vertices within machine precision
  for (size_t i = 0; i < vertices_M_host.size(); ++i) {
    EXPECT_NEAR(vertices_M_host[i][0], vertices_of_both_meshes[i][0],
                std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(vertices_M_host[i][1], vertices_of_both_meshes[i][1],
                std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(vertices_M_host[i][2], vertices_of_both_meshes[i][2],
                std::numeric_limits<double>::epsilon());
  }
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
  std::vector<uint8_t> collision_filter =
      SyclProximityEngineAttorney::get_collision_filter(impl);

  std::vector<uint8_t> expected_collision_filter{0, 0, 1, 0};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(collision_filter[i], expected_collision_filter[i]);
  }

  // Move geometries closer so that all elements are colliding and check
  // collision filter
  X_WGs[idB] = RigidTransformd(Vector3d{0, 0, 0.3});
  surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);
  collision_filter = SyclProximityEngineAttorney::get_collision_filter(impl);
  // Element 0 of A collides with element 0 of B
  // Element 1 of A collides with element 0 and 1 of B
  expected_collision_filter = {1, 0, 1, 1};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(collision_filter[i], expected_collision_filter[i]);
  }

  std::vector<size_t> prefix_sum =
      SyclProximityEngineAttorney::get_prefix_sum(impl);
  std::vector<size_t> expected_prefix_sum = {0, 1, 1, 2};
  EXPECT_EQ(prefix_sum, expected_prefix_sum);
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
  std::vector<uint8_t> collision_filter =
      SyclProximityEngineAttorney::get_collision_filter(impl);

  std::vector<uint8_t> expected_collision_filter{0, 0, 0, 0, 1, 0,
                                                 0, 0, 0, 0, 1, 0};
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(expected_collision_filter[i], collision_filter[i]);
  }

  // check compacted narrow_phase_check_indices_
  std::vector<size_t> narrow_phase_check_indices =
      SyclProximityEngineAttorney::get_narrow_phase_check_indices(impl);
  std::vector<size_t> expected_narrow_phase_check_indices{4, 10};
  ASSERT_EQ(narrow_phase_check_indices.size(),
            expected_narrow_phase_check_indices.size());
  for (size_t i = 0; i < narrow_phase_check_indices.size(); ++i) {
    EXPECT_EQ(narrow_phase_check_indices[i],
              expected_narrow_phase_check_indices[i]);
  }

  // Move meshes closer so all elements collide
  X_WGs[idB] = RigidTransformd(Vector3d{0, 0, 0.3});
  X_WGs[idC] = RigidTransformd(Vector3d{0, 0, 0.6});
  surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);
  collision_filter = SyclProximityEngineAttorney::get_collision_filter(impl);

  // With meshes closer, more elements should be colliding
  expected_collision_filter = {1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1};
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(expected_collision_filter[i], collision_filter[i]);
  }

  std::vector<size_t> prefix_sum =
      SyclProximityEngineAttorney::get_prefix_sum(impl);
  std::vector<size_t> expected_prefix_sum(expected_collision_filter.size());
  std::exclusive_scan(expected_collision_filter.begin(),
                      expected_collision_filter.end(),
                      expected_prefix_sum.begin(), 0);
  EXPECT_EQ(prefix_sum, expected_prefix_sum);

  // check compacted narrow_phase_check_indices_
  narrow_phase_check_indices =
      SyclProximityEngineAttorney::get_narrow_phase_check_indices(impl);
  expected_narrow_phase_check_indices = {0, 2, 4, 5, 6, 7, 8, 10, 11};
  ASSERT_EQ(narrow_phase_check_indices.size(),
            expected_narrow_phase_check_indices.size());
  for (size_t i = 0; i < narrow_phase_check_indices.size(); ++i) {
    EXPECT_EQ(narrow_phase_check_indices[i],
              expected_narrow_phase_check_indices[i]);
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
          TessellationStrategy::kDenseInteriorVertices));
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
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureB = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereB, meshB.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereB(*meshB);
  const hydroelastic::SoftGeometry soft_geometryB(
      hydroelastic::SoftMesh(std::move(meshB), std::move(pressureB)));
  const GeometryId sphereB_id = GeometryId::get_new_id();

  // Compute the candidate tets with the two BVHs
  std::vector<std::pair<int, int>> candidate_tetrahedra;
  const auto callback = [&candidate_tetrahedra, &soft_geometryA,
                         &soft_geometryB](int tet0,
                                          int tet1) -> BvttCallbackResult {
    const double min_A = soft_geometryA.pressure_field().EvaluateMin(tet0);
    const double max_A = soft_geometryA.pressure_field().EvaluateMax(tet0);
    const double min_B = soft_geometryB.pressure_field().EvaluateMin(tet1);
    const double max_B = soft_geometryB.pressure_field().EvaluateMax(tet1);
    if (!(max_A < min_B || max_B < min_A))
      candidate_tetrahedra.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };

  // Arbitrarily pose the spheres into a colliding configuration.
  const RigidTransformd X_WA =
      RigidTransformd(Vector3d{0.0 * radius, 0.0 * radius, 0.3 * radius});
  const RigidTransformd X_WB =
      RigidTransformd(Vector3d{1.0 * radius, 0.0 * radius, 0.3 * radius});
  const RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);

  bvhSphereA.Collide(bvhSphereB, X_AB, callback);

  // Convert cadidate tets to collision_filter_ that can be compared to one from
  // sycl_proximity_engine
  std::vector<uint8_t> expected_filter(soft_geometryA.mesh().num_elements() *
                                           soft_geometryB.mesh().num_elements(),
                                       0);
  for (auto [eA, eB] : candidate_tetrahedra) {
    const int i = eA * soft_geometryB.mesh().num_elements() + eB;
    expected_filter[i] = 1;
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
  const std::vector<uint8_t> collision_filter =
      SyclProximityEngineAttorney::get_collision_filter(impl);

  const int total_checks = SyclProximityEngineAttorney::get_total_checks(impl);

  ASSERT_EQ(total_checks, ssize(expected_filter));

  // Due to numerical tolerances in the CPU BVH leaf overlap test, there will
  // be false positives in the cpu filter. Therefore, we check that the sycl
  // filter is a subset of the cpu filter. Later on, we will verify that the cpu
  // narrow phase filters these out using true geometric quantites.
  std::vector<int> mismatch_indices;
  for (int i = 0; i < ssize(expected_filter); ++i) {
    EXPECT_LE(collision_filter[i], expected_filter[i]);
    if (collision_filter[i] < expected_filter[i]) {
      mismatch_indices.push_back(i);
    }
  }

  // Verify that each of the mismatches is TRULY a false positive from the CPU
  // broadphase. To do this, we compute the Aabb's of the element pairs in the
  // world frame, and then compute the intersection of those Aabb's and verify
  // that each intersection is empty.
  const auto CalcAabb = [](const Vector3d& a, const Vector3d& b,
                           const Vector3d& c, const Vector3d& d) {
    Vector3d min = a;
    Vector3d max = a;
    min = min.cwiseMin(b);
    max = max.cwiseMax(b);
    min = min.cwiseMin(c);
    max = max.cwiseMax(c);
    min = min.cwiseMin(d);
    max = max.cwiseMax(d);
    return std::make_pair(min, max);
  };

  for (int i : mismatch_indices) {
    int eA = i / soft_geometryB.mesh().num_elements();
    int eB = i - eA * soft_geometryB.mesh().num_elements();
    const auto [minA, maxA] =
        CalcAabb(X_WA * soft_geometryA.mesh().vertex(
                            soft_geometryA.mesh().element(eA).vertex(0)),
                 X_WA * soft_geometryA.mesh().vertex(
                            soft_geometryA.mesh().element(eA).vertex(1)),
                 X_WA * soft_geometryA.mesh().vertex(
                            soft_geometryA.mesh().element(eA).vertex(2)),
                 X_WA * soft_geometryA.mesh().vertex(
                            soft_geometryA.mesh().element(eA).vertex(3)));
    const auto [minB, maxB] =
        CalcAabb(X_WB * soft_geometryB.mesh().vertex(
                            soft_geometryB.mesh().element(eB).vertex(0)),
                 X_WB * soft_geometryB.mesh().vertex(
                            soft_geometryB.mesh().element(eB).vertex(1)),
                 X_WB * soft_geometryB.mesh().vertex(
                            soft_geometryB.mesh().element(eB).vertex(2)),
                 X_WB * soft_geometryB.mesh().vertex(
                            soft_geometryB.mesh().element(eB).vertex(3)));
    // Compute the bounds of the intersection of the Aabbs. The intersection is
    // empty if at least one of the dimensions has negative width.
    const Vector3d intersection_min = minA.cwiseMax(minB);
    const Vector3d intersection_max = maxA.cwiseMin(maxB);
    const Vector3d intersection_widths = intersection_max - intersection_min;
    EXPECT_LT(intersection_widths.minCoeff(), 0);
  }

  // Set the expected filter equal to test the prefix sum.
  for (int i : mismatch_indices) {
    expected_filter[i] = 0;
  }

  std::vector<size_t> prefix_sum =
      SyclProximityEngineAttorney::get_prefix_sum(impl);
  std::vector<size_t> expected_prefix_sum(expected_filter.size());
  std::exclusive_scan(expected_filter.begin(), expected_filter.end(),
                      expected_prefix_sum.begin(), 0);
  EXPECT_EQ(prefix_sum, expected_prefix_sum);

  // Get the narrow phase check indices
  const std::vector<size_t> narrow_phase_check_indices =
      SyclProximityEngineAttorney::get_narrow_phase_check_indices(impl);
  const size_t total_polygons =
      SyclProximityEngineAttorney::get_total_polygons(impl);
  const std::vector<size_t> valid_polygon_indices =
      SyclProximityEngineAttorney::get_valid_polygon_indices(impl);

  // Construct the element id pairs correspinding to each narrow_phase check
  // These id pairs will map to the global index that was used in the
  // collision_filter_ (row and column)
  std::vector<std::pair<int, int>> element_id_pairs;
  for (size_t i = 0; i < total_polygons; ++i) {
    size_t global_check_index =
        narrow_phase_check_indices[valid_polygon_indices[i]];
    int eA = global_check_index / soft_geometryB.mesh().num_elements();
    int eB = global_check_index - eA * soft_geometryB.mesh().num_elements();
    element_id_pairs.emplace_back(eA, eB);
  }
  std::vector<double> debug_polygon_vertices =
      SyclProximityEngineAttorney::get_debug_polygon_vertices(impl);

  std::unique_ptr<PolygonSurfaceMesh<double>> contact_surface;
  std::unique_ptr<PolygonSurfaceMeshFieldLinear<double, double>>
      contact_pressure;
  VolumeIntersector<PolyMeshBuilder<double>, Aabb> volume_intersector;
  volume_intersector.IntersectFields(
      soft_geometryA.pressure_field(), bvhSphereA,
      soft_geometryB.pressure_field(), bvhSphereB, X_AB, &contact_surface,
      &contact_pressure);
  fmt::print("ssize(compacted_polygon_areas): {}\n", total_polygons);
  fmt::print("contact surface num_faces: {}\n", contact_surface->num_faces());
  std::vector<int> polygons_found;
  std::vector<int> bad_area;
  std::vector<int> bad_centroid;
  std::vector<std::pair<int, int>> degenerate_tets;
  for (int i = 0; i < contact_surface->num_faces(); ++i) {
    const double expected_area = contact_surface->area(i);
    const Vector3d expected_centroid_M = contact_surface->element_centroid(i);
    // Transform by transforms of A since the contact surface is posed in frame
    // A.
    const Vector3d expected_centroid_W = X_WA * expected_centroid_M;
    const int tet0 = volume_intersector.tet0_of_polygon(i);
    const int tet1 = volume_intersector.tet1_of_polygon(i);
    const std::pair<int, int> tet_pair{tet0, tet1};
    const auto it =
        std::find(element_id_pairs.begin(), element_id_pairs.end(), tet_pair);
    // We expect to find polygons for every polygon in the cpu surface.
    EXPECT_TRUE(it != element_id_pairs.end());
    if (it != element_id_pairs.end()) {
      int index = (it - element_id_pairs.begin());
      polygons_found.push_back(index);
      if (std::abs(surfaces[0].areas()[index] - expected_area) >
          1e2 * std::numeric_limits<double>::epsilon()) {
        bad_area.push_back(index);
        std::cerr << fmt::format(
            "Bad area at index {} for tet pair ({}, {}): expected={}, "
            "got={}\n\n",
            index, tet0, tet1, expected_area, surfaces[0].areas()[index]);
        degenerate_tets.push_back(tet_pair);
      }
      const double centroid_error =
          (expected_centroid_W - surfaces[0].centroids()[index]).norm();
      if (centroid_error > 1e2 * std::numeric_limits<double>::epsilon() &&
          expected_area > 1e-15) {
        bad_centroid.push_back(index);
        std::cerr << fmt::format(
            "Bad centroid at index {} for tet pair ({}, {}) error: {} "
            "expected area: {}:, got area {}\n  "
            "expected={}\n  got=     "
            "{}\n\n",
            index, tet0, tet1, centroid_error, expected_area,
            surfaces[0].areas()[index],
            fmt_eigen(expected_centroid_W.transpose()),
            fmt_eigen(surfaces[0].centroids()[index].transpose()));
      }
    }
  }
  fmt::print("Polygons found by SYCL implementation: {}\n",
             ssize(polygons_found));
  fmt::print("Polygons with area difference beyond rounding error: {}\n",
             ssize(bad_area));
  EXPECT_EQ(bad_area.size(), 0);
  fmt::print(
      "Polygons with centroid difference beyond rounding error (in any of x, "
      "y, z): {}\n",
      ssize(bad_centroid));
  EXPECT_EQ(bad_centroid.size(), 0);

  std::sort(polygons_found.begin(), polygons_found.end());
  int counter = 0;
  for (int i = 0; i < static_cast<int>(surfaces[0].num_polygons()); ++i) {
    if (!std::binary_search(polygons_found.begin(), polygons_found.end(), i)) {
      if (surfaces[0].areas()[i] > 1e-15) {
        std::cerr << fmt::format(
            "Polygon with index {} and tet pair ({}, {}) has area {} and "
            "centroid {} in SYCL but not found in Drake\n",
            i, element_id_pairs[i].first, element_id_pairs[i].second,
            surfaces[0].areas()[i],
            fmt_eigen(surfaces[0].centroids()[i].transpose()));
        counter++;
      }
    }
  }
  fmt::print("Polygons found by SYCL implementation but NOT in Drake: {}\n",
             counter);
  EXPECT_EQ(counter, 0);
}

GTEST_TEST(SPETest, ThreeSpheresColliding) {
  constexpr double radius = 0.5;
  constexpr double resolution_hint_A = 0.5 * radius;
  constexpr double resolution_hint_B = 0.75 * radius;
  constexpr double resolution_hint_C = radius;
  constexpr double hydroelastic_modulus = 1e+7;

  // Sphere A
  const Sphere sphereA(radius);
  auto meshA =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereA, resolution_hint_A,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureA = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereA, meshA.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereA(*meshA);

  hydroelastic::SoftGeometry soft_geometryA(
      hydroelastic::SoftMesh(std::move(meshA), std::move(pressureA)));
  const GeometryId sphereA_id = GeometryId::get_new_id();

  // Sphere B
  const Sphere sphereB(radius);
  auto meshB =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereB, resolution_hint_B,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureB = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereB, meshB.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereB(*meshB);
  hydroelastic::SoftGeometry soft_geometryB(
      hydroelastic::SoftMesh(std::move(meshB), std::move(pressureB)));
  const GeometryId sphereB_id = GeometryId::get_new_id();

  // Sphere C
  const Sphere sphereC(radius);
  auto meshC =
      std::make_unique<VolumeMesh<double>>(MakeSphereVolumeMesh<double>(
          sphereC, resolution_hint_C,
          TessellationStrategy::kDenseInteriorVertices));
  auto pressureC = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeSpherePressureField(sphereC, meshC.get(), hydroelastic_modulus));
  const Bvh<Aabb, VolumeMesh<double>> bvhSphereC(*meshC);
  hydroelastic::SoftGeometry soft_geometryC(
      hydroelastic::SoftMesh(std::move(meshC), std::move(pressureC)));
  const GeometryId sphereC_id = GeometryId::get_new_id();

  // ARbitrarily pose the spheres into a colliding configuration.
  const RigidTransformd X_WA =
      RigidTransformd(Vector3d{0.2 * radius, 0.1 * radius, 0.3 * radius});
  const RigidTransformd X_WB =
      RigidTransformd(Vector3d{0.1 * radius, 0.2 * radius, 0.3 * radius});
  const RigidTransformd X_WC =
      RigidTransformd(Vector3d{0.2 * radius, 0.2 * radius, 0.3 * radius});
  const RigidTransformd X_AB = X_WA.InvertAndCompose(X_WB);
  const RigidTransformd X_AC = X_WA.InvertAndCompose(X_WC);
  const RigidTransformd X_BC = X_WB.InvertAndCompose(X_WC);

  // Compute the candidate tets.
  std::vector<std::pair<int, int>> candidate_tetrahedra_AB;
  const auto callback_AB = [&candidate_tetrahedra_AB, &soft_geometryA,
                            &soft_geometryB](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_A = soft_geometryA.pressure_field().EvaluateMin(tet0);
    const double max_A = soft_geometryA.pressure_field().EvaluateMax(tet0);
    const double min_B = soft_geometryB.pressure_field().EvaluateMin(tet1);
    const double max_B = soft_geometryB.pressure_field().EvaluateMax(tet1);
    if (!(max_A < min_B || max_B < min_A))
      candidate_tetrahedra_AB.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereA.Collide(bvhSphereB, X_AB, callback_AB);

  std::vector<std::pair<int, int>> candidate_tetrahedra_AC;
  const auto callback_AC = [&candidate_tetrahedra_AC, &soft_geometryA,
                            &soft_geometryC](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_A = soft_geometryA.pressure_field().EvaluateMin(tet0);
    const double max_A = soft_geometryA.pressure_field().EvaluateMax(tet0);
    const double min_C = soft_geometryC.pressure_field().EvaluateMin(tet1);
    const double max_C = soft_geometryC.pressure_field().EvaluateMax(tet1);
    if (!(max_A < min_C || max_C < min_A))
      candidate_tetrahedra_AC.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereA.Collide(bvhSphereC, X_AC, callback_AC);

  std::vector<std::pair<int, int>> candidate_tetrahedra_BC;
  const auto callback_BC = [&candidate_tetrahedra_BC, &soft_geometryB,
                            &soft_geometryC](int tet0,
                                             int tet1) -> BvttCallbackResult {
    const double min_B = soft_geometryB.pressure_field().EvaluateMin(tet0);
    const double max_B = soft_geometryB.pressure_field().EvaluateMax(tet0);
    const double min_C = soft_geometryC.pressure_field().EvaluateMin(tet1);
    const double max_C = soft_geometryC.pressure_field().EvaluateMax(tet1);
    if (!(max_B < min_C || max_C < min_B))
      candidate_tetrahedra_BC.emplace_back(tet0, tet1);

    return BvttCallbackResult::Continue;
  };
  bvhSphereB.Collide(bvhSphereC, X_BC, callback_BC);

  // Convert cadidate tets to collision_filter_ that can be compared to one
  // from sycl_proximity_engine
  const int num_A = soft_geometryA.mesh().num_elements();
  const int num_B = soft_geometryB.mesh().num_elements();
  const int num_C = soft_geometryC.mesh().num_elements();

  const int AB_size = num_A * num_B;
  const int AC_size = num_A * num_C;
  const int BC_size = num_B * num_C;

  std::vector<uint8_t> expected_filter(AB_size + AC_size + BC_size, 0);

  for (auto [eA, eB] : candidate_tetrahedra_AB) {
    const int i = eA * (num_B + num_C) + eB;
    expected_filter[i] = 1;
  }
  for (auto [eA, eC] : candidate_tetrahedra_AC) {
    const int i = eA * (num_B + num_C) + eC + num_B;
    expected_filter[i] = 1;
  }
  for (auto [eB, eC] : candidate_tetrahedra_BC) {
    const int i = eB * num_C + eC + (AB_size + AC_size);
    expected_filter[i] = 1;
  }

  // Create soft geometries
  const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>
      soft_geometries{{sphereA_id, soft_geometryA},
                      {sphereB_id, soft_geometryB},
                      {sphereC_id, soft_geometryC}};

  // Instantiate SyclProximityEngine to obtain collision filter
  drake::geometry::internal::sycl_impl::SyclProximityEngine engine(
      soft_geometries);

  // Update collision candidates
  engine.UpdateCollisionCandidates(
      {SortedPair<GeometryId>(sphereA_id, sphereB_id),
       SortedPair<GeometryId>(sphereA_id, sphereC_id),
       SortedPair<GeometryId>(sphereB_id, sphereC_id)});

  // Move spheres closer so that they collide
  const std::unordered_map<GeometryId, RigidTransformd> X_WGs{
      {sphereA_id, X_WA}, {sphereB_id, X_WB}, {sphereC_id, X_WC}};
  const auto surfaces = engine.ComputeSYCLHydroelasticSurface(X_WGs);

  // Get the total checks
  const auto impl = SyclProximityEngineAttorney::get_impl(engine);

  // Collision filter check
  const std::vector<uint8_t> collision_filter =
      SyclProximityEngineAttorney::get_collision_filter(impl);

  const int total_checks = SyclProximityEngineAttorney::get_total_checks(impl);

  ASSERT_EQ(total_checks, ssize(expected_filter));

  // Due to numerical tolerances in the CPU BVH leaf overlap test, there will
  // be false positives in the cpu filter. Therefore, we check that the sycl
  // filter is a subset of the cpu filter. Later on, we will verify that the
  // cpu narrow phase filters these out using true geometric quantites.
  std::vector<int> mismatch_indices;
  for (int i = 0; i < ssize(expected_filter); ++i) {
    EXPECT_LE(collision_filter[i], expected_filter[i]);
    if (collision_filter[i] < expected_filter[i]) {
      mismatch_indices.push_back(i);
    }
  }

  // Verify that each of the mismatches is TRULY a false positive from the CPU
  // broadphase. To do this, we compute the Aabb's of the element pairs in the
  // world frame, and then compute the intersection of those Aabb's and verify
  // that each intersection is empty.
  const auto CalcAabb = [](const Vector3d& a, const Vector3d& b,
                           const Vector3d& c, const Vector3d& d) {
    Vector3d min = a;
    Vector3d max = a;
    min = min.cwiseMin(b);
    max = max.cwiseMax(b);
    min = min.cwiseMin(c);
    max = max.cwiseMax(c);
    min = min.cwiseMin(d);
    max = max.cwiseMax(d);
    return std::make_pair(min, max);
  };

  for (int i : mismatch_indices) {
    Vector3d min0, max0, min1, max1;

    if (i > (AB_size + AC_size)) {
      int eB = (i - (AB_size + AC_size)) / num_C;
      int eC = (i - (AB_size + AC_size)) - eB * num_C;
      std::tie(min0, max0) =
          CalcAabb(X_WB * soft_geometryB.mesh().vertex(
                              soft_geometryB.mesh().element(eB).vertex(0)),
                   X_WB * soft_geometryB.mesh().vertex(
                              soft_geometryB.mesh().element(eB).vertex(1)),
                   X_WB * soft_geometryB.mesh().vertex(
                              soft_geometryB.mesh().element(eB).vertex(2)),
                   X_WB * soft_geometryB.mesh().vertex(
                              soft_geometryB.mesh().element(eB).vertex(3)));
      std::tie(min1, max1) =
          CalcAabb(X_WC * soft_geometryC.mesh().vertex(
                              soft_geometryC.mesh().element(eC).vertex(0)),
                   X_WC * soft_geometryC.mesh().vertex(
                              soft_geometryC.mesh().element(eC).vertex(1)),
                   X_WC * soft_geometryC.mesh().vertex(
                              soft_geometryC.mesh().element(eC).vertex(2)),
                   X_WC * soft_geometryC.mesh().vertex(
                              soft_geometryC.mesh().element(eC).vertex(3)));
    } else {
      int eA = i / (num_B + num_C);
      int eB = i - eA * (num_B + num_C);
      std::tie(min0, max0) =
          CalcAabb(X_WA * soft_geometryA.mesh().vertex(
                              soft_geometryA.mesh().element(eA).vertex(0)),
                   X_WA * soft_geometryA.mesh().vertex(
                              soft_geometryA.mesh().element(eA).vertex(1)),
                   X_WA * soft_geometryA.mesh().vertex(
                              soft_geometryA.mesh().element(eA).vertex(2)),
                   X_WA * soft_geometryA.mesh().vertex(
                              soft_geometryA.mesh().element(eA).vertex(3)));
      if (eB > num_B) {
        int eC = eB - num_B;
        std::tie(min1, max1) =
            CalcAabb(X_WC * soft_geometryC.mesh().vertex(
                                soft_geometryC.mesh().element(eC).vertex(0)),
                     X_WC * soft_geometryC.mesh().vertex(
                                soft_geometryC.mesh().element(eC).vertex(1)),
                     X_WC * soft_geometryC.mesh().vertex(
                                soft_geometryC.mesh().element(eC).vertex(2)),
                     X_WC * soft_geometryC.mesh().vertex(
                                soft_geometryC.mesh().element(eC).vertex(3)));
      } else {
        std::tie(min1, max1) =
            CalcAabb(X_WB * soft_geometryB.mesh().vertex(
                                soft_geometryB.mesh().element(eB).vertex(0)),
                     X_WB * soft_geometryB.mesh().vertex(
                                soft_geometryB.mesh().element(eB).vertex(1)),
                     X_WB * soft_geometryB.mesh().vertex(
                                soft_geometryB.mesh().element(eB).vertex(2)),
                     X_WB * soft_geometryB.mesh().vertex(
                                soft_geometryB.mesh().element(eB).vertex(3)));
      }
    }
    // Compute the bounds of the intersection of the Aabbs. The intersection
    // is empty if at least one of the dimensions has negative width.
    const Vector3d intersection_min = min0.cwiseMax(min1);
    const Vector3d intersection_max = max0.cwiseMin(max1);
    const Vector3d intersection_widths = intersection_max - intersection_min;
    EXPECT_LT(intersection_widths.minCoeff(), 0);
  }

  // Set the expected filter equal to test the prefix sum.
  for (int i : mismatch_indices) {
    expected_filter[i] = 0;
  }

  std::vector<size_t> prefix_sum =
      SyclProximityEngineAttorney::get_prefix_sum(impl);
  std::vector<size_t> expected_prefix_sum(expected_filter.size());
  std::exclusive_scan(expected_filter.begin(), expected_filter.end(),
                      expected_prefix_sum.begin(), 0);
  EXPECT_EQ(prefix_sum, expected_prefix_sum);

  // Get polygon areas and centroids
  const std::vector<double> polygon_areas =
      SyclProximityEngineAttorney::get_polygon_areas(impl);
  const std::vector<Vector3d> polygon_centroids =
      SyclProximityEngineAttorney::get_polygon_centroids(impl);

  // Get the narrow phase check indices
  const std::vector<size_t> narrow_phase_check_indices =
      SyclProximityEngineAttorney::get_narrow_phase_check_indices(impl);

  // Construct the element id pairs correspinding to each narrow_phase check
  // These id pairs will map to the global index that was used in the
  // collision_filter_ (row and column)
  std::vector<std::pair<int, int>> element_id_pairs;
  for (size_t i = 0; i < polygon_areas.size(); ++i) {
    size_t global_check_index = narrow_phase_check_indices[i];
    if (global_check_index > static_cast<size_t>(AB_size + AC_size)) {
      int eB = (global_check_index - (AB_size + AC_size)) / num_C;
      int eC = (global_check_index - (AB_size + AC_size)) - eB * num_C;
      element_id_pairs.emplace_back(eB, eC);
    } else {
      int eA = global_check_index / (num_B + num_C);
      int eB = global_check_index - eA * (num_B + num_C);
      element_id_pairs.emplace_back(eA, eB);
    }
  }
}

}  // namespace
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
