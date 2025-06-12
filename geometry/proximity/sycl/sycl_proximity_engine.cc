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

#include <oneapi/dpl/execution>  // For execution policies
#include <oneapi/dpl/numeric>    // For exclusive_scan
#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"
#include "drake/geometry/proximity/sycl/utils/sycl_contact_polygon.h"
#include "drake/geometry/proximity/sycl/utils/sycl_equilibrium_plane.h"
#include "drake/geometry/proximity/sycl/utils/sycl_hydroelastic_surface_creator.h"
#include "drake/geometry/proximity/sycl/utils/sycl_memory_manager.h"
#include "drake/geometry/proximity/sycl/utils/sycl_tetrahedron_slicing.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

#ifdef __SYCL_DEVICE_ONLY__
#define DRAKE_SYCL_DEVICE_INLINE [[sycl::device]]
#else
#define DRAKE_SYCL_DEVICE_INLINE
#endif

// Implementation class for SyclProximityEngine that contains all SYCL-specific
// code
class SyclProximityEngine::Impl {
 public:
  // Default constructor
  Impl() : q_device_(InitializeQueue()), mem_mgr_(q_device_) {}

  // Constructor that initializes with soft geometries
  Impl(const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>&
           soft_geometries)
      : q_device_(InitializeQueue()), mem_mgr_(q_device_) {
    DRAKE_THROW_UNLESS(soft_geometries.size() > 0);

    // Extract and sort geometry IDs for deterministic ordering
    std::vector<GeometryId> sorted_ids;
    sorted_ids.reserve(soft_geometries.size());
    for (const auto& [id, _] : soft_geometries) {
      sorted_ids.push_back(id);
    }
    std::sort(sorted_ids.begin(), sorted_ids.end());

    // Get number of geometries
    num_geometries_ = soft_geometries.size();

    // Allocate host memory for geometry IDs
    mesh_data_.geometry_ids =
        sycl::malloc_host<GeometryId>(num_geometries_, q_device_);

    // Allocate device memory for lookup arrays
    mesh_data_.element_offsets =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);
    mesh_data_.vertex_offsets =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);
    mesh_data_.element_counts =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);
    mesh_data_.vertex_counts =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);

    // First compute totals and build lookup data
    total_elements_ = 0;
    total_vertices_ = 0;

    // Use the sorted IDs to ensure deterministic ordering
    for (size_t id_index = 0; id_index < sorted_ids.size(); ++id_index) {
      const GeometryId& id = sorted_ids[id_index];
      const hydroelastic::SoftGeometry& soft_geometry = soft_geometries.at(id);
      const hydroelastic::SoftMesh& soft_mesh = soft_geometry.soft_mesh();
      const VolumeMesh<double>& mesh = soft_mesh.mesh();

      // Store the geometry's ID
      mesh_data_.geometry_ids[id_index] = id;

      // Store offsets and counts directly (no memcpy needed with shared memory)
      mesh_data_.element_offsets[id_index] = total_elements_;
      mesh_data_.vertex_offsets[id_index] = total_vertices_;

      const size_t num_elements = mesh.num_elements();
      const size_t num_vertices = mesh.num_vertices();
      mesh_data_.element_counts[id_index] = num_elements;
      mesh_data_.vertex_counts[id_index] = num_vertices;

      // Update totals
      total_elements_ += num_elements;
      total_vertices_ += num_vertices;
    }

    // Allocate device memory for all meshes
    mesh_data_.elements =
        sycl::malloc_device<std::array<int, 4>>(total_elements_, q_device_);
    mesh_data_.element_mesh_ids =
        sycl::malloc_device<size_t>(total_elements_, q_device_);

    mesh_data_.vertices_M =
        sycl::malloc_device<Vector3<double>>(total_vertices_, q_device_);
    mesh_data_.pressures =
        sycl::malloc_device<double>(total_vertices_, q_device_);
    mesh_data_.vertex_mesh_ids =
        sycl::malloc_device<size_t>(total_vertices_, q_device_);

    mesh_data_.inward_normals_M =
        sycl::malloc_device<std::array<Vector3<double>, 4>>(total_elements_,
                                                            q_device_);
    mesh_data_.min_pressures =
        sycl::malloc_device<double>(total_elements_, q_device_);
    mesh_data_.max_pressures =
        sycl::malloc_device<double>(total_elements_, q_device_);

    // Allocate combined gradient and pressure arrays
    mesh_data_.gradient_M_pressure_at_Mo =
        sycl::malloc_device<Vector4<double>>(total_elements_, q_device_);

    // Allocate even for world frame quantities
    mesh_data_.vertices_W =
        sycl::malloc_device<Vector3<double>>(total_vertices_, q_device_);
    mesh_data_.inward_normals_W =
        sycl::malloc_device<std::array<Vector3<double>, 4>>(total_elements_,
                                                            q_device_);
    mesh_data_.gradient_W_pressure_at_Wo =
        sycl::malloc_device<Vector4<double>>(total_elements_, q_device_);

    // Allocate device memory for transforms
    mesh_data_.transforms =
        sycl::malloc_host<double>(num_geometries_ * 12, q_device_);

    // Allocate device memory for element AABBs
    mesh_data_.element_aabb_min_W =
        sycl::malloc_device<Vector3<double>>(total_elements_, q_device_);
    mesh_data_.element_aabb_max_W =
        sycl::malloc_device<Vector3<double>>(total_elements_, q_device_);

    // Copy data for each mesh
    std::vector<sycl::event> transfer_events;  // Store all transfer events

    // Use the sorted IDs for deterministic ordering
    for (size_t id_index = 0; id_index < sorted_ids.size(); ++id_index) {
      const GeometryId& id = sorted_ids[id_index];
      const hydroelastic::SoftGeometry& soft_geometry = soft_geometries.at(id);
      const hydroelastic::SoftMesh& soft_mesh = soft_geometry.soft_mesh();
      const VolumeMesh<double>& mesh = soft_mesh.mesh();
      const VolumeMeshFieldLinear<double, double>& pressure_field =
          soft_mesh.pressure();

      // Direct access to shared memory values
      size_t element_offset = mesh_data_.element_offsets[id_index];
      size_t vertex_offset = mesh_data_.vertex_offsets[id_index];
      size_t num_elements = mesh_data_.element_counts[id_index];
      size_t num_vertices = mesh_data_.vertex_counts[id_index];

      const auto& mesh_elements = mesh.tetrahedra();
      for (size_t i = 0; i < num_elements; ++i) {
        const std::array<int, 4>& vertices = mesh_elements[i].getAllVertices();
        // Copy element by element to maintain lifetime safety
        transfer_events.push_back(
            q_device_.memcpy(mesh_data_.elements + element_offset + i,
                             &vertices, sizeof(std::array<int, 4>)));
      }
      // Fill in the mesh id for all elements in this mesh
      transfer_events.push_back(
          q_device_.fill(mesh_data_.element_mesh_ids + element_offset, id_index,
                         num_elements));

      // Vertices
      transfer_events.push_back(q_device_.memcpy(
          mesh_data_.vertices_M + vertex_offset, mesh.vertices().data(),
          num_vertices * sizeof(Vector3<double>)));

      // Pressures
      transfer_events.push_back(q_device_.memcpy(
          mesh_data_.pressures + vertex_offset, pressure_field.values().data(),
          num_vertices * sizeof(double)));

      // Fill in the mesh id for all vertices in this mesh
      transfer_events.push_back(q_device_.fill(
          mesh_data_.vertex_mesh_ids + vertex_offset, id_index, num_vertices));

      // Inward Normals
      transfer_events.push_back(q_device_.memcpy(
          mesh_data_.inward_normals_M + element_offset,
          mesh.inward_normals().data(),
          num_elements * sizeof(std::array<Vector3<double>, 4>)));

      // Min Pressures
      transfer_events.push_back(q_device_.memcpy(
          mesh_data_.min_pressures + element_offset,
          pressure_field.min_values().data(), num_elements * sizeof(double)));

      // Max Pressures
      transfer_events.push_back(q_device_.memcpy(
          mesh_data_.max_pressures + element_offset,
          pressure_field.max_values().data(), num_elements * sizeof(double)));

      // Create a temporary host buffer to pack gradient and pressure data
      std::vector<Vector4<double>> packed_gradient_pressure(num_elements);

      // Pack the gradient data (first 3 components) and pressure at Mo (4th
      // component)
      const auto& gradients = pressure_field.gradients();
      const auto& pressuresat_Mo = pressure_field.values_at_Mo();

      for (size_t i = 0; i < num_elements; ++i) {
        packed_gradient_pressure[i][0] = gradients[i][0];    // x component
        packed_gradient_pressure[i][1] = gradients[i][1];    // y component
        packed_gradient_pressure[i][2] = gradients[i][2];    // z component
        packed_gradient_pressure[i][3] = pressuresat_Mo[i];  // pressure at Mo
      }

      // Transfer the packed data in a single operation
      transfer_events.push_back(q_device_.memcpy(
          mesh_data_.gradient_M_pressure_at_Mo + element_offset,
          packed_gradient_pressure.data(),
          num_elements * sizeof(Vector4<double>)));
    }

    // ========================================
    // Some pre-processing for broad phase collision detection
    // ========================================
    // Stores at i the number of checks needs to be for the ith geometry
    collision_data_.total_checks_per_geometry =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);

    // geom_collision_filternum_cols[i] is the number of elements that need to
    // be checked with each of the elements of the ith geometry
    // Will be highest for 1st geometry and lowest for the last geometry (due to
    // symmetric nature of collision_filter - we are only consider upper
    // triangle)
    collision_data_.geom_collision_filter_num_cols =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);

    // Stores the exclusive scan of total checks per geometry
    collision_data_.geom_collision_filter_check_offsets =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);

    num_elements_in_last_geometry_ =
        mesh_data_.element_counts[num_geometries_ - 1];
    total_checks_ = 0;
    // Done entirely in CPU because its only looping over num_geometries
    for (size_t i = 0; i < num_geometries_ - 1; ++i) {
      const size_t num_elementsin_geometry = mesh_data_.element_counts[i];
      const size_t num_elementsin_rest_of_geometries =
          (mesh_data_.element_offsets[num_geometries_ - 1] +
           num_elements_in_last_geometry_) -
          mesh_data_.element_offsets[i + 1];
      collision_data_.geom_collision_filter_num_cols[i] =
          num_elementsin_rest_of_geometries;
      // We need to check each element in this geometry with each element in
      // the rest of the geometries
      collision_data_.total_checks_per_geometry[i] =
          num_elementsin_rest_of_geometries * num_elementsin_geometry;
      collision_data_.geom_collision_filter_check_offsets[i] = total_checks_;
      total_checks_ += collision_data_.total_checks_per_geometry[i];
    }
    collision_data_.total_checks_per_geometry[num_geometries_ - 1] = 0;

    // Allocate memory for polygon areas and centroids by estimating the narrow
    // phase checks to be 1% of total element checks
    estimated_narrow_phase_checks_ = std::max(
        static_cast<size_t>(1), static_cast<size_t>(total_checks_ / 100));
    // Similarly, we estimate the number of polygons to be 1% of the narrow
    // phase checks
    estimated_polygons_ =
        std::max(static_cast<size_t>(1),
                 static_cast<size_t>(estimated_narrow_phase_checks_ / 100));

    // Resize based on the estimated narrow phase checks
    current_polygon_areas_size_ = estimated_narrow_phase_checks_;
    polygon_data_.polygon_areas =
        sycl::malloc_device<double>(current_polygon_areas_size_, q_device_);
    // "3" is for each coordinate
    polygon_data_.polygon_centroids = sycl::malloc_device<Vector3<double>>(
        current_polygon_areas_size_, q_device_);
    polygon_data_.polygon_normals = sycl::malloc_device<Vector3<double>>(
        current_polygon_areas_size_, q_device_);
    polygon_data_.polygon_g_M =
        sycl::malloc_device<double>(current_polygon_areas_size_, q_device_);
    polygon_data_.polygon_g_N =
        sycl::malloc_device<double>(current_polygon_areas_size_, q_device_);
    polygon_data_.polygon_pressure_W =
        sycl::malloc_device<double>(current_polygon_areas_size_, q_device_);
    polygon_data_.polygon_geom_index_A =
        sycl::malloc_device<GeometryId>(current_polygon_areas_size_, q_device_);
    polygon_data_.polygon_geom_index_B =
        sycl::malloc_device<GeometryId>(current_polygon_areas_size_, q_device_);

    // Allocate memory for narrow_phase_check_indices
    current_narrow_phase_check_indices_size_ = estimated_narrow_phase_checks_;
    collision_data_.narrow_phase_check_indices = sycl::malloc_device<size_t>(
        current_narrow_phase_check_indices_size_, q_device_);
    collision_data_.narrow_phase_check_validity = sycl::malloc_device<uint8_t>(
        current_narrow_phase_check_indices_size_, q_device_);
    collision_data_.prefix_sum_narrow_phase_checks =
        sycl::malloc_device<size_t>(current_narrow_phase_check_indices_size_,
                                    q_device_);

    // Resize compacted data structures based on the estimated polygon sizes
    current_polygon_indices_size_ = estimated_polygons_;
    polygon_data_.compacted_polygon_areas =
        sycl::malloc_device<double>(current_polygon_indices_size_, q_device_);
    polygon_data_.compacted_polygon_centroids =
        sycl::malloc_device<Vector3<double>>(current_polygon_indices_size_,
                                             q_device_);
    polygon_data_.compacted_polygon_normals =
        sycl::malloc_device<Vector3<double>>(current_polygon_indices_size_,
                                             q_device_);
    polygon_data_.compacted_polygon_g_M =
        sycl::malloc_device<double>(current_polygon_indices_size_, q_device_);
    polygon_data_.compacted_polygon_g_N =
        sycl::malloc_device<double>(current_polygon_indices_size_, q_device_);
    polygon_data_.compacted_polygon_pressure_W =
        sycl::malloc_device<double>(current_polygon_indices_size_, q_device_);
    polygon_data_.compacted_polygon_geom_index_A =
        sycl::malloc_device<GeometryId>(current_polygon_indices_size_,
                                        q_device_);
    polygon_data_.compacted_polygon_geom_index_B =
        sycl::malloc_device<GeometryId>(current_polygon_indices_size_,
                                        q_device_);
    polygon_data_.valid_polygon_indices =
        sycl::malloc_device<size_t>(current_polygon_indices_size_, q_device_);

    // Generate collision filter for all checks
    collision_data_.collision_filter =
        sycl::malloc_device<uint8_t>(total_checks_, q_device_);
    collision_data_.prefix_sum_total_checks =
        sycl::malloc_device<size_t>(total_checks_, q_device_);

    collision_data_.collision_filter_host_body_index =
        sycl::malloc_host<size_t>(total_checks_, q_device_);

    // Fill in geometry index based on checks per geometry
    std::vector<sycl::event> collision_filter_host_body_indexfill_events;
    for (size_t i = 0; i < num_geometries_ - 1; ++i) {
      const size_t num_checks = collision_data_.total_checks_per_geometry[i];
      collision_filter_host_body_indexfill_events.push_back(q_device_.fill(
          collision_data_.collision_filter_host_body_index +
              collision_data_.geom_collision_filter_check_offsets[i],
          i, num_checks));
    }

    // Wait for all transfers to complete before returning
    sycl::event::wait_and_throw(transfer_events);
    sycl::event::wait_and_throw(collision_filter_host_body_indexfill_events);
  }

  // Copy constructor
  Impl(const Impl& other) : q_device_(other.q_device_), mem_mgr_(q_device_) {
    // TODO(huzaifa): Implement deep copy of SYCL resources
    // For now, we'll just create a shallow copy which isn't ideal
    collision_candidates_ = other.collision_candidates_;
    num_geometries_ = other.num_geometries_;
  }

  // Copy assignment operator
  Impl& operator=(const Impl& other) {
    if (this != &other) {
      // TODO(huzaifa): Implement deep copy of SYCL resources
      // For now, we'll just create a shallow copy which isn't ideal
      q_device_ = other.q_device_;
      collision_candidates_ = other.collision_candidates_;
      num_geometries_ = other.num_geometries_;
    }
    return *this;
  }

  // Destructor
  ~Impl() {
    // Free device memory
    if (num_geometries_ > 0) {
      sycl::free(mesh_data_.geometry_ids, q_device_);

      sycl::free(mesh_data_.element_offsets, q_device_);
      sycl::free(mesh_data_.vertex_offsets, q_device_);
      sycl::free(mesh_data_.element_counts, q_device_);
      sycl::free(mesh_data_.vertex_counts, q_device_);
      sycl::free(mesh_data_.element_mesh_ids, q_device_);
      sycl::free(mesh_data_.vertex_mesh_ids, q_device_);

      sycl::free(mesh_data_.elements, q_device_);
      sycl::free(mesh_data_.vertices_M, q_device_);
      sycl::free(mesh_data_.vertices_W, q_device_);
      sycl::free(mesh_data_.inward_normals_M, q_device_);
      sycl::free(mesh_data_.inward_normals_W, q_device_);
      sycl::free(mesh_data_.pressures, q_device_);
      sycl::free(mesh_data_.min_pressures, q_device_);
      sycl::free(mesh_data_.max_pressures, q_device_);
      sycl::free(mesh_data_.gradient_M_pressure_at_Mo, q_device_);
      sycl::free(mesh_data_.gradient_W_pressure_at_Wo, q_device_);
      sycl::free(mesh_data_.transforms, q_device_);

      sycl::free(collision_data_.geom_collision_filter_num_cols, q_device_);
      sycl::free(collision_data_.geom_collision_filter_check_offsets,
                 q_device_);
      sycl::free(collision_data_.collision_filter, q_device_);
      sycl::free(collision_data_.total_checks_per_geometry, q_device_);
      sycl::free(collision_data_.prefix_sum_total_checks, q_device_);

      sycl::free(polygon_data_.polygon_areas, q_device_);
      sycl::free(polygon_data_.polygon_centroids, q_device_);
      sycl::free(polygon_data_.polygon_normals, q_device_);
      sycl::free(polygon_data_.polygon_g_M, q_device_);
      sycl::free(polygon_data_.polygon_g_N, q_device_);
      sycl::free(polygon_data_.polygon_pressure_W, q_device_);
      sycl::free(polygon_data_.polygon_geom_index_A, q_device_);
      sycl::free(polygon_data_.polygon_geom_index_B, q_device_);

      sycl::free(collision_data_.narrow_phase_check_indices, q_device_);
      sycl::free(collision_data_.narrow_phase_check_validity, q_device_);
      sycl::free(collision_data_.prefix_sum_narrow_phase_checks, q_device_);
      sycl::free(polygon_data_.debug_polygon_vertices, q_device_);

      sycl::free(polygon_data_.compacted_polygon_areas, q_device_);
      sycl::free(polygon_data_.compacted_polygon_centroids, q_device_);
      sycl::free(polygon_data_.compacted_polygon_normals, q_device_);
      sycl::free(polygon_data_.compacted_polygon_g_M, q_device_);
      sycl::free(polygon_data_.compacted_polygon_g_N, q_device_);
      sycl::free(polygon_data_.compacted_polygon_pressure_W, q_device_);
      sycl::free(polygon_data_.compacted_polygon_geom_index_A, q_device_);
      sycl::free(polygon_data_.compacted_polygon_geom_index_B, q_device_);

      sycl::free(polygon_data_.valid_polygon_indices, q_device_);
    }
  }

  // Check if SYCL is available
  static bool is_available() {
    try {
      // Attempt to construct a default queue. If a SYCL runtime is available
      // and a default device can be selected, this construction will succeed.
      sycl::queue q;
      // Suppress unused variable warning
      (void)q;
      return true;
    } catch (const sycl::exception& /* e */) {
      // An exception during queue construction indicates that a default SYCL
      // device/queue is not available.
      return false;
    } catch (...) {
      // Catch any other unexpected exceptions.
      return false;
    }
  }

  // Update collision candidates
  void UpdateCollisionCandidates(
      const std::vector<SortedPair<GeometryId>>& collision_candidates) {
    collision_candidates_ = collision_candidates;
  }

  // Compute hydroelastic surfaces
  std::vector<SYCLHydroelasticSurface> ComputeSYCLHydroelasticSurface(
      const std::unordered_map<GeometryId, math::RigidTransform<double>>&
          X_WGs) {
    if (total_checks_ == 0) {
      return {};
    }

    auto collision_filtermemset_event = q_device_.memset(
        collision_data_.collision_filter, 0, total_checks_ * sizeof(uint8_t));

    // Get transfomers in host
    for (size_t geom_index = 0; geom_index < num_geometries_; ++geom_index) {
      GeometryId geometry_id = mesh_data_.geometry_ids[geom_index];
      // To maintain our orders of geometries we need to loop through the stored
      // geometry id's and query the X_WGs for that geometry id. Cannot iterate
      // over the unordered_map because it is not ordered
      const auto& X_WG = X_WGs.at(geometry_id);
      const auto& transform = X_WG.GetAsMatrix34();
#pragma unroll
      for (size_t i = 0; i < 12; ++i) {
        size_t row = i / 4;
        size_t col = i % 4;
        // Store transforms in row major order
        // transforms = [R_00, R_01, R_02, p_0, R_10, R_11, R_12, p_1, ...]
        mesh_data_.transforms[geom_index * 12 + i] = transform(row, col);
      }
    }

    // ========================================
    // Command group 1: Transform quantities to world frame
    // ========================================

    // Combine all transformation kernels into a single command group
    auto transform_vertices_event = q_device_.submit([&](sycl::handler& h) {
      // Transform vertices
      h.parallel_for(sycl::range<1>(total_vertices_), sycl::id<1>(0),
                     [=, vertices_M = mesh_data_.vertices_M,
                      vertices_W = mesh_data_.vertices_W,
                      vertex_mesh_ids = mesh_data_.vertex_mesh_ids,
                      transforms = mesh_data_.transforms](sycl::id<1> idx) {
                       const size_t vertex_index = idx[0];
                       const size_t mesh_index = vertex_mesh_ids[vertex_index];

                       const double x = vertices_M[vertex_index][0];
                       const double y = vertices_M[vertex_index][1];
                       const double z = vertices_M[vertex_index][2];
                       double T[12];
#pragma unroll
                       for (size_t i = 0; i < 12; ++i) {
                         T[i] = transforms[mesh_index * 12 + i];
                       }
                       double new_x = T[0] * x + T[1] * y + T[2] * z + T[3];
                       double new_y = T[4] * x + T[5] * y + T[6] * z + T[7];
                       double new_z = T[8] * x + T[9] * y + T[10] * z + T[11];

                       vertices_W[vertex_index][0] = new_x;
                       vertices_W[vertex_index][1] = new_y;
                       vertices_W[vertex_index][2] = new_z;
                     });
    });

    // Transform inward normals
    auto transform_elem_quantities_event1 =
        q_device_.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::range<1>(total_elements_), sycl::id<1>(0),
              [=, inward_normals_M = mesh_data_.inward_normals_M,
               inward_normals_W = mesh_data_.inward_normals_W,
               element_mesh_ids = mesh_data_.element_mesh_ids,
               transforms = mesh_data_.transforms](sycl::id<1> idx) {
                const size_t element_index = idx[0];
                const size_t mesh_index = element_mesh_ids[element_index];

                double T[12];
#pragma unroll
                for (size_t i = 0; i < 12; ++i) {
                  T[i] = transforms[mesh_index * 12 + i];
                }

                // Each element has 4 inward normals
                for (size_t j = 0; j < 4; ++j) {
                  const double nx = inward_normals_M[element_index][j][0];
                  const double ny = inward_normals_M[element_index][j][1];
                  const double nz = inward_normals_M[element_index][j][2];

                  // Only rotation
                  inward_normals_W[element_index][j][0] =
                      T[0] * nx + T[1] * ny + T[2] * nz;
                  inward_normals_W[element_index][j][1] =
                      T[4] * nx + T[5] * ny + T[6] * nz;
                  inward_normals_W[element_index][j][2] =
                      T[8] * nx + T[9] * ny + T[10] * nz;
                }
              });
        });

    // Transform pressure gradients
    auto transform_elem_quantities_event2 =
        q_device_.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::range<1>(total_elements_), sycl::id<1>(0),
              [=,
               gradient_M_pressure_at_Mo = mesh_data_.gradient_M_pressure_at_Mo,
               gradient_W_pressure_at_Wo = mesh_data_.gradient_W_pressure_at_Wo,
               element_mesh_ids = mesh_data_.element_mesh_ids,
               transforms = mesh_data_.transforms](sycl::id<1> idx) {
                const size_t element_index = idx[0];
                const size_t mesh_index = element_mesh_ids[element_index];

                double T[12];
#pragma unroll
                for (size_t i = 0; i < 12; ++i) {
                  T[i] = transforms[mesh_index * 12 + i];
                }
                // Each element has 1 pressure gradient
                const double gp_mx =
                    gradient_M_pressure_at_Mo[element_index][0];
                const double gp_my =
                    gradient_M_pressure_at_Mo[element_index][1];
                const double gp_mz =
                    gradient_M_pressure_at_Mo[element_index][2];
                const double p_mo = gradient_M_pressure_at_Mo[element_index][3];

                // Only rotation for the gradient pressures
                const double gp_wx = T[0] * gp_mx + T[1] * gp_my + T[2] * gp_mz;
                const double gp_wy = T[4] * gp_mx + T[5] * gp_my + T[6] * gp_mz;
                const double gp_wz =
                    T[8] * gp_mx + T[9] * gp_my + T[10] * gp_mz;

                // TODO(huzaifa): Check this computation
                // By equating the rotated pressure field with the original
                // pressure field, we can solve for the pressure at the origin
                // of the world frame
                const double p_wo =
                    p_mo - (gp_wx * T[3] + gp_wy * T[7] + gp_wz * T[11]);
                gradient_W_pressure_at_Wo[element_index][0] = gp_wx;
                gradient_W_pressure_at_Wo[element_index][1] = gp_wy;
                gradient_W_pressure_at_Wo[element_index][2] = gp_wz;
                gradient_W_pressure_at_Wo[element_index][3] = p_wo;
              });
        });

    // =========================================
    // Command group 2: Generate candidate tet pairs
    // =========================================
    auto element_aabb_event = q_device_.submit([&](sycl::handler& h) {
      h.depends_on(transform_vertices_event);
      // Lets first compute all AABBs irrespective if they are needed or not
      // Allocate device memory for element AABBs
      // While doing this, assign false to all elements that are not part of
      // geometries that are collision candidates
      h.parallel_for(
          sycl::range<1>(total_elements_),
          [=, elements = mesh_data_.elements,
           vertices_W = mesh_data_.vertices_W,
           element_mesh_ids = mesh_data_.element_mesh_ids,
           element_aabb_min_W = mesh_data_.element_aabb_min_W,
           element_aabb_max_W = mesh_data_.element_aabb_max_W,
           vertex_offsets = mesh_data_.vertex_offsets](sycl::id<1> idx) {
            const size_t element_index = idx[0];
            const size_t geom_index = element_mesh_ids[element_index];
            // Get the four vertex indices for this tetrahedron
            const std::array<int, 4>& tet_vertices = elements[element_index];
            const size_t vertex_mesh_offset = vertex_offsets[geom_index];
            // Initialize min/max to first vertex
            double min_x = vertices_W[vertex_mesh_offset + tet_vertices[0]][0];
            double min_y = vertices_W[vertex_mesh_offset + tet_vertices[0]][1];
            double min_z = vertices_W[vertex_mesh_offset + tet_vertices[0]][2];

            double max_x = min_x;
            double max_y = min_y;
            double max_z = min_z;

            // Find min/max across all four vertices
            for (int i = 1; i < 4; ++i) {
              const size_t vertex_idx = vertex_mesh_offset + tet_vertices[i];

              // Update min coordinates
              min_x = sycl::min(min_x, vertices_W[vertex_idx][0]);
              min_y = sycl::min(min_y, vertices_W[vertex_idx][1]);
              min_z = sycl::min(min_z, vertices_W[vertex_idx][2]);

              // Update max coordinates
              max_x = sycl::max(max_x, vertices_W[vertex_idx][0]);
              max_y = sycl::max(max_y, vertices_W[vertex_idx][1]);
              max_z = sycl::max(max_z, vertices_W[vertex_idx][2]);
            }

            // Store the results
            element_aabb_min_W[element_index][0] = min_x;
            element_aabb_min_W[element_index][1] = min_y;
            element_aabb_min_W[element_index][2] = min_z;

            element_aabb_max_W[element_index][0] = max_x;
            element_aabb_max_W[element_index][1] = max_y;
            element_aabb_max_W[element_index][2] = max_z;
          });
    });

    // Now generate collision filter with the AABBs that we computed
    auto generate_collision_filterevent =
        q_device_.submit([&](sycl::handler& h) {
          h.depends_on({element_aabb_event, collision_filtermemset_event});
          h.parallel_for(
              sycl::range<1>(total_checks_),
              [=, collision_filter = collision_data_.collision_filter,
               collision_filter_host_body_index =
                   collision_data_.collision_filter_host_body_index,
               geom_collision_filter_num_cols =
                   collision_data_.geom_collision_filter_num_cols,
               element_offsets = mesh_data_.element_offsets,
               element_aabb_min_W = mesh_data_.element_aabb_min_W,
               element_aabb_max_W = mesh_data_.element_aabb_max_W,
               total_checks_per_geometry =
                   collision_data_.total_checks_per_geometry,
               min_pressures = mesh_data_.min_pressures,
               max_pressures = mesh_data_.max_pressures](sycl::id<1> idx) {
                const size_t check_index = idx[0];
                const size_t host_body_index =
                    collision_filter_host_body_index[check_index];
                // What elements is this check_index checking?
                // host_body_index is the geometry index that element A belongs
                // to
                size_t num_of_checks_offset = 0;
                if (host_body_index > 0) {
                  num_of_checks_offset =
                      total_checks_per_geometry[host_body_index - 1];
                }
                const size_t geom_local_check_number =
                    check_index - num_of_checks_offset;

                const size_t A_element_index =
                    element_offsets[host_body_index] +
                    geom_local_check_number /
                        geom_collision_filter_num_cols[host_body_index];
                const size_t B_element_index =
                    element_offsets[host_body_index + 1] +
                    geom_local_check_number %
                        geom_collision_filter_num_cols[host_body_index];

                // Default to not colliding.
                // collision_filter[check_index] = 0;

                // First check if the pressure fields of the elements intersect
                if (max_pressures[B_element_index] <
                        min_pressures[A_element_index] ||
                    max_pressures[A_element_index] <
                        min_pressures[B_element_index]) {
                  return;
                }

                // We have two element index, now just check their AABB
                // A element AABB
                // min
                for (int i = 0; i < 3; ++i) {
                  if (element_aabb_max_W[B_element_index][i] <
                      element_aabb_min_W[A_element_index][i])
                    return;
                  if (element_aabb_max_W[A_element_index][i] <
                      element_aabb_min_W[B_element_index][i])
                    return;
                }

                collision_filter[check_index] = 1;
              });
        });
    generate_collision_filterevent.wait();

    // =========================================
    // Generate list of check_indices that are active
    // =========================================

    auto policy = oneapi::dpl::execution::make_device_policy(q_device_);

    // Perform the exclusive scan using USM pointers as iterators
    // We need to convert uint8_t collision_filter values to size_t for the
    // scan
    oneapi::dpl::transform_exclusive_scan(
        policy, collision_data_.collision_filter,
        collision_data_.collision_filter + total_checks_,
        collision_data_.prefix_sum_total_checks,  // output
        static_cast<size_t>(0),                   // initial value
        sycl::plus<size_t>(),                     // binary operation
        [](uint8_t x) {
          return static_cast<size_t>(x);
        });  // transform uint8_t to size_t
    q_device_.wait_and_throw();

    // Total checks needed for narrow phase
    total_narrow_phase_checks_ = 0;
    q_device_
        .memcpy(&total_narrow_phase_checks_,
                collision_data_.prefix_sum_total_checks + total_checks_ - 1,
                sizeof(size_t))
        .wait();
    // Last element check or not?
    uint8_t last_check_flag = 0;
    q_device_
        .memcpy(&last_check_flag,
                collision_data_.collision_filter + total_checks_ - 1,
                sizeof(uint8_t))
        .wait();
    // If last check is 1, then we need to add one more check
    total_narrow_phase_checks_ += static_cast<size_t>(last_check_flag);

    if (total_narrow_phase_checks_ == 0) {
      return {};
    }

    if (total_narrow_phase_checks_ > current_polygon_areas_size_) {
      // Give a 10 % bigger size
      size_t new_size = static_cast<size_t>(1.1 * total_narrow_phase_checks_);

      // Free old memory
      sycl::free(polygon_data_.polygon_areas, q_device_);
      sycl::free(polygon_data_.polygon_centroids, q_device_);
      sycl::free(polygon_data_.polygon_normals, q_device_);
      sycl::free(polygon_data_.polygon_g_M, q_device_);
      sycl::free(polygon_data_.polygon_g_N, q_device_);
      sycl::free(polygon_data_.polygon_pressure_W, q_device_);
      sycl::free(polygon_data_.polygon_geom_index_A, q_device_);
      sycl::free(polygon_data_.polygon_geom_index_B, q_device_);

      sycl::free(collision_data_.narrow_phase_check_validity, q_device_);
      sycl::free(collision_data_.prefix_sum_narrow_phase_checks, q_device_);
      sycl::free(collision_data_.narrow_phase_check_indices, q_device_);

      // Allocate new memory with larger size
      polygon_data_.polygon_areas =
          sycl::malloc_device<double>(new_size, q_device_);
      polygon_data_.polygon_centroids =
          sycl::malloc_device<Vector3<double>>(new_size, q_device_);
      polygon_data_.polygon_normals =
          sycl::malloc_device<Vector3<double>>(new_size, q_device_);
      polygon_data_.polygon_g_M =
          sycl::malloc_device<double>(new_size, q_device_);
      polygon_data_.polygon_g_N =
          sycl::malloc_device<double>(new_size, q_device_);
      polygon_data_.polygon_pressure_W =
          sycl::malloc_device<double>(new_size, q_device_);
      polygon_data_.polygon_geom_index_A =
          sycl::malloc_device<GeometryId>(new_size, q_device_);
      polygon_data_.polygon_geom_index_B =
          sycl::malloc_device<GeometryId>(new_size, q_device_);

      collision_data_.narrow_phase_check_validity =
          sycl::malloc_device<uint8_t>(new_size, q_device_);
      collision_data_.prefix_sum_narrow_phase_checks =
          sycl::malloc_device<size_t>(new_size, q_device_);
      collision_data_.narrow_phase_check_indices =
          sycl::malloc_device<size_t>(new_size, q_device_);
      current_polygon_areas_size_ = new_size;
    }

    /// Reset quantities that need to be reset across timesteps
    std::vector<sycl::event> fill_events;
    fill_events.push_back(q_device_.fill(
        collision_data_.narrow_phase_check_validity, static_cast<uint8_t>(1),
        current_polygon_areas_size_));  // All valid at the start
    fill_events.push_back(
        q_device_.fill(collision_data_.prefix_sum_narrow_phase_checks, 0,
                       current_polygon_areas_size_));

    auto fill_narrow_phase_check_indicesevent =
        q_device_.submit([&](sycl::handler& h) {
          h.depends_on(generate_collision_filterevent);
          h.parallel_for(sycl::range<1>(total_checks_),
                         [=,
                          narrow_phase_check_indices =
                              collision_data_.narrow_phase_check_indices,
                          prefix_sum_total_checks =
                              collision_data_.prefix_sum_total_checks,
                          collision_filter = collision_data_.collision_filter](
                             sycl::id<1> idx) {
                           const size_t check_index = idx[0];
                           if (collision_filter[check_index] == 1) {
                             size_t narrow_check_num =
                                 prefix_sum_total_checks[check_index];
                             narrow_phase_check_indices[narrow_check_num] =
                                 check_index;
                           }
                         });
        });

    // =========================================
    // Command group 4: Narrow phase collision detection
    // =========================================

    // Create dependency vector
    std::vector<sycl::event> dependencies = {
        generate_collision_filterevent, fill_narrow_phase_check_indicesevent,
        transform_elem_quantities_event1, transform_elem_quantities_event2};
    // Add polygon fill events to dependencies
    dependencies.insert(dependencies.end(), fill_events.begin(),
                        fill_events.end());

    sycl::event compute_contact_polygon_event;
    if (q_device_.get_device().get_info<sycl::info::device::device_type>() ==
        sycl::info::device_type::gpu) {
      compute_contact_polygon_event =
          LaunchContactPolygonComputation<DeviceCollisionData, DeviceMeshData,
                                          DevicePolygonData, DeviceType::GPU>(
              q_device_, dependencies, total_narrow_phase_checks_,
              collision_data_, mesh_data_, polygon_data_);
    } else {
      compute_contact_polygon_event =
          LaunchContactPolygonComputation<DeviceCollisionData, DeviceMeshData,
                                          DevicePolygonData, DeviceType::CPU>(
              q_device_, dependencies, total_narrow_phase_checks_,
              collision_data_, mesh_data_, polygon_data_);
    }
    compute_contact_polygon_event.wait_and_throw();

    // Exclusive scan to compact data into only the valid polygons found by
    // SYCL
    oneapi::dpl::transform_exclusive_scan(
        policy, collision_data_.narrow_phase_check_validity,
        collision_data_.narrow_phase_check_validity +
            total_narrow_phase_checks_,
        collision_data_.prefix_sum_narrow_phase_checks,  // output
        static_cast<size_t>(0),                          // initial value
        sycl::plus<size_t>(),                            // binary operation
        [](uint8_t x) {
          return static_cast<size_t>(x);
        });  // transform uint8_t to size_t
    q_device_.wait_and_throw();

    total_polygons_ = 0;
    q_device_
        .memcpy(&total_polygons_,
                collision_data_.prefix_sum_narrow_phase_checks +
                    total_narrow_phase_checks_ - 1,
                sizeof(size_t))
        .wait();
    // Last element check or not?
    last_check_flag = 0;
    q_device_
        .memcpy(&last_check_flag,
                collision_data_.narrow_phase_check_validity +
                    total_narrow_phase_checks_ - 1,
                sizeof(uint8_t))
        .wait();
    // If last check is 1, then we need to add one more check
    total_polygons_ += static_cast<size_t>(last_check_flag);

    if (total_polygons_ == 0) {
      return {};
    }

    if (total_polygons_ > current_polygon_indices_size_) {
      // Give a 10 % bigger size
      size_t new_size = static_cast<size_t>(1.1 * total_polygons_);

      // Free old memory
      sycl::free(polygon_data_.compacted_polygon_areas, q_device_);
      sycl::free(polygon_data_.compacted_polygon_centroids, q_device_);
      sycl::free(polygon_data_.compacted_polygon_normals, q_device_);
      sycl::free(polygon_data_.compacted_polygon_g_M, q_device_);
      sycl::free(polygon_data_.compacted_polygon_g_N, q_device_);
      sycl::free(polygon_data_.compacted_polygon_pressure_W, q_device_);
      sycl::free(polygon_data_.compacted_polygon_geom_index_A, q_device_);
      sycl::free(polygon_data_.compacted_polygon_geom_index_B, q_device_);
      sycl::free(polygon_data_.valid_polygon_indices, q_device_);

      // Allocate new memory with larger size
      polygon_data_.compacted_polygon_areas =
          sycl::malloc_device<double>(new_size, q_device_);
      polygon_data_.compacted_polygon_centroids =
          sycl::malloc_device<Vector3<double>>(new_size, q_device_);
      polygon_data_.compacted_polygon_normals =
          sycl::malloc_device<Vector3<double>>(new_size, q_device_);
      polygon_data_.compacted_polygon_g_M =
          sycl::malloc_device<double>(new_size, q_device_);
      polygon_data_.compacted_polygon_g_N =
          sycl::malloc_device<double>(new_size, q_device_);
      polygon_data_.compacted_polygon_pressure_W =
          sycl::malloc_device<double>(new_size, q_device_);
      polygon_data_.compacted_polygon_geom_index_A =
          sycl::malloc_device<GeometryId>(new_size, q_device_);
      polygon_data_.compacted_polygon_geom_index_B =
          sycl::malloc_device<GeometryId>(new_size, q_device_);
      polygon_data_.valid_polygon_indices =
          sycl::malloc_device<size_t>(new_size, q_device_);
      current_polygon_indices_size_ = new_size;
    }

    auto memset_event =
        q_device_.memset(polygon_data_.valid_polygon_indices, 0,
                         current_polygon_indices_size_ * sizeof(size_t));
    memset_event.wait_and_throw();
    auto fill_valid_polygon_indicesevent =
        q_device_.submit([&](sycl::handler& h) {
          h.depends_on(compute_contact_polygon_event);
          h.parallel_for(
              sycl::range<1>(total_narrow_phase_checks_),
              [=, valid_polygon_indices = polygon_data_.valid_polygon_indices,
               prefix_sum_narrow_phase_checks =
                   collision_data_.prefix_sum_narrow_phase_checks,
               narrow_phase_check_validity =
                   collision_data_.narrow_phase_check_validity](
                  sycl::id<1> idx) {
                const size_t check_index = idx[0];
                if (narrow_phase_check_validity[check_index] == 1) {
                  size_t valid_polygon_index =
                      prefix_sum_narrow_phase_checks[check_index];
                  valid_polygon_indices[valid_polygon_index] = check_index;
                }
              });
        });
    fill_valid_polygon_indicesevent.wait_and_throw();

    // Compact all the data to data only with valid polygons
    auto compact_event = q_device_.submit([&](sycl::handler& h) {
      h.depends_on({fill_valid_polygon_indicesevent});
      h.parallel_for(
          sycl::range<1>(total_polygons_),
          [=, compacted_polygon_areas = polygon_data_.compacted_polygon_areas,
           compacted_polygon_centroids =
               polygon_data_.compacted_polygon_centroids,
           compacted_polygon_normals = polygon_data_.compacted_polygon_normals,
           compacted_polygon_g_M = polygon_data_.compacted_polygon_g_M,
           compacted_polygon_g_N = polygon_data_.compacted_polygon_g_N,
           compacted_polygon_pressure_W =
               polygon_data_.compacted_polygon_pressure_W,
           compacted_polygon_geom_index_A =
               polygon_data_.compacted_polygon_geom_index_A,
           compacted_polygon_geom_index_B =
               polygon_data_.compacted_polygon_geom_index_B,
           valid_polygon_indices = polygon_data_.valid_polygon_indices,
           polygon_areas = polygon_data_.polygon_areas,
           polygon_centroids = polygon_data_.polygon_centroids,
           polygon_normals = polygon_data_.polygon_normals,
           polygon_g_M = polygon_data_.polygon_g_M,
           polygon_g_N = polygon_data_.polygon_g_N,
           polygon_pressure_W = polygon_data_.polygon_pressure_W,
           polygon_geom_index_A = polygon_data_.polygon_geom_index_A,
           polygon_geom_index_B =
               polygon_data_.polygon_geom_index_B](sycl::id<1> idx) {
            const size_t valid_polygon_index = idx[0];
            const size_t check_index =
                valid_polygon_indices[valid_polygon_index];
            compacted_polygon_areas[valid_polygon_index] =
                polygon_areas[check_index];
            compacted_polygon_centroids[valid_polygon_index] =
                polygon_centroids[check_index];
            compacted_polygon_normals[valid_polygon_index] =
                polygon_normals[check_index];
            compacted_polygon_g_M[valid_polygon_index] =
                polygon_g_M[check_index];
            compacted_polygon_g_N[valid_polygon_index] =
                polygon_g_N[check_index];
            compacted_polygon_pressure_W[valid_polygon_index] =
                polygon_pressure_W[check_index];
            compacted_polygon_geom_index_A[valid_polygon_index] =
                polygon_geom_index_A[check_index];
            compacted_polygon_geom_index_B[valid_polygon_index] =
                polygon_geom_index_B[check_index];
          });
    });

    compact_event.wait_and_throw();

    // For now return a vector
    return {CreateHydroelasticSurface(
        q_device_, polygon_data_.compacted_polygon_centroids,
        polygon_data_.compacted_polygon_areas,
        polygon_data_.compacted_polygon_pressure_W,
        polygon_data_.compacted_polygon_normals,
        polygon_data_.compacted_polygon_g_M,
        polygon_data_.compacted_polygon_g_N,
        polygon_data_.compacted_polygon_geom_index_A,
        polygon_data_.compacted_polygon_geom_index_B, total_polygons_)};
  }

 private:
  // Helper method to initialize SYCL queue
  static sycl::queue InitializeQueue() {
    try {
      sycl::queue q(sycl::gpu_selector_v);
      std::cout << "Using "
                << q.get_device().get_info<sycl::info::device::name>()
                << std::endl;
      return q;
    } catch (sycl::exception const& e) {
      std::cout << "Cannot select a GPU\n" << e.what() << std::endl;
      std::cout << "Using a CPU device" << std::endl;
      sycl::queue q(sycl::cpu_selector_v);
      std::cout << "Using "
                << q.get_device().get_info<sycl::info::device::name>()
                << std::endl;
      return q;
    }
  }

  friend class SyclProximityEngineTester;
  // We have a CPU queue for operations beneficial to perform on the host and a
  // device queue for operations beneficial to perform on the Accelerator.
  // Note: q_device_ HAS TO BE declared before mem_mgr_ since it needs to be
  // initialized first.
  sycl::queue q_device_;

  SyclMemoryManager mem_mgr_;
  DeviceMeshData mesh_data_;
  DeviceCollisionData collision_data_;
  DevicePolygonData polygon_data_;

  // The collision candidates.
  std::vector<SortedPair<GeometryId>> collision_candidates_;

  // Number of geometries
  size_t num_geometries_ = 0;

  size_t total_vertices_ = 0;
  size_t total_elements_ = 0;

  size_t current_polygon_areas_size_ =
      0;  // Current size of polygon_areas to prevent constant reallocation

  size_t current_narrow_phase_check_indices_size_ =
      0;  // Current size of narrow_phase_check_indices to prevent constant
          // reallocation

  size_t current_polygon_indices_size_ =
      0;  // Current size of valid_polygon_indices to prevent constant
          // reallocation

  size_t current_debug_polygon_vertices_size_ = 0;

  // Some helpers for broad phase collision detection
  size_t num_elements_in_last_geometry_ = 0;
  size_t total_checks_ = 0;
  size_t estimated_narrow_phase_checks_ =
      0;  // Estimated number of narrow phase checks (set to be 5% of total
          // element checks and used to size polygon_areas and
          // polygon_centroids)
  size_t total_narrow_phase_checks_ =
      0;  // Total number of narrow phase checks in the current time step
          // (updated in ComputeSYCLHydroelasticSurface)

  size_t total_polygons_ = 0;  // Total number of valid polygons found by SYCL
  size_t estimated_polygons_ = 0;  // Estimated number of polygons (set to be 1%
                                   // of the narrow phase checks)

  friend class SyclProximityEngineAttorney;
};

bool SyclProximityEngine::is_available() {
  return Impl::is_available();
}

SyclProximityEngine::SyclProximityEngine(
    const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>&
        soft_geometries)
    : impl_(std::make_unique<Impl>(soft_geometries)) {}

SyclProximityEngine::SyclProximityEngine() : impl_(std::make_unique<Impl>()) {}

SyclProximityEngine::~SyclProximityEngine() = default;

SyclProximityEngine::SyclProximityEngine(const SyclProximityEngine& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

SyclProximityEngine& SyclProximityEngine::operator=(
    const SyclProximityEngine& other) {
  if (this != &other) {
    *impl_ = *other.impl_;
  }
  return *this;
}

void SyclProximityEngine::UpdateCollisionCandidates(
    const std::vector<SortedPair<GeometryId>>& collision_candidates) {
  impl_->UpdateCollisionCandidates(collision_candidates);
}

std::vector<SYCLHydroelasticSurface>
SyclProximityEngine::ComputeSYCLHydroelasticSurface(
    const std::unordered_map<GeometryId, math::RigidTransform<double>>& X_WGs) {
  return impl_->ComputeSYCLHydroelasticSurface(X_WGs);
}

// SyclProximityEngineAttorney class definition
SyclProximityEngine::Impl* SyclProximityEngineAttorney::get_impl(
    SyclProximityEngine& engine) {
  return engine.impl_.get();
}
const SyclProximityEngine::Impl* SyclProximityEngineAttorney::get_impl(
    const SyclProximityEngine& engine) {
  return engine.impl_.get();
}
std::vector<uint8_t> SyclProximityEngineAttorney::get_collision_filter(
    SyclProximityEngine::Impl* impl) {
  size_t total_checks = SyclProximityEngineAttorney::get_total_checks(impl);
  std::vector<uint8_t> collision_filterhost(total_checks);
  auto q = impl->q_device_;
  auto collision_filter = impl->collision_data_.collision_filter;
  q.memcpy(collision_filterhost.data(), collision_filter,
           total_checks * sizeof(uint8_t))
      .wait();
  return collision_filterhost;
}

std::vector<size_t> SyclProximityEngineAttorney::get_prefix_sum(
    SyclProximityEngine::Impl* impl) {
  size_t total_checks = SyclProximityEngineAttorney::get_total_checks(impl);
  std::vector<size_t> prefix_sum_total_checkshost(total_checks);
  auto q = impl->q_device_;
  auto prefix_sum = impl->collision_data_.prefix_sum_total_checks;
  q.memcpy(prefix_sum_total_checkshost.data(), prefix_sum,
           total_checks * sizeof(size_t))
      .wait();
  return prefix_sum_total_checkshost;
}

std::vector<Vector3<double>> SyclProximityEngineAttorney::get_vertices_M(
    SyclProximityEngine::Impl* impl) {
  auto q = impl->q_device_;
  auto vertices_M = impl->mesh_data_.vertices_M;
  auto total_vertices = impl->total_vertices_;
  std::vector<Vector3<double>> vertices_Mhost(total_vertices);
  q.memcpy(vertices_Mhost.data(), vertices_M,
           total_vertices * sizeof(Vector3<double>))
      .wait();
  return vertices_Mhost;
}
std::vector<Vector3<double>> SyclProximityEngineAttorney::get_vertices_W(
    SyclProximityEngine::Impl* impl) {
  auto q = impl->q_device_;
  auto vertices_W = impl->mesh_data_.vertices_W;
  auto total_vertices = impl->total_vertices_;
  std::vector<Vector3<double>> vertices_Whost(total_vertices);
  q.memcpy(vertices_Whost.data(), vertices_W,
           total_vertices * sizeof(Vector3<double>))
      .wait();
  return vertices_Whost;
}
std::vector<std::array<int, 4>> SyclProximityEngineAttorney::get_elements(
    SyclProximityEngine::Impl* impl) {
  auto q = impl->q_device_;
  auto elements = impl->mesh_data_.elements;
  auto total_elements_ = impl->total_elements_;
  std::vector<std::array<int, 4>> elementshost(total_elements_);
  q.memcpy(elementshost.data(), elements,
           total_elements_ * sizeof(std::array<int, 4>))
      .wait();
  return elementshost;
}
double* SyclProximityEngineAttorney::get_pressures(
    SyclProximityEngine::Impl* impl) {
  return impl->mesh_data_.pressures;
}
Vector4<double>* SyclProximityEngineAttorney::get_gradient_M_pressure_at_Mo(
    SyclProximityEngine::Impl* impl) {
  return impl->mesh_data_.gradient_M_pressure_at_Mo;
}
Vector4<double>* SyclProximityEngineAttorney::get_gradient_W_pressure_at_Wo(
    SyclProximityEngine::Impl* impl) {
  return impl->mesh_data_.gradient_W_pressure_at_Wo;
}
size_t* SyclProximityEngineAttorney::get_collision_filter_host_body_index(
    SyclProximityEngine::Impl* impl) {
  return impl->collision_data_.collision_filter_host_body_index;
}
size_t SyclProximityEngineAttorney::get_total_checks(
    SyclProximityEngine::Impl* impl) {
  return impl->total_checks_;
}

size_t SyclProximityEngineAttorney::get_total_narrow_phase_checks(
    SyclProximityEngine::Impl* impl) {
  return impl->total_narrow_phase_checks_;
}

size_t SyclProximityEngineAttorney::get_total_polygons(
    SyclProximityEngine::Impl* impl) {
  return impl->total_polygons_;
}

std::vector<size_t> SyclProximityEngineAttorney::get_narrow_phase_check_indices(
    SyclProximityEngine::Impl* impl) {
  size_t total_narrow_phase_checks =
      SyclProximityEngineAttorney::get_total_narrow_phase_checks(impl);
  std::vector<size_t> narrow_phase_check_indiceshost(total_narrow_phase_checks);
  auto q = impl->q_device_;
  auto narrow_phase_check_indices =
      impl->collision_data_.narrow_phase_check_indices;
  q.memcpy(narrow_phase_check_indiceshost.data(), narrow_phase_check_indices,
           total_narrow_phase_checks * sizeof(size_t))
      .wait();
  return narrow_phase_check_indiceshost;
}

std::vector<size_t> SyclProximityEngineAttorney::get_valid_polygon_indices(
    SyclProximityEngine::Impl* impl) {
  size_t total_polygons = SyclProximityEngineAttorney::get_total_polygons(impl);
  std::vector<size_t> valid_polygon_indiceshost(total_polygons);
  auto q = impl->q_device_;
  auto valid_polygon_indices = impl->polygon_data_.valid_polygon_indices;
  q.memcpy(valid_polygon_indiceshost.data(), valid_polygon_indices,
           total_polygons * sizeof(size_t))
      .wait();
  return valid_polygon_indiceshost;
}

std::vector<double> SyclProximityEngineAttorney::get_polygon_areas(
    SyclProximityEngine::Impl* impl) {
  size_t total_narrow_phase_checks =
      SyclProximityEngineAttorney::get_total_narrow_phase_checks(impl);
  std::vector<double> polygon_areashost(total_narrow_phase_checks);
  auto q = impl->q_device_;
  auto polygon_areas = impl->polygon_data_.polygon_areas;
  q.memcpy(polygon_areashost.data(), polygon_areas,
           total_narrow_phase_checks * sizeof(double))
      .wait();
  return polygon_areashost;
}

std::vector<Vector3<double>> SyclProximityEngineAttorney::get_polygon_centroids(
    SyclProximityEngine::Impl* impl) {
  size_t total_narrow_phase_checks =
      SyclProximityEngineAttorney::get_total_narrow_phase_checks(impl);
  std::vector<Vector3<double>> polygon_centroidshost(total_narrow_phase_checks);
  auto q = impl->q_device_;
  auto polygon_centroids = impl->polygon_data_.polygon_centroids;
  q.memcpy(polygon_centroidshost.data(), polygon_centroids,
           total_narrow_phase_checks * sizeof(Vector3<double>))
      .wait();
  return polygon_centroidshost;
}

std::vector<double> SyclProximityEngineAttorney::get_debug_polygon_vertices(
    SyclProximityEngine::Impl* impl) {
  std::vector<double> debug_polygon_vertices_host(
      impl->current_debug_polygon_vertices_size_);
  auto q = impl->q_device_;
  auto debug_polygon_vertices = impl->polygon_data_.debug_polygon_vertices;
  q.memcpy(debug_polygon_vertices_host.data(), debug_polygon_vertices,
           impl->current_debug_polygon_vertices_size_ * sizeof(double))
      .wait();
  return debug_polygon_vertices_host;
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
