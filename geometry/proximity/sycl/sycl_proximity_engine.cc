#include "drake/geometry/proximity/sycl/sycl_proximity_engine.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
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

#include "drake/common/problem_size_logger.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"
#include "drake/geometry/proximity/sycl/utils/sycl_contact_polygon.h"
#include "drake/geometry/proximity/sycl/utils/sycl_equilibrium_plane.h"
#include "drake/geometry/proximity/sycl/utils/sycl_hydroelastic_surface_creator.h"
#include "drake/geometry/proximity/sycl/utils/sycl_memory_manager.h"
#include "drake/geometry/proximity/sycl/utils/sycl_naive_broad_phase.h"
#include "drake/geometry/proximity/sycl/utils/sycl_tetrahedron_slicing.h"
#include "drake/geometry/proximity/sycl/utils/sycl_timing_logger.h"
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

// Forward declarations for kernel names
class TransformVerticesKernel;
class TransformInwardNormalsKernel;
class TransformPressureGradientsKernel;
class ComputeElementAABBKernel;
class GenerateCollisionFilterKernel;
class FillNarrowPhaseCheckIndicesKernel;
class FillValidPolygonIndicesKernel;
class CompactPolygonDataKernel;

// Implementation class for SyclProximityEngine that contains all SYCL-specific
// code
class SyclProximityEngine::Impl {
 public:
  // Default constructor
  Impl()
      : q_device_(InitializeQueue()), mem_mgr_(q_device_), timing_logger_() {}

  // Constructor that initializes with soft geometries
  Impl(const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>&
           soft_geometries)
      : q_device_(InitializeQueue()), mem_mgr_(q_device_), timing_logger_() {
    DRAKE_THROW_UNLESS(soft_geometries.size() > 0);

    // #ifdef DRAKE_SYCL_TIMING_ENABLED
    //     timing_logger_.SetEnabled(true);
    // #endif

    // Extract and sort geometry IDs for deterministic ordering
    std::vector<GeometryId> sorted_ids;
    sorted_ids.reserve(soft_geometries.size());
    for (const auto& [id, _] : soft_geometries) {
      sorted_ids.push_back(id);
    }
    std::sort(sorted_ids.begin(), sorted_ids.end());

    // Get number of geometries
    num_geometries_ = soft_geometries.size();

    SyclMemoryHelper::AllocateMeshMemory(mem_mgr_, mesh_data_, num_geometries_);
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

    SyclMemoryHelper::AllocateMeshElementVerticesMemory(
        mem_mgr_, mesh_data_, total_elements_, total_vertices_);

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
        q_device_
            .memcpy(mesh_data_.elements + element_offset + i, &vertices,
                    sizeof(std::array<int, 4>))
            .wait();
      }

      q_device_
          .fill(mesh_data_.element_mesh_ids + element_offset, id_index,
                num_elements)
          .wait();

      // Vertices
      transfer_events.push_back(q_device_.memcpy(
          mesh_data_.vertices_M + vertex_offset, mesh.vertices().data(),
          num_vertices * sizeof(Vector3<double>)));

      // Pressures
      transfer_events.push_back(q_device_.memcpy(
          mesh_data_.pressures + vertex_offset, pressure_field.values().data(),
          num_vertices * sizeof(double)));

      // Fill in the mesh id for all vertices in this mesh
      q_device_
          .fill(mesh_data_.vertex_mesh_ids + vertex_offset, id_index,
                num_vertices)
          .wait();

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

      q_device_
          .memcpy(mesh_data_.gradient_M_pressure_at_Mo + element_offset,
                  packed_gradient_pressure.data(),
                  num_elements * sizeof(Vector4<double>))
          .wait();
    }

    // ========================================
    // Some pre-processing for broad phase collision detection
    // ========================================
    SyclMemoryHelper::AllocateGeometryCollisionMemory(mem_mgr_, collision_data_,
                                                      num_geometries_);

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
    // Resize compacted data structures based on the estimated polygon sizes
    current_polygon_indices_size_ = estimated_polygons_;

    SyclMemoryHelper::AllocateTotalChecksCollisionMemory(
        mem_mgr_, collision_data_, total_checks_);
    SyclMemoryHelper::AllocateNarrowPhaseChecksCollisionMemory(
        mem_mgr_, collision_data_, estimated_narrow_phase_checks_);

    SyclMemoryHelper::AllocateFullPolygonMemory(mem_mgr_, polygon_data_,
                                                estimated_narrow_phase_checks_);

    SyclMemoryHelper::AllocateCompactPolygonMemory(mem_mgr_, polygon_data_,
                                                   estimated_polygons_);

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
      SyclMemoryHelper::FreeMeshMemory(mem_mgr_, mesh_data_);
      SyclMemoryHelper::FreeCollisionMemory(mem_mgr_, collision_data_);
      SyclMemoryHelper::FreeFullPolygonMemory(mem_mgr_, polygon_data_);
      SyclMemoryHelper::FreeCompactPolygonMemory(mem_mgr_, polygon_data_);
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
    // // Performance analysis: Static counter for time steps
    // static size_t time_step_counter = 0;
    // ++time_step_counter;

    if (total_checks_ == 0) {
      return {};
    }
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.StartKernel("unpack_transforms");
#endif
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
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.EndKernel("unpack_transforms");
    timing_logger_.StartKernel("transform_and_broad_phase");
#endif
    // ========================================
    // Command group 1: Transform quantities to world frame
    // ========================================

    // Combine all transformation kernels into a single command group
    auto transform_vertices_event = q_device_.submit([&](sycl::handler& h) {
      // Transform vertices
      const size_t work_group_size = 64;
      const size_t global_vertices =
          RoundUpToWorkGroupSize(total_vertices_, work_group_size);
      h.parallel_for<TransformVerticesKernel>(
          sycl::nd_range<1>(sycl::range<1>(global_vertices),
                            sycl::range<1>(work_group_size)),
          [=, vertices_M = mesh_data_.vertices_M,
           vertices_W = mesh_data_.vertices_W,
           vertex_mesh_ids = mesh_data_.vertex_mesh_ids,
           transforms = mesh_data_.transforms,
           total_vertices_ = total_vertices_]
#ifdef __NVPTX__
          [[sycl::reqd_work_group_size(64)]]
#endif
          (sycl::nd_item<1> item) {
            const size_t vertex_index = item.get_global_id(0);
            if (vertex_index >= total_vertices_) return;

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
          const size_t work_group_size = 256;
          const size_t global_elements =
              RoundUpToWorkGroupSize(total_elements_, work_group_size);
          h.parallel_for<TransformInwardNormalsKernel>(
              sycl::nd_range<1>(sycl::range<1>(global_elements),
                                sycl::range<1>(work_group_size)),
              [=, inward_normals_M = mesh_data_.inward_normals_M,
               inward_normals_W = mesh_data_.inward_normals_W,
               element_mesh_ids = mesh_data_.element_mesh_ids,
               transforms = mesh_data_.transforms,
               total_elements_ = total_elements_]
#ifdef __NVPTX__
              [[sycl::reqd_work_group_size(256)]]
#endif
              (sycl::nd_item<1> item) {
                const size_t element_index = item.get_global_id(0);
                if (element_index >= total_elements_) return;

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
          const size_t work_group_size = 256;
          const size_t global_elements =
              RoundUpToWorkGroupSize(total_elements_, work_group_size);
          h.parallel_for<TransformPressureGradientsKernel>(
              sycl::nd_range<1>(sycl::range<1>(global_elements),
                                sycl::range<1>(work_group_size)),
              [=,
               gradient_M_pressure_at_Mo = mesh_data_.gradient_M_pressure_at_Mo,
               gradient_W_pressure_at_Wo = mesh_data_.gradient_W_pressure_at_Wo,
               element_mesh_ids = mesh_data_.element_mesh_ids,
               transforms = mesh_data_.transforms,
               total_elements_ = total_elements_]
#ifdef __NVPTX__
              [[sycl::reqd_work_group_size(256)]]
#endif
              (sycl::nd_item<1> item) {
                const size_t element_index = item.get_global_id(0);
                if (element_index >= total_elements_) return;

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
    // Command group 2: Generate candidate tet pairs using NaiveBroadPhase
    // =========================================
    auto [element_aabb_event, generate_collision_filterevent] = NaiveBroadPhase(
        q_device_, mesh_data_, collision_data_, total_elements_, total_checks_,
        transform_vertices_event, collision_filtermemset_event);
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

#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.EndKernel("transform_and_broad_phase");
#endif
    if (total_narrow_phase_checks_ == 0) {
      return {};
    }
    drake::common::ProblemSizeLogger::GetInstance().AddCount(
        "SYCLCandidateTets", total_narrow_phase_checks_);

    if (total_narrow_phase_checks_ > current_polygon_areas_size_) {
      // Give a 10 % bigger size
      size_t new_size = static_cast<size_t>(1.1 * total_narrow_phase_checks_);

      // Free old memory
      SyclMemoryHelper::FreeFullPolygonMemory(mem_mgr_, polygon_data_);
      SyclMemoryHelper::FreeNarrowPhaseChecksCollisionMemory(mem_mgr_,
                                                             collision_data_);

      // Allocate new memory with larger size
      SyclMemoryHelper::AllocateFullPolygonMemory(mem_mgr_, polygon_data_,
                                                  new_size);
      SyclMemoryHelper::AllocateNarrowPhaseChecksCollisionMemory(
          mem_mgr_, collision_data_, new_size);

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
          const size_t work_group_size = 1024;
          const size_t global_checks =
              RoundUpToWorkGroupSize(total_checks_, work_group_size);
          h.parallel_for<FillNarrowPhaseCheckIndicesKernel>(
              sycl::nd_range<1>(sycl::range<1>(global_checks),
                                sycl::range<1>(work_group_size)),
              [=,
               narrow_phase_check_indices =
                   collision_data_.narrow_phase_check_indices,
               prefix_sum_total_checks =
                   collision_data_.prefix_sum_total_checks,
               collision_filter = collision_data_.collision_filter,
               total_checks_ = total_checks_]
#ifdef __NVPTX__
              [[sycl::reqd_work_group_size(1024)]]
#endif
              (sycl::nd_item<1> item) {
                const size_t check_index = item.get_global_id(0);
                if (check_index >= total_checks_) return;
                if (collision_filter[check_index] == 1) {
                  size_t narrow_check_num =
                      prefix_sum_total_checks[check_index];
                  narrow_phase_check_indices[narrow_check_num] = check_index;
                }
              });
        });

    // Create dependency vector
    std::vector<sycl::event> dependencies = {
        generate_collision_filterevent, fill_narrow_phase_check_indicesevent,
        transform_elem_quantities_event1, transform_elem_quantities_event2};
    // Add polygon fill events to dependencies
    dependencies.insert(dependencies.end(), fill_events.begin(),
                        fill_events.end());
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.StartKernel("compute_contact_polygons");
#endif
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
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.EndKernel("compute_contact_polygons");
#endif
    if (total_polygons_ == 0) {
      return {};
    }
    drake::common::ProblemSizeLogger::GetInstance().AddCount("SYCFacesInserted",
                                                             total_polygons_);
#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.StartKernel("compact_polygon_data");
#endif
    if (total_polygons_ > current_polygon_indices_size_) {
      // Give a 10 % bigger size
      size_t new_size = static_cast<size_t>(1.1 * total_polygons_);

      SyclMemoryHelper::FreeCompactPolygonMemory(mem_mgr_, polygon_data_);

      // Allocate new memory with larger size
      SyclMemoryHelper::AllocateCompactPolygonMemory(mem_mgr_, polygon_data_,
                                                     new_size);
      current_polygon_indices_size_ = new_size;
    }

    auto memset_event =
        q_device_.memset(polygon_data_.valid_polygon_indices, 0,
                         current_polygon_indices_size_ * sizeof(size_t));
    memset_event.wait_and_throw();
    auto fill_valid_polygon_indicesevent =
        q_device_.submit([&](sycl::handler& h) {
          h.depends_on(compute_contact_polygon_event);
          h.parallel_for<FillValidPolygonIndicesKernel>(
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
      h.parallel_for<CompactPolygonDataKernel>(
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

#ifdef DRAKE_SYCL_TIMING_ENABLED
    timing_logger_.EndKernel("compact_polygon_data");
#endif

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
  }  // namespace sycl_impl

 private:
  // Helper method to initialize SYCL queue
  static sycl::queue InitializeQueue() {
    try {
#ifdef DRAKE_SYCL_TIMING_ENABLED
      //   sycl::queue q(sycl::gpu_selector_v,
      //                 sycl::property::queue::enable_profiling());
      sycl::queue q(sycl::gpu_selector_v);
#else
      sycl::queue q(sycl::gpu_selector_v);
#endif
      std::cout << "Using "
                << q.get_device().get_info<sycl::info::device::name>()
                << std::endl;
      return q;
    } catch (sycl::exception const& e) {
      std::cout << "Cannot select a GPU\n" << e.what() << std::endl;
      std::cout << "Using a CPU device" << std::endl;
#ifdef DRAKE_SYCL_TIMING_ENABLED
      //   sycl::queue q(sycl::cpu_selector_v,
      //                 sycl::property::queue::enable_profiling());
      sycl::queue q(sycl::cpu_selector_v);
#else
      sycl::queue q(sycl::cpu_selector_v);
#endif
      std::cout << "Using "
                << q.get_device().get_info<sycl::info::device::name>()
                << std::endl;
      return q;
    }
  }

  friend class SyclProximityEngineTester;
  // We have a CPU queue for operations beneficial to perform on the host
  // and a device queue for operations beneficial to perform on the
  // Accelerator. Note: q_device_ HAS TO BE declared before mem_mgr_ since
  // it needs to be initialized first.
  sycl::queue q_device_;

  SyclMemoryManager mem_mgr_;
  DeviceMeshData mesh_data_;
  DeviceCollisionData collision_data_;
  DevicePolygonData polygon_data_;

  // Timing logger for kernel performance analysis
  SyclTimingLogger timing_logger_;

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
  size_t estimated_polygons_ = 0;  // Estimated number of polygons (set to
                                   // be 1% of the narrow phase checks)

  friend class SyclProximityEngineAttorney;
};  // namespace sycl_impl

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

void SyclProximityEngine::PrintTimingStats() const {
#ifdef DRAKE_SYCL_TIMING_ENABLED
  SyclProximityEngineAttorney::PrintTimingStats(impl_.get());
#endif
}

void SyclProximityEngine::PrintTimingStatsJson(const std::string& path) const {
#ifdef DRAKE_SYCL_TIMING_ENABLED
  SyclProximityEngineAttorney::PrintTimingStatsJson(impl_.get(), path);
#endif
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

void SyclProximityEngineAttorney::PrintTimingStats(
    SyclProximityEngine::Impl* impl) {
#ifdef DRAKE_SYCL_TIMING_ENABLED
  impl->timing_logger_.PrintStats();
#endif
}

void SyclProximityEngineAttorney::PrintTimingStatsJson(
    SyclProximityEngine::Impl* impl, const std::string& path) {
#ifdef DRAKE_SYCL_TIMING_ENABLED
  impl->timing_logger_.PrintStatsJson(path);
#endif
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
