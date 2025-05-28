#pragma once

#include <array>

#include <sycl/sycl.hpp>

#include "drake/geometry/proximity/sycl/utils/sycl_memory_manager.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Forward declarations for kernel names
class ComputeElementAABBKernel;
class GenerateCollisionFilterKernel;

// Helper function to round up to nearest multiple of work group size
SYCL_EXTERNAL inline size_t RoundUpToWorkGroupSize(size_t n,
                                                   size_t work_group_size) {
  return ((n + work_group_size - 1) / work_group_size) * work_group_size;
}

/**
 * @brief Performs naive broad phase collision detection using AABB and pressure
 * field intersection
 *
 * This function implements a naive broad phase collision detection algorithm
 * that:
 * 1. Computes AABBs for all tetrahedral elements
 * 2. Generates collision filter based on AABB overlap and pressure field
 * intersection
 *
 * @param q_device SYCL queue for device execution
 * @param mesh_data Device mesh data containing vertices, elements, and pressure
 * information
 * @param collision_data Device collision data for storing collision filter
 * results
 * @param total_elements Total number of tetrahedral elements across all
 * geometries
 * @param total_checks Total number of collision checks to perform
 * @param transform_vertices_event Event to wait for before starting AABB
 * computation
 * @param collision_filter_memset_event Event to wait for before starting
 * collision filter generation
 * @return std::pair<sycl::event, sycl::event> Pair of events for AABB
 * computation and collision filter generation
 */
inline std::pair<sycl::event, sycl::event> NaiveBroadPhase(
    sycl::queue& q_device, const DeviceMeshData& mesh_data,
    const DeviceCollisionData& collision_data, size_t total_elements,
    size_t total_checks, sycl::event transform_vertices_event,
    sycl::event collision_filter_memset_event) {
  // =========================================
  // Command group 1: Compute element AABBs
  // =========================================
  auto element_aabb_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on(transform_vertices_event);

    // Compute all AABBs irrespective if they are needed or not
    // Allocate device memory for element AABBs
    // While doing this, assign false to all elements that are not part of
    // geometries that are collision candidates
    const size_t work_group_size = 256;
    const size_t global_elements =
        RoundUpToWorkGroupSize(total_elements, work_group_size);

    h.parallel_for<ComputeElementAABBKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, elements = mesh_data.elements, vertices_W = mesh_data.vertices_W,
         element_mesh_ids = mesh_data.element_mesh_ids,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         vertex_offsets = mesh_data.vertex_offsets,
         total_elements_ = total_elements]

#ifdef __NVPTX__
        [[sycl::reqd_work_group_size(256)]]
#endif
        (sycl::nd_item<1> item) {
          const size_t element_index = item.get_global_id(0);
          if (element_index >= total_elements_) return;

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

  // =========================================
  // Command group 2: Generate collision filter with the AABBs that we computed
  // =========================================
  auto generate_collision_filter_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on({element_aabb_event, collision_filter_memset_event});

    const size_t work_group_size = 1024;
    const size_t global_checks =
        RoundUpToWorkGroupSize(total_checks, work_group_size);

    h.parallel_for<GenerateCollisionFilterKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_checks),
                          sycl::range<1>(work_group_size)),
        [=, collision_filter = collision_data.collision_filter,
         collision_filter_host_body_index =
             collision_data.collision_filter_host_body_index,
         geom_collision_filter_check_offsets =
             collision_data.geom_collision_filter_check_offsets,
         geom_collision_filter_num_cols =
             collision_data.geom_collision_filter_num_cols,
         element_offsets = mesh_data.element_offsets,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         min_pressures = mesh_data.min_pressures,
         max_pressures = mesh_data.max_pressures, total_checks_ = total_checks]

#ifdef __NVPTX__
        [[sycl::reqd_work_group_size(1024)]]
#endif
        (sycl::nd_item<1> item) {
          const size_t check_index = item.get_global_id(0);
          if (check_index >= total_checks_) return;

          const size_t host_body_index =
              collision_filter_host_body_index[check_index];

          // What elements is this check_index checking?
          // host_body_index is the geometry index that element A belongs to
          size_t num_of_checks_offset =
              geom_collision_filter_check_offsets[host_body_index];
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
          if (max_pressures[B_element_index] < min_pressures[A_element_index] ||
              max_pressures[A_element_index] < min_pressures[B_element_index]) {
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

  return {element_aabb_event, generate_collision_filter_event};
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake