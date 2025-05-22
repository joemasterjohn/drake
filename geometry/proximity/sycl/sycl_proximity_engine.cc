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

#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/sycl/sycl_hydroelastic_surface.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Implementation class for SyclProximityEngine that contains all SYCL-specific
// code
class SyclProximityEngine::Impl {
 public:
  // Default constructor
  Impl() = default;

  // Constructor that initializes with soft geometries
  Impl(const std::unordered_map<GeometryId, hydroelastic::SoftGeometry>&
           soft_geometries) {
    // Initialize SYCL queues
    q_device_ = sycl::queue(sycl::gpu_selector_v);
    q_host_ = sycl::queue(sycl::cpu_selector_v);

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
    soft_geometry_ids_ = new GeometryId[num_geometries_];

    // Allocate device memory for lookup arrays
    sh_element_offsets_ = sycl::malloc_host<size_t>(num_geometries_, q_device_);
    sh_vertex_offsets_ = sycl::malloc_host<size_t>(num_geometries_, q_device_);
    sh_element_counts_ = sycl::malloc_host<size_t>(num_geometries_, q_device_);
    sh_vertex_counts_ = sycl::malloc_host<size_t>(num_geometries_, q_device_);

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
      soft_geometry_ids_[id_index] = id;

      // Store offsets and counts directly (no memcpy needed with shared memory)
      sh_element_offsets_[id_index] = total_elements_;
      sh_vertex_offsets_[id_index] = total_vertices_;

      const size_t num_elements = mesh.num_elements();
      const size_t num_vertices = mesh.num_vertices();
      sh_element_counts_[id_index] = num_elements;
      sh_vertex_counts_[id_index] = num_vertices;

      // Update totals
      total_elements_ += num_elements;
      total_vertices_ += num_vertices;
    }

    // Allocate device memory for all meshes
    elements_ =
        sycl::malloc_device<std::array<int, 4>>(total_elements_, q_device_);
    sh_element_mesh_ids_ =
        sycl::malloc_device<size_t>(total_elements_, q_device_);

    vertices_M_ =
        sycl::malloc_device<Vector3<double>>(total_vertices_, q_device_);
    sh_vertex_mesh_ids_ =
        sycl::malloc_device<size_t>(total_vertices_, q_device_);

    inward_normals_M_ = sycl::malloc_device<std::array<Vector3<double>, 4>>(
        total_elements_, q_device_);
    edge_vectors_M_ = sycl::malloc_device<std::array<Vector3<double>, 6>>(
        total_elements_, q_device_);
    pressures_ = sycl::malloc_device<double>(total_elements_, q_device_);
    min_pressures_ = sycl::malloc_device<double>(total_elements_, q_device_);
    max_pressures_ = sycl::malloc_device<double>(total_elements_, q_device_);

    // Allocate combined gradient and pressure arrays
    gradient_M_pressure_at_Mo_ =
        sycl::malloc_device<Vector4<double>>(total_elements_, q_device_);

    // Allocate even for world frame quantities
    vertices_W_ =
        sycl::malloc_device<Vector3<double>>(total_vertices_, q_device_);
    inward_normals_W_ = sycl::malloc_device<std::array<Vector3<double>, 4>>(
        total_elements_, q_device_);
    edge_vectors_W_ = sycl::malloc_device<std::array<Vector3<double>, 6>>(
        total_elements_, q_device_);
    gradient_W_pressure_at_Wo_ =
        sycl::malloc_device<Vector4<double>>(total_elements_, q_device_);

    // Allocate device memory for transforms
    transforms_ = sycl::malloc_host<double>(num_geometries_ * 12, q_device_);

    // Allocate device memory for element AABBs
    element_aabb_min_W_ =
        sycl::malloc_device<Vector3<double>>(total_elements_, q_device_);
    element_aabb_max_W_ =
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
      size_t element_offset = sh_element_offsets_[id_index];
      size_t vertex_offset = sh_vertex_offsets_[id_index];
      size_t num_elements = sh_element_counts_[id_index];
      size_t num_vertices = sh_vertex_counts_[id_index];

      // Copy mesh data using the offsets with async operations
      transfer_events.push_back(q_device_.memcpy(
          elements_ + element_offset, mesh.pack_element_vertices().data(),
          num_elements * sizeof(std::array<int, 4>)));

      // Fill in the mesh id for all elements in this mesh
      transfer_events.push_back(q_device_.fill(
          sh_element_mesh_ids_ + element_offset, id_index, num_elements));

      // Vertices
      transfer_events.push_back(
          q_device_.memcpy(vertices_M_ + vertex_offset, mesh.vertices().data(),
                           num_vertices * sizeof(Vector3<double>)));

      // Fill in the mesh id for all vertices in this mesh
      transfer_events.push_back(q_device_.fill(
          sh_vertex_mesh_ids_ + vertex_offset, id_index, num_vertices));

      // Inward Normals
      transfer_events.push_back(q_device_.memcpy(
          inward_normals_M_ + element_offset, mesh.inward_normals().data(),
          num_elements * sizeof(std::array<Vector3<double>, 4>)));

      // Edge Vectors
      transfer_events.push_back(q_device_.memcpy(
          edge_vectors_M_ + element_offset, mesh.edge_vectors().data(),
          num_elements * sizeof(std::array<Vector3<double>, 6>)));

      // Pressures
      transfer_events.push_back(q_device_.memcpy(
          pressures_ + element_offset, pressure_field.values().data(),
          num_elements * sizeof(double)));

      // Min Pressures
      transfer_events.push_back(q_device_.memcpy(
          min_pressures_ + element_offset, pressure_field.min_values().data(),
          num_elements * sizeof(double)));

      // Max Pressures
      transfer_events.push_back(q_device_.memcpy(
          max_pressures_ + element_offset, pressure_field.max_values().data(),
          num_elements * sizeof(double)));

      // Create a temporary host buffer to pack gradient and pressure data
      std::vector<Vector4<double>> packed_gradient_pressure(num_elements);

      // Pack the gradient data (first 3 components) and pressure at Mo (4th
      // component)
      const auto& gradients = pressure_field.gradients();
      const auto& pressures_at_Mo = pressure_field.values_at_Mo();

      for (size_t i = 0; i < num_elements; ++i) {
        packed_gradient_pressure[i][0] = gradients[i][0];     // x component
        packed_gradient_pressure[i][1] = gradients[i][1];     // y component
        packed_gradient_pressure[i][2] = gradients[i][2];     // z component
        packed_gradient_pressure[i][3] = pressures_at_Mo[i];  // pressure at Mo
      }

      // Transfer the packed data in a single operation
      transfer_events.push_back(
          q_device_.memcpy(gradient_M_pressure_at_Mo_ + element_offset,
                           packed_gradient_pressure.data(),
                           num_elements * sizeof(Vector4<double>)));
    }

    // ========================================
    // Some pre-processing for broad phase collision detection
    // ========================================
    // Stores at i the number of checks needs to be for the ith geometry
    total_checks_per_geometry_ =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);

    // geom_collision_filter_num_cols[i] is the number of elements that need to
    // be checked with each of the elements of the ith geometry
    // Will be highest for 1st geometry and lowest for the last geometry (due to
    // symmetric nature of collision_filter - we are only consider upper
    // triangle)
    geom_collision_filter_num_cols_ =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);

    // Stores the exclusive scan of total checks per geometry
    geom_collision_filter_check_offsets_ =
        sycl::malloc_host<size_t>(num_geometries_, q_device_);

    num_elements_in_last_geometry_ = sh_element_counts_[num_geometries_ - 1];
    total_checks_ = 0;
    // Done entirely in CPU because its only looping over num_geometries
    for (size_t i = 0; i < num_geometries_; ++i) {
      const size_t num_elements_in_geometry = sh_element_counts_[i];
      // sh_element_offsets_ stores  [0, E_0, E_0 + E_1, E_0 + E_1 +
      // E_2, ...] where E_i is the number of elements in the i-th
      // geometry sh_element_offsets_[i] = E_i -> number of elements
      // in the i-th geometry Ex: 1st geometry has to do checks with
      // all other geometries except itself 2nd geometry has to do
      // checks with all other geometries except itself and the first
      // geometry (because A check B  = B check A)...
      const size_t num_elements_in_rest_of_geometries =
          sh_element_offsets_[num_geometries_ - 1 - i] +
          num_elements_in_last_geometry_ - num_elements_in_geometry;
      geom_collision_filter_num_cols_[i] = num_elements_in_rest_of_geometries;
      // We need to check each element in this geometry with each element in
      // the rest of the geometries
      total_checks_per_geometry_[i] =
          num_elements_in_rest_of_geometries * num_elements_in_geometry;
      geom_collision_filter_check_offsets_[i] = total_checks_;
      total_checks_ += total_checks_per_geometry_[i];
    }

    // Generate collision filter for all checks
    collision_filter_ = sycl::malloc_host<bool>(total_checks_, q_device_);
    // memset all to 0 for now (will be filled in when we have AABBs for each
    // element)
    auto collision_filter_memset_event =
        q_device_.memset(collision_filter_, 0, total_checks_ * sizeof(bool));

    collision_filter_host_body_index_ =
        sycl::malloc_host<size_t>(total_checks_, q_device_);

    // Fill in geometry index based on checks per geometry
    std::vector<sycl::event> collision_filter_host_body_index_fill_events;
    for (size_t i = 0; i < num_geometries_; ++i) {
      const size_t num_checks = total_checks_per_geometry_[i];
      collision_filter_host_body_index_fill_events.push_back(
          q_device_.fill(collision_filter_host_body_index_ +
                             geom_collision_filter_check_offsets_[i],
                         i, num_checks));
    }

    // Wait for all transfers to complete before returning
    sycl::event::wait_and_throw(transfer_events);
    collision_filter_memset_event.wait();
    sycl::event::wait_and_throw(collision_filter_host_body_index_fill_events);
  }

  // Copy constructor
  Impl(const Impl& other) {
    // TODO(huzaifa): Implement deep copy of SYCL resources
    // For now, we'll just create a shallow copy which isn't ideal
    q_device_ = other.q_device_;
    q_host_ = other.q_host_;
    collision_candidates_ = other.collision_candidates_;
    num_geometries_ = other.num_geometries_;
  }

  // Copy assignment operator
  Impl& operator=(const Impl& other) {
    if (this != &other) {
      // TODO(huzaifa): Implement deep copy of SYCL resources
      // For now, we'll just create a shallow copy which isn't ideal
      q_device_ = other.q_device_;
      q_host_ = other.q_host_;
      collision_candidates_ = other.collision_candidates_;
      num_geometries_ = other.num_geometries_;
    }
    return *this;
  }

  // Destructor
  ~Impl() {
    // Free device memory
    if (num_geometries_ > 0) {
      delete[] soft_geometry_ids_;

      sycl::free(sh_element_offsets_, q_device_);
      sycl::free(sh_vertex_offsets_, q_device_);
      sycl::free(sh_element_counts_, q_device_);
      sycl::free(sh_vertex_counts_, q_device_);
      sycl::free(sh_element_mesh_ids_, q_device_);
      sycl::free(sh_vertex_mesh_ids_, q_device_);

      sycl::free(elements_, q_device_);
      sycl::free(vertices_M_, q_device_);
      sycl::free(vertices_W_, q_device_);
      sycl::free(inward_normals_M_, q_device_);
      sycl::free(inward_normals_W_, q_device_);
      sycl::free(edge_vectors_M_, q_device_);
      sycl::free(edge_vectors_W_, q_device_);
      sycl::free(pressures_, q_device_);
      sycl::free(min_pressures_, q_device_);
      sycl::free(max_pressures_, q_device_);
      sycl::free(gradient_M_pressure_at_Mo_, q_device_);
      sycl::free(gradient_W_pressure_at_Wo_, q_device_);
      sycl::free(transforms_, q_device_);

      sycl::free(geom_collision_filter_num_cols_, q_device_);
      sycl::free(geom_collision_filter_check_offsets_, q_device_);
      sycl::free(collision_filter_, q_device_);
      sycl::free(total_checks_per_geometry_, q_device_);
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
    // Get transfomers in host
    for (size_t geom_index = 0; geom_index < num_geometries_; ++geom_index) {
      GeometryId geometry_id = soft_geometry_ids_[geom_index];
      // To maintain our orders of geometries we need to loop through the stored
      // geometry id's and query the X_WGs for that geometry id. Cannot iterate
      // over the unordered_map because it is not ordered
      const auto& X_WG = X_WGs.at(geometry_id);
      const auto& transform = X_WG.GetAsMatrix34();
      for (size_t i = 0; i < 12; ++i) {
        size_t row = i / 4;
        size_t col = i % 4;
        // Store transforms in row major order
        // transforms_ = [R_00, R_01, R_02, p_0, R_10, R_11, R_12, p_1, ...]
        transforms_[geom_index * 12 + i] = transform(row, col);
      }
    }

    // ========================================
    // Command group 1: Transform quantities to world frame
    // ========================================

    // Combine all transformation kernels into a single command group
    auto transform_vertices_event = q_device_.submit([&](sycl::handler& h) {
      // Transform vertices
      h.parallel_for(sycl::range<1>(total_vertices_), sycl::id<1>(0),
                     [=, vertices_M_ = vertices_M_, vertices_W_ = vertices_W_,
                      sh_vertex_mesh_ids_ = sh_vertex_mesh_ids_,
                      transforms_ = transforms_](sycl::id<1> idx) {
                       const size_t vertex_index = idx[0];
                       const size_t mesh_index =
                           sh_vertex_mesh_ids_[vertex_index];

                       const double x = vertices_M_[vertex_index][0];
                       const double y = vertices_M_[vertex_index][1];
                       const double z = vertices_M_[vertex_index][2];
                       double T[12];
                       for (size_t i = 0; i < 12; ++i) {
                         T[i] = transforms_[mesh_index * 12 + i];
                       }
                       double new_x = T[0] * x + T[1] * y + T[2] * z + T[3];
                       double new_y = T[4] * x + T[5] * y + T[6] * z + T[7];
                       double new_z = T[8] * x + T[9] * y + T[10] * z + T[11];

                       vertices_W_[vertex_index][0] = new_x;
                       vertices_W_[vertex_index][1] = new_y;
                       vertices_W_[vertex_index][2] = new_z;
                     });
    });

    // Transform inward normals
    auto transform_elem_quantities_event1 =
        q_device_.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::range<1>(total_elements_), sycl::id<1>(0),
              [=, inward_normals_M_ = inward_normals_M_,
               inward_normals_W_ = inward_normals_W_,
               sh_element_mesh_ids_ = sh_element_mesh_ids_,
               transforms_ = transforms_](sycl::id<1> idx) {
                const size_t element_index = idx[0];
                const size_t mesh_index = sh_element_mesh_ids_[element_index];

                double T[12];
                for (size_t i = 0; i < 12; ++i) {
                  T[i] = transforms_[mesh_index * 12 + i];
                }

                // Each element has 4 inward normals
                for (size_t j = 0; j < 4; ++j) {
                  const double nx = inward_normals_M_[element_index][j][0];
                  const double ny = inward_normals_M_[element_index][j][1];
                  const double nz = inward_normals_M_[element_index][j][2];

                  // Only rotation
                  inward_normals_W_[element_index][j][0] =
                      T[0] * nx + T[1] * ny + T[2] * nz;
                  inward_normals_W_[element_index][j][1] =
                      T[4] * nx + T[5] * ny + T[6] * nz;
                  inward_normals_W_[element_index][j][2] =
                      T[8] * nx + T[9] * ny + T[10] * nz;
                }
              });
        });

    // Transform edge vectors
    auto transform_elem_quantities_event2 =
        q_device_.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::range<1>(total_elements_), sycl::id<1>(0),
              [=, edge_vectors_M_ = edge_vectors_M_,
               edge_vectors_W_ = edge_vectors_W_,
               sh_element_mesh_ids_ = sh_element_mesh_ids_,
               transforms_ = transforms_](sycl::id<1> idx) {
                const size_t element_index = idx[0];
                const size_t mesh_index = sh_element_mesh_ids_[element_index];

                double T[12];
                for (size_t i = 0; i < 12; ++i) {
                  T[i] = transforms_[mesh_index * 12 + i];
                }

                // Each element has 6 edge vectors
                for (size_t j = 0; j < 6; ++j) {
                  const double vx = edge_vectors_M_[element_index][j][0];
                  const double vy = edge_vectors_M_[element_index][j][1];
                  const double vz = edge_vectors_M_[element_index][j][2];

                  // Only rotation
                  edge_vectors_W_[element_index][j][0] =
                      T[0] * vx + T[1] * vy + T[2] * vz;
                  edge_vectors_W_[element_index][j][1] =
                      T[4] * vx + T[5] * vy + T[6] * vz;
                  edge_vectors_W_[element_index][j][2] =
                      T[8] * vx + T[9] * vy + T[10] * vz;
                }
              });
        });

    // Transform pressure gradients
    auto transform_elem_quantities_event3 =
        q_device_.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::range<1>(total_elements_), sycl::id<1>(0),
              [=, gradient_M_pressure_at_Mo_ = gradient_M_pressure_at_Mo_,
               gradient_W_pressure_at_Wo_ = gradient_W_pressure_at_Wo_,
               sh_element_mesh_ids_ = sh_element_mesh_ids_,
               transforms_ = transforms_](sycl::id<1> idx) {
                const size_t element_index = idx[0];
                const size_t mesh_index = sh_element_mesh_ids_[element_index];

                double T[12];
                for (size_t i = 0; i < 12; ++i) {
                  T[i] = transforms_[mesh_index * 12 + i];
                }
                // Each element has 1 pressure gradient
                const double gp_mx =
                    gradient_M_pressure_at_Mo_[element_index][0];
                const double gp_my =
                    gradient_M_pressure_at_Mo_[element_index][1];
                const double gp_mz =
                    gradient_M_pressure_at_Mo_[element_index][2];
                const double p_mo =
                    gradient_M_pressure_at_Mo_[element_index][3];

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
                gradient_W_pressure_at_Wo_[element_index][0] = gp_wx;
                gradient_W_pressure_at_Wo_[element_index][1] = gp_wy;
                gradient_W_pressure_at_Wo_[element_index][2] = gp_wz;
                gradient_W_pressure_at_Wo_[element_index][3] = p_wo;
              });
        });

    // Wait for the transformation to complete
    transform_vertices_event.wait();

    // =========================================
    // Command group 2: Generate candidate tet pairs
    // =========================================
    auto element_aabb_event = q_device_.submit([&](sycl::handler& h) {
      // Lets first compute all AABBs irrespective if they are needed or not
      // Allocate device memory for element AABBs
      // While doing this, assign false to all elements that are not part of
      // geometries that are collision candidates
      h.parallel_for(
          sycl::range<1>(total_elements_),
          [=, elements_ = elements_, vertices_W_ = vertices_W_,
           sh_element_mesh_ids_ = sh_element_mesh_ids_,
           element_aabb_min_W_ = element_aabb_min_W_,
           element_aabb_max_W_ = element_aabb_max_W_,
           sh_vertex_offsets_ = sh_vertex_offsets_](sycl::id<1> idx) {
            const size_t element_index = idx[0];
            const size_t geom_index = sh_element_mesh_ids_[element_index];
            // Get the four vertex indices for this tetrahedron
            const std::array<int, 4>& tet_vertices = elements_[element_index];
            const size_t vertex_mesh_offset = sh_vertex_offsets_[geom_index];
            // Initialize min/max to first vertex
            double min_x = vertices_W_[vertex_mesh_offset + tet_vertices[0]][0];
            double min_y = vertices_W_[vertex_mesh_offset + tet_vertices[0]][1];
            double min_z = vertices_W_[vertex_mesh_offset + tet_vertices[0]][2];

            double max_x = min_x;
            double max_y = min_y;
            double max_z = min_z;

            // Find min/max across all four vertices
            for (int i = 1; i < 4; ++i) {
              const size_t vertex_idx = vertex_mesh_offset + tet_vertices[i];

              // Update min coordinates
              min_x = sycl::min(min_x, vertices_W_[vertex_idx][0]);
              min_y = sycl::min(min_y, vertices_W_[vertex_idx][1]);
              min_z = sycl::min(min_z, vertices_W_[vertex_idx][2]);

              // Update max coordinates
              max_x = sycl::max(max_x, vertices_W_[vertex_idx][0]);
              max_y = sycl::max(max_y, vertices_W_[vertex_idx][1]);
              max_z = sycl::max(max_z, vertices_W_[vertex_idx][2]);
            }

            // Store the results
            element_aabb_min_W_[element_index][0] = min_x;
            element_aabb_min_W_[element_index][1] = min_y;
            element_aabb_min_W_[element_index][2] = min_z;

            element_aabb_max_W_[element_index][0] = max_x;
            element_aabb_max_W_[element_index][1] = max_y;
            element_aabb_max_W_[element_index][2] = max_z;
          });
    });
    element_aabb_event.wait();

    // Now generate collision filter with the AABBs that we computed
    auto generate_collision_filter_event =
        q_device_.submit([&](sycl::handler& h) {
          h.parallel_for(
              sycl::range<1>(total_checks_),
              [=, collision_filter_ = collision_filter_,
               collision_filter_host_body_index_ =
                   collision_filter_host_body_index_,
               geom_collision_filter_num_cols_ =
                   geom_collision_filter_num_cols_,
               sh_element_offsets_ = sh_element_offsets_,
               element_aabb_min_W_ = element_aabb_min_W_,
               element_aabb_max_W_ = element_aabb_max_W_,
               total_checks_per_geometry_ =
                   total_checks_per_geometry_](sycl::id<1> idx) {
                const size_t check_index = idx[0];
                const size_t host_body_index =
                    collision_filter_host_body_index_[check_index];
                // What elements is this check_index checking?
                // host_body_index is the geometry index that element A belongs
                // to
                size_t num_of_checks_offset = 0;
                if (host_body_index > 0) {
                  num_of_checks_offset =
                      total_checks_per_geometry_[host_body_index - 1];
                }
                const size_t geom_local_check_number =
                    check_index - num_of_checks_offset;

                const size_t A_element_index =
                    sh_element_offsets_[host_body_index] +
                    geom_local_check_number /
                        geom_collision_filter_num_cols_[host_body_index];
                const size_t B_element_index =
                    sh_element_offsets_[host_body_index + 1] +
                    geom_local_check_number %
                        geom_collision_filter_num_cols_[host_body_index];

                // We have two element index, now just check their AABB
                // A element AABB
                // min
                const double A_element_aabb_min_W_x =
                    element_aabb_min_W_[A_element_index][0];
                const double A_element_aabb_min_W_y =
                    element_aabb_min_W_[A_element_index][1];
                const double A_element_aabb_min_W_z =
                    element_aabb_min_W_[A_element_index][2];
                // max
                const double A_element_aabb_max_W_x =
                    element_aabb_max_W_[A_element_index][0];
                const double A_element_aabb_max_W_y =
                    element_aabb_max_W_[A_element_index][1];
                const double A_element_aabb_max_W_z =
                    element_aabb_max_W_[A_element_index][2];

                // B element AABB
                // min
                const double B_element_aabb_min_W_x =
                    element_aabb_min_W_[B_element_index][0];
                const double B_element_aabb_min_W_y =
                    element_aabb_min_W_[B_element_index][1];
                const double B_element_aabb_min_W_z =
                    element_aabb_min_W_[B_element_index][2];
                // max
                const double B_element_aabb_max_W_x =
                    element_aabb_max_W_[B_element_index][0];
                const double B_element_aabb_max_W_y =
                    element_aabb_max_W_[B_element_index][1];
                const double B_element_aabb_max_W_z =
                    element_aabb_max_W_[B_element_index][2];

                collision_filter_[check_index] =
                    !(A_element_aabb_max_W_x < B_element_aabb_min_W_x ||
                      A_element_aabb_min_W_x > B_element_aabb_max_W_x ||
                      A_element_aabb_max_W_y < B_element_aabb_min_W_y ||
                      A_element_aabb_min_W_y > B_element_aabb_max_W_y ||
                      A_element_aabb_max_W_z < B_element_aabb_min_W_z ||
                      A_element_aabb_min_W_z > B_element_aabb_max_W_z);
              });
        });
    generate_collision_filter_event.wait();

    std::vector<sycl::event> transform_events{transform_elem_quantities_event1,
                                              transform_elem_quantities_event2,
                                              transform_elem_quantities_event3};
    sycl::event::wait(transform_events);

    // Placeholder that returns an empty vector
    std::vector<SYCLHydroelasticSurface> sycl_hydroelastic_surfaces;
    return sycl_hydroelastic_surfaces;
  }

 private:
  friend class SyclProximityEngineTester;
  // We have a CPU queue for operations beneficial to perform on the host and a
  // device queue for operations beneficial to perform on the Accelerator.
  sycl::queue q_device_;
  sycl::queue q_host_;
  // The collision candidates.
  std::vector<SortedPair<GeometryId>> collision_candidates_;
  // GeometryIds of soft geometries (host-side)
  GeometryId* soft_geometry_ids_ = nullptr;
  // Number of geometries
  size_t num_geometries_ = 0;

  // SYCL shared arrays for geometry lookup
  size_t* sh_element_offsets_ =
      nullptr;  // Element offset for each geometry -> points to memory on host
                // accessible on device through PCIe
  size_t* sh_vertex_offsets_ =
      nullptr;  // Vertex offset for each geometry -> points to memory on host
                // accessible on device through PCIe
  size_t* sh_element_counts_ =
      nullptr;  // Number of elements for each geometry -> points to memory on
                // host accessible on device through PCIe
  size_t* sh_vertex_counts_ =
      nullptr;  // Number of vertices for each geometry -> points to memory on
                // host accessible on device through PCIe

  size_t* sh_vertex_mesh_ids_ = nullptr;   // Mesh id for each vertex -> points
                                           // to memory in device global memory
  size_t* sh_element_mesh_ids_ = nullptr;  // Mesh id for each element -> points
                                           // to memory in device global memory

  Vector3<double>* element_aabb_min_W_ =
      nullptr;  // Minimum AABB for each element
  Vector3<double>* element_aabb_max_W_ =
      nullptr;  // Maximum AABB for each element
  size_t total_vertices_ = 0;
  size_t total_elements_ = 0;
  /*
  A hydroelastic geometry contains one mesh. Elements are tetrahedra.
  All data is stored in contiguous arrays, with each geometry's data
  at a specific offset in these arrays.
  */

  // Mesh element data - accessed by element_offset + local_element_index
  std::array<int, 4>* elements_ =
      nullptr;  // Elements as 4 vertex indices -
                // Note the 4 vertex indicies are
                // geometry indices to the geometry and thus are in
                // [0,num_vertices_per_geometry)
  std::array<Vector3<double>, 4>* inward_normals_M_ =
      nullptr;  // Inward normals in mesh frame
  std::array<Vector3<double>, 4>* inward_normals_W_ =
      nullptr;  // Inward normals in world frame
  std::array<Vector3<double>, 6>* edge_vectors_M_ =
      nullptr;  // Edge vectors in mesh frame
  std::array<Vector3<double>, 6>* edge_vectors_W_ =
      nullptr;  // Edge vectors in world frame

  // Mesh vertex data - accessed by vertex_offset + local_vertex_index
  Vector3<double>* vertices_M_ = nullptr;  // Vertices in mesh frame
  Vector3<double>* vertices_W_ = nullptr;  // Vertices in world frame

  // Pressure field data - accessed by element_offset + local_element_index
  double* pressures_ = nullptr;      // Pressure values
  double* min_pressures_ = nullptr;  // Minimum pressure values
  double* max_pressures_ = nullptr;  // Maximum pressure values

  // Combined gradient and pressure values to optimize cache utilization
  // First 3 components are gradient, 4th component is pressure at origin
  Vector4<double>* gradient_M_pressure_at_Mo_ = nullptr;  // In mesh frame
  Vector4<double>* gradient_W_pressure_at_Wo_ = nullptr;  // In world frame

  double* transforms_ =
      nullptr;  // Pointer to memory residing on host but accesible on device
                // (all transfers through PCIe)

  // Some helpers for broad phase collision detection
  size_t num_elements_in_last_geometry_ = 0;
  size_t* total_checks_per_geometry_ =
      nullptr;  // total_checks_per_geometry_[i]= number of element AABB checks
                // for the i-th geometry
  size_t* collision_filter_host_body_index_ =
      nullptr;  // collision_filter_host_body_index_[i] = Geometry index to
                // which eleemnt A belongs to - This is then used to index into
                // and geom_collision_filter_num_cols_ to get the number of
                // checks for element A (the num columns in that block of the
                // collision filter)
  bool* collision_filter_ = nullptr;  // collision_filter_[i]= 1 if the i-th
                                      // geometry is a collision candidate,
                                      // 0 otherwise
  size_t* geom_collision_filter_check_offsets_ = nullptr;
  size_t* geom_collision_filter_num_cols_ = nullptr;
  size_t total_checks_ = 0;

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
bool* SyclProximityEngineAttorney::get_collision_filter(
    SyclProximityEngine::Impl* impl) {
  return impl->collision_filter_;
}
std::vector<Vector3<double>> SyclProximityEngineAttorney::get_vertices_M(
    SyclProximityEngine::Impl* impl) {
  auto q = impl->q_device_;
  auto vertices_M = impl->vertices_M_;
  auto total_vertices = impl->total_vertices_;
  std::vector<Vector3<double>> vertices_M_host(total_vertices);
  q.memcpy(vertices_M_host.data(), vertices_M,
           total_vertices * sizeof(Vector3<double>));
  return vertices_M_host;
}
std::vector<Vector3<double>> SyclProximityEngineAttorney::get_vertices_W(
    SyclProximityEngine::Impl* impl) {
  auto q = impl->q_device_;
  auto vertices_W = impl->vertices_W_;
  auto total_vertices = impl->total_vertices_;
  std::vector<Vector3<double>> vertices_W_host(total_vertices);
  q.memcpy(vertices_W_host.data(), vertices_W,
           total_vertices * sizeof(Vector3<double>));
  return vertices_W_host;
}
std::vector<std::array<int, 4>> SyclProximityEngineAttorney::get_elements(
    SyclProximityEngine::Impl* impl) {
  auto q = impl->q_device_;
  auto elements = impl->elements_;
  auto total_elements = impl->total_elements_;
  std::vector<std::array<int, 4>> elements_host(total_elements);
  q.memcpy(elements_host.data(), elements,
           total_elements * sizeof(std::array<int, 4>));
  return elements_host;
}
double* SyclProximityEngineAttorney::get_pressures(
    SyclProximityEngine::Impl* impl) {
  return impl->pressures_;
}
Vector4<double>* SyclProximityEngineAttorney::get_gradient_M_pressure_at_Mo(
    SyclProximityEngine::Impl* impl) {
  return impl->gradient_M_pressure_at_Mo_;
}
Vector4<double>* SyclProximityEngineAttorney::get_gradient_W_pressure_at_Wo(
    SyclProximityEngine::Impl* impl) {
  return impl->gradient_W_pressure_at_Wo_;
}
size_t* SyclProximityEngineAttorney::get_collision_filter_host_body_index(
    SyclProximityEngine::Impl* impl) {
  return impl->collision_filter_host_body_index_;
}
size_t SyclProximityEngineAttorney::get_total_checks(
    SyclProximityEngine::Impl* impl) {
  return impl->total_checks_;
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
