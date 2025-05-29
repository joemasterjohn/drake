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
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Tetrahedon slice with EqPlane helper code from mesh_plane_intersection.cc

/* This table essentially assigns an index to each edge in the tetrahedron.
 Each edge is represented by its pair of vertex indexes. */
using TetrahedronEdge = std::pair<int, int>;
constexpr std::array<std::pair<int, int>, 6> kTetEdges = {
    // base formed by vertices 0, 1, 2.
    TetrahedronEdge{0, 1}, TetrahedronEdge{1, 2}, TetrahedronEdge{2, 0},
    // pyramid with top at node 3.
    TetrahedronEdge{0, 3}, TetrahedronEdge{1, 3}, TetrahedronEdge{2, 3}};

/* Marching tetrahedra tables. Each entry in these tables have an index value
 based on a binary encoding of the signs of the plane's signed distance
 function evaluated at all tetrahedron vertices. Therefore, with four
 vertices and two possible signs, we have a total of 16 entries. We encode
 the table indexes in binary so that a "1" and "0" correspond to a vertex
 with positive or negative signed distance, respectively. The least
 significant bit (0) corresponds to vertex 0 in the tetrahedron, and the
 most significant bit (3) is vertex 3. */

/* Each entry of kMarchingTetsEdgeTable stores a vector of edges.
 Based on the signed distance values, these edges are the ones that
 intersect the plane. Edges are numbered according to the table kTetEdges.
 The edges have been ordered such that a polygon formed by visiting the
 listed edge's intersection vertices in the array order has a right-handed
 normal pointing in the direction of the plane's normal. The accompanying
 unit tests verify this.

 A -1 is a sentinel value indicating no edge encoding. The number of
 intersecting edges is equal to the index of the *first* -1 (with an implicit
 logical -1 at index 4).  */
// clang-format off
constexpr std::array<std::array<int, 4>, 16> kMarchingTetsEdgeTable = {
                                /* bits    3210 */
    std::array<int, 4>{-1, -1, -1, -1}, /* 0000 */
    std::array<int, 4>{0, 3, 2, -1},    /* 0001 */
    std::array<int, 4>{0, 1, 4, -1},    /* 0010 */
    std::array<int, 4>{4, 3, 2, 1},     /* 0011 */
    std::array<int, 4>{1, 2, 5, -1},    /* 0100 */
    std::array<int, 4>{0, 3, 5, 1},     /* 0101 */
    std::array<int, 4>{0, 2, 5, 4},     /* 0110 */
    std::array<int, 4>{3, 5, 4, -1},    /* 0111 */
    std::array<int, 4>{3, 4, 5, -1},    /* 1000 */
    std::array<int, 4>{4, 5, 2, 0},     /* 1001 */
    std::array<int, 4>{1, 5, 3, 0},     /* 1010 */
    std::array<int, 4>{1, 5, 2, -1},    /* 1011 */
    std::array<int, 4>{1, 2, 3, 4},     /* 1100 */
    std::array<int, 4>{0, 4, 1, -1},    /* 1101 */
    std::array<int, 4>{0, 2, 3, -1},    /* 1110 */
    std::array<int, 4>{-1, -1, -1, -1}  /* 1111 */};



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
    pressures_ = sycl::malloc_device<double>(total_vertices_, q_device_);
    sh_vertex_mesh_ids_ =
        sycl::malloc_device<size_t>(total_vertices_, q_device_);

    inward_normals_M_ = sycl::malloc_device<std::array<Vector3<double>, 4>>(
        total_elements_, q_device_);
    edge_vectors_M_ = sycl::malloc_device<std::array<Vector3<double>, 6>>(
        total_elements_, q_device_);
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

      // Pressures
      transfer_events.push_back(q_device_.memcpy(
          pressures_ + vertex_offset, pressure_field.values().data(),
          num_vertices * sizeof(double)));

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
    for (size_t i = 0; i < num_geometries_ - 1; ++i) {
      const size_t num_elements_in_geometry = sh_element_counts_[i];
      const size_t num_elements_in_rest_of_geometries =
          (sh_element_offsets_[num_geometries_ - 1] +
           num_elements_in_last_geometry_) -
          sh_element_offsets_[i + 1];
      geom_collision_filter_num_cols_[i] = num_elements_in_rest_of_geometries;
      // We need to check each element in this geometry with each element in
      // the rest of the geometries
      total_checks_per_geometry_[i] =
          num_elements_in_rest_of_geometries * num_elements_in_geometry;
      geom_collision_filter_check_offsets_[i] = total_checks_;
      total_checks_ += total_checks_per_geometry_[i];
    }

    // Generate collision filter for all checks
    collision_filter_ = sycl::malloc_device<uint8_t>(total_checks_, q_device_);
    // Required in ComputeSYCLHydroelasticSurface
    prefix_sum_ = sycl::malloc_device<size_t>(total_checks_, q_device_);
    // memset all to 0 for now (will be filled in when we have AABBs for each
    // element)
    auto collision_filter_memset_event =
        q_device_.memset(collision_filter_, 0, total_checks_ * sizeof(uint8_t));
    collision_filter_memset_event.wait();

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
      sycl::free(prefix_sum_, q_device_);
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
      return std::vector<SYCLHydroelasticSurface>();
    }

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

    // Now generate collision filter with the AABBs that we computed
    auto generate_collision_filter_event =
        q_device_.submit([&](sycl::handler& h) {
          h.depends_on(element_aabb_event);
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
               total_checks_per_geometry_ = total_checks_per_geometry_,
               min_pressures_ = min_pressures_,
               max_pressures_ = max_pressures_](sycl::id<1> idx) {
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

                // Default to not colliding.
                // collision_filter_[check_index] = 0;

                // First check if the pressure fields of the elements intersect
                if (max_pressures_[B_element_index] <
                        min_pressures_[A_element_index] ||
                    max_pressures_[A_element_index] <
                        min_pressures_[B_element_index]) {
                  return;
                }

                // We have two element index, now just check their AABB
                // A element AABB
                // min
                for (int i = 0; i < 3; ++i) {
                  if (element_aabb_max_W_[B_element_index][i] <
                      element_aabb_min_W_[A_element_index][i])
                    return;
                  if (element_aabb_max_W_[A_element_index][i] <
                      element_aabb_min_W_[B_element_index][i])
                    return;
                }

                collision_filter_[check_index] = 1;
              });
        });
    generate_collision_filter_event.wait();

    // =========================================
    // Generate list of check_indices that are active
    // =========================================

    auto policy = oneapi::dpl::execution::make_device_policy(q_device_);

    // Perform the exclusive scan using USM pointers as iterators
    // We need to convert uint8_t collision_filter_ values to size_t for the
    // scan
    oneapi::dpl::transform_exclusive_scan(
        policy, collision_filter_, collision_filter_ + total_checks_,
        prefix_sum_,             // output
        static_cast<size_t>(0),  // initial value
        sycl::plus<size_t>(),    // binary operation
        [](uint8_t x) {
          return static_cast<size_t>(x);
        });  // transform uint8_t to size_t
    q_device_.wait_and_throw();

    // Total checks needed for narrow phase
    size_t total_narrow_phase_checks = 0;
    q_device_
        .memcpy(&total_narrow_phase_checks, prefix_sum_ + total_checks_ - 1,
                sizeof(size_t))
        .wait();
    // Last element check or not?
    uint8_t last_check_flag = 0;
    q_device_
        .memcpy(&last_check_flag, collision_filter_ + total_checks_ - 1,
                sizeof(uint8_t))
        .wait();
    // If last check is 1, then we need to add one more check
    total_narrow_phase_checks += static_cast<size_t>(last_check_flag);

    // Now we need to get the index of all the narrow phase checks
    size_t* narrow_phase_check_indices =
        sycl::malloc_device<size_t>(total_narrow_phase_checks, q_device_);

    auto fill_narrow_phase_check_indices_event =
        q_device_.submit([&](sycl::handler& h) {
          h.depends_on(generate_collision_filter_event);
          h.parallel_for(
              sycl::range<1>(total_checks_),
              [=, narrow_phase_check_indices = narrow_phase_check_indices,
               prefix_sum_ = prefix_sum_,
               collision_filter_ = collision_filter_](sycl::id<1> idx) {
                const size_t check_index = idx[0];
                if (collision_filter_[check_index] == 1) {
                  size_t narrow_check_num = prefix_sum_[check_index];
                  narrow_phase_check_indices[narrow_check_num] = check_index;
                }
              });
        });

    // =========================================
    // Command group 4: Narrow phase collision detection
    // =========================================
    // Maintain a list of invalidated narrow phase checks
    // 1 is invalid, 0 is valid
    uint8_t* invalidated_narrow_phase_checks_ =
        sycl::malloc_device<uint8_t>(total_narrow_phase_checks, q_device_);
    // Initialize all to 0 -> All are valid
    q_device_
        .memset(invalidated_narrow_phase_checks_, 0,
                total_narrow_phase_checks * sizeof(uint8_t))
        .wait();

    constexpr size_t LOCAL_SIZE = 128;  // 128 per work group
    // Number of threads involved in the computations of one check
    // Minimum that can be set here is 4 -> Limits on shared memory requirements
    // Maximum can be set to 16 -> Limit for meaningful parallelization without
    // most threads waiting most of the time
    constexpr size_t NUM_THREADS_PER_CHECK = 4;
    constexpr size_t NUM_CHECKS_IN_WORK_GROUP =
        LOCAL_SIZE / NUM_THREADS_PER_CHECK;
    
    // Calculate total threads needed (4 threads per check)
    const size_t TOTAL_THREADS_NEEDED = total_narrow_phase_checks * NUM_THREADS_PER_CHECK;
    // Number of work groups
    const size_t NUM_GROUPS =
        (TOTAL_THREADS_NEEDED + LOCAL_SIZE - 1) / LOCAL_SIZE;
    // Calculation of the number of doubles to be stored in shared memory per
    // check Eq. Plane - 2 x 3 = 6 doubles (normal and point), Polygon verticies
    // - 8 x 3 = 24 doubles, Inward normals - 2 x 4 x 3 = 24 doubles (Normals of
    // 4 faces), Vertices of elements - 2 x 4 x 3 = 24 doubles (Vertices of 4
    // faces), Edge Vectors - 2 x 6 x 3 = 36 doubles (Edge vectors of 6 edges)
    // Total 114 doubles
    // Offsets are required to index the local memory
    constexpr size_t EQ_PLANE_OFFSET = 0;
    constexpr size_t EQ_PLANE_DOUBLES = 6;

    constexpr size_t VERTEX_A_OFFSET = EQ_PLANE_OFFSET + EQ_PLANE_DOUBLES;
    constexpr size_t VERTEX_A_DOUBLES = 12;
    constexpr size_t VERTEX_B_OFFSET = VERTEX_A_OFFSET + VERTEX_A_DOUBLES;
    constexpr size_t VERTEX_B_DOUBLES = 12;

    constexpr size_t INWARD_NORMAL_A_OFFSET =
        VERTEX_B_OFFSET + VERTEX_B_DOUBLES;
    constexpr size_t INWARD_NORMAL_A_DOUBLES = 12;
    constexpr size_t INWARD_NORMAL_B_OFFSET =
        INWARD_NORMAL_A_OFFSET + INWARD_NORMAL_A_DOUBLES;
    constexpr size_t INWARD_NORMAL_B_DOUBLES = 12;

    constexpr size_t EDGE_A_OFFSET =
        INWARD_NORMAL_B_OFFSET + INWARD_NORMAL_B_DOUBLES;
    constexpr size_t EDGE_A_DOUBLES = 18;
    constexpr size_t EDGE_B_OFFSET = EDGE_A_OFFSET + EDGE_A_DOUBLES;
    constexpr size_t EDGE_B_DOUBLES = 18;

    constexpr size_t POLYGON_OFFSET = EDGE_B_OFFSET + EDGE_B_DOUBLES;
    constexpr size_t POLYGON_DOUBLES = 8;

    // Used varylingly through the kernel to express more parallelism
    constexpr size_t RANDOM_SCRATCH_OFFSET = POLYGON_OFFSET + POLYGON_DOUBLES;
    constexpr size_t RANDOM_SCRATCH_DOUBLES = 8;

    // Calculate total doubles for verification
    constexpr size_t VERTEX_DOUBLES = VERTEX_A_DOUBLES + VERTEX_B_DOUBLES;
    constexpr size_t INWARD_NORMAL_DOUBLES =
        INWARD_NORMAL_A_DOUBLES + INWARD_NORMAL_B_DOUBLES;
    constexpr size_t EDGE_DOUBLES = EDGE_A_DOUBLES + EDGE_B_DOUBLES;

    constexpr size_t DOUBLES_PER_CHECK = EQ_PLANE_DOUBLES + VERTEX_DOUBLES +
                                         INWARD_NORMAL_DOUBLES + EDGE_DOUBLES +
                                         POLYGON_DOUBLES + RANDOM_SCRATCH_DOUBLES;

    
    // Additionally lets have a random scratch space for storing INTS
    // These will also be used varyingly throughout the kernel to express parallelism
    constexpr size_t RANDOM_SCRATCH_INTS_OFFSET = 0;
    constexpr size_t RANDOM_SCRATCH_INTS = 8;

    // We need to compute the equilibrium plane for each check
    // We will use the first NUM_CHECKS_IN_WORK_GROUP checks in the work group
    // to compute the equilibrium plane
    // We will then use the remaining checks to compute the contact polygon

    auto compute_contact_polygon_event = q_device_.submit([&](sycl::handler&
                                                                  h) {
      h.depends_on({fill_narrow_phase_check_indices_event,
                    transform_elem_quantities_event1,
                    transform_elem_quantities_event2,
                    transform_elem_quantities_event3});
      // Shared Local Memory (SLM) is stored as
      // [Eq_plane_i, Vertices_A_i, Vertices_B_i, Inward_normals_A_i,
      // Inward_normals_B_i, Edge_vectors_A_i, Edge_vectors_B_i,
      // Polygon_i, Eq_plane_i+1, Vertices_A_i+1,
      // Vertices_B_i+1, Inward_normals_A_i+1, Inward_normals_B_i+1,
      // Edge_vectors_A_i+1, Edge_vectors_B_i+1, Polygon_i+1, ...]
      // Always Polygon A stored first and then Polygon B (for the
      // quantities which we need both off)
      sycl::local_accessor<double, 1> slm(
          LOCAL_SIZE / NUM_THREADS_PER_CHECK * DOUBLES_PER_CHECK, h);
      sycl::local_accessor<int, 1> slm_ints(
          LOCAL_SIZE / NUM_THREADS_PER_CHECK * RANDOM_SCRATCH_INTS, h);
      h.parallel_for(
          sycl::nd_range<1>{NUM_GROUPS * LOCAL_SIZE, LOCAL_SIZE},
          [=, narrow_phase_check_indices = narrow_phase_check_indices,
           gradient_W_pressure_at_Wo_ = gradient_W_pressure_at_Wo_,
           sh_element_offsets_ = sh_element_offsets_,
           sh_vertex_offsets_ = sh_vertex_offsets_,
           sh_element_mesh_ids_ = sh_element_mesh_ids_, elements_ = elements_,
           vertices_W_ = vertices_W_, inward_normals_W_ = inward_normals_W_,
           edge_vectors_W_ = edge_vectors_W_,
           geom_collision_filter_num_cols_ = geom_collision_filter_num_cols_,
           total_checks_per_geometry_ = total_checks_per_geometry_,
           collision_filter_host_body_index_ =
               collision_filter_host_body_index_,
           invalidated_narrow_phase_checks_ =
               invalidated_narrow_phase_checks_](sycl::nd_item<1> item) {
            size_t global_id = item.get_global_id(0);
            // Early return for extra threads
            if (global_id >= TOTAL_THREADS_NEEDED) return;
            size_t local_id = item.get_local_id(0);
            // In a group we have NUM_CHECKS_IN_WORK_GROUP checks
            // This gives us which check number in [0, NUM_CHECKS_IN_WORK_GROUP)
            // this item is working on
            // NUM_THREADS_PER_CHECK threads will have the same
            // group_check_number
            // It ranges from [0, NUM_CHECKS_IN_WORK_GROUP)
            size_t group_local_check_number = local_id / NUM_THREADS_PER_CHECK;

            // This offset is used to compute the positions each of the
            // quantities for reading and writing to slm
            size_t slm_offset = group_local_check_number * DOUBLES_PER_CHECK;
            
            // Separate offset for slm_ints array
            size_t slm_ints_offset = group_local_check_number * RANDOM_SCRATCH_INTS;

            // Each check has NUM_THREADS_PER_CHECK workers.
            // This index helps identify the check local worker id
            // It ranges for [0, NUM_THREADS_PER_CHECK)
            size_t check_local_item_id = local_id % NUM_THREADS_PER_CHECK;

            // Get global element ids
            size_t narrow_phase_check_index = global_id / NUM_THREADS_PER_CHECK;
            // global check index
            size_t global_check_index =
                narrow_phase_check_indices[narrow_phase_check_index];

            // For these checks, get the global element indicies
            // Same logic as the broad phase collision
            const size_t host_body_index =
                collision_filter_host_body_index_[global_check_index];

            // Same logic as broad phase
            size_t num_of_checks_offset = 0;
            if (host_body_index > 0) {
              num_of_checks_offset =
                  total_checks_per_geometry_[host_body_index - 1];
            }
            const size_t geom_local_check_number =
                global_check_index - num_of_checks_offset;

            const size_t A_element_index =
                sh_element_offsets_[host_body_index] +
                geom_local_check_number /
                    geom_collision_filter_num_cols_[host_body_index];
            const size_t B_element_index =
                sh_element_offsets_[host_body_index + 1] +
                geom_local_check_number %
                    geom_collision_filter_num_cols_[host_body_index];

            // We only need one thread to compute the Equilibrium Plane
            // for each check, however we have potentially multiple threads
            // per check.
            // We cannot use the first "total_narrow_phase_check" items since we
            // need to store the equilibirum planes in shared memory which is
            // work group local Thus, make sure only 1 thread in the check group
            // computes the equilibrium plane and we choose this to be the 1st
            // thread
            if (check_local_item_id == 0) {
              // Get individual quanities for quick access from registers
              double gradP_A_Wo_x =
                  gradient_W_pressure_at_Wo_[A_element_index][0];
              double gradP_A_Wo_y =
                  gradient_W_pressure_at_Wo_[A_element_index][1];
              double gradP_A_Wo_z =
                  gradient_W_pressure_at_Wo_[A_element_index][2];
              double p_A_Wo = gradient_W_pressure_at_Wo_[A_element_index][3];
              double gradP_B_Wo_x =
                  gradient_W_pressure_at_Wo_[B_element_index][0];
              double gradP_B_Wo_y =
                  gradient_W_pressure_at_Wo_[B_element_index][1];
              double gradP_B_Wo_z =
                  gradient_W_pressure_at_Wo_[B_element_index][2];
              double p_B_Wo = gradient_W_pressure_at_Wo_[B_element_index][3];

              // In frame W, the two linear functions are:
              //      f₀(p_Wo) = grad_f0_W.dot(p_Wo) + f0_Wo.
              //      f₁(p_Wo) = grad_f1_W.dot(p_Wo) + f1_Wo.
              // Their equilibrium plane is:
              //   (grad_f0_W - grad_f1_W).dot(p_Wo) + (f0_Wo - f1_Wo) = 0.
              // Its perpendicular (but not necessarily unit-length) vector
              // is:
              //           n_W = grad_f0_W - grad_f1_W,
              // which is in the direction of increasing f₀ and decreasing f₁.
              double n_W_x = gradP_A_Wo_x - gradP_B_Wo_x;
              double n_W_y = gradP_A_Wo_y - gradP_B_Wo_y;
              double n_W_z = gradP_A_Wo_z - gradP_B_Wo_z;

              double n_W_norm =
                  sycl::sqrt(n_W_x * n_W_x + n_W_y * n_W_y + n_W_z * n_W_z);
              double n_W_x_normalized = n_W_x / n_W_norm;
              double n_W_y_normalized = n_W_y / n_W_norm;
              double n_W_z_normalized = n_W_z / n_W_norm;

              if (n_W_norm <= 0.0) {
                // Invalidate this check -> It will not generate a contact
                // surface
                invalidated_narrow_phase_checks_[narrow_phase_check_index] = 1;
              }

              // Normal has to point in the direction of increasing field_0
              // and decreasing field_1
              // Normalized pressure gradient is:
              const double gradP_A_W_norm = sycl::sqrt(
                  gradP_A_Wo_x * gradP_A_Wo_x + gradP_A_Wo_y * gradP_A_Wo_y +
                  gradP_A_Wo_z * gradP_A_Wo_z);
              const double gradP_A_W_normalized_x =
                  gradP_A_Wo_x / gradP_A_W_norm;
              const double gradP_A_W_normalized_y =
                  gradP_A_Wo_y / gradP_A_W_norm;
              const double gradP_A_W_normalized_z =
                  gradP_A_Wo_z / gradP_A_W_norm;

              const double cos_theta_A =
                  n_W_x_normalized * gradP_A_W_normalized_x +
                  n_W_y_normalized * gradP_A_W_normalized_y +
                  n_W_z_normalized * gradP_A_W_normalized_z;

              const double kCosAlpha = sycl::cos(5. * M_PI / 8.);
              if (cos_theta_A < kCosAlpha) {
                // Invalidate this check -> It will not generate a contact
                // surface
                invalidated_narrow_phase_checks_[narrow_phase_check_index] = 1;
              }

              const double gradP_B_W_norm = sycl::sqrt(
                  gradP_B_Wo_x * gradP_B_Wo_x + gradP_B_Wo_y * gradP_B_Wo_y +
                  gradP_B_Wo_z * gradP_B_Wo_z);
              const double gradP_B_W_normalized_x =
                  gradP_B_Wo_x / gradP_B_W_norm;
              const double gradP_B_W_normalized_y =
                  gradP_B_Wo_y / gradP_B_W_norm;
              const double gradP_B_W_normalized_z =
                  gradP_B_Wo_z / gradP_B_W_norm;

              // Normal given negative direction because we need to check that
              // its pointing along the decreasing field
              const double cos_theta_B =
                  -n_W_x_normalized * gradP_B_W_normalized_x +
                  -n_W_y_normalized * gradP_B_W_normalized_y +
                  -n_W_z_normalized * gradP_B_W_normalized_z;
              if (cos_theta_B < kCosAlpha) {
                // Invalidate this check -> It will not generate a contact
                // surface
                invalidated_narrow_phase_checks_[narrow_phase_check_index] = 1;
              }

              // Using the unit normal vector nhat_W, the plane equation (1)
              // becomes:
              //
              //          nhat_W.dot(p_Wo) + Δ = 0,
              //
              // where Δ = (f0_Wo - f1_Wo)/‖n_W‖. One such p_Wo is:
              //
              //                          p_Wo = -Δ * nhat_W
              //
              double p_WQ_x = ((p_B_Wo - p_A_Wo) / n_W_norm) * n_W_x_normalized;
              double p_WQ_y = ((p_B_Wo - p_A_Wo) / n_W_norm) * n_W_y_normalized;
              double p_WQ_z = ((p_B_Wo - p_A_Wo) / n_W_norm) * n_W_z_normalized;

              // Write for Eq plane
              slm[slm_offset + EQ_PLANE_OFFSET] = n_W_x_normalized;
              slm[slm_offset + EQ_PLANE_OFFSET + 1] = n_W_y_normalized;
              slm[slm_offset + EQ_PLANE_OFFSET + 2] = n_W_z_normalized;
              slm[slm_offset + EQ_PLANE_OFFSET + 3] = p_WQ_x;
              slm[slm_offset + EQ_PLANE_OFFSET + 4] = p_WQ_y;
              slm[slm_offset + EQ_PLANE_OFFSET + 5] = p_WQ_z;
            }

            // Move vertices and edge vectors to slm
            // Some quantities required for indexing
            const size_t geom_index_A = sh_element_mesh_ids_[A_element_index];
            const std::array<int, 4>& tet_vertices_A =
                elements_[A_element_index];
            const size_t vertex_mesh_offset_A =
                sh_vertex_offsets_[geom_index_A];

            // Vertices of element B
            const size_t geom_index_B = sh_element_mesh_ids_[B_element_index];
            const std::array<int, 4>& tet_vertices_B =
                elements_[B_element_index];
            const size_t vertex_mesh_offset_B =
                sh_vertex_offsets_[geom_index_B];

            // Loop is over x,y,z
            for (size_t i = 0; i < 3; i++) {
              // Quantities that we have "4" of
              for (size_t llid = check_local_item_id; llid < 4;
                   llid += NUM_THREADS_PER_CHECK) {
                // All 4 vertices moved at once by our sub items
                slm[slm_offset + VERTEX_A_OFFSET + llid * 3 + i] =
                    vertices_W_[vertex_mesh_offset_A + tet_vertices_A[llid]][i];
                slm[slm_offset + VERTEX_B_OFFSET + llid * 3 + i] =
                    vertices_W_[vertex_mesh_offset_B + tet_vertices_B[llid]][i];
                }
              // Quantities that we have "6" of
              for (size_t llid = check_local_item_id; llid < 6;
                   llid += NUM_THREADS_PER_CHECK) {
                // Edge vectors of element A
                slm[slm_offset + EDGE_A_OFFSET + llid * 3 + i] =
                    edge_vectors_W_[A_element_index][llid][i];

                // Edge vectors of element B
                slm[slm_offset + EDGE_B_OFFSET + llid * 3 + i] =
                    edge_vectors_W_[B_element_index][llid][i];
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Slice element A with Eq Plane

            // Compute signed distance of all vertices of element A with Eq plane
            // Parallelization based on distance computation

            for(size_t llid = check_local_item_id; llid < 4; llid += NUM_THREADS_PER_CHECK) {
             // Each thread gets 1 vertex of element A in slm
             const double vertex_A_x = slm[slm_offset + VERTEX_A_OFFSET + llid * 3 + 0];
             const double vertex_A_y = slm[slm_offset + VERTEX_A_OFFSET + llid * 3 + 1];
             const double vertex_A_z = slm[slm_offset + VERTEX_A_OFFSET + llid * 3 + 2];
             // Each thread accesses the same Eq plane from slm
             // TODO(huzaifa) - Will we have shMem bank conflict on Nvidia GPUs?
             // Need to know if SYCL backend compiler propertly recognizes that this is a broadcast operation
             // Normals
             const double normal_x = slm[slm_offset + EQ_PLANE_OFFSET];
             const double normal_y = slm[slm_offset + EQ_PLANE_OFFSET + 1];
             const double normal_z = slm[slm_offset + EQ_PLANE_OFFSET + 2];
             // Point on the plane
             const double point_on_plane_x = slm[slm_offset + EQ_PLANE_OFFSET + 3];
             const double point_on_plane_y = slm[slm_offset + EQ_PLANE_OFFSET + 4];
             const double point_on_plane_z = slm[slm_offset + EQ_PLANE_OFFSET + 5];
             // Compute the dispalcement of the plane from the origin of the frame (world in this case) as simple dot product
             const double displacement = normal_x * point_on_plane_x + normal_y * point_on_plane_y + normal_z * point_on_plane_z;

             // Compute signed distance of this vertex with Eq plane
             // +ve height indicates point is above the plane
             // -ve height indicates point is below the plane
             // Store these in our random scratch space
             slm[slm_offset + RANDOM_SCRATCH_OFFSET + llid] = normal_x * vertex_A_x + normal_y * vertex_A_y + normal_z * vertex_A_z - displacement;
            }
            item.barrier(sycl::access::fence_space::local_space);
            
            // Let one thread compute intersection code and store this in the shared memory for other threads
            if(check_local_item_id == 0) {
              int intersection_code = 0;
              for(size_t llid = 0; llid < 4; llid++) {
                if(slm[slm_offset + RANDOM_SCRATCH_OFFSET + llid] > 0.0) {
                  intersection_code |= (1 << llid);
                }
              }
              slm_ints[slm_ints_offset] = intersection_code;
            }
            item.barrier(sycl::access::fence_space::local_space);
            
            // Now go back to using NUM_THREADS_PER_CHECK threads to compute the polygon vertices
            for(size_t llid = check_local_item_id; llid < 4; llid += NUM_THREADS_PER_CHECK) {
              const int edge_index = kMarchingTetsEdgeTable[slm_ints[slm_ints_offset]][llid];

              // Only proceed if we are not at the end of edge list
              if(edge_index != -1) {
                // Get the tet edge
               const TetrahedronEdge& tet_edge = kTetEdges[edge_index];
                // Get the heights of these vertices from the scratch space
               const double height_0 = slm[slm_offset + RANDOM_SCRATCH_OFFSET + tet_edge.first];
               const double height_1 = slm[slm_offset + RANDOM_SCRATCH_OFFSET + tet_edge.second];

               // Compute the intersection point
               const double t = height_0 / (height_0 - height_1);


               // Compute polygon vertices
               // Loop is over x,y,z
               for(size_t i = 0; i < 3; i++) {
                const double vertex_0 = slm[slm_offset + VERTEX_A_OFFSET + tet_edge.first * 3 + i];
                const double vertex_1 = slm[slm_offset + VERTEX_A_OFFSET + tet_edge.second * 3 + i];


                const double intersection = vertex_0 + t * (vertex_1 - vertex_0);


                // Store the intersection point in the polygon
                slm[slm_offset + POLYGON_OFFSET + llid * 3 + i] = intersection;
               }


              }
              
            }




            
            


            // Move inward normals, edge vectors
            // Loop is over x,y,z
            for (size_t i = 0; i < 3; i++) {
              // Quantities that we have "4" of
              for (size_t llid = check_local_item_id; llid < 4;
                   llid += NUM_THREADS_PER_CHECK) {
                // Inward normals of element A
                slm[slm_offset + INWARD_NORMAL_A_OFFSET + llid * 3 + i] =
                    inward_normals_W_[A_element_index][llid][i];

                // Inward normals of element B
                slm[slm_offset + INWARD_NORMAL_B_OFFSET + llid * 3 + i] =
                    inward_normals_W_[B_element_index][llid][i];
              }
              // Quantity that we have "8" of - For now set all the verticies of
              // the polygon to double max so that we know all are stale
              for (size_t llid = check_local_item_id; llid < 8;
                   llid += NUM_THREADS_PER_CHECK) {
                // Polygon vertices
                slm[slm_offset + POLYGON_OFFSET + llid * 3 + i] =
                    std::numeric_limits<double>::max();
              }
            }
          });
    });

    sycl::free(narrow_phase_check_indices, q_device_);
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
  uint8_t* collision_filter_ = nullptr;  // collision_filter_[i]= 1 if the i-th
                                         // geometry is a collision candidate,
                                         // 0 otherwise
  size_t* geom_collision_filter_check_offsets_ = nullptr;
  size_t* geom_collision_filter_num_cols_ = nullptr;
  size_t total_checks_ = 0;

  // Internal use
  size_t* prefix_sum_ = nullptr;  // prefix_sum_[i] = prefix sum of the first i
                                  // elements of the collision filter

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
  std::vector<uint8_t> collision_filter_host(total_checks);
  auto q = impl->q_device_;
  auto collision_filter = impl->collision_filter_;
  q.memcpy(collision_filter_host.data(), collision_filter,
           total_checks * sizeof(uint8_t))
      .wait();
  return collision_filter_host;
}

std::vector<size_t> SyclProximityEngineAttorney::get_prefix_sum(
    SyclProximityEngine::Impl* impl) {
  size_t total_checks = SyclProximityEngineAttorney::get_total_checks(impl);
  std::vector<size_t> prefix_sum_host(total_checks);
  auto q = impl->q_device_;
  auto prefix_sum = impl->prefix_sum_;
  q.memcpy(prefix_sum_host.data(), prefix_sum, total_checks * sizeof(size_t))
      .wait();
  return prefix_sum_host;
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
