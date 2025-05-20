#include "drake/geometry/proximity/sycl/sycl_proximity_engine.h"

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
    q_device_ = sycl::queue(sycl::default_selector_v);
    q_host_ = sycl::queue(sycl::cpu_selector_v);

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
    size_t total_elements = 0;
    size_t total_vertices = 0;

    size_t id_index = 0;
    for (const auto& [id, soft_geometry] : soft_geometries) {
      const hydroelastic::SoftMesh& soft_mesh = soft_geometry.soft_mesh();
      const VolumeMesh<double>& mesh = soft_mesh.mesh();

      // Store the geometry's ID
      soft_geometry_ids_[id_index] = id;

      // Store offsets and counts directly (no memcpy needed with shared memory)
      sh_element_offsets_[id_index] = total_elements;
      sh_vertex_offsets_[id_index] = total_vertices;

      const size_t num_elements = mesh.num_elements();
      const size_t num_vertices = mesh.num_vertices();
      sh_element_counts_[id_index] = num_elements;
      sh_vertex_counts_[id_index] = num_vertices;

      // Update totals
      total_elements += num_elements;
      total_vertices += num_vertices;
      id_index++;
    }

    // Allocate device memory for all meshes
    elements_ =
        sycl::malloc_device<std::array<int, 4>>(total_elements, q_device_);
    vertices_M_ =
        sycl::malloc_device<Vector3<double>>(total_vertices, q_device_);
    inward_normals_M_ = sycl::malloc_device<std::array<Vector3<double>, 4>>(
        total_elements, q_device_);
    edge_vectors_M_ = sycl::malloc_device<std::array<Vector3<double>, 6>>(
        total_elements, q_device_);
    pressures_ = sycl::malloc_device<double>(total_elements, q_device_);
    min_pressures_ = sycl::malloc_device<double>(total_elements, q_device_);
    max_pressures_ = sycl::malloc_device<double>(total_elements, q_device_);

    // Allocate combined gradient and pressure arrays
    gradient_M_pressure_at_Mo_ =
        sycl::malloc_device<Vector4<double>>(total_elements, q_device_);

    // Allocate even for world frame quantities
    vertices_W_ =
        sycl::malloc_device<Vector3<double>>(total_vertices, q_device_);
    inward_normals_W_ = sycl::malloc_device<std::array<Vector3<double>, 4>>(
        total_elements, q_device_);
    edge_vectors_W_ = sycl::malloc_device<std::array<Vector3<double>, 6>>(
        total_elements, q_device_);
    gradient_W_pressure_at_Wo_ =
        sycl::malloc_device<Vector4<double>>(total_elements, q_device_);

    // Allocate device memory for transforms
    transforms_ =
        sycl::malloc_host<double>(num_geometries_ * transform_size_, q_device_);

    // Copy data for each mesh
    id_index = 0;
    std::vector<sycl::event> transfer_events;  // Store all transfer events

    for (const auto& [id, soft_geometry] : soft_geometries) {
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

      // Vertices
      transfer_events.push_back(
          q_device_.memcpy(vertices_M_ + vertex_offset, mesh.vertices().data(),
                           num_vertices * sizeof(Vector3<double>)));

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

      id_index++;
    }

    // Wait for all transfers to complete before returning
    sycl::event::wait_and_throw(transfer_events);
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
    size_t geom_index = 0;
    for (auto geometry_id : soft_geometry_ids_) {
      // To maintain our orders of geometries we need to loop through the store
      // geometry id's and query the X_WGs for that geometry id Cannot iterate
      // over the unordered_map because it is not ordered
      const auto& X_WG = X_WGs.at(geometry_id);
      const auto& transform = X_WG.GetAsMatrix34();
      for (size_t i = 0; i < transform_size_; ++i) {
        size_t row = i / 4;
        size_t col = i % 4;
        // Store transforms in row major order
        // transforms_ = [R_00, R_01, R_02, p_0, R_10, R_11, R_12, p_1, ...]
        transforms_[geom_index * transform_size_ + i] = transform(row, col);
      }
      geom_index++;
    }

    // ========================================
    // Kernel 1: Transform quantities to world frame
    // ========================================
    // Transform all verticies
    // auto transform_event = q_device_.submit([&](sycl::handler& h) {
    //   h.parallel_for(sycl::range<1>(num_geometries_ * num_vertices_),
    //                  [=](sycl::id<1> idx) {
    //                    const size_t vertex_index = idx[0];
    //                    const size_t mesh_index =
    //                        vertex_index / sh_vertex_counts_[];

    //                    const Vector3<double> vertex =
    //                    vertices_M_[vertex_index]; double T[transform_size_];
    //                    for (size_t i = 0; i < transform_size_; ++i) {
    //                      T[i] = transforms_[mesh_index * transform_size_ +
    //                      i];
    //                    }
    //                    double new_x = T[0] * x + T[1] * y + T[2] * z + T[3];
    //                    double new_y = T[4] * x + T[5] * y + T[6] * z + T[7];
    //                    double new_z = T[8] * x + T[9] * y + T[10] * z +
    //                    T[11];
    //                  });
    // });

    // ========================================
    // Kernel 2: Generate candidate tet pairs
    // ========================================

    // Placeholder that returns an empty vector
    std::vector<SYCLHydroelasticSurface> sycl_hydroelastic_surfaces;
    return sycl_hydroelastic_surfaces;
  }

 private:
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
  // Get the transforms in just a simple double*
  constexpr size_t transform_size_ = 12;

  // SYCL shared arrays for geometry lookup
  size_t* sh_element_offsets_ = nullptr;  // Element offset for each geometry
  size_t* sh_vertex_offsets_ = nullptr;   // Vertex offset for each geometry
  size_t* sh_element_counts_ = nullptr;  // Number of elements for each geometry
  size_t* sh_vertex_counts_ = nullptr;   // Number of vertices for each geometry

  /*
  A hydroelastic geometry contains one mesh. Elements are tetrahedra.
  All data is stored in contiguous arrays, with each geometry's data
  at a specific offset in these arrays.
  */

  // Mesh element data - accessed by element_offset + local_element_index
  std::array<int, 4>* elements_ = nullptr;  // Elements as 4 vertex indices
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
};

// SyclProximityEngine implementation that forwards to the Impl class

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

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
