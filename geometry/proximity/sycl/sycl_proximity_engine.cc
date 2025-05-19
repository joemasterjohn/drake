#include "sycl_proximity_engine.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

bool SyclProximityEngine::is_available() {
  return sycl::queue::get_default_queue() != nullptr;
}

SyclProximityEngine::SyclProximityEngine(
    const std::unordered_map<GeometryId, SoftGeometry>& soft_geometries) {
  // Initialize SYCL queues
  q_device_ = sycl::queue(sycl::default_selector_v);
  q_host_ = sycl::queue(sycl::cpu_selector_v);

  // Get number of geometries
  num_geometries_ = soft_geometries.size();

  // Allocate device memory for lookup arrays
  sh_element_offsets_ = sycl::malloc_device<size_t>(num_geometries_, q_device_);
  sh_vertex_offsets_ = sycl::malloc_device<size_t>(num_geometries_, q_device_);
  sh_element_counts_ = sycl::malloc_device<size_t>(num_geometries_, q_device_);
  sh_vertex_counts_ = sycl::malloc_device<size_t>(num_geometries_, q_device_);
  // First compute totals and build lookup data
  size_t total_elements = 0;
  size_t total_vertices = 0;

  for (const auto& [id, soft_geometry] : soft_geometries) {
    DRAKE_THROW_UNLESS(soft_geometry.hydroelastic_type(id) ==
                       HydroelasticType::kSoft);

    const SoftMesh& soft_mesh = soft_geometry.soft_mesh();
    const VolumeMesh<double>& mesh = soft_mesh.mesh();

    // Store the geometry's ID
    soft_geometry_ids_[id] = id;

    // Store offsets and counts
    sh_element_offsets_[id] = total_elements;
    sh_vertex_offsets_[id] = total_vertices;

    const size_t num_elements = mesh.num_elements();
    const size_t num_vertices = mesh.num_vertices();
    sh_element_counts_[id] = num_elements;
    sh_vertex_counts_[id] = num_vertices;

    // Update totals
    total_elements += num_elements;
    total_vertices += num_vertices;
  }

  // Allocate device memory for all meshes
  elements_ =
      sycl::malloc_device<std::array<int, 4>>(total_elements, q_device_);
  vertices_M_ = sycl::malloc_device<Vector3<double>>(total_vertices, q_device_);
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
  vertices_W_ = sycl::malloc_device<Vector3<double>>(total_vertices, q_device_);
  inward_normals_W_ = sycl::malloc_device<std::array<Vector3<double>, 4>>(
      total_elements, q_device_);
  edge_vectors_W_ = sycl::malloc_device<std::array<Vector3<double>, 6>>(
      total_elements, q_device_);
  gradient_W_pressure_at_Wo_ =
      sycl::malloc_device<Vector4<double>>(total_elements, q_device_);

  // Copy data for each mesh
  int geo_idx = 0;
  std::vector<sycl::event> transfer_events;  // Store all transfer events

  for (const auto& [id, soft_geometry] : soft_geometries) {
    const SoftMesh& soft_mesh = soft_geometry.soft_mesh();
    const VolumeMesh<double>& mesh = soft_mesh.mesh();
    const VolumeMeshFieldLinear<double, double>& pressure_field =
        soft_mesh.pressure();

    const size_t element_offset = sh_element_offsets_[geo_idx];
    const size_t vertex_offset = sh_vertex_offsets_[geo_idx];
    const size_t num_elements = sh_element_counts_[geo_idx];
    const size_t num_vertices = sh_vertex_counts_[geo_idx];

    // Copy mesh data using the offsets with async operations
    transfer_events.push_back(q_device_.memcpy_async(
        elements_ + element_offset, mesh.pack_element_vertices().data(),
        num_elements * sizeof(std::array<int, 4>)));

    // Vertices
    transfer_events.push_back(q_device_.memcpy_async(
        vertices_M_ + vertex_offset, mesh.vertices().data(),
        num_vertices * sizeof(Vector3<double>)));

    // Inward Normals
    transfer_events.push_back(q_device_.memcpy_async(
        inward_normals_M_ + element_offset, mesh.inward_normals().data(),
        num_elements * sizeof(std::array<Vector3<double>, 4>)));

    // Edge Vectors
    transfer_events.push_back(q_device_.memcpy_async(
        edge_vectors_M_ + element_offset, mesh.edge_vectors().data(),
        num_elements * sizeof(std::array<Vector3<double>, 6>)));

    // Pressures
    transfer_events.push_back(q_device_.memcpy_async(
        pressures_ + element_offset, pressure_field.values().data(),
        num_elements * sizeof(double)));

    // Min Pressures
    transfer_events.push_back(q_device_.memcpy_async(
        min_pressures_ + element_offset, pressure_field.min_values().data(),
        num_elements * sizeof(double)));

    // Max Pressures
    transfer_events.push_back(q_device_.memcpy_async(
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
        q_device_.memcpy_async(gradient_M_pressure_at_Mo_ + element_offset,
                               packed_gradient_pressure.data(),
                               num_elements * sizeof(Vector4<double>)));

    geo_idx++;
  }

  // Wait for all transfers to complete before returning
  sycl::event::wait_and_throw(transfer_events);

  // Prefetch geometry lookup data to the device - We will only need it on
  // device from here on out
  q_device_.prefetch(sh_element_offsets_, num_geometries_ * sizeof(size_t));
  q_device_.prefetch(sh_vertex_offsets_, num_geometries_ * sizeof(size_t));
  q_device_.prefetch(sh_element_counts_, num_geometries_ * sizeof(size_t));
  q_device_.prefetch(sh_vertex_counts_, num_geometries_ * sizeof(size_t));
}

SyclProximityEngine::~SyclProximityEngine() {
  // TODO(huzaifa): Implement
}

SyclProximityEngine::SyclProximityEngine(const SyclProximityEngine& other) {
  // TODO(huzaifa): Implement
}

SyclProximityEngine& SyclProximityEngine::operator=(
    const SyclProximityEngine& other) {
  // TODO(huzaifa): Implement
  return *this;
}

std::vector<SYCLHydroelasticSurface>
SyclProximityEngine::ComputeSYCLHydroelasticSurface(
    const std::unordered_map<GeometryId, math::RigidTransform<double>>& X_WGs) {
  // TODO(huzaifa): Implement
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake
