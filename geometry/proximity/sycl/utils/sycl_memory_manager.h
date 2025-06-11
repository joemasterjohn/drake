#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <sycl/sycl.hpp>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Helper class to manage SYCL device memory allocations and transfers
class SyclMemoryManager {
 public:
  explicit SyclMemoryManager(sycl::queue& queue) : queue_(queue) {}

  // Allocate device memory for basic types
  template <typename T>
  inline T* AllocateDevice(size_t count) {
    return sycl::malloc_device<T>(count, queue_);
  }

  // Allocate host-accessible memory for basic types
  template <typename T>
  inline T* AllocateHost(size_t count) {
    return sycl::malloc_host<T>(count, queue_);
  }

  // Free device/host memory
  template <typename T>
  inline void Free(T* ptr) {
    if (ptr != nullptr) {
      sycl::free(ptr, queue_);
    }
  }

  // Copy data from host to device
  template <typename T>
  inline sycl::event CopyToDevice(T* device_ptr, const T* host_ptr,
                                  size_t count) {
    return queue_.memcpy(device_ptr, host_ptr, count * sizeof(T));
  }

  // Copy data from device to host
  template <typename T>
  inline sycl::event CopyToHost(T* host_ptr, const T* device_ptr,
                                size_t count) {
    return queue_.memcpy(host_ptr, device_ptr, count * sizeof(T));
  }

  // Fill device memory with a value
  template <typename T>
  inline sycl::event Fill(T* device_ptr, const T& value, size_t count) {
    return queue_.fill(device_ptr, value, count);
  }

  // Memset device memory to zero
  template <typename T>
  inline sycl::event Memset(T* device_ptr, size_t count) {
    return queue_.memset(device_ptr, 0, count * sizeof(T));
  }

 private:
  sycl::queue& queue_;
};

// Structure to hold all device memory pointers for mesh data
struct DeviceMeshData {
  // Element data
  std::array<int, 4>* elements = nullptr;
  size_t* element_mesh_ids = nullptr;
  std::array<Vector3<double>, 4>* inward_normals_M = nullptr;
  std::array<Vector3<double>, 4>* inward_normals_W = nullptr;
  double* min_pressures = nullptr;
  double* max_pressures = nullptr;
  Vector4<double>* gradient_M_pressure_at_Mo = nullptr;
  Vector4<double>* gradient_W_pressure_at_Wo = nullptr;
  Vector3<double>* element_aabb_min_W = nullptr;
  Vector3<double>* element_aabb_max_W = nullptr;

  // Vertex data
  Vector3<double>* vertices_M = nullptr;
  Vector3<double>* vertices_W = nullptr;
  double* pressures = nullptr;
  size_t* vertex_mesh_ids = nullptr;

  // Lookup arrays (host accessible)
  size_t* element_offsets = nullptr;
  size_t* vertex_offsets = nullptr;
  size_t* element_counts = nullptr;
  size_t* vertex_counts = nullptr;
  GeometryId* geometry_ids = nullptr;
  double* transforms = nullptr;
};

// Structure to hold collision detection memory
struct DeviceCollisionData {
  // Broad phase data
  uint8_t* collision_filter = nullptr;
  size_t* collision_filter_host_body_index = nullptr;
  size_t* total_checks_per_geometry = nullptr;
  size_t* geom_collision_filter_num_cols = nullptr;
  size_t* geom_collision_filter_check_offsets = nullptr;
  size_t* prefix_sum_total_checks = nullptr;

  // Narrow phase data
  size_t* narrow_phase_check_indices = nullptr;
  uint8_t* narrow_phase_check_validity = nullptr;
  size_t* prefix_sum_narrow_phase_checks = nullptr;
};

// Structure to hold polygon data memory
struct DevicePolygonData {
  // Raw polygon data
  double* polygon_areas = nullptr;
  Vector3<double>* polygon_centroids = nullptr;
  Vector3<double>* polygon_normals = nullptr;
  double* polygon_g_M = nullptr;
  double* polygon_g_N = nullptr;
  double* polygon_pressure_W = nullptr;
  GeometryId* polygon_geom_index_A = nullptr;
  GeometryId* polygon_geom_index_B = nullptr;

  // Compacted polygon data
  double* compacted_polygon_areas = nullptr;
  Vector3<double>* compacted_polygon_centroids = nullptr;
  Vector3<double>* compacted_polygon_normals = nullptr;
  double* compacted_polygon_g_M = nullptr;
  double* compacted_polygon_g_N = nullptr;
  double* compacted_polygon_pressure_W = nullptr;
  GeometryId* compacted_polygon_geom_index_A = nullptr;
  GeometryId* compacted_polygon_geom_index_B = nullptr;

  size_t* valid_polygon_indices = nullptr;

  // Debug data
  double* debug_polygon_vertices = nullptr;
};

// Helper functions for memory allocation and initialization
class SyclMemoryHelper {
 public:
  // Allocate all mesh-related device memory
  static inline void AllocateMeshMemory(SyclMemoryManager& mem_mgr,
                                        DeviceMeshData& mesh_data,
                                        size_t num_geometries) {
    // Allocate lookup arrays (host accessible)
    mesh_data.element_offsets = mem_mgr.AllocateHost<size_t>(num_geometries);
    mesh_data.vertex_offsets = mem_mgr.AllocateHost<size_t>(num_geometries);
    mesh_data.element_counts = mem_mgr.AllocateHost<size_t>(num_geometries);
    mesh_data.vertex_counts = mem_mgr.AllocateHost<size_t>(num_geometries);
    mesh_data.geometry_ids = mem_mgr.AllocateHost<GeometryId>(num_geometries);
    mesh_data.transforms = mem_mgr.AllocateHost<double>(num_geometries * 12);
  }

  static inline void AllocateMeshElementVerticesMemory(
      SyclMemoryManager& mem_mgr, DeviceMeshData& mesh_data,
      size_t total_elements, size_t total_vertices) {
    // Allocate element data
    mesh_data.elements =
        mem_mgr.AllocateDevice<std::array<int, 4>>(total_elements);
    mesh_data.element_mesh_ids = mem_mgr.AllocateDevice<size_t>(total_elements);
    mesh_data.inward_normals_M =
        mem_mgr.AllocateDevice<std::array<Vector3<double>, 4>>(total_elements);
    mesh_data.inward_normals_W =
        mem_mgr.AllocateDevice<std::array<Vector3<double>, 4>>(total_elements);
    mesh_data.min_pressures = mem_mgr.AllocateDevice<double>(total_elements);
    mesh_data.max_pressures = mem_mgr.AllocateDevice<double>(total_elements);
    mesh_data.gradient_M_pressure_at_Mo =
        mem_mgr.AllocateDevice<Vector4<double>>(total_elements);
    mesh_data.gradient_W_pressure_at_Wo =
        mem_mgr.AllocateDevice<Vector4<double>>(total_elements);
    mesh_data.element_aabb_min_W =
        mem_mgr.AllocateDevice<Vector3<double>>(total_elements);
    mesh_data.element_aabb_max_W =
        mem_mgr.AllocateDevice<Vector3<double>>(total_elements);

    // Allocate vertex data
    mesh_data.vertices_M =
        mem_mgr.AllocateDevice<Vector3<double>>(total_vertices);
    mesh_data.vertices_W =
        mem_mgr.AllocateDevice<Vector3<double>>(total_vertices);
    mesh_data.pressures = mem_mgr.AllocateDevice<double>(total_vertices);
    mesh_data.vertex_mesh_ids = mem_mgr.AllocateDevice<size_t>(total_vertices);
  }

  // Allocate collision detection memory
  static inline void AllocateCollisionMemory(
      SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
      size_t num_geometries, size_t total_checks,
      size_t estimated_narrow_phase_checks) {
    // Broad phase data
    collision_data.collision_filter =
        mem_mgr.AllocateDevice<uint8_t>(total_checks);
    collision_data.collision_filter_host_body_index =
        mem_mgr.AllocateHost<size_t>(total_checks);
    collision_data.total_checks_per_geometry =
        mem_mgr.AllocateHost<size_t>(num_geometries);
    collision_data.geom_collision_filter_num_cols =
        mem_mgr.AllocateHost<size_t>(num_geometries);
    collision_data.geom_collision_filter_check_offsets =
        mem_mgr.AllocateHost<size_t>(num_geometries);
    collision_data.prefix_sum_total_checks =
        mem_mgr.AllocateDevice<size_t>(total_checks);

    // Narrow phase data
    collision_data.narrow_phase_check_indices =
        mem_mgr.AllocateDevice<size_t>(estimated_narrow_phase_checks);
    collision_data.narrow_phase_check_validity =
        mem_mgr.AllocateDevice<uint8_t>(estimated_narrow_phase_checks);
    collision_data.prefix_sum_narrow_phase_checks =
        mem_mgr.AllocateDevice<size_t>(estimated_narrow_phase_checks);
  }

  // Allocate polygon memory
  static inline void AllocateFullPolygonMemory(
      SyclMemoryManager& mem_mgr, DevicePolygonData& polygon_data,
      size_t estimated_narrow_phase_checks) {
    // Raw polygon data
    polygon_data.polygon_areas =
        mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
    polygon_data.polygon_centroids =
        mem_mgr.AllocateDevice<Vector3<double>>(estimated_narrow_phase_checks);
    polygon_data.polygon_normals =
        mem_mgr.AllocateDevice<Vector3<double>>(estimated_narrow_phase_checks);
    polygon_data.polygon_g_M =
        mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
    polygon_data.polygon_g_N =
        mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
    polygon_data.polygon_pressure_W =
        mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
    polygon_data.polygon_geom_index_A =
        mem_mgr.AllocateDevice<GeometryId>(estimated_narrow_phase_checks);
    polygon_data.polygon_geom_index_B =
        mem_mgr.AllocateDevice<GeometryId>(estimated_narrow_phase_checks);
  }

  static inline void AllocateCompactPolygonMemory(
      SyclMemoryManager& mem_mgr, DevicePolygonData& polygon_data,
      size_t estimated_polygons) {
    // Compacted polygon data
    polygon_data.compacted_polygon_areas =
        mem_mgr.AllocateDevice<double>(estimated_polygons);
    polygon_data.compacted_polygon_centroids =
        mem_mgr.AllocateDevice<Vector3<double>>(estimated_polygons);
    polygon_data.compacted_polygon_normals =
        mem_mgr.AllocateDevice<Vector3<double>>(estimated_polygons);
    polygon_data.compacted_polygon_g_M =
        mem_mgr.AllocateDevice<double>(estimated_polygons);
    polygon_data.compacted_polygon_g_N =
        mem_mgr.AllocateDevice<double>(estimated_polygons);
    polygon_data.compacted_polygon_pressure_W =
        mem_mgr.AllocateDevice<double>(estimated_polygons);
    polygon_data.compacted_polygon_geom_index_A =
        mem_mgr.AllocateDevice<GeometryId>(estimated_polygons);
    polygon_data.compacted_polygon_geom_index_B =
        mem_mgr.AllocateDevice<GeometryId>(estimated_polygons);

    polygon_data.valid_polygon_indices =
        mem_mgr.AllocateDevice<size_t>(estimated_polygons);
  }

  // Free all mesh memory
  static inline void FreeMeshMemory(SyclMemoryManager& mem_mgr,
                                    DeviceMeshData& mesh_data) {
    // Element data
    mem_mgr.Free(mesh_data.elements);
    mem_mgr.Free(mesh_data.element_mesh_ids);
    mem_mgr.Free(mesh_data.inward_normals_M);
    mem_mgr.Free(mesh_data.inward_normals_W);
    mem_mgr.Free(mesh_data.min_pressures);
    mem_mgr.Free(mesh_data.max_pressures);
    mem_mgr.Free(mesh_data.gradient_M_pressure_at_Mo);
    mem_mgr.Free(mesh_data.gradient_W_pressure_at_Wo);
    mem_mgr.Free(mesh_data.element_aabb_min_W);
    mem_mgr.Free(mesh_data.element_aabb_max_W);

    // Vertex data
    mem_mgr.Free(mesh_data.vertices_M);
    mem_mgr.Free(mesh_data.vertices_W);
    mem_mgr.Free(mesh_data.pressures);
    mem_mgr.Free(mesh_data.vertex_mesh_ids);

    // Lookup arrays
    mem_mgr.Free(mesh_data.element_offsets);
    mem_mgr.Free(mesh_data.vertex_offsets);
    mem_mgr.Free(mesh_data.element_counts);
    mem_mgr.Free(mesh_data.vertex_counts);
    mem_mgr.Free(mesh_data.geometry_ids);
    mem_mgr.Free(mesh_data.transforms);
  }

  // Free collision memory
  static inline void FreeCollisionMemory(SyclMemoryManager& mem_mgr,
                                         DeviceCollisionData& collision_data) {
    mem_mgr.Free(collision_data.collision_filter);
    mem_mgr.Free(collision_data.collision_filter_host_body_index);
    mem_mgr.Free(collision_data.total_checks_per_geometry);
    mem_mgr.Free(collision_data.geom_collision_filter_num_cols);
    mem_mgr.Free(collision_data.geom_collision_filter_check_offsets);
    mem_mgr.Free(collision_data.prefix_sum_total_checks);
    mem_mgr.Free(collision_data.narrow_phase_check_indices);
    mem_mgr.Free(collision_data.narrow_phase_check_validity);
    mem_mgr.Free(collision_data.prefix_sum_narrow_phase_checks);
  }

  // Free polygon memory
  static inline void FreeFullPolygonMemory(SyclMemoryManager& mem_mgr,
                                           DevicePolygonData& polygon_data) {
    // Raw polygon data
    mem_mgr.Free(polygon_data.polygon_areas);
    mem_mgr.Free(polygon_data.polygon_centroids);
    mem_mgr.Free(polygon_data.polygon_normals);
    mem_mgr.Free(polygon_data.polygon_g_M);
    mem_mgr.Free(polygon_data.polygon_g_N);
    mem_mgr.Free(polygon_data.polygon_pressure_W);
    mem_mgr.Free(polygon_data.polygon_geom_index_A);
    mem_mgr.Free(polygon_data.polygon_geom_index_B);

    // Debug data
    mem_mgr.Free(polygon_data.debug_polygon_vertices);
  }

  static inline void FreeCompactPolygonMemory(SyclMemoryManager& mem_mgr,
                                              DevicePolygonData& polygon_data) {
    // Compacted polygon data
    mem_mgr.Free(polygon_data.compacted_polygon_areas);
    mem_mgr.Free(polygon_data.compacted_polygon_centroids);
    mem_mgr.Free(polygon_data.compacted_polygon_normals);
    mem_mgr.Free(polygon_data.compacted_polygon_g_M);
    mem_mgr.Free(polygon_data.compacted_polygon_g_N);
    mem_mgr.Free(polygon_data.compacted_polygon_pressure_W);
    mem_mgr.Free(polygon_data.compacted_polygon_geom_index_A);
    mem_mgr.Free(polygon_data.compacted_polygon_geom_index_B);
    mem_mgr.Free(polygon_data.valid_polygon_indices);
  }

  static inline void FreePolygonMemory(SyclMemoryManager& mem_mgr,
                                       DevicePolygonData& polygon_data) {
    FreeFullPolygonMemory(mem_mgr, polygon_data);
    FreeCompactPolygonMemory(mem_mgr, polygon_data);
  }
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake