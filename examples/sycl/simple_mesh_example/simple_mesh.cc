#include "drake/examples/sycl/simple_mesh_example/simple_mesh.h"

#include <sycl/sycl.hpp>

namespace drake {

// Constructor taking pre-allocated USM memory pointers
SimpleMesh::SimpleMesh(Vector3<double>* p_MV, int* elements, size_t num_points,
                       size_t num_elements) {
  p_MV_ = p_MV;
  elements_ = elements;
  num_points_ = num_points;
  num_elements_ = num_elements;
}

// No destructor needed - memory management is handled by the user

SimpleMesh::SimpleMesh(const SimpleMesh& other) {
  p_MV_ = other.p_MV_;
  elements_ = other.elements_;
  num_points_ = other.num_points_;
  num_elements_ = other.num_elements_;
}

SimpleMesh::SimpleMesh(SimpleMesh&& other) {
  p_MV_ = other.p_MV_;
  elements_ = other.elements_;
  num_points_ = other.num_points_;
  num_elements_ = other.num_elements_;
}

SimpleMesh& SimpleMesh::operator=(const SimpleMesh& other) {
  p_MV_ = other.p_MV_;
  elements_ = other.elements_;
  num_points_ = other.num_points_;
  num_elements_ = other.num_elements_;
  return *this;
}

SimpleMesh& SimpleMesh::operator=(SimpleMesh&& other) {
  p_MV_ = other.p_MV_;
  elements_ = other.elements_;
  num_points_ = other.num_points_;
  num_elements_ = other.num_elements_;
  return *this;
}

}  // namespace drake
