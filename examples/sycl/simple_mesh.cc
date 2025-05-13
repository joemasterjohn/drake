#include "drake/examples/sycl/simple_mesh.h"
#include <sycl/sycl.hpp>

namespace drake {

// Constructor taking std::vector
SimpleMesh::SimpleMesh(const std::vector<Vector3<double>>& p_MV, 
                      const std::vector<int>& elements,
                      sycl::queue& q) {
    
    num_points_ = p_MV.size();
    num_elements_ = elements.size() / 4;  // Each tet has 4 vertices
    q_ = &q;  // Store queue for deallocation

    // Allocate USM memory for the vertex positions and copy data
    p_MV_ = static_cast<Vector3<double>*>(
        sycl::malloc_shared(num_points_ * sizeof(Vector3<double>), q));
    q.prefetch(p_MV_, num_points_ * sizeof(Vector3<double>));

    // Allocate USM memory for the elements and copy data
    elements_ = static_cast<int*>(
        sycl::malloc_shared(elements.size() * sizeof(int), q));
    // q.mem_advise(elements_, elements.size() * sizeof(int), sycl::property::memory_advice::read_only);
    q.prefetch(elements_, elements.size() * sizeof(int));
}

// Constructor taking raw pointers
SimpleMesh::SimpleMesh(const Vector3<double>* p_MV, 
                      const int* elements,
                      size_t num_points,
                      size_t num_elements,
                      sycl::queue& q) {
    
    num_points_ = num_points;
    num_elements_ = num_elements;
    q_ = &q;  // Store queue for deallocation

    // Allocate USM memory for the vertex positions and copy data
    p_MV_ = static_cast<Vector3<double>*>(
        sycl::malloc_shared(num_points_ * sizeof(Vector3<double>), q));
    q.prefetch(p_MV_, num_points_ * sizeof(Vector3<double>));
    
    // Allocate USM memory for the elements and copy data
    // 4 here is because each element is a tet and has 4 vertices
    elements_ = static_cast<int*>(
        sycl::malloc_shared(num_elements_ * 4 * sizeof(int), q));
    // q.mem_advise(elements_, num_elements_ * 4 * sizeof(int), sycl::property::memory_advice::read_only);
    q.prefetch(elements_, num_elements_ * 4 * sizeof(int));
}

SimpleMesh::~SimpleMesh() {
    // Use sycl::free instead of delete[] for USM memory
    sycl::free(p_MV_, *q_);
    sycl::free(elements_, *q_);
}

SimpleMesh::SimpleMesh(const SimpleMesh& other) {
    num_points_ = other.num_points_;
    num_elements_ = other.num_elements_;
    q_ = other.q_;
}

SimpleMesh::SimpleMesh(SimpleMesh&& other) {
    num_points_ = other.num_points_;
    num_elements_ = other.num_elements_;
    q_ = other.q_;
}



}  // namespace drake


