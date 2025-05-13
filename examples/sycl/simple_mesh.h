#include "drake/common/eigen_types.h"
#include <sycl/sycl.hpp>
#include <vector>

// Define SYCL_EXTERNAL if not already defined
#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL [[sycl::external]]
#endif

namespace drake {
// A simple mesh class that contains all the information about the mesh 
// necessary on the GPU for narrow and broad phase collision detection.

// Contract:
// Class allocates USM memory using malloc_shared and copies data provided by user.
// The destructor frees this memory using sycl::free.
class SimpleMesh {
    private:
        Vector3<double>* p_MV_; // vertex positions in frame M (USM memory)
        int* elements_; // elements[4*i,....,4*i + 3] are tets (USM memory)
        size_t num_points_; // number of vertices
        size_t num_elements_; // number of tets
        sycl::queue* q_; // Store queue for deallocation
    public:
        // Constructor taking std::vector
        SimpleMesh(const std::vector<Vector3<double>>& p_MV, 
                  const std::vector<int>& elements,
                  sycl::queue& q);
        
        // Constructor taking raw pointers
        SimpleMesh(const Vector3<double>* p_MV, 
                  const int* elements,
                  size_t num_points,
                  size_t num_elements,
                  sycl::queue& q);
                  
        ~SimpleMesh();

        // // Copy assignment operator
        SYCL_EXTERNAL SimpleMesh& operator=(const SimpleMesh& other);

        // Move assignment operator
        SYCL_EXTERNAL SimpleMesh& operator=(SimpleMesh&& other);

        // Copy constructor
        SimpleMesh(const SimpleMesh& other);

        // Move constructor
        SimpleMesh(SimpleMesh&& other);

        // Getters
        size_t num_points() const { return num_points_; }
        size_t num_elements() const { return num_elements_; }
        Vector3<double>* p_MV() const { return p_MV_; }
        int* elements() const { return elements_; }
};
}  // namespace drake
