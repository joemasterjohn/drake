#include "drake/common/eigen_types.h"
#include <sycl/sycl.hpp>
#include <vector>

namespace drake {
// A simple mesh class that contains all the information about the mesh 
// necessary on the GPU for narrow and broad phase collision detection.

// Contract:
// Class does NOT allocate or deallocate USM memory.
// The user is responsible for allocating and deallocating USM memory and
// passing pointers to this class.
class SimpleMesh {
    private:
        Vector3<double>* p_MV_; // vertex positions in frame M (USM memory)
        int* elements_; // elements[4*i,....,4*i + 3] are tets (USM memory)
        size_t num_points_; // number of vertices
        size_t num_elements_; // number of tets
    public:
        // Constructor taking pre-allocated USM memory pointers
        SimpleMesh(Vector3<double>* p_MV, 
                  int* elements,
                  size_t num_points,
                  size_t num_elements);
        
        // Default destructor (no memory deallocation)
        ~SimpleMesh() = default;

        // Copy assignment operator
        SimpleMesh& operator=(const SimpleMesh& other);

        // Move assignment operator
        SimpleMesh& operator=(SimpleMesh&& other);

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
