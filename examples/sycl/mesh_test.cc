#include "drake/examples/sycl/simple_mesh.h"
#include <fmt/format.h>
#include <random>
#include "drake/math/rigid_transform.h"

using namespace drake;
using namespace drake::math;

// Populate allocated USM memory with random positions
void InitializeRandomPositions(Vector3<double>* p_MV, size_t num_points) {
    for (size_t i = 0; i < num_points; i++) {
        p_MV[i] = Vector3<double>(rand() % 100, rand() % 100, rand() % 100);
    }
}

// Populate allocated USM memory with random elements
void InitializeRandomElements(int* elements, size_t num_elements) {
    // Assuming 4 vertices per element (tet)
    for (size_t i = 0; i < num_elements * 4; i++) {
        elements[i] = rand() % 100;
    }
}

// Construct a bunch of simple meshes and do transforms on them using SYCL
int main() {
    sycl::queue q(sycl::gpu_selector_v);
    
    // Create 5 meshes each with 100 vertices and 25 elements (each with 4 vertices)
    size_t num_meshes = 5;
    size_t points_per_mesh = 100;
    size_t elements_per_mesh = 100;
    
    // Allocate memory for meshes using USM
    SimpleMesh* meshes = sycl::malloc_shared<SimpleMesh>(num_meshes, q);
    
    // Allocate memory for transforms
    RigidTransformd* X_MBs = sycl::malloc_shared<RigidTransformd>(num_meshes, q);
    
    // Allocate memory for vertices and elements directly using USM
    Vector3<double>** vertex_arrays = sycl::malloc_shared<Vector3<double>*>(num_meshes, q);
    int** element_arrays = sycl::malloc_shared<int*>(num_meshes, q);
    
    // Initialize data and create meshes with pre-allocated memory
    for (size_t i = 0; i < num_meshes; i++) {
        // Allocate memory for vertices
        vertex_arrays[i] = sycl::malloc_shared<Vector3<double>>(points_per_mesh, q);
        InitializeRandomPositions(vertex_arrays[i], points_per_mesh);
        
        // Allocate memory for elements (4 vertices per element)
        element_arrays[i] = sycl::malloc_shared<int>(elements_per_mesh * 4, q);
        InitializeRandomElements(element_arrays[i], elements_per_mesh);
        
        // Create mesh with pre-allocated memory
        meshes[i] = SimpleMesh(vertex_arrays[i], element_arrays[i], 
                               points_per_mesh, elements_per_mesh);
        
        // Initialize transform
        X_MBs[i] = RigidTransformd::Identity();
    }

    // Transform the meshes with a kernel
    // Parallelize across the meshes
    sycl::range<1> num_items{num_meshes};
    
    auto e = q.parallel_for(num_items, [=](auto idx) { 
        // Get the mesh and transform
        SimpleMesh& mesh = meshes[idx];
        RigidTransformd& X_MB = X_MBs[idx];
        // Apply the transform to the mesh
        for (size_t i = 0; i < mesh.num_points(); ++i) {
            mesh.p_MV()[i] = X_MB * mesh.p_MV()[i];
        }     
    });

    e.wait();

    // Deallocate memory in reverse order of allocation
    for (size_t i = 0; i < num_meshes; i++) {
        sycl::free(element_arrays[i], q);
        sycl::free(vertex_arrays[i], q);
    }
    sycl::free(element_arrays, q);
    sycl::free(vertex_arrays, q);
    sycl::free(X_MBs, q);
    sycl::free(meshes, q);
}
