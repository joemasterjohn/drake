#include "drake/examples/sycl/simple_mesh.h"
#include <fmt/format.h>
#include <random>
#include "drake/math/rigid_transform.h"

using namespace drake;
using namespace drake::math;
void InitializeRandomPositions(std::vector<Vector3<double>>& p_MV) {
    for (size_t i = 0; i < p_MV.size(); i++) {
        p_MV[i] = Vector3<double>(rand() % 100, rand() % 100, rand() % 100);
    }
}

void InitializeRandomElements(std::vector<int>& elements) {
    for (size_t i = 0; i < elements.size(); i++) {
        elements[i] = rand() % 100;
    }
}

// Construct a bunch of simple meshes and do transforms on them using SYCL
int main() {

    sycl::queue q(sycl::cpu_selector_v);
    // Create 5 meshes each with 100 vertices and 100 elements
    size_t num_meshes = 5;
    // Allocate memory for these meshes using USM
    SimpleMesh* meshes = sycl::malloc_shared<SimpleMesh>(num_meshes, q);
    // Each mesh will undergo a rigid transform
    RigidTransformd* X_MBs = sycl::malloc_shared<RigidTransformd>(num_meshes, q);

    // Store transformed output meshes
    SimpleMesh* output_meshes = sycl::malloc_shared<SimpleMesh>(num_meshes, q);
    for (size_t i = 0; i < num_meshes; i++) {   
        std::vector<Vector3<double>> p_MV(100);
        InitializeRandomPositions(p_MV);

        std::vector<int> elements(100);
        InitializeRandomElements(elements);

        // Class private members within SimpleMesh are allocated using USM
        SimpleMesh mesh(p_MV, elements, q);
        meshes[i] = mesh;
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

    

    // Deallocate memory
    sycl::free(meshes, q);
    sycl::free(X_MBs, q);
    sycl::free(output_meshes, q);



}
