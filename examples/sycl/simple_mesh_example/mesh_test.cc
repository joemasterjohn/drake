#include "drake/examples/sycl/simple_mesh_example/simple_mesh.h"
#include <fmt/format.h>
#include <random>
#include <chrono>
#include <vector>
#include <numeric>
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

// Run the mesh transformation benchmark with the given queue
std::pair<double, double> runBenchmarkOnce(sycl::queue& q, const std::string& device_name, 
                                         size_t num_meshes, size_t points_per_mesh, 
                                         size_t elements_per_mesh, bool print_results = true) {
    auto start = std::chrono::high_resolution_clock::now();
    
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
    
    auto kernel_start = std::chrono::high_resolution_clock::now();
    
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
    
    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_duration = kernel_end - kernel_start;

    // Deallocate memory in reverse order of allocation
    for (size_t i = 0; i < num_meshes; i++) {
        sycl::free(element_arrays[i], q);
        sycl::free(vertex_arrays[i], q);
    }
    sycl::free(element_arrays, q);
    sycl::free(vertex_arrays, q);
    sycl::free(X_MBs, q);
    sycl::free(meshes, q);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = end - start;
    
    if (print_results) {
        fmt::print("  Run results:\n");
        fmt::print("    - Kernel execution time: {:.3f} ms\n", kernel_duration.count());
        fmt::print("    - Total execution time:  {:.3f} ms\n", total_duration.count());
    }
    
    return {kernel_duration.count(), total_duration.count()};
}

void runBenchmark(sycl::queue& q, const std::string& device_name, 
                size_t num_meshes, size_t points_per_mesh, size_t elements_per_mesh, 
                int num_runs = 5) {
    fmt::print("{} benchmark:\n", device_name);
    
    std::vector<double> kernel_times;
    std::vector<double> total_times;
    
    // Add one extra run as warm-up but print all runs
    int total_runs = num_runs + 1;
    
    for (int i = 0; i < total_runs; i++) {
        fmt::print("  Run {} of {} {}:\n", i + 1, total_runs, 
                 (i == 0 ? "(warm-up, excluded from average)" : ""));
        auto [kernel_time, total_time] = runBenchmarkOnce(q, device_name, 
                                                       num_meshes, points_per_mesh, 
                                                       elements_per_mesh);
        
        // Only include the last num_runs (exclude the first/warm-up run)
        if (i > 0) {
            kernel_times.push_back(kernel_time);
            total_times.push_back(total_time);
        }
    }
    
    // Calculate averages (only from runs after the warm-up)
    double avg_kernel_time = std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0) / num_runs;
    double avg_total_time = std::accumulate(total_times.begin(), total_times.end(), 0.0) / num_runs;
    
    fmt::print("\n{} AVERAGE RESULTS (over {} runs, excluding warm-up):\n", device_name, num_runs);
    fmt::print("  - Average kernel execution time: {:.3f} ms\n", avg_kernel_time);
    fmt::print("  - Average total execution time:  {:.3f} ms\n\n", avg_total_time);
}

// Construct a bunch of simple meshes and do transforms on them using SYCL
int main() {
    size_t num_meshes = 1000;
    size_t points_per_mesh = 100;
    size_t elements_per_mesh = 100;
    int num_runs = 5;  // This is the number of runs to include in average (excluding warm-up)
    
    try {
        fmt::print("Running on {} meshes with {} points per mesh and {} elements per mesh\n", 
                 num_meshes, points_per_mesh, elements_per_mesh);
        fmt::print("Each benchmark will run {} times plus one warm-up run (total: {})\n", 
                 num_runs, num_runs + 1);
        fmt::print("Warm-up run will be excluded from average calculations\n\n");
        
        // Create GPU queue
        sycl::queue gpu_queue(sycl::gpu_selector_v);
        fmt::print("Running on GPU: {}\n", 
                 gpu_queue.get_device().get_info<sycl::info::device::name>());
        
        // Create CPU queue
        sycl::queue cpu_queue(sycl::cpu_selector_v);
        fmt::print("Running on CPU: {}\n", 
                 cpu_queue.get_device().get_info<sycl::info::device::name>());
        
        fmt::print("\nRunning performance comparison...\n\n");
        
        // Run benchmarks
        runBenchmark(gpu_queue, "GPU", num_meshes, points_per_mesh, elements_per_mesh, num_runs);
        runBenchmark(cpu_queue, "CPU", num_meshes, points_per_mesh, elements_per_mesh, num_runs);
        
        return 0;
    } catch (sycl::exception const& e) {
        fmt::print("SYCL exception caught: {}\n", e.what());
        return 1;
    } catch (std::exception const& e) {
        fmt::print("Standard exception caught: {}\n", e.what());
        return 1;
    }
}
