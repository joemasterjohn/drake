#include "drake/examples/sycl/simple_mesh_example/simple_mesh.h"
#include <fmt/format.h>
#include <random>
#include <chrono>
#include <vector>
#include <numeric>
#include <fstream>
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
                                         size_t elements_per_mesh, bool print_results = false) {
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

std::pair<double, double> runBenchmark(sycl::queue& q, const std::string& device_name, 
                size_t num_meshes, size_t points_per_mesh, size_t elements_per_mesh, 
                int num_runs = 5, bool print_details = false) {
    if (print_details) {
        fmt::print("{} benchmark for meshes={}, points={}, elements={}:\n", 
                  device_name, num_meshes, points_per_mesh, elements_per_mesh);
    }
    
    std::vector<double> kernel_times;
    std::vector<double> total_times;
    
    // Add one extra run as warm-up but print all runs
    int total_runs = num_runs + 1;
    
    for (int i = 0; i < total_runs; i++) {
        if (print_details) {
            fmt::print("  Run {} of {} {}:\n", i + 1, total_runs, 
                     (i == 0 ? "(warm-up, excluded from average)" : ""));
        }
        auto [kernel_time, total_time] = runBenchmarkOnce(q, device_name, 
                                                        num_meshes, points_per_mesh, 
                                                        elements_per_mesh, print_details);
        
        // Only include the last num_runs (exclude the first/warm-up run)
        if (i > 0) {
            kernel_times.push_back(kernel_time);
            total_times.push_back(total_time);
        }
    }
    
    // Calculate averages (only from runs after the warm-up)
    double avg_kernel_time = std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0) / num_runs;
    double avg_total_time = std::accumulate(total_times.begin(), total_times.end(), 0.0) / num_runs;
    
    if (print_details) {
        fmt::print("\n{} AVERAGE RESULTS (over {} runs, excluding warm-up):\n", device_name, num_runs);
        fmt::print("  - Average kernel execution time: {:.3f} ms\n", avg_kernel_time);
        fmt::print("  - Average total execution time:  {:.3f} ms\n\n", avg_total_time);
    }
    
    return {avg_kernel_time, avg_total_time};
}

// Construct a bunch of simple meshes and do transforms on them using SYCL
int main(int argc, char** argv) {
    const int num_runs = 5;  // Constant number of runs for all benchmarks
    
    // Powers of 10 from 1 to 1,000,000
    std::vector<size_t> parameter_values = {1, 10, 100, 1000, 10000, 100000, 1000000};
    
    try {
        // Create output file for results
        std::ofstream results_file("mesh_benchmark_results.csv");
        if (!results_file) {
            fmt::print("Error opening results file!\n");
            return 1;
        }
        
        // Write CSV header
        results_file << "device,num_meshes,points_per_mesh,elements_per_mesh,kernel_time_ms,total_time_ms\n";
        
        // Create GPU queue
        sycl::queue gpu_queue(sycl::gpu_selector_v);
        fmt::print("Running on GPU: {}\n", 
                 gpu_queue.get_device().get_info<sycl::info::device::name>());
        
        // Create CPU queue
        sycl::queue cpu_queue(sycl::cpu_selector_v);
        fmt::print("Running on CPU: {}\n", 
                 cpu_queue.get_device().get_info<sycl::info::device::name>());
        
        fmt::print("\nRunning benchmark for all parameter combinations...\n");
        fmt::print("Each configuration will run {} times plus one warm-up run\n", num_runs);
        
        // Total number of combinations
        size_t total_combinations = parameter_values.size() * parameter_values.size() * parameter_values.size();
        size_t completed = 0;
        
        // Test all combinations with both GPU and CPU
        for (size_t num_meshes : parameter_values) {
            for (size_t points_per_mesh : parameter_values) {
                for (size_t elements_per_mesh : parameter_values) {
                    // Skip combinations that could cause out-of-memory issues
                    if (num_meshes * points_per_mesh * elements_per_mesh > 1e9) {
                        fmt::print("Skipping potentially out-of-memory combination: meshes={}, points={}, elements={}\n", 
                                 num_meshes, points_per_mesh, elements_per_mesh);
                        completed += 2;  // Count as completed for both GPU and CPU
                        continue;
                    }
                    
                    fmt::print("Running benchmark ({}/{}) for meshes={}, points={}, elements={}\n", 
                             completed+1, total_combinations*2, num_meshes, points_per_mesh, elements_per_mesh);
                    
                    try {
                        // GPU benchmark
                        auto [gpu_kernel_time, gpu_total_time] = runBenchmark(gpu_queue, "GPU", 
                                                                          num_meshes, points_per_mesh, elements_per_mesh, 
                                                                          num_runs);
                        
                        // Write GPU results to CSV
                        results_file << "GPU," << num_meshes << "," << points_per_mesh << "," 
                                    << elements_per_mesh << "," << gpu_kernel_time << "," 
                                    << gpu_total_time << "\n";
                        completed++;
                        
                        fmt::print("Running benchmark ({}/{}) for meshes={}, points={}, elements={}\n", 
                                 completed+1, total_combinations*2, num_meshes, points_per_mesh, elements_per_mesh);
                        
                        // CPU benchmark
                        auto [cpu_kernel_time, cpu_total_time] = runBenchmark(cpu_queue, "CPU", 
                                                                          num_meshes, points_per_mesh, elements_per_mesh, 
                                                                          num_runs);
                        
                        // Write CPU results to CSV
                        results_file << "CPU," << num_meshes << "," << points_per_mesh << "," 
                                    << elements_per_mesh << "," << cpu_kernel_time << "," 
                                    << cpu_total_time << "\n";
                        completed++;
                        
                        // Flush results to file periodically
                        results_file.flush();
                        
                    } catch (std::exception const& e) {
                        fmt::print("Exception during benchmark for meshes={}, points={}, elements={}: {}\n", 
                                 num_meshes, points_per_mesh, elements_per_mesh, e.what());
                        completed += 2;  // Count as completed for both GPU and CPU
                    }
                }
            }
        }
        
        fmt::print("\nBenchmark completed. Results written to mesh_benchmark_results.csv\n");
        
        return 0;
    } catch (sycl::exception const& e) {
        fmt::print("SYCL exception caught: {}\n", e.what());
        return 1;
    } catch (std::exception const& e) {
        fmt::print("Standard exception caught: {}\n", e.what());
        return 1;
    }
}
