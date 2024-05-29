Within the stubdev folder, the following libraries and unit tests:

# cuda_cholesky libary

CUDA Cholesky factorization and solve library. The most important code is in "cuda_cholesky.cuh", which contains three CUDA device functions to perform Cholesky factorization, forward substitution, and backward substitution. "cuda_cholesky.cu" contains driver code for the unit test "cuda_cholesky_test.cc." The .cu file and .cc file can be viewed separately as a functional unit test and will not be used in the actual CUDA SAP solver.

```bash
bazel run stubdev:cuda_cholesky_test
```

# cuda_eigen_debug library

This is a unit test used to benchmark the performance of a hand-rolled (just simply a for loop) matrix-vector multiplication and the performance of a matrix-vector multiplication using the .row() function from Eigen. The unit test "cuda_eigen_debug_test.cc" performs a runtime benchmark of a 60x60 matrix multiplied by a 60x1 vector using these two versions of the code, averaged over 1000 runs.

To run the unit test, call

```bash
bazel run stubdev:cuda_eigen_debug_test
```
# cuda_gpu_collision library

Library that contains code to perform collision checks on the GPU. For now, only spheres and half-planes are supported. This library checks for valid collisions, reports the collision point, collision rotation matrix, and corresponding collision data.

The "cuda_gpu_collision.cuh" contains geometry class definitions (Sphere and Plane), collision data structures, and CUDA device functions used for collision detection and the collision data assembling process.

The "cuda_gpu_collision.cu" contains function implementations to initialize the GPU collision class and retrieval functions from GPU to CPU for debugging.

# cuda_sap_solver library

Library which contains the main cuda_sap_solver. cuda Kernels and device functions can be found in "cuda_sap_solver.cu". cuda data access device functions and related data struct definition can be found in "cuda_sap_solver.cuh". 

# cuda_sap_qp library

Essentially, this is a duplicate of cuda_sap_solver but with slight modification to construct a simple QP problem with a constraint being impose for x larger than 6. 

This library was constructed during the development of cuda_sap_solver to assist debugging and correctbess validation, and should only be used as a reference as many performance optimization made into the cuda_sap_solver were not propagated back to cuda_sap_qp. 

The unit test "cuda_sap_qp_test.cc" checks for correctness of various free motion velocity and various initial guesses. The unit test checks for the correctness of the converged solution, number of newton iterations needed for convergence, and number of line search loop. You can run the unit test with command:

```bash
bazel run stubdev:cuda_sap_qp_test
```

# cuda_sap_cpu_wrapper

The CPU entry point for the simulation contains one CollisionGPUData and one SAPGPUData. This encapsulation abstracts the entire simulation into three major functions: init (initialization), step (perform specified steps of simulation on the GPU), and destroy (delete related data structs and release GPU memory).


# demo

The "cuda_sap_test.cc" replicates the timing benchmark as presented in the 2024 summer intern final presentations. The benchmark testing runs simulations with 4, 7, 10, 11, 12, 13, 14, 15, and 22 spheres, and for each case, runs simulations with batch sizes of 1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, and 10000, 15000.

The file also performs a write-out of the simulation into CSV files. Note that running in Bazel environments doesn't allow proper file write-out, so the write-out portion of the demo code is commented out. By default, if the demo is invoked through "bazel run," the write-out part will be skipped, and the code will start performing the timing benchmark immediately. You can run the program using the following command:

```bash
bazel run stubdev:cuda_sap_test
```

If you would like to have the simulation data saved for post-processing through Paraview, at the Drake root directory, you can use the following commands to call the executable from the bazel-bin directory:

```bash
bazel build stubdev:cuda_sap_test
cd bazel-bin
cd stubdev
./cuda_sap_test
```

The simulation data is saved as CSV files, which can be visualized directly in Paraview, as we are only using sphere geometry. In Paraview, after selecting the corresponding CSV sequence in focus, add a "Table to Points" filter, select "pos_x," "pos_y," and "pos_z" to represent point positions, choose the visualization representation as "Point Gaussian," and change the radius to the corresponding radius (in this demo, 0.03). The animation can then be viewed or exported in Paraview.


