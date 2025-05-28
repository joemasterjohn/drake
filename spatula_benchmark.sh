# Warm up runs for all cases
ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/spatula_slip_control:spatula_slip_control -- --use_sycl=true --simulation_sec=1 --mesh_res=20
ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/spatula_slip_control:spatula_slip_control -- --use_sycl=false --simulation_sec=1 --mesh_res=20
ONEAPI_DEVICE_SELECTOR=opencl:* bazel run //examples/hydroelastic/spatula_slip_control:spatula_slip_control -- --use_sycl=true --simulation_sec=1 --mesh_res=20


#Drake CPU and CUDA
for use_sycl in true false; do
  for mesh_res in 2 5 10 20; do
    echo "Running with --use_sycl=$use_sycl --mesh_res=$mesh_res"
    ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/spatula_slip_control:spatula_slip_control -- --use_sycl=$use_sycl --simulation_sec=6 --mesh_res=$mesh_res
  done
done
# Json outputs in drake/performance_jsons/

# OpenCL CPU
for use_sycl in true; do
  for mesh_res in 2 5 10 20; do
    echo "Running with --use_sycl=$use_sycl --mesh_res=$mesh_res"
    ONEAPI_DEVICE_SELECTOR=opencl:* bazel run //examples/hydroelastic/spatula_slip_control:spatula_slip_control -- --use_sycl=$use_sycl --simulation_sec=6 --mesh_res=$mesh_res
  done
done
# Json outputs in drake/performance_jsons_opencl/