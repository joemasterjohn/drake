# Warm up runs
ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=true --num_grippers=1 --num_peppers=1 --object_spacing=0.1 --print_perf=true

# Define object counts and spacing values
OBJECT_COUNTS=(1 2 5 10 20 33 50 100)

echo "SYCL GPU runs sparser:"
for count in "${OBJECT_COUNTS[@]}"; do
    echo "Running with $count grippers/peppers..."
    ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=true --num_grippers=$count --num_peppers=$count --object_spacing=0.15 --config_name="0.15_$count" --print_perf=true --visualize=false
done

# SYCL GPU runs
echo "SYCL GPU runs sparse:"
for count in "${OBJECT_COUNTS[@]}"; do
    echo "Running with $count grippers/peppers..."
    ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=true --num_grippers=$count --num_peppers=$count --object_spacing=0.1 --config_name="0.1_$count" --print_perf=true --visualize=false
done




# SYCL GPU runs
echo "SYCL GPU runs dense:"
for count in "${OBJECT_COUNTS[@]}"; do
    echo "Running with $count grippers/peppers..."
    ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=true --num_grippers=$count --num_peppers=$count --object_spacing=0.05 --config_name="0.05_$count" --print_perf=true --visualize=false
done
