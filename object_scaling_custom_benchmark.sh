# Warm up runs

bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=false --num_grippers=1 --num_peppers=1 --object_spacing=0.1

ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=true --num_grippers=1 --num_peppers=1 --object_spacing=0.1

# Define object counts and spacing values
OBJECT_COUNTS=(1 2 5 10 20)
SPACING_VALUES=(0.05 0.1)


# Spread out distribution runs (0.1 spacing)
echo "Running spread out distribution benchmarks (spacing=0.1)..."

# Drake CPU runs
# echo "Drake CPU runs:"
# for count in "${OBJECT_COUNTS[@]}"; do
#     echo "Running with $count grippers/peppers..."
#     bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=false --num_grippers=$count --num_peppers=$count --object_spacing=0.1 --config_name="0.1_$count"
# done

# SYCL GPU runs
echo "SYCL GPU runs:"
for count in "${OBJECT_COUNTS[@]}"; do
    echo "Running with $count grippers/peppers..."
    ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=true --num_grippers=$count --num_peppers=$count --object_spacing=0.1 --config_name="0.1_$count"
done

# Dense distribution runs (0.05 spacing)
echo "Running dense distribution benchmarks (spacing=0.05)..."

# Drake CPU runs
# echo "Drake CPU runs:"
# for count in "${OBJECT_COUNTS[@]}"; do
#     echo "Running with $count grippers/peppers..."
#     bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=false --num_grippers=$count --num_peppers=$count --object_spacing=0.05 --config_name="0.05_$count"
# done

# SYCL GPU runs
echo "SYCL GPU runs:"
for count in "${OBJECT_COUNTS[@]}"; do
    echo "Running with $count grippers/peppers..."
    ONEAPI_DEVICE_SELECTOR=cuda:* bazel run //examples/hydroelastic/objects_scaling:objects_scaling_run_dynamics -- --use_sycl=true --num_grippers=$count --num_peppers=$count --object_spacing=0.05 --config_name="0.05_$count"
done







