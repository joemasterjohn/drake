#!/usr/bin/env bash


# This is a special sleep function which returns the number of seconds slept as
# the "error code" or return code" so that we can easily see that we are in
# fact actually obtaining the return code of each process as it finishes.
my_sleep() {
    seconds_to_sleep="$1"
    sleep "$seconds_to_sleep"
    return "$seconds_to_sleep"
}

binary="/home/joemasterjohn/tri/drake/bazel-bin/examples/hydroelastic/two_spheres/two_spheres_run_dynamics"
data_dir="/home/joemasterjohn/tri/drake/examples/hydroelastic/two_spheres/data/"

# Create an array of whatever commands you want to run as subprocesses
procs=()  # bash array
output=()

procs+=("${binary} --filename=models/embedded/pepper5.sdf")
procs+=("${binary} --filename=models/embedded/pepper9.sdf")
procs+=("${binary} --filename=models/embedded/pepper15.sdf")
procs+=("${binary} --filename=models/embedded/pepper19.sdf")
procs+=("${binary} --filename=models/embedded/pepper25.sdf")
output+=("${data_dir}/embedded5.txt")
output+=("${data_dir}/embedded9.txt")
output+=("${data_dir}/embedded15.txt")
output+=("${data_dir}/embedded19.txt")
output+=("${data_dir}/embedded25.txt")

procs+=("${binary} --filename=models/body_fitted/pepper8.sdf")
procs+=("${binary} --filename=models/body_fitted/pepper6.sdf")
procs+=("${binary} --filename=models/body_fitted/pepper4.sdf")
procs+=("${binary} --filename=models/body_fitted/pepper2.sdf")
procs+=("${binary} --filename=models/body_fitted/pepper.sdf")
output+=("${data_dir}/body_fitted8.txt")
output+=("${data_dir}/body_fitted6.txt")
output+=("${data_dir}/body_fitted4.txt")
output+=("${data_dir}/body_fitted2.txt")
output+=("${data_dir}/body_fitted.txt")

procs+=("${binary} --filename=models/body_fitted_extended/pepper8.sdf")
procs+=("${binary} --filename=models/body_fitted_extended/pepper6.sdf")
procs+=("${binary} --filename=models/body_fitted_extended/pepper4.sdf")
procs+=("${binary} --filename=models/body_fitted_extended/pepper2.sdf")
procs+=("${binary} --filename=models/body_fitted_extended/pepper.sdf")
output+=("${data_dir}/body_fitted_extended8.txt")
output+=("${data_dir}/body_fitted_extended6.txt")
output+=("${data_dir}/body_fitted_extended4.txt")
output+=("${data_dir}/body_fitted_extended2.txt")
output+=("${data_dir}/body_fitted_extended.txt")

num_procs=${#procs[@]}  # number of processes
echo "num_procs = $num_procs"

# run commands as subprocesses and store pids in an array
pids=()  # bash array
for (( i=0; i<"$num_procs"; i++ )); do
    echo "cmd = ${procs[$i]}"
    ${procs[$i]} > "${output[$i]}" &  # run the cmd as a subprocess
    # store pid of last subprocess started; see:
    # https://unix.stackexchange.com/a/30371/114401
    pids+=("$!")
    echo "    pid = ${pids[$i]}"
done

# OPTION 1 (comment this option out if using Option 2 below): wait for all pids
for pid in "${pids[@]}"; do
    wait "$pid"
    return_code="$?"
    echo "PID = $pid; return_code = $return_code"
done
echo "All $num_procs processes have ended."
