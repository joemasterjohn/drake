#!/bin/bash

z0="$1"
vx="$2"
wy="$3"
t="$4"

./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=1e-2 --data_file=data_1e-2.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=4e-3 --data_file=data_4e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=1e-3 --data_file=data_1e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=4e-4 --data_file=data_4e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=1e-4 --data_file=data_1e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=1e-5 --data_file=data_1e-5.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=1e-2 --data_file=data_spec_1e-2.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=4e-3 --data_file=data_spec_4e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=1e-3 --data_file=data_spec_1e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=4e-4 --data_file=data_spec_4e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=1e-4 --data_file=data_spec_1e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --mbp_dt=1e-5 --data_file=data_spec_1e-5.txt &
