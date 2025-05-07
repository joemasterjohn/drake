#!/bin/bash

z0="$1"
vx="$2"
wy="$3"
t="$4"

./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-2 --data_file=data_1e-2.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=7.5e-3 --data_file=data_7.5e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=5e-3 --data_file=data_5e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=2.5e-3 --data_file=data_2.5e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-3 --data_file=data_1e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=5e-4 --data_file=data_5e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-4 --data_file=data_1e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-5 --data_file=data_1e-5.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=-1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-2 --data_file=data_spec_1e-2.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=-1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=7.5e-3 --data_file=data_spec_7.5e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=-1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=5e-3 --data_file=data_spec_5e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=-1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=2.5e-3 --data_file=data_spec_2.5e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=-1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-3 --data_file=data_spec_1e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=-1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=5e-4 --data_file=data_spec_5e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=-1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-4 --data_file=data_spec_1e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=-1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-5 --data_file=data_spec_1e-5.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-2 --data_file=data_single_spec_1e-2.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=7.5e-3 --data_file=data_single_spec_7.5e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=5e-3 --data_file=data_single_spec_5e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=2.5e-3 --data_file=data_single_spec_2.5e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-3 --data_file=data_single_spec_1e-3.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=5e-4 --data_file=data_single_spec_5e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-4 --data_file=data_single_spec_1e-4.txt &
./bazel-bin/examples/hydroelastic/ball_plate/ball_plate_run_dynamics --use_speculative --num_speculative=1 --mode=data --z0="$z0" --vx="$vx" --wy="$wy" --num_dofs=3 --simulation_time="$t" --dt=1e-5 --data_file=data_single_spec_1e-5.txt &
