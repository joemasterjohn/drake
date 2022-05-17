#!/usr/bin/env python3

import subprocess
import os
import numpy as np

# Default arguments
prog_default = ['./bazel-bin/examples/multibody/spinning_coin/spinning_coin',
                '--simulation_time=5']
  

coin_radius = 0.02426

#translational = [1 + i*0.5 for i in range(10)]
#ratio = [0.1, 0.2, 0.4, 0.75, 1.0, 2.0, 4.0, 5.0, 7.5, 10]

translational = [1]
ratio = np.logspace(-1, 1, 10)

def ensure_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def run(prog, output_dir):
  for vy in translational:
    for alpha in ratio:
      wz = alpha * vy / coin_radius
      other_args = ['--vy={}'.format(vy),
                    '--wz={}'.format(wz),
                    '--output_filename={}/run_{}_{}'.format(output_dir, vy, wz),
                    '--epsilon_filename={}/epsilon.txt'.format(output_dir)]
      subprocess.call(prog + other_args)

def write_gnuplot_file(output_dir):

  file_prefix = '''
set autoscale xfix
set yrange [0:0.3]

set title 'Spinning Coin Velocity Ratio' font ",18" # Set graph title, set title font size to 18
 
set terminal jpeg size 1200,630          # Set the output format to jpeg, set dimensions to 1200x630
set output 'output.jpg'                  # Set output file to output.jpg

plot \
'''  
    
  gnuplot_file = open(output_dir + '/plot.txt', 'w')
  gnuplot_file.write(file_prefix)

  for vy in translational:
    for alpha in ratio:
      wz = alpha * vy / coin_radius
      gnuplot_file.write('     \'run_{}_{}\' using 1:2 notitle with linespoints linetype 6 linewidth 3, \\\n'.format(vy, wz))

  gnuplot_file.close()


def run_timestep_convergence(prog, output_dir):
  vy = 1
  alpha = 1
  wz = alpha * vy / coin_radius
  #mbp_dt = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
  #mbp_dt = [5e-05]
  for i in range(1, 7):
      dt = pow(0.2, i)
      other_args = ['--vy={}'.format(vy),
                    '--wz={}'.format(wz),
                    '--output_filename={}/run_{}'.format(output_dir, i),
                    '--epsilon_filename={}/epsilon.txt'.format(output_dir),
                    '--mbt_dt={}'.format(dt)]
      print("\nrunning: {}\n".format(" ".join(prog + other_args)))
      subprocess.call(prog + other_args)

def run_mesh_convergence(prog, output_dir):
  vy = 1.0
  alpha = 1
  wz = alpha * vy / coin_radius
  for i in range(1, 7):
      other_args = ['--vy={}'.format(vy),
                    '--wz={}'.format(wz),
                    '--output_filename={}/run_{}'.format(output_dir, i),
                    '--epsilon_filename={}/epsilon.txt'.format(output_dir),
                    '--mbt_dt={}'.format(0.001),
                    '--coin_file=coin_{}.sdf'.format(2**(i+1))]
      print("\nrunning: {}\n".format(" ".join(prog + other_args)))
      subprocess.call(prog + other_args)


def run_continuous_test(prog, output_dir):
  vy = 1.0
  a = [0.1, 1, 10]
  for alpha in a:
      wz = alpha * vy / coin_radius
      other_args = ['--vy={}'.format(vy),
                    '--wz={}'.format(wz),
                    '--output_filename={}/run_{}'.format(output_dir, alpha),
                    '--epsilon_filename={}/epsilon.txt'.format(output_dir),
                    '--mbt_dt=0']
      subprocess.call(prog + other_args)


def do_main():
 
#  # Discrete Hydro / Normal Resolution Surface
#  output_dir = "paper_experiments/" + "discrete_hydro_high_res"
#  prog = prog_default.copy()
#  prog.append('--mbt_dt=0.001')
#  
#  ensure_dir(output_dir)
#  run(prog, output_dir)
#  write_gnuplot_file(output_dir)
#  
#  # Discrete Hydro / Low Resolution Surface
#  output_dir = "paper_experiments/" + "discrete_hydro_low_res"
#  prog = prog_default.copy()
#  prog.append('--mbt_dt=0.001')
#  prog.append('--low_res_contact_surface')
#  prog.append('--dalpha_threshold=200')
#  
#  ensure_dir(output_dir)
#  run(prog, output_dir)
#  write_gnuplot_file(output_dir)
 
#  # Discrete Hydro / Low Resolution Surface Convergence
#  output_dir = "paper_experiments/" + "discrete_hydro_timestep_convergence"
#  prog = prog_default.copy()
#  prog.append('--low_res_contact_surface')
#  prog.append('--dalpha_threshold=1000000')
#  
#  ensure_dir(output_dir)
#  run_timestep_convergence(prog, output_dir)
#  write_gnuplot_file(output_dir)
 
  # Discrete Hydro / Low Resolution Surface Convergence
  output_dir = "paper_experiments/" + "discrete_hydro_mesh_convergence"
  prog = prog_default.copy()
  prog.append('--low_res_contact_surface')
  prog.append('--dalpha_threshold=1000000')
  
  ensure_dir(output_dir)
  run_mesh_convergence(prog, output_dir)
  write_gnuplot_file(output_dir)


#  # Discrete Hydro / Low Resolution Surface Convergence
#  output_dir = "paper_experiments/" + "continous_convergence_test"
#  prog = prog_default.copy()
#  prog.append('--dalpha_threshold=20000')
#  
#  ensure_dir(output_dir)
#  run_continuous_test(prog, output_dir)
#  write_gnuplot_file(output_dir)

#  # Continuous Hydro
#  output_dir = "paper_experiments/" + "continuous_hydro"
#  prog = prog_default.copy()
#  prog.append('--mbt_dt=0')
#  prog.append('--simulator_integration_scheme=implicit_euler')

#  ensure_dir(output_dir)
#  run(prog, output_dir)
#  write_gnuplot_file(output_dir)
  
#  # Point Contact
#  output_dir = "paper_experiments/" + "point"
#  prog = prog_default.copy()
#  prog.append('--mbt_dt=0.001')
#  prog.append('--point_contact')
#  
#  ensure_dir(output_dir)
#  run(prog, output_dir)
#  write_gnuplot_file(output_dir)

if __name__ == '__main__':
  do_main()
