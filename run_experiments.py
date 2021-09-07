#!/usr/bin/env python3

import subprocess
import os
  
# Default arguments
prog_default = ['./bazel-bin/examples/multibody/spinning_coin/spinning_coin',
                '--simulation_time=5']
  
translational = [1 + i*0.5 for i in range(10)]

coin_radius = 0.02426

ratio = [(0.01 + i*0.05) for i in range(10)]

def ensure_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def run(prog, output_dir):
  for vy in translational:
    for alpha in ratio:
      wz = alpha * vy / coin_radius
      other_args = ['--vy={}'.format(vy),
                    '--wz={}'.format(wz),
                    '--output_filename={}/run_{}_{}'.format(output_dir, vy, wz)]
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
#  prog.append('--mbt_dt=0.0001')
#  prog.append('--low_res_contact_surface')
#  
#  ensure_dir(output_dir)
#  run(prog, output_dir)
#  write_gnuplot_file(output_dir)
  
  # Continuous Hydro
  output_dir = "paper_experiments/" + "continuous_hydro"
  prog = prog_default.copy()
  prog.append('--mbt_dt=0')
  prog.append('--simulator_integration_scheme=implicit_euler')

  ensure_dir(output_dir)
  run(prog, output_dir)
  write_gnuplot_file(output_dir)
  
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
