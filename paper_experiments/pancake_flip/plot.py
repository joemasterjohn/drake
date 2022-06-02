#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patches as patches
import csv
import numpy as np
from scipy import interpolate

constraints = []
discrete_pair_time = []
solve_time = []
with open('data.txt', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        constraints.append(float(row[0]))
        discrete_pair_time.append(float(row[1]) / 5.)
        solve_time.append(float(row[2]) / 5.)

print(constraints)
print(discrete_pair_time)
print(solve_time)

# plot l2 norm
fig, ax = plt.subplots()
ax.plot(constraints, discrete_pair_time, 'x-', color='black', markersize=12, markeredgewidth=2, label='Geometry')
ax.plot(constraints, solve_time, 'o-', color='black', markersize=12, markeredgewidth=2, label='SAP')

#dt_line = np.linspace(150, 2000)
#ax.plot(dt_line.tolist(), (0.001 * dt_line).tolist(), linestyle='--', dashes=(20, 8), color='black')

plt.xlabel(r'Mean Number of Constraints [-]', fontsize=20)
plt.ylabel(r'Mean Wall-clock time [ms]', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

#plt.locator_params(axis='y', nbins=5)

ax.tick_params(which='minor', length=6,  width=2, direction="in")
ax.tick_params(which='major', length=10, width=2, direction="in")
for axis in ['top', 'bottom', 'left', 'right']:
 ax.spines[axis].set_linewidth(2)  # change width

ax.set_xlim(left=100, right=2000)
ax.set_ylim(bottom=0.5, top=25)

plt.xscale('log', basex=10)
plt.yscale('log', basey=10)

plt.margins(0.1)
plt.legend(fontsize=18)
fig.set_tight_layout(True)
fig.savefig("constraints_vs_time.png")
