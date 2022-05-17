#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patches as patches
import csv
import numpy as np
from scipy import interpolate

def read(filename, m):
    t = []
    traj = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            t.append(float(row[0]))
            traj.append(float(row[6]))

    m["t"] = np.array(t)
    m["traj"] = np.array(traj).transpose()

num_runs = 6

runs = [dict()]
runs_interp = [dict()]

reference_map = dict()
read("run_{}".format(num_runs), reference_map)


# open each file and get (x,y)
for i in range(1, num_runs):
    m = dict()
    read("run_{}".format(i), m)
    runs.append(m)

# interpolate (x,y) to reference trajectory
for i in range(1, num_runs):
    m = runs[i]
    m_interp = dict()
    f = interpolate.interp1d(m["t"], m["traj"])
    t = reference_map["t"]
    m_interp["t"] = t
    m_interp["traj"] = f(t)
    runs_interp.append(m_interp)

l2 = []
linf = []

# calculate L1, L2, Linf norm
for i in range(1, num_runs):
    x_dt = runs_interp[i]["traj"]
    x = reference_map["traj"]
    l2.append(np.linalg.norm(x_dt - x, ord=2) / np.linalg.norm(x, ord=2))
    linf.append(np.linalg.norm(x_dt - x, ord=np.inf))

dt = np.power(0.2, range(1, 6))
dt_list = dt.tolist()

# plot l2 norm
fig, ax = plt.subplots()
ax.plot(dt_list, l2, 'ko', markersize=10, fillstyle='none', markeredgewidth=2)

dt_line = np.linspace(1e-4, 1e-0)

ax.plot(dt_line.tolist(), (5*dt_line).tolist(), markersize=10, fillstyle='none', markeredgewidth=2, linestyle='dotted')
plt.xlabel(r'$\delta t$', fontsize=20)
plt.ylabel(r'$\frac{|| x_{\delta t}(t) - x(t) ||_2 }{ || x(t) ||_2 }$', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.locator_params(axis='y', nbins=5)

ax.tick_params(width=2, length=2, direction="in")
for axis in ['top', 'bottom', 'left', 'right']:
 ax.spines[axis].set_linewidth(2)  # change width

ax.set_xlim(left=1e-4, right=1e-0)

plt.xscale('log', basex=10)
plt.yscale('log', basey=10)

plt.margins(0.1)
fig.set_tight_layout(True)
fig.savefig("l2_error.png")


## plot linf norm
#fig, ax = plt.subplots()
#ax.plot(dt, linf, 'ko', markersize=10, fillstyle='none', markeredgewidth=2)
#plt.xlabel(r'$\delta t$', fontsize=20)
#plt.ylabel(r'$|| x_{\delta t}(t) - x(t) ||_{\inf}$', fontsize=20)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#
#plt.locator_params(axis='y', nbins=5)
#
#ax.tick_params(width=2, length=2, direction="in")
#for axis in ['top', 'bottom', 'left', 'right']:
# ax.spines[axis].set_linewidth(2)  # change width
#
#plt.xscale('log', basex=10)
#plt.yscale('log', basey=10)
#
#plt.margins(0.1)
#fig.set_tight_layout(True)
#fig.savefig("linf_error.png")
