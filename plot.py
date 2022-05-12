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
            traj.append([float(row[5]), float(row[6])])

    m["t"] = np.array(t)
    m["traj"] = np.array(traj).transpose()

# norm of trajectory x = [[x0, y0], ...]
def norm2(x):
    return np.sqrt(np.sum((x[:,0] ** 2.0) + x[:,1] ** 2.0))

def normInf(x):
    return np.sqrt(np.max(np.sqrt(x[:,0] ** 2.0 + x[:,1] ** 2.0)))

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
    x = reference_map["traj"] - runs_interp[i]["traj"]
    #l2.append(np.linalg.norm(x, ord=2))
    #linf.append(np.linalg.norm(x, ord=np.inf))
    l2.append(norm2(x))
    linf.append(normInf(x))

dt = np.power(0.2, range(1, 6)).tolist()

# plot l2 norm
fig, ax = plt.subplots()
ax.plot(dt, l2, 'ko', markersize=10, fillstyle='none', markeredgewidth=2)
plt.xlabel('$\delta t$', fontsize=20)
plt.ylabel('$|| x_{\delta t}(t) - x_h(t) ||_2$', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.locator_params(axis='y', nbins=5)

ax.tick_params(width=2, length=2, direction="in")
ax.invert_xaxis()
for axis in ['top', 'bottom', 'left', 'right']:
 ax.spines[axis].set_linewidth(2)  # change width

plt.xscale('log', basex=5)

plt.margins(0.1)
fig.set_tight_layout(True)
fig.savefig("l2_error.png")


# plot linf norm
fig, ax = plt.subplots()
ax.plot(dt, linf, 'ko', markersize=10, fillstyle='none', markeredgewidth=2)
plt.xlabel('$\delta t$', fontsize=20)
plt.ylabel('$|| x_{\delta t}(t) - x_h(t) ||_{\infinity}$', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.locator_params(axis='y', nbins=5)

ax.tick_params(width=2, length=2, direction="in")
ax.invert_xaxis()
for axis in ['top', 'bottom', 'left', 'right']:
 ax.spines[axis].set_linewidth(2)  # change width

plt.xscale('log', basex=5)

plt.margins(0.1)
fig.set_tight_layout(True)
fig.savefig("linf_error.png")
