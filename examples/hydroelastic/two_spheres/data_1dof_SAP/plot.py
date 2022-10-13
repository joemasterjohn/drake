#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patches as patches
import csv
import numpy as np

embedded_files = ["embedded5.txt",
                  "embedded9.txt", 
                  "embedded15.txt", 
                  "embedded19.txt", 
                  "embedded25.txt"]
embedded_traj = []
embedded_time = []
embedded_max_contacts = []
embedded_avg_contacts = []

for i in embedded_files:
  lines = [float(x) for x in open(i).read().strip().split()]
  embedded_traj.append(np.array(lines[0:-3]))
  embedded_time.append(lines[-3])
  embedded_max_contacts.append(lines[-2])
  embedded_avg_contacts.append(lines[-1])

# body fitted
body_fitted_files = ["body_fitted8.txt",
                     "body_fitted6.txt", 
                     "body_fitted4.txt", 
                     "body_fitted2.txt", 
                     "body_fitted.txt"]
body_fitted_traj = []
body_fitted_time = []
body_fitted_max_contacts = []
body_fitted_avg_contacts = []

for i in body_fitted_files:
  lines = [float(x) for x in open(i).read().strip().split()]
  body_fitted_traj.append(np.array(lines[0:-3]))
  body_fitted_time.append(lines[-3])
  body_fitted_max_contacts.append(lines[-2])
  body_fitted_avg_contacts.append(lines[-1])


# body fitted extended
body_fitted_extended_files = ["body_fitted_extended8.txt",
                              "body_fitted_extended6.txt", 
                              "body_fitted_extended4.txt", 
                              "body_fitted_extended2.txt", 
                              "body_fitted_extended.txt"]
body_fitted_extended_traj = []
body_fitted_extended_time = []
body_fitted_extended_max_contacts = []
body_fitted_extended_avg_contacts = []

for i in body_fitted_extended_files:
  lines = [float(x) for x in open(i).read().strip().split()]
  body_fitted_extended_traj.append(np.array(lines[0:-3]))
  body_fitted_extended_time.append(lines[-3])
  body_fitted_extended_max_contacts.append(lines[-2])
  body_fitted_extended_avg_contacts.append(lines[-1])

# Convergence to finest geometry within own method
embedded_convergence_error = []
for i in range(len(embedded_traj)-1):
  embedded_convergence_error.append(np.linalg.norm(embedded_traj[i] - embedded_traj[-1]) / np.linalg.norm(embedded_traj[-1]))

body_fitted_convergence_error = []
for i in range(len(body_fitted_traj)-1):
  body_fitted_convergence_error.append(np.linalg.norm(body_fitted_traj[i] - body_fitted_traj[-1]) / np.linalg.norm(body_fitted_traj[-1]))

body_fitted_extended_convergence_error = []
for i in range(len(body_fitted_extended_traj)-1):
  body_fitted_extended_convergence_error.append(np.linalg.norm(body_fitted_extended_traj[i] - body_fitted_extended_traj[-1]) / np.linalg.norm(body_fitted_extended_traj[-1]))

fig, ax = plt.subplots()
#ax.plot(embedded_avg_contacts[0:-1], embedded_convergence_error, color='r',  marker='o', markersize=10, fillstyle='none', markeredgewidth=2, label="embedded error")
#ax.plot(body_fitted_avg_contacts[0:-1], body_fitted_convergence_error, color='g', marker='x', markersize=10, fillstyle='none', markeredgewidth=2, label="body fitted error")
#ax.plot(body_fitted_extended_avg_contacts[0:-1], body_fitted_extended_convergence_error, color='b', marker='+', markersize=10, fillstyle='none', markeredgewidth=2, label="body fitted (extended) error")
ax.plot(embedded_max_contacts[0:-1], embedded_convergence_error, color='r',  marker='o', markersize=10, fillstyle='none', markeredgewidth=2, label="embedded error")
ax.plot(body_fitted_max_contacts[0:-1], body_fitted_convergence_error, color='g', marker='x', markersize=10, fillstyle='none', markeredgewidth=2, label="body fitted error")
ax.plot(body_fitted_extended_max_contacts[0:-1], body_fitted_extended_convergence_error, color='b', marker='+', markersize=10, fillstyle='none', markeredgewidth=2, label="body fitted (extended) error")
ax.legend()
plt.xlabel('Max Number of Contacts', fontsize=20)
plt.ylabel('Trajectory Relative Error', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.locator_params(axis='y', nbins=5)

ax.tick_params(width=2, length=6, direction="in")
#ax.set_ylim(bottom=0.64, top=0.66)
#ax.set_xlim(left=0.09, right=11)
for axis in ['top', 'bottom', 'left', 'right']:
 ax.spines[axis].set_linewidth(2)  # change width

# Set axis, legend, and save figure
plt.xscale('log', basex=10)
plt.yscale('log', basey=10)

plt.margins(0.15)
fig.set_tight_layout(True)
fig.savefig("trajector_error.png")

# Convergence to body fitted with extended field
#e1_err = np.linalg.norm(e1 - ep4) / np.linalg.norm(ep4)
#e2_err = np.linalg.norm(e2 - ep4) / np.linalg.norm(ep4)
#e3_err = np.linalg.norm(e3 - ep4) / np.linalg.norm(ep4)
#e4_err = np.linalg.norm(e4 - ep4) / np.linalg.norm(ep4)
#
#embedded_to_body_fitted_error = [e1_err, e2_err, e3_err, e4_err]
#
#fig, ax = plt.subplots()
#ax.plot(embedded_mesh_avg_contacts, embedded_to_body_fitted_error, 'ko', markersize=10, fillstyle='none', markeredgewidth=2, label="embedded error")
#ax.legend()
#plt.xlabel('Average Number of Contacts', fontsize=20)
#plt.ylabel('Trajectory Relative Error', fontsize=20)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#
#plt.locator_params(axis='y', nbins=5)
#
#ax.tick_params(width=2, length=6, direction="in")
##ax.set_ylim(bottom=0.64, top=0.66)
##ax.set_xlim(left=0.09, right=11)
#for axis in ['top', 'bottom', 'left', 'right']:
# ax.spines[axis].set_linewidth(2)  # change width
#
## Set axis, legend, and save figure
#plt.xscale('log', basex=10)
#plt.yscale('log', basey=10)
#
#plt.margins(0.15)
#fig.set_tight_layout(True)
#fig.savefig("embedded_to_body_fitted_trajector_error.png")

