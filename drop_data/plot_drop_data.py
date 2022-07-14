#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patches as patches
import csv
import numpy as np

import sys

time = list()
phi = list()
force = list()
contacts = list()

phi2 = list()
force2 = list()
contacts2 = list()

with open('drop_data_dissipation_16.txt', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        time.append(float(row[0]))
        phi.append(float(row[2]))
        force.append(float(row[3]))
        contacts.append(float(row[4]))
        phi2.append(float(row[6]))
        force2.append(float(row[7]))
        contacts2.append(float(row[8]))

force.insert(0,0)
force.pop(-1)

force2.insert(0,0)
force2.pop(-1)

# Force vs. time
fig, ax = plt.subplots()
ax.plot(time, force, color='black')
plt.xlabel('$t\;[s]$', fontsize=30)
plt.ylabel('$F_z\;[N]$', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=6)

ax.tick_params(width=2, length=6, direction="in")
#ax.set_ylim(bottom=-0.5, top=0.1)
ax.set_xlim(left=0, right=0.75)
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width

plt.margins(0.15)
fig.set_tight_layout(True)
fig.savefig("embedded_force_vs_time.png")

# Num contacts vs. time
fig, ax = plt.subplots()
ax.plot(time, contacts, color='black')
plt.xlabel('$t\;[s]$', fontsize=30)
plt.ylabel('$contacts$', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=6)

ax.tick_params(width=2, length=6, direction="in")
#ax.set_ylim(bottom=-0.5, top=0.1)
ax.set_xlim(left=0, right=0.75)
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width

plt.margins(0.15)
fig.set_tight_layout(True)
fig.savefig("embedded_contacts_vs_time.png")


# Phi vs. Force
phi_vs_force = list(zip(phi, force))
[phi, force] = list(zip(*sorted(phi_vs_force)))

fig, ax = plt.subplots()
ax.plot(phi, force, color='black')
plt.xlabel('$\phi\;[m]$', fontsize=30)
plt.ylabel('$F_z\;[N]$', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=6)

ax.tick_params(width=2, length=6, direction="in")
#ax.set_ylim(bottom=-0.5, top=0.1)
ax.set_xlim(left=-0.005, right=0.01)
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width

plt.margins(0.15)
fig.set_tight_layout(True)
fig.savefig("embedded_force_vs_phi.png")

# Phi vs. Force (zoomed)
fig, ax = plt.subplots()
ax.plot(phi, force, color='black')
plt.xlabel('$\phi\;[m]$', fontsize=30)
plt.ylabel('$F_z\;[N]$', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=6)

ax.tick_params(width=2, length=6, direction="in")
#ax.set_ylim(bottom=-0.5, top=0.1)
ax.set_xlim(left=-0.002, right=0.002)
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width

plt.margins(0.15)
fig.set_tight_layout(True)
fig.savefig("embedded_force_vs_phi_zoom.png")

# -----------------------------------------------------------
# -----------------------------------------------------------

# Force vs. time
fig, ax = plt.subplots()
ax.plot(time, force2, color='black')
plt.xlabel('$t\;[s]$', fontsize=30)
plt.ylabel('$F_z\;[N]$', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=6)

ax.tick_params(width=2, length=6, direction="in")
#ax.set_ylim(bottom=-0.5, top=0.1)
ax.set_xlim(left=0, right=0.75)
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width

plt.margins(0.15)
fig.set_tight_layout(True)
fig.savefig("internal_force_vs_time.png")

# Num contacts vs. time
fig, ax = plt.subplots()
ax.plot(time, contacts2, color='black')
plt.xlabel('$t\;[s]$', fontsize=30)
plt.ylabel('$contacts$', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=6)

ax.tick_params(width=2, length=6, direction="in")
#ax.set_ylim(bottom=-0.5, top=0.1)
ax.set_xlim(left=0, right=0.75)
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width

plt.margins(0.15)
fig.set_tight_layout(True)
fig.savefig("internal_contacts_vs_time.png")

# Phi vs. Force
phi_vs_force2 = list(zip(phi2, force2))
[phi2, force2] = list(zip(*sorted(phi_vs_force2)))

fig, ax = plt.subplots()
ax.plot(phi2, force2, color='black')
plt.xlabel('$\phi\;[m]$', fontsize=30)
plt.ylabel('$F_z\;[N]$', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=6)

ax.tick_params(width=2, length=6, direction="in")
#ax.set_ylim(bottom=-0.5, top=0.1)
ax.set_xlim(left=-0.005, right=0.01)
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width

plt.margins(0.15)
fig.set_tight_layout(True)
fig.savefig("internal_force_vs_phi.png")

# Phi vs. Force (zoomed)
fig, ax = plt.subplots()
ax.plot(phi2, force2, color='black')
plt.xlabel('$\phi\;[m]$', fontsize=30)
plt.ylabel('$F_z\;[N]$', fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=6)

ax.tick_params(width=2, length=6, direction="in")
#ax.set_ylim(bottom=-0.5, top=0.1)
ax.set_xlim(left=-0.002, right=0.002)
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width

plt.margins(0.15)
fig.set_tight_layout(True)
fig.savefig("internal_force_vs_phi_zoom.png")

