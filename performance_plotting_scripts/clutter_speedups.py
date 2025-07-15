#!/usr/bin/env python3
"""
GPU Performance Comparison Script

This script compares multiple GPU/CPU versions by plotting broad phase and narrow phase timing 
vs number of objects. It creates both line plots and bar plots with multiple lines/bars - one for each version.

Usage:
    python object_scaling_oldVsNew.py --folders <folder1> <folder2> ... --legends <legend1> <legend2> ...

Example:
    python object_scaling_oldVsNew.py --folders performance_jsons_bvh_1s performance_jsons_bvh_2s performance_jsons_cpu --legends "GPU Version 1" "GPU Version 2" "CPU Version"

The script expects the data to be formatted identically for all versions.
For CPU versions, the legend name must contain "cpu" (case insensitive).
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import numpy as np
import pandas as pd
from utils import get_data

def calculate_actual_objects_clutter(obp):
    """
    Calculate the actual number of objects based on OBP (objects per pile).
    Each pile has 1 table, and each pile has obp objects.
    Formula: 1 + obp
    """
    return int(obp) * 4 + 5
def calculate_number_of_elements_clutter(obp, sr, problem_size_data):
    # Get hydroelastic_bodies array
    hydroelastic_bodies = problem_size_data.get("hydroelastic_bodies", [])
    total_tets = 0
    for body_data in hydroelastic_bodies:
        body_name = body_data.get("body", "")
        tetrahedra = int(body_data.get("tetrahedra", 0))
        total_tets += tetrahedra
        
    return total_tets
def get_corrected_timing(timing_data,raw_timing_data,key):
    """
    Get corrected timing for sycl by excluding JIT compilation time.
    For sycl-cpu/sycl-gpu: read the txt file and remove the first timing
    """
    calls = int(timing_data.get("calls", 1))
    first_time = float(raw_timing_data[key])
    total_time = float(timing_data.get("total_us", 0))
    return (total_time - first_time) / (calls - 1)

def _slope_indicator(ax, x0, y0, exponent, label,
                     length_dec=0.6,             # ← shorter than before
                     **line_kw):
    """
    Add a faint slope reference O(n^exponent) starting at (x0, y0).

    length_dec : how many powers of ten the arrow spans along x.
    """
    x1 = x0 * 10 ** length_dec
    y1 = y0 * (x1 / x0) ** exponent

    defaults = dict(ls="--", lw=1.0, color="0.4", alpha=0.6, zorder=1,
                    solid_capstyle="butt")
    defaults.update(line_kw)

    ax.plot([x0, x1], [y0, y1], **defaults)
    ax.annotate(label, xy=(x1, y1), xytext=(4, -2),
                textcoords="offset points", fontsize=8,
                ha="left", va="center", color=defaults["color"])


def _slope_indicator_nlogn(ax, x0, y0, label=r"$n\log n$",
                           length_dec=0.6, **line_kw):
    """
    Draw a short dashed curve showing the asymptotic shape of n log n
    on log–log axes.  The curve starts at (x0, y0) and spans
    `length_dec` decades in x.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x0, y0 : float      start point in data coordinates
    label : str         text to annotate at the curve end
    length_dec : float  horizontal length in decades
    **line_kw :         forwarded to ax.plot
    """
    x1 = x0 * 10 ** length_dec
    xs = np.logspace(np.log10(x0), np.log10(x1), 32)

    # Scale factor k such that k * x0 * log(x0) == y0
    k = y0 / (x0 * np.log(x0))
    ys = k * xs * np.log(xs)

    defaults = dict(ls="--", lw=1.0, color="0.4",
                    alpha=0.6, zorder=1)
    defaults.update(line_kw)

    ax.plot(xs, ys, **defaults)
    ax.annotate(label, (xs[-1], ys[-1]),
                xytext=(4, -2), textcoords="offset points",
                ha="left", va="center", fontsize=8,
                color=defaults["color"])

def plot_hydroelastic_query_perf_speedup_clutter(
        gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, xaxis_type):
    """
    Two‑row grid (raw timings | speed‑up) with:
      • identical y‑limits within each row (so left & right columns align),
      • internal O(n²) and O(n) slope indicators,
      • colour‑blind palette & compact legends.
    """
    # 0  Cosmetic defaults -------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers = ["o", "s", "D", "^", "v"]
    lstyles = ["-", "--", "-.", ":"]

    # 1  Build tidy table of raw timings ----------------------------------
    rows_raw = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for obp in objects_per_pile:
            for sr in sphere_resolutions:
                timing_dict = store[folder_names[f_idx]][obp][sr] \
                                  ["timing_overall"].get("timings", {})
                hq_time = timing_dict.get("HydroelasticQuery", {}).get("avg_us")
                if xaxis_type == "default":
                    rows_raw.append(dict(Legend=legend,
                                     SphereResolution=sr,
                                     Bodies=calculate_actual_objects_clutter(obp),
                                     HQTime_us=hq_time))
                elif xaxis_type == "num_elements":
                    rows_raw.append(dict(Legend=legend,
                                     SphereResolution=sr,
                                     Bodies=calculate_number_of_elements_clutter(obp, sr, cpu_data[folder_names[f_idx]][obp][sr]["problem_size"]),
                                     HQTime_us=hq_time))

    df_raw = pd.DataFrame(rows_raw)

    # 2  Speed‑up (= CPU / GPU) -------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd = []
    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["SphereResolution", "Bodies"])["HQTime_us"])
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["SphereResolution", "Bodies"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["HQTime_us"]
        rows_spd.append(sub)
    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3  Prepare grid – sharey='row' keeps y identical per row ------------
    n_cols = len(sphere_resolutions)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    # Pre‑compute common y‑limits
    y_raw_min, y_raw_max = df_raw["HQTime_us"].min(), df_raw["HQTime_us"].max()
    y_spd_min, y_spd_max = df_spd["SpeedUp"].min(), df_spd["SpeedUp"].max()

    # Nice padding
    y_raw_min *= 0.8
    y_raw_max *= 1.25
    y_spd_min *= 0.8
    y_spd_max *= 1.25

    sphere_resolution_titles = {}
    for sr in sphere_resolutions:
        sphere_resolution_titles[sr] = f"Sphere Resolution - {sr}"

    # 4  Plot raw timings (row 0) -----------------------------------------
    for c, sr in enumerate(sphere_resolutions):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) & (df_raw["SphereResolution"] == sr)]
            if d.empty:
                continue
            color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            ax.loglog(d["Bodies"], d["HQTime_us"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=color, label=legend, zorder=3)

        ax.set_ylim(y_raw_min, y_raw_max)
        ax.set_title(sphere_resolution_titles.get(sr, sr), fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("HydroelasticQuery time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # Slope indicators – place safely inside limits
        xs = df_raw[df_raw["SphereResolution"] == sr]["Bodies"].values
        ys = df_raw[df_raw["SphereResolution"] == sr]["HQTime_us"].values
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator(ax, x0, y0, 2, r"$n^{2}$")          # quadratic
        _slope_indicator(ax, x0, y0 / 4, 1, r"$n$")          # linear (below)

    # 5  Plot speed‑up (row 1) --------------------------------------------
    for c, sr in enumerate(sphere_resolutions):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) & (df_spd["SphereResolution"] == sr)]
            if d.empty:
                continue
            ax.loglog(d["Bodies"], d["SpeedUp"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=sns.color_palette()[i % 10],
                      label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if xaxis_type == "default":
            ax.set_xlabel("Number of Geometries $n$", fontsize=12)
        elif xaxis_type == "num_elements":
            ax.set_xlabel("Number of Tetrahedra $n$", fontsize=12)
        if c == 0:
            ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    # 6  Finish ------------------------------------------------------------
    return fig, axes
def plot_broad_phase_perf_speedup_clutter(cpu_data, gpu_data,
                                  folder_names, legend_names,
                                  objects_per_pile, sphere_resolutions, xaxis_type, with_fcl_time=False):
    """
    Two‑row grid (raw BroadPhase timings | GPU speed‑up) for clutter.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray shape (2, len(objects_per_pile), len(sphere_resolutions))
    """
    # 0 ▸ cosmetics --------------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers  = ["o", "s", "D", "^", "v"]
    lstyles  = ["-", "--", "-.", ":"]

    # 1 ▸ assemble tidy raw‑timing table ----------------------------------
    rows_raw  = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for obp in objects_per_pile:
            for sr in sphere_resolutions:
                if legend == cpu_label:                # CPU path
                    bp_dict = (store[folder_names[f_idx]][obp][sr]
                               ["timing_overall"].get("timings", {})
                               .get("BroadPhase", {}))
                    bp_time = bp_dict.get("avg_us")
                    fcl_dict = (store[folder_names[f_idx]][obp][sr]
                               ["timing_overall"].get("timings", {})
                               .get("FCLBroadPhase", {}))
                    fcl_time = fcl_dict.get("avg_us")
                    if with_fcl_time:
                        bp_time = bp_time + fcl_time
                else:                                  # GPU path
                    bp_dict = (store[folder_names[f_idx]][obp][sr]
                               ["kernel_timing"]
                               .get("kernel_timings", {})
                               .get("transform_and_broad_phase", {}))
                    bp_time = get_corrected_timing(bp_dict, gpu_data[folder_names[f_idx]][obp][sr]["raw_timing_data"], "transform_and_broad_phase")
                    fcl_dict = (gpu_data[folder_names[f_idx]][obp][sr]
                               ["timing_overall"].get("timings", {})
                               .get("FCLBroadPhase", {}))
                    fcl_time = fcl_dict.get("avg_us")
                    if with_fcl_time:
                        bp_time = bp_time + fcl_time
                if xaxis_type == "default":
                    rows_raw.append(dict(
                        Legend     = legend,
                        SphereResolution    = sr,
                        Bodies     = calculate_actual_objects_clutter(obp),
                        BPTime_us  = bp_time,
                        FCLTime_us = fcl_time
                    ))
                elif xaxis_type == "num_elements":
                    rows_raw.append(dict(
                        Legend     = legend,
                        SphereResolution    = sr,
                        Bodies     = calculate_number_of_elements_clutter(obp, sr, cpu_data[folder_names[f_idx]][obp][sr]["problem_size"]),
                        BPTime_us  = bp_time,
                        FCLTime_us = fcl_time
                    ))

    df_raw = pd.DataFrame(rows_raw)

    # 2 ▸ compute speed‑ups ------------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd    = []

    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["SphereResolution", "Bodies"])["BPTime_us"])

    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["SphereResolution", "Bodies"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["BPTime_us"]
        rows_spd.append(sub)

    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3 ▸ figure grid – sharey='row' keeps y equal in each row ------------
    n_cols = len(sphere_resolutions)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    # common y‑limits
    if with_fcl_time:
        y_raw_min, y_raw_max = df_raw["FCLTime_us"].min(), df_raw["BPTime_us"].max()
    else:
        y_raw_min, y_raw_max = df_raw["BPTime_us"].min(), df_raw["BPTime_us"].max()
    y_spd_min, y_spd_max = df_spd["SpeedUp"].min(), df_spd["SpeedUp"].max()
    y_raw_min *= 0.8;  y_raw_max *= 1.25
    y_spd_min *= 0.8;  y_spd_max *= 1.25
    if xaxis_type == "num_elements":
        # Global x-axis limits for consistent ranges across all plots
        x_min = df_raw["Bodies"].min() * 0.8
        x_max = df_raw["Bodies"].max() * 1.25

    spacing_titles = {}
    for sr in sphere_resolutions:
        spacing_titles[sr] = f"Sphere Resolution - {sr}"

    # 4 ▸ row‑0 : raw timings ---------------------------------------------
    for c, sr in enumerate(sphere_resolutions):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) &
                       (df_raw["SphereResolution"] == sr)]
            if d.empty:
                continue
            color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            ax.loglog(d["Bodies"], d["BPTime_us"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=color,
                      label=legend, zorder=3)
            if with_fcl_time:
                if legend != cpu_label:
                    ax.loglog(d["Bodies"], d["FCLTime_us"],
                            marker=markers[i % len(markers)],
                            ls=lstyles[i % len(lstyles)],
                            ms=5, lw=1.8, color=sns.color_palette()[2],
                            label=f"Geometry-BP", zorder=3, alpha=0.5)

        ax.set_ylim(y_raw_min, y_raw_max)
        if xaxis_type == "num_elements":
            ax.set_xlim(x_min, x_max)
        ax.set_title(spacing_titles.get(sr, sr),
                     fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("BroadPhase time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # internal slope indicators
        xs = df_raw[df_raw["SphereResolution"] == sr]["Bodies"].values
        ys = df_raw[df_raw["SphereResolution"] == sr]["BPTime_us"].values
        x0 = np.percentile(xs, 10)
        y0 = np.percentile(ys, 50)
        _slope_indicator_nlogn(ax, x0, y0)   # nlogn

    # 5 ▸ row‑1 : speed‑up -------------------------------------------------
    for c, sr in enumerate(sphere_resolutions):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) &
                       (df_spd["SphereResolution"] == sr)]
            if d.empty:
                continue
            ax.loglog(d["Bodies"], d["SpeedUp"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8,
                      color=sns.color_palette()[i % 10],
                      label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        if xaxis_type == "num_elements":
            ax.set_xlim(x_min, x_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if xaxis_type == "default":
            ax.set_xlabel("Number of Geometries $n$", fontsize=12)
        elif xaxis_type == "num_elements":
            ax.set_xlabel("Number of Tetrahedra $n$", fontsize=12)
        if c == 0:
            ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    # 6 ▸ finish -----------------------------------------------------------
    return fig, axes




def plot_advance_to_query_perf_speedup(
        gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, xaxis_type):
    """
    Two‑row grid (raw timings | speed‑up) with:
      • identical y‑limits within each row (so left & right columns align),
      • internal O(n²) and O(n) slope indicators,
      • colour‑blind palette & compact legends.
    Uses advance_to timing data instead of HydroelasticQuery.
    """
    # 0  Cosmetic defaults -------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers = ["o", "s", "D", "^", "v"]
    lstyles = ["-", "--", "-.", ":"]

    # 1  Build tidy table of raw timings ----------------------------------
    rows_raw = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for sr in sphere_resolutions:
            for obp in objects_per_pile:
                advance_to_data = store[folder_names[f_idx]][obp][sr]["advance_to"]
                advance_to_time = advance_to_data.get("advance_to_time", 0.0)
                # Convert to microseconds for consistency with other plots
                # advance_to_time_us = advance_to_time * 1000.0
                advance_to_time_us = advance_to_time
                if xaxis_type == "default":
                    rows_raw.append(dict(Legend=legend,
                                   SphereResolution=sr,
                                   Bodies=calculate_actual_objects_clutter(obp),
                                   AdvanceToTime_us=advance_to_time_us))
                elif xaxis_type == "num_elements":
                    rows_raw.append(dict(Legend=legend,
                                   SphereResolution=sr,
                                   Bodies=calculate_number_of_elements_clutter(obp, sr, cpu_data[folder_names[f_idx]][obp][sr]["problem_size"]),
                                   AdvanceToTime_us=advance_to_time_us))

    df_raw = pd.DataFrame(rows_raw)

    # 2  Speed‑up (= CPU / GPU) -------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd = []
    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["SphereResolution", "Bodies"])["AdvanceToTime_us"])
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["SphereResolution", "Bodies"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["AdvanceToTime_us"]
        rows_spd.append(sub)
    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3  Prepare grid – sharey='row' keeps y identical per row ------------
    n_cols = len(sphere_resolutions)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    # Pre‑compute common y‑limits
    y_raw_min, y_raw_max = df_raw["AdvanceToTime_us"].min(), df_raw["AdvanceToTime_us"].max()
    y_spd_min, y_spd_max = df_spd["SpeedUp"].min(), df_spd["SpeedUp"].max()

    # Nice padding
    y_raw_min *= 0.8
    y_raw_max *= 1.25
    y_spd_min *= 0.8
    y_spd_max *= 1.25
    
    x_min = df_raw["Bodies"].min() * 0.8
    x_max = df_raw["Bodies"].max() * 1.25

    sphere_resolution_titles = {}
    for sr in sphere_resolutions:
        sphere_resolution_titles[sr] = f"Sphere Resolution - {sr}"

    # 4  Plot raw timings (row 0) -----------------------------------------
    for c, sr in enumerate(sphere_resolutions):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) & (df_raw["SphereResolution"] == sr)]
            if d.empty:
                continue
            color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            ax.loglog(d["Bodies"], d["AdvanceToTime_us"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=color, label=legend, zorder=3)

        ax.set_ylim(y_raw_min, y_raw_max)
        ax.set_xlim(x_min, x_max)
        ax.set_title(sphere_resolution_titles.get(sr, sr), fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("AdvanceTo time [s]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # Slope indicators – place safely inside limits
        xs = df_raw[df_raw["SphereResolution"] == sr]["Bodies"].values
        ys = df_raw[df_raw["SphereResolution"] == sr]["AdvanceToTime_us"].values
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator(ax, x0, y0, 2, r"$n^{2}$")          # quadratic
        _slope_indicator(ax, x0, y0 / 4, 1, r"$n$")          # linear (below)

    # 5  Plot speed‑up (row 1) --------------------------------------------
    for c, sr in enumerate(sphere_resolutions):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) & (df_spd["SphereResolution"] == sr)]
            if d.empty:
                continue
            ax.loglog(d["Bodies"], d["SpeedUp"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=sns.color_palette()[i % 10],
                      label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if xaxis_type == "default":
            ax.set_xlabel("Number of Bodies $n$", fontsize=12)
        elif xaxis_type == "num_elements":
            ax.set_xlabel("Number of Tetrahedra $n$", fontsize=12)
        if c == 0:
            ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    # 6  Finish ------------------------------------------------------------
    return fig, axes


def plot_narrow_phase_query_perf_speedup_clutter(cpu_data, gpu_data,
                                         folder_names, legend_names,
                                         objects_per_pile, sphere_resolutions, xaxis_type):
    """
    Two‑row figure:
        • Row 0 – raw Narrow‑phase timings   (CPU + GPUs).
        • Row 1 – CPU / GPU speed‑up.

    CPU and GPU rows are matched on (Spacing, gpp) so the speed‑up
    column never turns into NaNs even when the "candidate‑tets"
    averages differ slightly.
    """
    # 0 ▸ visual defaults --------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers = ["o", "s", "D", "^", "v"]
    lstyles = ["-", "--", "-.", ":"]

    # 1 ▸ gather raw timings ----------------------------------------------
    rows_raw = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for obp in objects_per_pile:
            for sr in sphere_resolutions:
                if legend == cpu_label:                         # CPU
                    nph_dict = (store[folder_names[f_idx]][obp][sr]
                                ["timing_overall"].get("timings", {})
                                .get("NarrowPhase", {}))
                    nph_time = nph_dict.get("avg_us")
                    if xaxis_type == "num_elements":
                        prob_sz  = (store[folder_names[f_idx]][obp][sr]
                                    ["problem_size"].get("problem_sizes", {})
                                    .get("CandidateTets", {}).get("avg", None))
                else:                                           # GPU
                    nph_dict = (store[folder_names[f_idx]][obp][sr]
                                ["kernel_timing"]
                                .get("kernel_timings", {})
                                .get("compute_contact_polygons", {}))
                    nph_time = get_corrected_timing(nph_dict, gpu_data[folder_names[f_idx]][obp][sr]["raw_timing_data"], "compute_contact_polygons")
                    if xaxis_type == "num_elements":
                        prob_sz  = (store[folder_names[f_idx]][obp][sr]
                                    ["problem_size"].get("problem_sizes", {})
                                    .get("SYCLCandidateTets", {}).get("avg", None))

                if xaxis_type == "default":
                    rows_raw.append(dict(
                        Legend      = legend,
                        SphereResolution     = sr,
                        ObjectsPerPile         = calculate_actual_objects_clutter(obp),               # <- control variable
                        NPTime_us   = nph_time
                    ))
                elif xaxis_type == "num_elements":
                    rows_raw.append(dict(
                        Legend      = legend,
                        SphereResolution     = sr,
                        ObjectsPerPile         = calculate_actual_objects_clutter(obp),               # <- control variable
                        TetsProcess = prob_sz,
                        NPTime_us   = nph_time
                    ))

    # build tidy DataFrame, keep only finite values
    df_raw = (pd.DataFrame(rows_raw)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["NPTime_us"]))
    if xaxis_type == "num_elements":
        df_raw = df_raw[df_raw["TetsProcess"] > 0]
    if df_raw.empty:
        raise ValueError("No finite narrow‑phase timings were found.")

    # 2 ▸ compute speed‑ups (keyed on Spacing + gpp) -----------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]

    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["SphereResolution", "ObjectsPerPile"])["NPTime_us"])

    rows_spd = []
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(
            sub.set_index(["SphereResolution", "ObjectsPerPile"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["NPTime_us"]
        rows_spd.append(sub)

    df_spd = (pd.concat(rows_spd, ignore_index=True)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["SpeedUp"]))

    # 3 ▸ global limits --------------------------------------------------
    y_raw_min = df_raw["NPTime_us"].min() * 0.8
    y_raw_max = df_raw["NPTime_us"].max() * 1.25
    if df_spd.empty:                               # fall‑back baseline
        y_spd_min, y_spd_max = 0.8, 1.25
    else:
        y_spd_min = df_spd["SpeedUp"].min() * 0.8
        y_spd_max = df_spd["SpeedUp"].max() * 1.25
    
    if xaxis_type == "num_elements":
        # Global x-axis limits for consistent ranges across all plots
        x_min = df_raw["TetsProcess"].min() * 0.8
        x_max = df_raw["TetsProcess"].max() * 1.25

    # 4 ▸ figure grid ------------------------------------------------------
    n_cols = len(sphere_resolutions)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    spacing_titles = {}
    for sr in sphere_resolutions:
        spacing_titles[sr] = f"Sphere Resolution - {sr}"

    # 5 ▸ row‑0 : raw timings ---------------------------------------------
    for c, sr in enumerate(sphere_resolutions):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) &
                       (df_raw["SphereResolution"] == sr)]
            if d.empty:
                continue
            colour = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            if xaxis_type == "default":
                ax.loglog(d["ObjectsPerPile"], d["NPTime_us"],
                        marker=markers[i % len(markers)],
                            ls=lstyles[i % len(lstyles)],
                            ms=5, lw=1.8, color=colour,
                            label=legend, zorder=3)
            elif xaxis_type == "num_elements":
                ax.loglog(d["TetsProcess"], d["NPTime_us"],
                        marker=markers[i % len(markers)],
                            ls=lstyles[i % len(lstyles)],
                            ms=5, lw=1.8, color=colour,
                            label=legend, zorder=3)

        ax.set_ylim(y_raw_min, y_raw_max)
        if xaxis_type == "num_elements":
            ax.set_xlim(x_min, x_max)
        ax.set_title(spacing_titles.get(sr, sr),
                     fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("Narrow‑phase time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # slope indicators
        if xaxis_type == "default":
            xs = df_raw["ObjectsPerPile"].values
            ys = df_raw["NPTime_us"].values
        elif xaxis_type == "num_elements":
            xs = df_raw["TetsProcess"].values
            ys = df_raw["NPTime_us"].values
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 70)
        _slope_indicator(ax, x0, y0,   2, r"$n^{2}$")
        _slope_indicator(ax, x0, y0/4, 1, r"$n$")

    # 6 ▸ row‑1 : speed‑up -------------------------------------------------
    for c, sr in enumerate(sphere_resolutions):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) &
                       (df_spd["SphereResolution"] == sr)]
            if d.empty:
                continue
            if xaxis_type == "default":
                ax.loglog(d["ObjectsPerPile"], d["SpeedUp"],
                        marker=markers[i % len(markers)],
                            ls=lstyles[i % len(lstyles)],
                            ms=5, lw=1.8, color=sns.color_palette()[i % 10],
                            label=legend, zorder=3)
            elif xaxis_type == "num_elements":
                ax.loglog(d["TetsProcess"], d["SpeedUp"],
                        marker=markers[i % len(markers)],
                            ls=lstyles[i % len(lstyles)],
                            ms=5, lw=1.8, color=sns.color_palette()[i % 10],
                            label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        if xaxis_type == "num_elements":
            ax.set_xlim(x_min, x_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if xaxis_type == "default":
            ax.set_xlabel("Number of Geometries $n$", fontsize=12)
        elif xaxis_type == "num_elements":
            ax.set_xlabel("Tet Pairs to process $n$", fontsize=12)
        if c == 0:
            ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    return fig, axes
def main():       
    # Create plots directory
    plot_dir = "plots_gpu_comparison_clutter_Aug13"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    

    # Plot broad phase timing vs actual number of objects (log-log)
    fig, axes = plot_hydroelastic_query_perf_speedup_clutter(gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "default")
    
    plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png", dpi=300)
    print(f"Saved hydroelastic query perf speedup plot to {base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_hydroelastic_query_perf_speedup_clutter(gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "num_elements")
    
    plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_perf_speedup_vs_num_elements.png", dpi=300)
    print(f"Saved hydroelastic query perf speedup vs num elements plot to {base_dir}/{plot_dir}/hydroelastic_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup_clutter(cpu_data, gpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "default")
    
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png", dpi=300)
    print(f"Saved broad phase query perf speedup plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup_clutter(cpu_data, gpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "num_elements")
    
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png", dpi=300)
    print(f"Saved broad phase query perf speedup vs num elements plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup_clutter(cpu_data, gpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "default", with_fcl_time=True)
    
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl.png", dpi=300)
    print(f"Saved broad phase query perf speedup with fcl plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup_clutter(cpu_data, gpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "num_elements", with_fcl_time=True)
    
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl_vs_num_elements.png", dpi=300)
    print(f"Saved broad phase query perf speedup with fcl vs num elements plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_narrow_phase_query_perf_speedup_clutter(cpu_data, gpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "default")
    
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png", dpi=300)
    print(f"Saved narrow phase query perf speedup plot to {base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    
    fig, axes = plot_narrow_phase_query_perf_speedup_clutter(cpu_data, gpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "num_elements")
    
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_query_perf_speedup_vs_num_elements.png", dpi=300)
    print(f"Saved narrow phase query perf speedup vs num elements plot to {base_dir}/{plot_dir}/narrow_phase_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_advance_to_query_perf_speedup(gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "num_elements")
    
    plt.savefig(f"{base_dir}/{plot_dir}/advance_to_query_perf_speedup_vs_num_elements.png", dpi=300)
    print(f"Saved advance_to query perf speedup vs num elements plot to {base_dir}/{plot_dir}/advance_to_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_advance_to_query_perf_speedup(gpu_data, cpu_data, folder_names, legend_names, objects_per_pile, sphere_resolutions, "default")
    
    plt.savefig(f"{base_dir}/{plot_dir}/advance_to_query_perf_speedup_vs_obp.png", dpi=300)
    print(f"Saved advance_to query perf speedup vs obp plot to {base_dir}/{plot_dir}/advance_to_query_perf_speedup_vs_obp.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare multiple GPU/CPU versions with line plots and bar plots')
    parser.add_argument('--folders', nargs='+', required=True, help='Names of the performance data folders')
    parser.add_argument('--legends', nargs='+', required=True, help='Legend names for each version')
    args = parser.parse_args()
    
    # Validate that we have the same number of folders and legends
    if len(args.folders) != len(args.legends):
        print("Error: Number of folders must match number of legends")
        sys.exit(1)
    
    base_dir = os.path.dirname(os.getcwd())
    demo_name = "clutter"
    objects_per_pile = ["1", "2", "5", "10", "20", "33", "50"]
    sphere_resolutions = ["0.0050", "0.0100", "0.0200", "0.0400"]
    
    folder_names = args.folders
    legend_names = args.legends
    
    # Store all data in a nested dictionary: all_data[folder_name][spacing][num_gpp][data_type]
    gpu_data = {}
    cpu_data = {}
    
    for i, folder_name in enumerate(folder_names):
        legend_lower = legend_names[i].lower()
        if "cpu" in legend_lower:
            # CPU data
            if folder_name not in cpu_data:
                cpu_data[folder_name] = {}
            for obp in objects_per_pile:
                if obp not in cpu_data[folder_name]:
                    cpu_data[folder_name][obp] = {}
                for sr in sphere_resolutions:
                    if sr not in cpu_data[folder_name][obp]:
                        cpu_data[folder_name][obp][sr] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_drake-cpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    cpu_data[folder_name][obp][sr]["timing_overall"] = data_timing_overall
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_drake-cpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    cpu_data[folder_name][obp][sr]["problem_size"] = data_problem_size
                    
                    json_path_advance_to = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_drake-cpu_timing_advance_to.json"
                    data_advance_to = get_data(json_path_advance_to)
                    cpu_data[folder_name][obp][sr]["advance_to"] = data_advance_to
        else:
            # GPU data
            if folder_name not in gpu_data:
                gpu_data[folder_name] = {}
            for obp in objects_per_pile:
                if obp not in gpu_data[folder_name]:
                    gpu_data[folder_name][obp] = {}
                for sr in sphere_resolutions:
                    if sr not in gpu_data[folder_name][obp]:
                        gpu_data[folder_name][obp][sr] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_sycl-gpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    gpu_data[folder_name][obp][sr]["timing_overall"] = data_timing_overall
                    # Kernel timing
                    json_path_kernel_timing = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_sycl-gpu_timing.json"
                    data_kernel_timing = get_data(json_path_kernel_timing)
                    gpu_data[folder_name][obp][sr]["kernel_timing"] = data_kernel_timing
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_sycl-gpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    gpu_data[folder_name][obp][sr]["problem_size"] = data_problem_size
                    
                    json_path_advance_to = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_sycl-gpu_timing_advance_to.json"
                    data_advance_to = get_data(json_path_advance_to)
                    gpu_data[folder_name][obp][sr]["advance_to"] = data_advance_to
                    
                    txt_base_path = f"{base_dir}/{folder_name}/{demo_name}_{obp}_1.0000_{sr}_3_sycl-gpu_timing_"
                    
                    kernel_keys = ["unpack_transforms", "transform_and_broad_phase", "device_to_host_memcpy", "compute_contact_polygons", "compact_polygon_data"]
                    gpu_data[folder_name][obp][sr]["raw_timing_data"] = {}
                    for key in kernel_keys:
                        txt_path = f"{txt_base_path}{key}.txt"
                        with open(txt_path, "r") as f:
                            third_line = f.readlines()[2]
                            third_line_split_cost = float(third_line.split(" ")[1])
                            gpu_data[folder_name][obp][sr]["raw_timing_data"][key] = third_line_split_cost
                            
                            
                            
                    
    main()