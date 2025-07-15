import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import numpy as np
from utils import get_data
import pandas as pd

def calculate_actual_objects(gpp):
    """
    Calculate the actual number of objects based on GPP (grippers per pepper).
    Each gripper has 2 bodies, each pepper has 1 body, and there's 1 table.
    Formula: 2 * gpp + gpp + 1 = 3 * gpp + 1
    """
    return 3 * int(gpp) + 1
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

def calculate_number_of_elements_objects_scaling(gpp, problem_size_data):
    num_floors = 1
    num_peppers = int(gpp)
    num_grippers = int(gpp)
    
    # Initialize element counts
    floor_elements = 0
    pepper_elements = 0
    left_gripper_elements = 0
    right_gripper_elements = 0
    
    # Get hydroelastic_bodies array
    hydroelastic_bodies = problem_size_data.get("hydroelastic_bodies", [])
    
    # Process each body in the array
    for body_data in hydroelastic_bodies:
        body_name = body_data.get("body", "")
        tetrahedra = int(body_data.get("tetrahedra", 0))
        
        if body_name == "Floor":
            floor_elements = tetrahedra * num_floors
        elif body_name == "yellow_bell_pepper_no_stem":
            pepper_elements = tetrahedra * num_peppers
        elif body_name == "left_finger_bubble":
            left_gripper_elements = tetrahedra * num_grippers
        elif body_name == "right_finger_bubble":
            right_gripper_elements = tetrahedra * num_grippers
    
    gripper_elements = left_gripper_elements + right_gripper_elements
    return floor_elements + pepper_elements + gripper_elements

def plot_hydroelastic_query_perf_speedup(
        gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp, xaxis_type):
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
        for spacing in spacings:
            for gpp in num_gpp:
                timing_dict = store[folder_names[f_idx]][spacing][gpp] \
                                  ["timing_overall"].get("timings", {})
                hq_time = timing_dict.get("HydroelasticQuery", {}).get("avg_us")
                
                if xaxis_type == "default":
                    rows_raw.append(dict(Legend=legend,
                                        Spacing=spacing,
                                        Bodies=calculate_actual_objects(gpp),
                                        HQTime_us=hq_time))
                elif xaxis_type == "num_elements":
                    rows_raw.append(dict(Legend=legend,
                                        Spacing=spacing,
                                        Bodies=calculate_number_of_elements_objects_scaling(gpp, cpu_data[folder_names[f_idx]][spacing][gpp]["problem_size"]),
                                        HQTime_us=hq_time))

    df_raw = pd.DataFrame(rows_raw)

    # 2  Speed‑up (= CPU / GPU) -------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd = []
    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Spacing", "Bodies"])["HQTime_us"])
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["Spacing", "Bodies"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["HQTime_us"]
        rows_spd.append(sub)
    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3  Prepare grid – sharey='row' keeps y identical per row ------------
    n_cols = len(spacings)
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

    spacing_titles = {}
    for spacing in spacings:
        if(spacing == "0.1"):
            spacing_titles[spacing] = "Sparse - 0.1"
        elif(spacing == "0.15"):
            spacing_titles[spacing] = "Sparse - 0.15"
        elif(spacing == "0.05"):
            spacing_titles[spacing] = "Dense - 0.05"

    # 4  Plot raw timings (row 0) -----------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) & (df_raw["Spacing"] == spacing)]
            if d.empty:
                continue
            color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            ax.loglog(d["Bodies"], d["HQTime_us"],
                      marker=markers[i % len(markers)],
                      ls=lstyles[i % len(lstyles)],
                      ms=5, lw=1.8, color=color, label=legend, zorder=3)

        ax.set_ylim(y_raw_min, y_raw_max)
        ax.set_title(spacing_titles.get(spacing, spacing), fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("HydroelasticQuery time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # Slope indicators – place safely inside limits
        xs = df_raw[df_raw["Spacing"] == spacing]["Bodies"].values
        ys = df_raw[df_raw["Spacing"] == spacing]["HQTime_us"].values
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator(ax, x0, y0, 2, r"$n^{2}$")          # quadratic
        _slope_indicator(ax, x0, y0 / 4, 1, r"$n$")          # linear (below)

    # 5  Plot speed‑up (row 1) --------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) & (df_spd["Spacing"] == spacing)]
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

def plot_broad_phase_perf_speedup(cpu_data, gpu_data,
                                  folder_names, legend_names,
                                  spacings, num_gpp, xaxis_type, with_fcl_time=False):
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
        for spacing in spacings:
            for gpp in num_gpp:
                if legend == cpu_label:                # CPU path
                    bp_dict = (store[folder_names[f_idx]][spacing][gpp]
                               ["timing_overall"].get("timings", {})
                               .get("BroadPhase", {}))
                    bp_time = bp_dict.get("avg_us")
                    fcl_bp_dict = (store[folder_names[f_idx]][spacing][gpp]
                               ["timing_overall"].get("timings", {})
                               .get("FCLBroadPhase", {}))
                    fcl_bp_time = fcl_bp_dict.get("avg_us")
                    if with_fcl_time:
                        bp_time = bp_time + fcl_bp_time
                else:                                  # GPU path
                    bp_dict = (store[folder_names[f_idx]][spacing][gpp]
                               ["kernel_timing"]
                               .get("kernel_timings", {})
                               .get("transform_and_broad_phase", {}))
                    bp_time = get_corrected_timing(bp_dict, gpu_data[folder_names[f_idx]][spacing][gpp]["raw_timing_data"], "transform_and_broad_phase")
                    fcl_bp_dict = (store[folder_names[f_idx]][spacing][gpp]
                               ["timing_overall"].get("timings", {})
                               .get("FCLBroadPhase", {}))
                    fcl_bp_time = fcl_bp_dict.get("avg_us")
                    if with_fcl_time:
                        bp_time = bp_time + fcl_bp_time
                if xaxis_type == "default":
                    rows_raw.append(dict(
                        Legend     = legend,
                        Spacing    = spacing,
                        Bodies     = calculate_actual_objects(gpp),
                        BPTime_us  = bp_time,
                        FCLTime_us = fcl_bp_time
                    ))
                elif xaxis_type == "num_elements":
                    rows_raw.append(dict(
                        Legend     = legend,
                        Spacing    = spacing,
                        Bodies     = calculate_number_of_elements_objects_scaling(gpp, cpu_data[folder_names[f_idx]][spacing][gpp]["problem_size"]),
                        BPTime_us  = bp_time,
                        FCLTime_us = fcl_bp_time
                    ))

    df_raw = pd.DataFrame(rows_raw)

    # 2 ▸ compute speed‑ups ------------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd    = []

    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Spacing", "Bodies"])["BPTime_us"])

    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["Spacing", "Bodies"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["BPTime_us"]
        rows_spd.append(sub)

    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3 ▸ figure grid – sharey='row' keeps y equal in each row ------------
    n_cols = len(spacings)
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

    spacing_titles = {}
    for spacing in spacings:
        if(spacing == "0.1"):
            spacing_titles[spacing] = "0.1"
        elif(spacing == "0.15"):
            spacing_titles[spacing] = "0.15"
        elif(spacing == "0.05"):
            spacing_titles[spacing] = "0.05"

    # 4 ▸ row‑0 : raw timings ---------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) &
                       (df_raw["Spacing"] == spacing)]
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
        ax.set_title(spacing_titles.get(spacing, spacing),
                     fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("BroadPhase time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # internal slope indicators
        xs = df_raw[df_raw["Spacing"] == spacing]["Bodies"].values
        ys = df_raw[df_raw["Spacing"] == spacing]["BPTime_us"].values
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator_nlogn(ax, x0, y0)   # nlogn

    # 5 ▸ row‑1 : speed‑up -------------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) &
                       (df_spd["Spacing"] == spacing)]
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

def plot_narrow_phase_query_perf_speedup(cpu_data, gpu_data,
                                         folder_names, legend_names,
                                         spacings, num_gpp, xaxis_type):
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
        for spacing in spacings:
            for gpp in num_gpp:
                if legend == cpu_label:                         # CPU
                    nph_dict = (store[folder_names[f_idx]][spacing][gpp]
                                ["timing_overall"].get("timings", {})
                                .get("NarrowPhase", {}))
                    nph_time = nph_dict.get("avg_us")
                    if xaxis_type == "num_elements":
                        prob_sz  = (store[folder_names[f_idx]][spacing][gpp]
                                    ["problem_size"].get("problem_sizes", {})
                                    .get("CandidateTets", {}).get("avg", None))
                else:                                           # GPU
                    nph_dict = (store[folder_names[f_idx]][spacing][gpp]
                                ["kernel_timing"]
                                .get("kernel_timings", {})
                                .get("compute_contact_polygons", {}))
                    nph_time = get_corrected_timing(nph_dict, gpu_data[folder_names[f_idx]][spacing][gpp]["raw_timing_data"], "compute_contact_polygons")
                    if xaxis_type == "num_elements":
                        prob_sz  = (store[folder_names[f_idx]][spacing][gpp]
                                    ["problem_size"].get("problem_sizes", {})
                                    .get("SYCLCandidateTets", {}).get("avg", None))
                if xaxis_type == "default":
                    rows_raw.append(dict(
                        Legend      = legend,
                        Spacing     = spacing,
                        gpp         = calculate_actual_objects(gpp),               
                        NPTime_us   = nph_time
                    ))
                elif xaxis_type == "num_elements":
                    rows_raw.append(dict(
                        Legend      = legend,
                        Spacing     = spacing,
                        gpp         = calculate_actual_objects(gpp),               
                        TetsProcess = prob_sz,
                        NPTime_us   = nph_time
                    ))

    # build tidy DataFrame, keep only finite values
    df_raw = (pd.DataFrame(rows_raw)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["NPTime_us"]))
    if df_raw.empty:
        raise ValueError("No finite narrow‑phase timings were found.")

    # 2 ▸ compute speed‑ups (keyed on Spacing + gpp) -----------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]

    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Spacing", "gpp"])["NPTime_us"])

    rows_spd = []
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(
            sub.set_index(["Spacing", "gpp"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["NPTime_us"]
        rows_spd.append(sub)

    df_spd = (pd.concat(rows_spd, ignore_index=True)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["SpeedUp"]))

    # 3 ▸ global y‑limits --------------------------------------------------
    y_raw_min = df_raw["NPTime_us"].min() * 0.8
    y_raw_max = df_raw["NPTime_us"].max() * 1.25
    if df_spd.empty:                               # fall‑back baseline
        y_spd_min, y_spd_max = 0.8, 1.25
    else:
        y_spd_min = df_spd["SpeedUp"].min() * 0.8
        y_spd_max = df_spd["SpeedUp"].max() * 1.25

    # Global x-axis limits for consistent ranges across all plots
    if xaxis_type == "num_elements":
        # Global x-axis limits for consistent ranges across all plots
        x_min = df_raw["TetsProcess"].min() * 0.8
        x_max = df_raw["TetsProcess"].max() * 1.25

    # 4 ▸ figure grid ------------------------------------------------------
    n_cols = len(spacings)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 6.5),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    spacing_titles = {}
    for spacing in spacings:
        if(spacing == "0.1"):
            spacing_titles[spacing] = "0.1"
        elif(spacing == "0.15"):
            spacing_titles[spacing] = "0.15"
        elif(spacing == "0.05"):
            spacing_titles[spacing] = "0.05"

    # 5 ▸ row‑0 : raw timings ---------------------------------------------
    for c, spacing in enumerate(spacings):
        ax = axes[0, c]
        for i, legend in enumerate(legend_names):
            d = df_raw[(df_raw["Legend"] == legend) &
                       (df_raw["Spacing"] == spacing)]
            if d.empty:
                continue
            colour = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
            
            if xaxis_type == "default":
                ax.loglog(d["gpp"], d["NPTime_us"],
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
        ax.set_title(spacing_titles.get(spacing, spacing),
                     fontsize=13, weight="bold")
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        if c == 0:
            ax.set_ylabel("Narrow‑phase time [µs]", fontsize=12)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

        # slope indicators
        if xaxis_type == "default":
            xs = df_raw["gpp"].values
            ys = df_raw["NPTime_us"].values
        elif xaxis_type == "num_elements":
            xs = df_raw["TetsProcess"].values
            ys = df_raw["NPTime_us"].values
        if len(xs) and len(ys):
            x0 = np.percentile(xs, 25)
            y0 = np.percentile(ys, 60)
            _slope_indicator(ax, x0, y0,   2, r"$n^{2}$")
            _slope_indicator(ax, x0, y0/4, 1, r"$n$")

    # 6 ▸ row‑1 : speed‑up -------------------------------------------------
    for c, sr in enumerate(spacings):
        ax = axes[1, c]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend) &
                       (df_spd["Spacing"] == sr)]
            if d.empty:
                continue
            if xaxis_type == "default":
                ax.loglog(d["gpp"], d["SpeedUp"],
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
    demo_name = "objects_scaling"
    spacings = ["0.05", "0.1", "0.15"]
    num_gpp = ["1", "2", "5", "10", "20", "33", "50", "100"]
    
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
            for spacing in spacings:
                if spacing not in cpu_data[folder_name]:
                    cpu_data[folder_name][spacing] = {}
                for gpp in num_gpp:
                    if gpp not in cpu_data[folder_name][spacing]:
                        cpu_data[folder_name][spacing][gpp] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_drake-cpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    cpu_data[folder_name][spacing][gpp]["timing_overall"] = data_timing_overall
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_drake-cpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    cpu_data[folder_name][spacing][gpp]["problem_size"] = data_problem_size
        else:
            # GPU data
            if folder_name not in gpu_data:
                gpu_data[folder_name] = {}
            for spacing in spacings:
                if spacing not in gpu_data[folder_name]:
                    gpu_data[folder_name][spacing] = {}
                for gpp in num_gpp:
                    if gpp not in gpu_data[folder_name][spacing]:
                        gpu_data[folder_name][spacing][gpp] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    gpu_data[folder_name][spacing][gpp]["timing_overall"] = data_timing_overall
                    # Kernel timing
                    json_path_kernel_timing = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_timing.json"
                    data_kernel_timing = get_data(json_path_kernel_timing)
                    gpu_data[folder_name][spacing][gpp]["kernel_timing"] = data_kernel_timing
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    gpu_data[folder_name][spacing][gpp]["problem_size"] = data_problem_size
                    
                    txt_base_path = f"{base_dir}/{folder_name}/{demo_name}_{spacing}_{gpp}_sycl-gpu_timing_"
                    
                    kernel_keys = ["unpack_transforms", "transform_and_broad_phase", "device_to_host_memcpy", "compute_contact_polygons", "compact_polygon_data"]
                    gpu_data[folder_name][spacing][gpp]["raw_timing_data"] = {}
                    for key in kernel_keys:
                        txt_path = f"{txt_base_path}{key}.txt"
                        with open(txt_path, "r") as f:
                            third_line = f.readlines()[2]
                            third_line_split_cost = float(third_line.split(" ")[1])
                            gpu_data[folder_name][spacing][gpp]["raw_timing_data"][key] = third_line_split_cost
                    
    # Create plots directory
    plot_dir = "plots_gpu_comparison_object_scaling_Aug20"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    

    # Plot broad phase timing vs actual number of objects (log-log)
    fig, axes = plot_hydroelastic_query_perf_speedup(gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp, xaxis_type="default")
    plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png", dpi=600)
    print(f"Saved hydroelastic query perf speedup plot to {base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_hydroelastic_query_perf_speedup(gpu_data, cpu_data, folder_names, legend_names, spacings, num_gpp, "num_elements")
    plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_perf_speedup_vs_num_elements.png", dpi=600)
    print(f"Saved hydroelastic query perf speedup vs num elements plot to {base_dir}/{plot_dir}/hydroelastic_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp, xaxis_type="default", with_fcl_time=False)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png", dpi=600)
    print(f"Saved broad phase query perf speedup plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp, xaxis_type="num_elements", with_fcl_time=False)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png", dpi=600)
    print(f"Saved broad phase query perf speedup vs num elements plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp, xaxis_type="default", with_fcl_time=True)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl.png", dpi=600)
    print(f"Saved broad phase query perf speedup with fcl plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp, xaxis_type="num_elements", with_fcl_time=True)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements_with_fcl.png", dpi=600)
    print(f"Saved broad phase query perf speedup vs num elements with fcl plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements_with_fcl.png")
    plt.show()
    plt.close()
    

    fig, axes = plot_narrow_phase_query_perf_speedup(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp, xaxis_type="default")
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png", dpi=600)
    print(f"Saved narrow phase query perf speedup plot to {base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_narrow_phase_query_perf_speedup(cpu_data, gpu_data, folder_names, legend_names, spacings, num_gpp, xaxis_type="num_elements")
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_query_perf_speedup_vs_num_elements.png", dpi=600)
    print(f"Saved narrow phase query perf speedup vs num elements plot to {base_dir}/{plot_dir}/narrow_phase_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
if __name__ == "__main__":
    main()