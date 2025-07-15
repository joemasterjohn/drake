import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import numpy as np
import pandas as pd
from utils import get_data


def get_corrected_timing(timing_data,raw_timing_data,key):
    """
    Get corrected timing for sycl by excluding JIT compilation time.
    For sycl-cpu/sycl-gpu: read the txt file and remove the first timing
    """
    calls = int(timing_data.get("calls", 1))
    first_time = float(raw_timing_data[key])
    total_time = float(timing_data.get("total_us", 0))
    return (total_time - first_time) / (calls - 1)

def calculate_number_of_elements_spatula(env, problem_size_data):
    # Get hydroelastic_bodies array
    hydroelastic_bodies = problem_size_data.get("hydroelastic_bodies", [])
    total_tets = 0
    for body_data in hydroelastic_bodies:
        body_name = body_data.get("body", "")
        tetrahedra = int(body_data.get("tetrahedra", 0))
        total_tets += tetrahedra
        
    return total_tets
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

def plot_hydroelastic_query_perf_speedup_spatula(
        gpu_data, cpu_data, folder_names, legend_names, envs, xaxis_type):
    """
    Two‑row grid (raw timings | speed‑up) with a single column where the
    x‑axis is the number of environments. Keeps log‑log axes, colour‑blind
    palette, and compact legends.
    """
    # 0  Cosmetic defaults -------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers = ["o", "s", "D", "^", "v"]
    lstyles = ["-", "--", "-.", ":"]

    # 1  Build tidy table of raw timings ----------------------------------
    rows_raw = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    env_ints = sorted(int(e) for e in envs)
    env_strs = [str(e) for e in env_ints]

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for env_str, env_int in zip(env_strs, env_ints):
            timing_dict = store[folder_names[f_idx]][env_str]["timing_overall"].get("timings", {})
            hq_time = timing_dict.get("HydroelasticQuery", {}).get("avg_us")
            if xaxis_type == "default":
                rows_raw.append(dict(Legend=legend,
                                    Env=env_int,
                                    HQTime_us=hq_time))
            elif xaxis_type == "num_elements":
                rows_raw.append(dict(Legend=legend,
                                    Env=calculate_number_of_elements_spatula(env_int, cpu_data[folder_names[f_idx]][env_str]["problem_size"]),
                                    HQTime_us=hq_time))

    df_raw = pd.DataFrame(rows_raw)

    # 2  Speed‑up (= CPU / GPU) -------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd = []
    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Env"])["HQTime_us"])
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["Env"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["HQTime_us"]
        rows_spd.append(sub)
    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3  Prepare grid – sharey='row' keeps y identical per row ------------
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.5),
                             sharex=True, sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    axes = np.array(axes).reshape(2, 1)

    # Pre‑compute common y‑limits
    y_raw_min, y_raw_max = df_raw["HQTime_us"].min(), df_raw["HQTime_us"].max()
    y_spd_min, y_spd_max = df_spd["SpeedUp"].min(), df_spd["SpeedUp"].max()

    # Nice padding
    y_raw_min *= 0.8
    y_raw_max *= 1.25
    y_spd_min *= 0.8
    y_spd_max *= 1.25

    # 4  Plot raw timings (row 0) -----------------------------------------
    ax = axes[0, 0]
    for i, legend in enumerate(legend_names):
        d = df_raw[(df_raw["Legend"] == legend)].copy().sort_values("Env")
        if d.empty:
            continue
        color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
        ax.loglog(d["Env"], d["HQTime_us"],
                  marker=markers[i % len(markers)],
                  ls=lstyles[i % len(lstyles)],
                  ms=5, lw=1.8, color=color, label=legend, zorder=3)

    ax.set_ylim(y_raw_min, y_raw_max)
    ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
    ax.set_ylabel("HydroelasticQuery time [µs]", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # Slope indicators – place safely inside limits
    xs = df_raw["Env"].values
    ys = df_raw["HQTime_us"].values
    if len(xs) and len(ys):
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator(ax, x0, y0, 2, r"$n^{2}$")          # quadratic
        _slope_indicator(ax, x0, y0 / 4, 1, r"$n$")          # linear (below)

    # 5  Plot speed‑up (row 1) --------------------------------------------
    ax = axes[1, 0]
    for i, legend in enumerate(gpu_legends):
        d = df_spd[(df_spd["Legend"] == legend)].copy().sort_values("Env")
        if d.empty:
            continue
        ax.loglog(d["Env"], d["SpeedUp"],
                  marker=markers[i % len(markers)],
                  ls=lstyles[i % len(lstyles)],
                  ms=5, lw=1.8, color=sns.color_palette()[i % 10],
                  label=legend, zorder=3)

    ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
    ax.set_ylim(y_spd_min, y_spd_max)
    ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
    if xaxis_type == "default":
        ax.set_xlabel("Number of Environments $n$", fontsize=12)
    elif xaxis_type == "num_elements":
        ax.set_xlabel("Number of Tetrahedra $n$", fontsize=12)
    ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # 6  Finish ------------------------------------------------------------
    return fig, axes

def plot_broad_phase_perf_speedup_spatula(cpu_data, gpu_data,
                                  folder_names, legend_names,
                                  envs, xaxis_type, with_fcl_time=False):
    """
    Two‑row grid (raw BroadPhase timings | GPU speed‑up) vs number of
    environments, using a single column. Keeps log‑log axes and styling.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray shape (2, 1)
    """
    # 0 ▸ cosmetics --------------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers  = ["o", "s", "D", "^", "v"]
    lstyles  = ["-", "--", "-.", ":"]

    # 1 ▸ assemble tidy raw‑timing table ----------------------------------
    rows_raw  = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    env_ints = sorted(int(e) for e in envs)
    env_strs = [str(e) for e in env_ints]

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for env_str, env_int in zip(env_strs, env_ints):
            if legend == cpu_label:                # CPU path
                bp_dict = (store[folder_names[f_idx]][env_str]
                           ["timing_overall"].get("timings", {})
                           .get("BroadPhase", {}))
                bp_time = bp_dict.get("avg_us")
                fcl_bp_dict = (store[folder_names[f_idx]][env_str]
                           ["timing_overall"].get("timings", {})
                           .get("FCLBroadPhase", {}))
                fcl_bp_time = fcl_bp_dict.get("avg_us")
                if with_fcl_time:
                    bp_time = bp_time + fcl_bp_time
            else:                                  # GPU path
                bp_dict = (store[folder_names[f_idx]][env_str]
                           ["kernel_timing"]
                           .get("kernel_timings", {})
                           .get("transform_and_broad_phase", {}))
                bp_time = get_corrected_timing(bp_dict, gpu_data[folder_names[f_idx]][env_str]["raw_timing_data"], "transform_and_broad_phase")
                fcl_bp_dict = (store[folder_names[f_idx]][env_str]
                           ["timing_overall"]
                           .get("timings", {})
                           .get("FCLBroadPhase", {}))
                fcl_bp_time = fcl_bp_dict.get("avg_us")
                if with_fcl_time:
                    bp_time = bp_time + fcl_bp_time

            if xaxis_type == "default":
                rows_raw.append(dict(
                    Legend    = legend,
                    Env       = env_int,
                    BPTime_us = bp_time,
                    FCLTime_us = fcl_bp_time
                ))
            elif xaxis_type == "num_elements":
                rows_raw.append(dict(
                    Legend    = legend,
                    Env       = calculate_number_of_elements_spatula(env_int, cpu_data[folder_names[f_idx]][env_str]["problem_size"]),
                    BPTime_us = bp_time,
                    FCLTime_us = fcl_bp_time
                ))

    df_raw = pd.DataFrame(rows_raw)

    # 2 ▸ compute speed‑ups ------------------------------------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]
    rows_spd    = []

    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Env"])["BPTime_us"])

    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(sub.set_index(["Env"]).index).values
        sub["SpeedUp"] = sub["CPU_us"] / sub["BPTime_us"]
        rows_spd.append(sub)

    df_spd = pd.concat(rows_spd, ignore_index=True)

    # 3 ▸ figure grid – sharey='row' keeps y equal in each row ------------
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.5),
                             sharex=True, sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    axes = np.array(axes).reshape(2, 1)

    # common y‑limits
    if with_fcl_time:
        y_raw_min, y_raw_max = df_raw["FCLTime_us"].min(), df_raw["BPTime_us"].max()
    else:
        y_raw_min, y_raw_max = df_raw["BPTime_us"].min(), df_raw["BPTime_us"].max()
    y_spd_min, y_spd_max = df_spd["SpeedUp"].min(), df_spd["SpeedUp"].max()
    y_raw_min *= 0.8;  y_raw_max *= 1.25
    y_spd_min *= 0.8;  y_spd_max *= 1.25

    # 4 ▸ row‑0 : raw timings ---------------------------------------------
    ax = axes[0, 0]
    for i, legend in enumerate(legend_names):
        d = df_raw[(df_raw["Legend"] == legend)].copy().sort_values("Env")
        if d.empty:
            continue
        color = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
        ax.loglog(d["Env"], d["BPTime_us"],
                  marker=markers[i % len(markers)],
                  ls=lstyles[i % len(lstyles)],
                  ms=5, lw=1.8, color=color,
                  label=legend, zorder=3)
        if with_fcl_time:
            if legend != cpu_label:
                ax.loglog(d["Env"], d["FCLTime_us"],
                        marker=markers[i % len(markers)],
                        ls=lstyles[i % len(lstyles)],
                        ms=5, lw=1.8, color=sns.color_palette()[2],
                        label=f"Geometry-BP", zorder=3, alpha=0.5)

    ax.set_ylim(y_raw_min, y_raw_max)
    ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
    ax.set_ylabel("BroadPhase time [µs]", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # internal slope indicator (n log n)
    xs = df_raw["Env"].values
    ys = df_raw["BPTime_us"].values
    if len(xs) and len(ys):
        x0 = np.percentile(xs, 25)
        y0 = np.percentile(ys, 30)
        _slope_indicator_nlogn(ax, x0, y0)

    # 5 ▸ row‑1 : speed‑up -------------------------------------------------
    ax = axes[1, 0]
    for i, legend in enumerate(gpu_legends):
        d = df_spd[(df_spd["Legend"] == legend)].copy().sort_values("Env")
        if d.empty:
            continue
        ax.loglog(d["Env"], d["SpeedUp"],
                  marker=markers[i % len(markers)],
                  ls=lstyles[i % len(lstyles)],
                  ms=5, lw=1.8,
                  color=sns.color_palette()[i % 10],
                  label=legend, zorder=3)

    ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
    ax.set_ylim(y_spd_min, y_spd_max)
    ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
    if xaxis_type == "default":
        ax.set_xlabel("Number of Environments $n$", fontsize=12)
    elif xaxis_type == "num_elements":
        ax.set_xlabel("Number of Tetrahedra $n$", fontsize=12)
    ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # 6 ▸ finish -----------------------------------------------------------
    return fig, axes

def plot_narrow_phase_query_perf_speedup_spatula(cpu_data, gpu_data,
                                         folder_names, legend_names,
                                         envs, xaxis_type):
    """
    Two‑row figure with a single column where the x‑axis is the number of
    environments:
        • Row 0 – raw Narrow‑phase timings   (CPU + GPUs).
        • Row 1 – CPU / GPU speed‑up.

    Keeps log‑log axes, colour‑blind palette, and compact legends.
    """
    # 0 ▸ visual defaults --------------------------------------------------
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    markers = ["o", "s", "D", "^", "v"]
    lstyles = ["-", "--", "-.", ":"]

    # 1 ▸ gather raw timings ----------------------------------------------
    rows_raw = []
    cpu_label = next(lbl for lbl in legend_names if "cpu" in lbl.lower())

    env_ints = sorted(int(e) for e in envs)
    env_strs = [str(e) for e in env_ints]

    for f_idx, legend in enumerate(legend_names):
        store = cpu_data if legend == cpu_label else gpu_data
        for env_str, env_int in zip(env_strs, env_ints):
            if legend == cpu_label:                         # CPU
                nph_dict = (store[folder_names[f_idx]][env_str]
                            ["timing_overall"].get("timings", {})
                            .get("NarrowPhase", {}))
                nph_time = nph_dict.get("avg_us")
                if xaxis_type == "num_elements":
                    prob_sz  = (store[folder_names[f_idx]][env_str]
                                ["problem_size"].get("problem_sizes", {})
                                .get("CandidateTets", {}).get("avg", None))
            else:                                           # GPU
                nph_dict = (store[folder_names[f_idx]][env_str]
                            ["kernel_timing"]
                            .get("kernel_timings", {})
                            .get("compute_contact_polygons", {}))
                nph_time = get_corrected_timing(nph_dict, gpu_data[folder_names[f_idx]][env_str]["raw_timing_data"], "compute_contact_polygons")
                if xaxis_type == "num_elements":
                    prob_sz  = (store[folder_names[f_idx]][env_str]
                                ["problem_size"].get("problem_sizes", {})
                                .get("SYCLCandidateTets", {}).get("avg", None))
            if xaxis_type == "default":
                rows_raw.append(dict(
                    Legend    = legend,
                    Env       = env_int,
                    NPTime_us = nph_time
                ))
            elif xaxis_type == "num_elements":
                rows_raw.append(dict(
                    Legend    = legend,
                    Env       = env_int,
                    TetsProcess = prob_sz,
                    NPTime_us = nph_time
                ))

    # build tidy DataFrame, keep only finite values
    df_raw = (pd.DataFrame(rows_raw)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["NPTime_us"]))
    if df_raw.empty:
        raise ValueError("No finite narrow‑phase timings were found.")

    # 2 ▸ compute speed‑ups (keyed on Env) -----------------------
    gpu_legends = [l for l in legend_names if l != cpu_label]

    cpu_tbl = (df_raw[df_raw["Legend"] == cpu_label]
               .set_index(["Env"])["NPTime_us"])

    rows_spd = []
    for legend in gpu_legends:
        sub = df_raw[df_raw["Legend"] == legend].copy()
        sub["CPU_us"] = cpu_tbl.reindex(
            sub.set_index(["Env"]).index).values
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
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.5),
                             sharex=True, sharey="row",
                             gridspec_kw=dict(hspace=0.10, wspace=0.15))
    axes = np.array(axes).reshape(2, 1)

    # 5 ▸ row‑0 : raw timings ---------------------------------------------
    ax = axes[0, 0]
    for i, legend in enumerate(legend_names):
        d = df_raw[(df_raw["Legend"] == legend)].copy().sort_values("Env")
        if d.empty:
            continue
        colour = "0.25" if legend == cpu_label else sns.color_palette()[i % 10]
        if xaxis_type == "default":
            ax.loglog(d["Env"], d["NPTime_us"],
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
    ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
    ax.set_ylabel("Narrow‑phase time [µs]", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # slope indicators
    if xaxis_type == "default":
        xs = df_raw["Env"].values
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
    if xaxis_type == "default":
        ax = axes[1, 0]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend)].copy().sort_values("Env")
            if d.empty:
                continue
            ax.loglog(d["Env"], d["SpeedUp"],
                    marker=markers[i % len(markers)],
                    ls=lstyles[i % len(lstyles)],
                    ms=5, lw=1.8,
                    color=sns.color_palette()[i % 10],
                    label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        ax.set_xlabel("Number of Environments $n$", fontsize=12)
        ax.set_ylabel("CPU / GPU speed‑up", fontsize=12)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    elif xaxis_type == "num_elements":
        ax = axes[1, 0]
        for i, legend in enumerate(gpu_legends):
            d = df_spd[(df_spd["Legend"] == legend)]
            if d.empty:
                continue
            ax.loglog(d["TetsProcess"], d["SpeedUp"],
                    marker=markers[i % len(markers)],
                    ls=lstyles[i % len(lstyles)],
                    ms=5, lw=1.8,
                    color=sns.color_palette()[i % 10],
                    label=legend, zorder=3)

        ax.axhline(1.0, color="0.3", lw=0.8, alpha=0.7)
        ax.set_ylim(y_spd_min, y_spd_max)
        ax.grid(True, ls="-", lw=0.3, color="0.8", which="both")
        ax.set_xlabel("Tets Pairs to process $n$", fontsize=12)
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
    demo_name = "spatula_slip_control_5"
    envs = ["1", "10", "20", "50", "80", "100", "200", "500", "800"]
    
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
            for env in envs:
                if env not in cpu_data[folder_name]:
                    cpu_data[folder_name][env] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{env}_drake-cpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    cpu_data[folder_name][env]["timing_overall"] = data_timing_overall
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{env}_drake-cpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    cpu_data[folder_name][env]["problem_size"] = data_problem_size
                    
                    json_path_advance_to = f"{base_dir}/{folder_name}/{demo_name}_{env}_drake-cpu_timing_advance_to.json"
                    data_advance_to = get_data(json_path_advance_to)
                    cpu_data[folder_name][env]["advance_to"] = data_advance_to
        else:
            # GPU data
            if folder_name not in gpu_data:
                gpu_data[folder_name] = {}
            for env in envs:
                if env not in gpu_data[folder_name]:
                    gpu_data[folder_name][env] = {}
                    # Timing overall
                    json_path_timing_overall = f"{base_dir}/{folder_name}/{demo_name}_{env}_sycl-gpu_timing_overall.json"
                    data_timing_overall = get_data(json_path_timing_overall)
                    gpu_data[folder_name][env]["timing_overall"] = data_timing_overall
                    # Kernel timing
                    json_path_kernel_timing = f"{base_dir}/{folder_name}/{demo_name}_{env}_sycl-gpu_timing.json"
                    data_kernel_timing = get_data(json_path_kernel_timing)
                    gpu_data[folder_name][env]["kernel_timing"] = data_kernel_timing
                    
                    json_path_problem_size = f"{base_dir}/{folder_name}/{demo_name}_{env}_sycl-gpu_problem_size.json"
                    data_problem_size = get_data(json_path_problem_size)
                    gpu_data[folder_name][env]["problem_size"] = data_problem_size
                    
                    json_path_advance_to = f"{base_dir}/{folder_name}/{demo_name}_{env}_sycl-gpu_timing_advance_to.json"
                    data_advance_to = get_data(json_path_advance_to)
                    gpu_data[folder_name][env]["advance_to"] = data_advance_to
                    
                    txt_base_path = f"{base_dir}/{folder_name}/{demo_name}_{env}_sycl-gpu_timing_"
                    
                    kernel_keys = ["unpack_transforms", "transform_and_broad_phase", "device_to_host_memcpy", "compute_contact_polygons", "compact_polygon_data"]
                    gpu_data[folder_name][env]["raw_timing_data"] = {}
                    for key in kernel_keys:
                        txt_path = f"{txt_base_path}{key}.txt"
                        with open(txt_path, "r") as f:
                            third_line = f.readlines()[2]
                            third_line_split_cost = float(third_line.split(" ")[1])
                            gpu_data[folder_name][env]["raw_timing_data"][key] = third_line_split_cost
                    
    # Create plots directory
    plot_dir = "plots_gpu_comparison_spatula_Aug19"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    

    # Plot broad phase timing vs actual number of objects (log-log)
    fig, axes = plot_hydroelastic_query_perf_speedup_spatula(gpu_data, cpu_data, folder_names, legend_names, envs, xaxis_type="default")
    plt.savefig(f"{base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png", dpi=600)
    print(f"Saved hydroelastic query perf speedup plot to {base_dir}/{plot_dir}/hydroelastic_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    
    fig, axes = plot_broad_phase_perf_speedup_spatula(cpu_data, gpu_data, folder_names, legend_names, envs, xaxis_type="default", with_fcl_time=False)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png", dpi=600)
    print(f"Saved broad phase query perf speedup plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup_spatula(cpu_data, gpu_data, folder_names, legend_names, envs, xaxis_type="num_elements", with_fcl_time=False)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png", dpi=600)
    print(f"Saved broad phase query perf speedup plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup_spatula(cpu_data, gpu_data, folder_names, legend_names, envs, xaxis_type="default", with_fcl_time=True)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl.png", dpi=600)
    print(f"Saved broad phase query perf speedup plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_broad_phase_perf_speedup_spatula(cpu_data, gpu_data, folder_names, legend_names, envs, xaxis_type="num_elements", with_fcl_time=True)
    plt.savefig(f"{base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl_vs_num_elements.png", dpi=600)
    print(f"Saved broad phase query perf speedup plot to {base_dir}/{plot_dir}/broad_phase_query_perf_speedup_with_fcl_vs_num_elements.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_narrow_phase_query_perf_speedup_spatula(cpu_data, gpu_data, folder_names, legend_names, envs, xaxis_type="default")
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png", dpi=600)
    print(f"Saved narrow phase query perf speedup plot to {base_dir}/{plot_dir}/narrow_phase_query_perf_speedup.png")
    plt.show()
    plt.close()
    
    fig, axes = plot_narrow_phase_query_perf_speedup_spatula(cpu_data, gpu_data, folder_names, legend_names, envs, xaxis_type="num_elements")
    plt.savefig(f"{base_dir}/{plot_dir}/narrow_phase_query_perf_speedup_tet_pairs.png", dpi=600)
    print(f"Saved narrow phase query perf speedup plot to {base_dir}/{plot_dir}/narrow_phase_query_perf_speedup_tet_pairs.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()