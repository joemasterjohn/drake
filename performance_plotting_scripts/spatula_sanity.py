import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from utils import get_data
import pandas as pd
import numpy as np


def plot_candidate_tets_vs_envs(all_data, run_types, envs, ax=None):
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Prepare data
    plot_data = []
    for run_type in run_types:
        for env in envs:
            problem_sizes = all_data[run_type][env]["problem_size"].get("problem_sizes", {})
            if run_type.startswith("sycl"):
                tets = problem_sizes.get("SYCLCandidateTets", {}).get("avg", None)
            else:
                tets = problem_sizes.get("CandidateTets", {}).get("avg", None)

            plot_data.append({
                "NumEnv": int(env),
                "CandidateTets": tets,
                "RunType": run_type
            })

    # Create DataFrame and plot
    df = pd.DataFrame(plot_data)
    sns.lineplot(data=df, x="NumEnv", y="CandidateTets", hue="RunType", marker="o", ax=ax)

    # Customize
    if len(envs) > 6:
        # For many values, show every other tick mark
        tick_indices = list(range(0, len(envs), 2))
        if len(envs) - 1 not in tick_indices:  # Always show the last value
            tick_indices.append(len(envs) - 1)
        tick_values = [int(envs[i]) for i in tick_indices]
    else:
            tick_values = [int(e) for e in envs]
    ax.set_xticks(tick_values)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("Number of Environments", fontsize=14)
    ax.set_ylabel("Tet Candidates - Broad Phase", fontsize=14)
    ax.set_title("Spatula", fontsize=15, fontweight='bold')

    ax.legend(fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_faces_inserted_vs_envs(all_data, run_types, envs, ax=None):
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Prepare data
    plot_data = []
    for run_type in run_types:
        for env in envs:
            problem_sizes = all_data[run_type][env]["problem_size"].get("problem_sizes", {})
            if run_type.startswith("sycl"):
                faces = problem_sizes.get("SYCFacesInserted", {}).get("avg", None)
            else:
                faces = problem_sizes.get("FacesInserted", {}).get("avg", None)

            plot_data.append({
                "NumEnv": int(env),
                "FacesInserted": faces,
                "RunType": run_type
            })

    # Create DataFrame and plot
    df = pd.DataFrame(plot_data)
    sns.lineplot(data=df, x="NumEnv", y="FacesInserted", hue="RunType", marker="o", ax=ax)

    # Customize
    if len(envs) > 6:
        # For many values, show every other tick mark
        tick_indices = list(range(0, len(envs), 2))
        if len(envs) - 1 not in tick_indices:  # Always show the last value
            tick_indices.append(len(envs) - 1)
        tick_values = [int(envs[i]) for i in tick_indices]
    else:
        tick_values = [int(e) for e in envs]
    ax.set_xticks(tick_values)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("Number of Environments", fontsize=14)
    ax.set_ylabel("Contact Faces Generated - Narrow Phase", fontsize=14)
    ax.set_title("Spatula", fontsize=15, fontweight='bold')

    ax.legend(fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def main():
    base_dir = os.path.dirname(os.getcwd())
    demo_name = "spatula_slip_control_5"
    envs = ["1", "10", "20", "50", "80", "100", "200", "500", "800"]

    run_types = ["sycl-gpu", "drake-cpu"]
    perf_folder = "performance_jsons_spatula_slip_control_scale_convex"

    # Store all data in a nested dictionary: all_data[run_type][env][data_type]
    all_data = {run_type: {} for run_type in run_types}
    for run_type in run_types:
        for env in envs:
            all_data[run_type][env] = {}

            # Problem size
            json_path_problem_size = f"{base_dir}/{perf_folder}/{demo_name}_{env}_{run_type}_problem_size.json"
            data_problem_size = get_data(json_path_problem_size)
            all_data[run_type][env]["problem_size"] = data_problem_size

            # Timing overall
            json_path_timing_overall = f"{base_dir}/{perf_folder}/{demo_name}_{env}_{run_type}_timing_overall.json"
            data_timing_overall = get_data(json_path_timing_overall)
            all_data[run_type][env]["timing_overall"] = data_timing_overall

            # Advance to timing
            json_path_advance_to = f"{base_dir}/{perf_folder}/{demo_name}_{env}_{run_type}_timing_advance_to.json"
            data_advance_to = get_data(json_path_advance_to)
            all_data[run_type][env]["advance_to"] = data_advance_to

            # Kernel timing for SYCL
            if run_type == "sycl-gpu":
                json_path_kernel_timing = f"{base_dir}/{perf_folder}/{demo_name}_{env}_{run_type}_timing.json"
                data_kernel_timing = get_data(json_path_kernel_timing)
                all_data[run_type][env]["kernel_timing"] = data_kernel_timing

    # Plot Faces Inserted vs environments
    plot_faces_inserted_vs_envs(all_data, run_types, envs)
    plot_dir = "plots_gpu_comparison_spatula_Aug19"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plt.savefig(f"{base_dir}/{plot_dir}/spatula_faces_inserted_vs_env.png", dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/spatula_faces_inserted_vs_env.png")
    plt.show()
    plt.close()

    # Plot Candidate Tets vs environments
    plot_candidate_tets_vs_envs(all_data, run_types, envs)
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plt.savefig(f"{base_dir}/{plot_dir}/spatula_candidate_tets_vs_env.png", dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/spatula_candidate_tets_vs_env.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()


