import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from utils import get_data
import pandas as pd
import numpy as np

def calculate_actual_objects_clutter(obp):
    """
    Calculate the actual number of objects based on OBP (objects per pile).
    Each pile has 1 table, and each pile has obp objects.
    Formula: 1 + obp
    """
    return int(obp) * 4 + 5
def plot_candidate_tets_vs_obp(all_data, run_types, objects_per_pile, sphere_resolutions, ax=None):
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots dynamically based on number of sphere resolutions
    n_cols = len(sphere_resolutions)
    if ax is None:
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    
    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])
    
    # Prepare data for each spacing
    for i, sr in enumerate(sphere_resolutions):
        plot_data = []
        for run_type in run_types:
            for obp in objects_per_pile:
                # Get the candidate tets data
                problem_sizes = all_data[run_type][obp][sr]["problem_size"].get("problem_sizes", {})
                if run_type.startswith("sycl"):
                    tets = problem_sizes.get("SYCLCandidateTets", {}).get("avg", None)
                else:
                    tets = problem_sizes.get("CandidateTets", {}).get("avg", None)
                
                plot_data.append({
                    "ObjectsPerPile": calculate_actual_objects_clutter(obp),
                    "CandidateTets": tets,
                    "RunType": run_type
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="ObjectsPerPile", y="CandidateTets", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        if len(objects_per_pile) > 6:
            # For many values, show every other tick mark
            tick_indices = list(range(0, len(objects_per_pile), 2))
            if len(objects_per_pile) - 1 not in tick_indices:  # Always show the last value
                tick_indices.append(len(objects_per_pile) - 1)
            tick_values = [int(calculate_actual_objects_clutter(objects_per_pile[i])) for i in tick_indices]
        else:
            tick_values = [int(calculate_actual_objects_clutter(obp)) for obp in objects_per_pile]
        current_ax.set_xticks(tick_values)
        current_ax.set_xlabel("Total Geometries", fontsize=14)
        current_ax.set_ylabel("Tet Candidates - Broad Phase" if i == 0 else "", fontsize=14)
        title = f"Sphere Resolution - {sphere_resolutions[i]}"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return axes


def plot_faces_inserted_vs_obp(all_data, run_types, objects_per_pile, sphere_resolutions, ax=None):
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots dynamically based on number of sphere resolutions
    n_cols = len(sphere_resolutions)
    if ax is None:
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    
    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])
    
    # Prepare data for each spacing
    for i, sr in enumerate(sphere_resolutions):
        plot_data = []
        for run_type in run_types:
            for obp in objects_per_pile:
                # Get the faces inserted data
                problem_sizes = all_data[run_type][obp][sr]["problem_size"].get("problem_sizes", {})
                if run_type.startswith("sycl"):
                    faces = problem_sizes.get("SYCFacesInserted", {}).get("avg", None)
                else:
                    faces = problem_sizes.get("FacesInserted", {}).get("avg", None)
                
                plot_data.append({
                    "ObjectsPerPile": calculate_actual_objects_clutter(obp),
                    "FacesInserted": faces,
                    "RunType": run_type
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="ObjectsPerPile", y="FacesInserted", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        # Show fewer tick marks to prevent overlap
        if len(objects_per_pile) > 6:
            # For many values, show every other tick mark
            tick_indices = list(range(0, len(objects_per_pile), 2))
            if len(objects_per_pile) - 1 not in tick_indices:  # Always show the last value
                tick_indices.append(len(objects_per_pile) - 1)
            tick_values = [int(calculate_actual_objects_clutter(objects_per_pile[i])) for i in tick_indices]
        else:
            tick_values = [int(calculate_actual_objects_clutter(obp)) for obp in objects_per_pile]
        current_ax.set_xticks(tick_values)
        current_ax.set_xlabel("Total Geometries", fontsize=14)
        current_ax.set_ylabel("Contact Faces Generated - Narrow Phase" if i == 0 else "", fontsize=14)
        title = f"Sphere Resolution - {sphere_resolutions[i]}"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return axes
def main():
    base_dir = os.path.dirname(os.getcwd())
    demo_name = "clutter"
    objects_per_pile = ["1", "2", "5", "10", "20", "33", "50"]
    sphere_resolutions = ["0.0050", "0.0100", "0.0200", "0.0400"]
    
    
    run_types = ["sycl-gpu", "drake-cpu"]
    perf_folder = "performance_jsons_clutter_convex_Aug13"

    # Store all data in a nested dictionary: all_data[run_type][spacing][num_gpp][data_type]
    all_data = {run_type: {} for run_type in run_types}
    for run_type in run_types:
        for obp in objects_per_pile:
            all_data[run_type][obp] = {}
            for sr in sphere_resolutions:
                all_data[run_type][obp][sr] = {}
                # Problem size
                json_path_problem_size = f"{base_dir}/{perf_folder}/{demo_name}_{obp}_1.0000_{sr}_3_{run_type}_problem_size.json"
                data_problem_size = get_data(json_path_problem_size)
                all_data[run_type][obp][sr]["problem_size"] = data_problem_size
                
                # Timing overall
                json_path_timing_overall = f"{base_dir}/{perf_folder}/{demo_name}_{obp}_1.0000_{sr}_3_{run_type}_timing_overall.json"
                data_timing_overall = get_data(json_path_timing_overall)
                all_data[run_type][obp][sr]["timing_overall"] = data_timing_overall
                
                # Advance to timing
                json_path_advance_to = f"{base_dir}/{perf_folder}/{demo_name}_{obp}_1.0000_{sr}_3_{run_type}_timing_advance_to.json"
                data_advance_to = get_data(json_path_advance_to)
                all_data[run_type][obp][sr]["advance_to"] = data_advance_to

                if(run_type == "sycl-gpu" or run_type == "sycl-cpu"):
                    json_path_kernel_timing = f"{base_dir}/{perf_folder}/{demo_name}_{obp}_1.0000_{sr}_3_{run_type}_timing.json"
                    data_kernel_timing = get_data(json_path_kernel_timing)
                    all_data[run_type][obp][sr]["kernel_timing"] = data_kernel_timing
                   
    plot_faces_inserted_vs_obp(all_data, run_types, objects_per_pile, sphere_resolutions)
    plot_dir = "plots_gpu_comparison_clutter_Aug13"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plt.savefig(f"{base_dir}/{plot_dir}/clutter_faces_inserted_vs_obp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/clutter_faces_inserted_vs_obp.png")
    plt.show()
    plt.close()
    
    plot_candidate_tets_vs_obp(all_data, run_types, objects_per_pile, sphere_resolutions)
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plt.savefig(f"{base_dir}/{plot_dir}/clutter_candidate_tets_vs_obp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/clutter_candidate_tets_vs_obp.png")
    plt.show()
    plt.close()
    
        
if __name__ == "__main__":
    main()