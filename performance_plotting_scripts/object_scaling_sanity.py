import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from utils import get_data
import pandas as pd
import numpy as np

def calculate_actual_objects(gpp):
    """
    Calculate the actual number of objects based on GPP (grippers per pepper).
    Each gripper has 2 bodies, each pepper has 1 body, and there's 1 table.
    Formula: 2 * gpp + gpp + 1 = 3 * gpp + 1
    """
    return 3 * int(gpp) + 1
def plot_candidate_tets_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots dynamically based on number of spacings
    n_cols = len(spacings)
    if ax is None:
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    
    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        for run_type in run_types:
            for gpp in num_gpp:
                # Get the candidate tets data
                problem_sizes = all_data[run_type][spacing][gpp]["problem_size"].get("problem_sizes", {})
                if run_type.startswith("sycl"):
                    tets = problem_sizes.get("SYCLCandidateTets", {}).get("avg", None)
                else:
                    tets = problem_sizes.get("CandidateTets", {}).get("avg", None)
                
                plot_data.append({
                    "NumGpp": int(calculate_actual_objects(gpp)),
                    "CandidateTets": tets,
                    "RunType": run_type
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="NumGpp", y="CandidateTets", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        # Show fewer tick marks to prevent overlap
        if len(num_gpp) > 6:
            # For many values, show every other tick mark
            tick_indices = list(range(0, len(num_gpp), 2))
            if len(num_gpp) - 1 not in tick_indices:  # Always show the last value
                tick_indices.append(len(num_gpp) - 1)
            tick_values = [int(calculate_actual_objects(num_gpp[i])) for i in tick_indices]
        else:
            tick_values = [int(calculate_actual_objects(gpp)) for gpp in num_gpp]
        
        current_ax.set_xticks(tick_values)
        current_ax.tick_params(axis='x', rotation=45)
        current_ax.set_xlabel("Total Geometries", fontsize=14)
        current_ax.set_ylabel("Tet Candidates - Broad Phase" if i == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "0.1"
        elif(spacing == "0.15"):
            title = "0.15"
        elif(spacing == "0.05"):
            title = "0.05"
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
  

def plot_faces_inserted_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):

    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots dynamically based on number of spacings
    n_cols = len(spacings)
    if ax is None:
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    
    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        for run_type in run_types:
            for gpp in num_gpp:
                # Get the faces inserted data
                problem_sizes = all_data[run_type][spacing][gpp]["problem_size"].get("problem_sizes", {})
                if run_type.startswith("sycl"):
                    faces = problem_sizes.get("SYCFacesInserted", {}).get("avg", None)
                else:
                    faces = problem_sizes.get("FacesInserted", {}).get("avg", None)
                
                plot_data.append({
                    "NumGpp": int(calculate_actual_objects(gpp)),
                    "FacesInserted": faces,
                    "RunType": run_type
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="NumGpp", y="FacesInserted", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        # Show fewer tick marks to prevent overlap
        if len(num_gpp) > 6:
            # For many values, show every other tick mark
            tick_indices = list(range(0, len(num_gpp), 2))
            if len(num_gpp) - 1 not in tick_indices:  # Always show the last value
                tick_indices.append(len(num_gpp) - 1)
            tick_values = [int(calculate_actual_objects(num_gpp[i])) for i in tick_indices]
        else:
            tick_values = [int(calculate_actual_objects(gpp)) for gpp in num_gpp]
        current_ax.set_xticks(tick_values)
        current_ax.tick_params(axis='x', rotation=45)
        current_ax.set_xlabel("Total Geometries", fontsize=14)
        current_ax.set_ylabel("Contact Faces Generated - Narrow Phase" if i == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "0.1"
        elif(spacing == "0.15"):
            title = "0.15"
        elif(spacing == "0.05"):
            title = "0.05"
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
    demo_name = "objects_scaling"
    spacings = ["0.05", "0.1", "0.15"]
    num_gpp = ["1", "2", "5", "10", "20", "33", "50", "100"]
    # Make list of all run combos spacings_num_gpp
    runs = [f"{spacing}_{gpp}" for spacing in spacings for gpp in num_gpp]
    
    
    run_types = ["sycl-gpu", "drake-cpu"]
    perf_folder = "performance_jsons_bvh_Aug20"

    # Store all data in a nested dictionary: all_data[run_type][spacing][num_gpp][data_type]
    all_data = {run_type: {} for run_type in run_types}
    for run_type in run_types:
        for spacing in spacings:
            all_data[run_type][spacing] = {}
            for gpp in num_gpp:
                all_data[run_type][spacing][gpp] = {}
                # Problem size
                json_path_problem_size = f"{base_dir}/{perf_folder}/{demo_name}_{spacing}_{gpp}_{run_type}_problem_size.json"
                data_problem_size = get_data(json_path_problem_size)
                all_data[run_type][spacing][gpp]["problem_size"] = data_problem_size
                
                # Timing overall
                json_path_timing_overall = f"{base_dir}/{perf_folder}/{demo_name}_{spacing}_{gpp}_{run_type}_timing_overall.json"
                data_timing_overall = get_data(json_path_timing_overall)
                all_data[run_type][spacing][gpp]["timing_overall"] = data_timing_overall

                if(run_type == "sycl-gpu" or run_type == "sycl-cpu"):
                    json_path_kernel_timing = f"{base_dir}/{perf_folder}/{demo_name}_{spacing}_{gpp}_{run_type}_timing.json"
                    data_kernel_timing = get_data(json_path_kernel_timing)
                    all_data[run_type][spacing][gpp]["kernel_timing"] = data_kernel_timing
    
    
    # Analyze distribution of candidate tets and faces inserted for the dense vs sparse object placements
    plot_faces_inserted_vs_num_gpp(all_data, run_types, spacings, num_gpp)
    plot_dir = "plots_gpu_comparison_object_scaling_Aug20"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plt.savefig(f"{base_dir}/{plot_dir}/object_scaling_faces_inserted_vs_num_gpp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/object_scaling_faces_inserted_vs_num_gpp.png")
    plt.show()
    plt.close()
    
    plot_candidate_tets_vs_num_gpp(all_data, run_types, spacings, num_gpp)
    plt.savefig(f"{base_dir}/{plot_dir}/object_scaling_candidate_tets_vs_num_gpp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/{plot_dir}/object_scaling_candidate_tets_vs_num_gpp.png")
    plt.show()
    plt.close()
    
    
    
    
    
               

if __name__ == "__main__":
    main()