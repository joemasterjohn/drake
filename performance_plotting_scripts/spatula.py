import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from utils import plot_candidate_tets_vs_resolution, plot_faces_inserted_vs_resolution, plot_timing_overall_vs_resolution, plot_broad_narrow_misc_bar, get_data


"""
Relevant data:
- problem_size
For sycl-gpu and sycl-cpu:
  SYCFacesInserted : Number of faces after narrow phase
  SYCLCandidateTets : Number of candidate tetrahedra after broad phase
For drake-cpu:
  FacesInserted : Number of faces after narrow phase
  CandidateTets : Number of candidate tetrahedra after broad phase
  
  - Timing Overall
For sycl-gpu and sycl-cpu:
  HydroelasticQuery - avg_us : Avg time taken for full hydroelastic query
For drake-cpu:
  HydroelasticQuery - avg_us : Avg time taken for full hydroelastic query
  BroadPhase - avg_us : Avg time taken for broad phase
  NarrowPhase - avg_us : Avg time taken for narrow phase
  
  - kernel_timing
For sycl-gpu and sycl-cpu:
  - transform_and_broad_phase - avg_us : Avg time taken for transform and broad phase
  - compute_contact_polygons - avg_us : Avg time taken for narrow phase  
"""



def main():
    base_dir = os.path.dirname(os.getcwd())
    demo_name = "spatula_slip_control"
    resolutions = [2, 5, 10, 20]
    run_types = ["sycl-gpu", "sycl-cpu", "drake-cpu"]
    perf_folder = "performance_jsons"

    # Store all data in a nested dictionary: all_data[run_type][resolution][data_type]
    all_data = {run_type: {} for run_type in run_types}

    for resolution in resolutions:
        for run_type in run_types:
            all_data[run_type][resolution] = {}
            # Problem size
            json_path_problem_size = f"{base_dir}/{perf_folder}/{demo_name}_{resolution}_{run_type}_problem_size.json"
            data_problem_size = get_data(json_path_problem_size)
            all_data[run_type][resolution]["problem_size"] = data_problem_size
        
            # Overall timing and CPU timing
            json_path_timing_overall = f"{base_dir}/{perf_folder}/{demo_name}_{resolution}_{run_type}_timing_overall.json"
            data_timing_overall = get_data(json_path_timing_overall)
            all_data[run_type][resolution]["timing_overall"] = data_timing_overall
            
            if(run_type == "sycl-gpu" or run_type == "sycl-cpu"):
                json_path_kernel_timing = f"{base_dir}/{perf_folder}/{demo_name}_{resolution}_{run_type}_timing.json"
                data_kernel_timing = get_data(json_path_kernel_timing)
                all_data[run_type][resolution]["kernel_timing"] = data_kernel_timing
    
    
    # X axis: Resolution
    # Y axis: sycl-gpu and drake-cpu CandidateTets
    plot_candidate_tets_vs_resolution(
        all_data,
        run_types=["sycl-gpu", "sycl-cpu", "drake-cpu"],
        resolutions=resolutions
    )
    if not os.path.exists(f"{base_dir}/plots"):
        os.makedirs(f"{base_dir}/plots")
    plt.savefig(f"{base_dir}/plots/spatula_candidate_tets_vs_resolution.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/spatula_candidate_tets_vs_resolution.png")
    plt.show()
    
    plot_faces_inserted_vs_resolution(
        all_data,
        run_types=["sycl-gpu", "sycl-cpu", "drake-cpu"],
        resolutions=resolutions
    )
    plt.savefig(f"{base_dir}/plots/spatula_faces_inserted_vs_resolution.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/spatula_faces_inserted_vs_resolution.png")
    plt.show()
    
    plot_timing_overall_vs_resolution(
        all_data,
        run_types=["sycl-gpu", "sycl-cpu", "drake-cpu"],
        # run_types=["sycl-gpu", "drake-cpu"],
        resolutions=resolutions
    )
    plt.savefig(f"{base_dir}/plots/spatula_timing_overall_vs_resolution.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/spatula_timing_overall_vs_resolution.png")
    plt.show()
    
    plot_broad_narrow_misc_bar(
        all_data,
        run_types=["sycl-gpu", "drake-cpu"],
        resolutions=resolutions
    )
    plt.savefig(f"{base_dir}/plots/spatula_broad_narrow_misc_vs_resolution.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/spatula_broad_narrow_misc_vs_resolution.png")
    plt.show()
    
if __name__ == "__main__":
    main()