import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from utils import plot_faces_inserted_vs_num_gpp, get_data, plot_candidate_tets_vs_num_gpp, plot_broad_narrow_misc_vs_num_gpp, plot_narrow_phase_timing_vs_num_gpp, plot_narrow_phase_timing_vs_candidate_tets
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
    demo_name = "objects_scaling"
    spacings = ["0.1", "0.05"]
    num_gpp = ["1", "2", "5", "10", "20"]
    # Make list of all run combos spacings_num_gpp
    runs = [f"{spacing}_{gpp}" for spacing in spacings for gpp in num_gpp]
    
    
    run_types = ["sycl-gpu", "drake-cpu"]
    perf_folder = "performance_jsons"

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
    if not os.path.exists(f"{base_dir}/plots"):
        os.makedirs(f"{base_dir}/plots")
    plt.savefig(f"{base_dir}/plots/object_scaling_faces_inserted_vs_num_gpp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/object_scaling_faces_inserted_vs_num_gpp.png")
    plt.show()
    plt.close()
    
    plot_candidate_tets_vs_num_gpp(all_data, run_types, spacings, num_gpp)
    plt.savefig(f"{base_dir}/plots/object_scaling_candidate_tets_vs_num_gpp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/object_scaling_candidate_tets_vs_num_gpp.png")
    plt.show()
    plt.close()
    
    
    # Analyze distribution of timing overall for the dense vs sparse object placements
    plot_broad_narrow_misc_vs_num_gpp(all_data, run_types, spacings, ["1", "2"])
    plt.savefig(f"{base_dir}/plots/object_scaling_timing_overall_1_2_vs_num_gpp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/object_scaling_timing_overall_1_2_vs_num_gpp.png")
    plt.show()
    plt.close()
    
    plot_broad_narrow_misc_vs_num_gpp(all_data, run_types, spacings, ["1", "2", "5"])
    plt.savefig(f"{base_dir}/plots/object_scaling_timing_overall_1_2_5_vs_num_gpp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/object_scaling_timing_overall_1_2_5_vs_num_gpp.png")
    plt.show()
    plt.close()
    
    plot_broad_narrow_misc_vs_num_gpp(all_data, run_types, spacings, ["1", "2", "5", "10"])
    plt.savefig(f"{base_dir}/plots/object_scaling_timing_overall_1_2_5_10_vs_num_gpp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/object_scaling_timing_overall_1_2_5_10_vs_num_gpp.png")
    plt.show()
    plt.close() 
    
    plot_broad_narrow_misc_vs_num_gpp(all_data, run_types, spacings, num_gpp)
    plt.savefig(f"{base_dir}/plots/object_scaling_timing_overall_vs_num_gpp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/object_scaling_timing_overall_vs_num_gpp.png")
    plt.show()
    plt.close()
    
    
    
    # Narrow phase vs num_gpp
    plot_narrow_phase_timing_vs_num_gpp(all_data, run_types, spacings, num_gpp)
    plt.savefig(f"{base_dir}/plots/object_scaling_narrow_phase_timing_vs_num_gpp.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/object_scaling_narrow_phase_timing_vs_num_gpp.png")
    plt.show()
    plt.close()
    
    
    plot_narrow_phase_timing_vs_candidate_tets(all_data, run_types, spacings, num_gpp)
    plt.savefig(f"{base_dir}/plots/object_scaling_narrow_phase_timing_vs_candidate_tets.png",dpi=600)
    print("Saved plot to ", f"{base_dir}/plots/object_scaling_narrow_phase_timing_vs_candidate_tets.png")
    plt.show()
    plt.close()
    
    
    
    
    
    
               

if __name__ == "__main__":
    main()