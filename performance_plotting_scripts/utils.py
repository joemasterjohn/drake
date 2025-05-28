import matplotlib.pyplot as plt
import seaborn as sns
import json
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
def get_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def get_corrected_timing(timing_data, run_type):
    """
    Get corrected timing for sycl-cpu by excluding JIT compilation time.
    For sycl-cpu: (total_us - max_us) / (calls - 1)
    For others: use avg_us
    """
    if run_type == "sycl-cpu":
        total_us = timing_data.get("total_us", 0)
        max_us = timing_data.get("max_us", 0)
        calls = timing_data.get("calls", 1)
        if calls > 1:
            return (total_us - max_us) / (calls - 1)
        else:
            return timing_data.get("avg_us", 0)
    else:
        return timing_data.get("avg_us", 0)

def plot_candidate_tets_vs_resolution(all_data, run_types, resolutions, ax=None):
    """
    Plots CandidateTets vs Resolution for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        resolutions: List of resolution values
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    # Prepare data for plotting
    plot_data = []
    for run_type in run_types:
        for res in resolutions:
            # sycl-gpu: 'SYCLCandidateTets', drake-cpu: 'CandidateTets'
            problem_sizes = all_data[run_type][res]["problem_size"].get("problem_sizes", {})
            if run_type.startswith("sycl"):
                tets = problem_sizes.get("SYCLCandidateTets", {}).get("avg", None)
            else:
                tets = problem_sizes.get("CandidateTets", {}).get("avg", None)
            plot_data.append({
                "Resolution": res,
                "CandidateTets": tets,
                "RunType": run_type
            })
    df = pd.DataFrame(plot_data)
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Resolution", y="CandidateTets", hue="RunType", marker="o", ax=ax)
    ax.set_xticks(resolutions)
    ax.set_xlabel("Mesh Res. (mm)", fontsize=16)
    ax.set_ylabel("Candidates - Broad Phase", fontsize=16)
    ax.legend(fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    return ax 



def plot_faces_inserted_vs_resolution(all_data, run_types, resolutions, ax=None):
    """
    Plots FacesInserted vs Resolution for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        resolutions: List of resolution values
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    # Prepare data for plotting
    plot_data = []
    for run_type in run_types:
        for res in resolutions:
            # sycl-gpu: 'SYCFacesInserted', drake-cpu: 'FacesInserted'
            problem_sizes = all_data[run_type][res]["problem_size"].get("problem_sizes", {})
            if run_type.startswith("sycl"):
                tets = problem_sizes.get("SYCFacesInserted", {}).get("avg", None)
            else:
                tets = problem_sizes.get("FacesInserted", {}).get("avg", None)
            plot_data.append({
                "Resolution": res,
                "FacesInserted": tets,
                "RunType": run_type
            })
    df = pd.DataFrame(plot_data)
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Resolution", y="FacesInserted", hue="RunType", marker="o", ax=ax)
    ax.set_xticks(resolutions)
    ax.set_xlabel("Mesh Res. (mm)", fontsize=16)
    ax.set_ylabel("Faces Inserted - Narrow Phase", fontsize=16)
    ax.legend(fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    return ax 

def plot_faces_inserted_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots FacesInserted vs NumGpp for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
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
                    "NumGpp": int(gpp),
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
        current_ax.set_xticks([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Faces Inserted - Narrow Phase" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
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

def plot_candidate_tets_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots CandidateTets vs NumGpp for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
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
                    "NumGpp": int(gpp),
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
        current_ax.set_xticks([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Candidates - Broad Phase" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
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

def plot_broad_narrow_misc_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots a grouped stacked bar plot comparing BroadPhase, NarrowPhase, and Misc times for sycl-gpu and drake-cpu.
    The full bar is HydroelasticQueryTime, with segments for BroadPhase, NarrowPhase, and Misc.
    Uses two subplots for different spacings.
    Args:
        all_data: Nested dict as in object_scaling.py
        run_types: List of run types (should include 'sycl-gpu' and 'drake-cpu')
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Define colors for different timing components
    colors = {
        'BroadPhase': '#1f77b4',
        'NarrowPhase': '#ff7f0e',
        'Misc': '#2ca02c',
    }
    
    # Plot for each spacing
    for spacing_idx, spacing in enumerate(spacings):
        current_ax = axes[spacing_idx]
        
        bar_width = 0.35
        x = list(range(len(num_gpp)))
        run_type_offsets = {'sycl-gpu': -bar_width/2, 'drake-cpu': bar_width/2}
        
        for i, run_type in enumerate(['sycl-gpu', 'drake-cpu']):
            if run_type not in run_types:
                continue
                
            broad_vals, narrow_vals, misc_vals, total_vals = [], [], [], []
            
            for gpp in num_gpp:
                timings = all_data[run_type][spacing][gpp]["timing_overall"].get("timings", {})
                hq_data = timings.get("HydroelasticQuery", {})
                hq = get_corrected_timing(hq_data, run_type)
                
                if run_type == 'sycl-gpu':
                    kernel_timing = all_data[run_type][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                    broad_data = kernel_timing.get("transform_and_broad_phase", {})
                    narrow_data = kernel_timing.get("compute_contact_polygons", {})
                    broad = get_corrected_timing(broad_data, run_type)
                    narrow = get_corrected_timing(narrow_data, run_type)
                else:
                    broad_data = timings.get("BroadPhase", {})
                    narrow_data = timings.get("NarrowPhase", {})
                    broad = get_corrected_timing(broad_data, run_type)
                    narrow = get_corrected_timing(narrow_data, run_type)
                
                misc = max(hq - broad - narrow, 0)
                broad_vals.append(broad)
                narrow_vals.append(narrow)
                misc_vals.append(misc)
                total_vals.append(hq)
            # Plot stacked bars for this run_type
            xpos = [xi + run_type_offsets[run_type] for xi in x]
            
            # Create stacked bars
            b1 = current_ax.bar(xpos, broad_vals, bar_width, color=colors['BroadPhase'], 
                               label=f'BroadPhase' if spacing_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            b2 = current_ax.bar(xpos, narrow_vals, bar_width, bottom=broad_vals, color=colors['NarrowPhase'], 
                               label=f'NarrowPhase' if spacing_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            bottoms = [b + n for b, n in zip(broad_vals, narrow_vals)]
            b3 = current_ax.bar(xpos, misc_vals, bar_width, bottom=bottoms, color=colors['Misc'], 
                               label=f'Misc' if spacing_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
        
        # Customize the subplot
        current_ax.set_xticks(x)
        current_ax.set_xticklabels(num_gpp)
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Time (us)" if spacing_idx == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Add subtitle indicating bar grouping only on first subplot
        if spacing_idx == 0:
            current_ax.text(0.5, -0.15, "(left: sycl-gpu, right: drake-cpu)", 
                           transform=current_ax.transAxes, ha='center', fontsize=12)
        
        # Only show legend on the first subplot
        if spacing_idx == 0:
            handles, labels = current_ax.get_legend_handles_labels()
            seen = set()
            new_handles, new_labels = [], []
            for h, l in zip(handles, labels):
                if l not in seen and l:
                    new_handles.append(h)
                    new_labels.append(l)
                    seen.add(l)
            current_ax.legend(new_handles, new_labels, fontsize=12, title_fontsize=13)
        
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return axes

def plot_narrow_phase_timing_vs_num_gpp(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots NarrowPhase timing vs NumGpp for given run_types using seaborn line plots.
    Args:
        all_data: Nested dict as in object_scaling.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        for run_type in run_types:
            for gpp in num_gpp:
                # Get the narrow phase timing data
                if run_type == 'sycl-gpu':
                    kernel_timing = all_data[run_type][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                    narrow_data = kernel_timing.get("compute_contact_polygons", {})
                    narrow_time = get_corrected_timing(narrow_data, run_type)
                else:
                    timings = all_data[run_type][spacing][gpp]["timing_overall"].get("timings", {})
                    narrow_data = timings.get("NarrowPhase", {})
                    narrow_time = get_corrected_timing(narrow_data, run_type)
                
                plot_data.append({
                    "NumGpp": int(gpp),
                    "NarrowPhaseTime": narrow_time,
                    "RunType": run_type
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="NumGpp", y="NarrowPhaseTime", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        current_ax.set_xticks([int(gpp) for gpp in num_gpp])
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Narrow Phase Time (us)" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
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

def plot_narrow_phase_timing_vs_candidate_tets(all_data, run_types, spacings, num_gpp, ax=None):
    """
    Plots NarrowPhase timing vs CandidateTets for given run_types using seaborn line plots.
    Shows the relationship between broad phase output and narrow phase performance.
    Args:
        all_data: Nested dict as in object_scaling.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        spacings: List of spacings (e.g., ["0.1", "0.05"])
        num_gpp: List of num_gpp values (e.g., ["1", "2", "5", "10", "20"])
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Prepare data for each spacing
    for i, spacing in enumerate(spacings):
        plot_data = []
        for run_type in run_types:
            for gpp in num_gpp:
                # Get the candidate tets data
                problem_sizes = all_data[run_type][spacing][gpp]["problem_size"].get("problem_sizes", {})
                if run_type.startswith("sycl"):
                    candidate_tets = problem_sizes.get("SYCLCandidateTets", {}).get("avg", None)
                else:
                    candidate_tets = problem_sizes.get("CandidateTets", {}).get("avg", None)
                
                # Get the narrow phase timing data
                if run_type == 'sycl-gpu':
                    kernel_timing = all_data[run_type][spacing][gpp]["kernel_timing"].get("kernel_timings", {})
                    narrow_data = kernel_timing.get("compute_contact_polygons", {})
                    narrow_time = get_corrected_timing(narrow_data, run_type)
                else:
                    timings = all_data[run_type][spacing][gpp]["timing_overall"].get("timings", {})
                    narrow_data = timings.get("NarrowPhase", {})
                    narrow_time = get_corrected_timing(narrow_data, run_type)
                
                plot_data.append({
                    "CandidateTets": candidate_tets,
                    "NarrowPhaseTime": narrow_time,
                    "RunType": run_type,
                    "NumGpp": int(gpp)  # Keep this for reference/debugging
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(plot_data)
        
        # Plot on the current subplot
        current_ax = axes[i]
        sns.lineplot(data=df, x="CandidateTets", y="NarrowPhaseTime", hue="RunType", 
                    marker="o", ax=current_ax)
        
        # Customize the subplot
        current_ax.set_xlabel("Candidate Tets - Broad Phase", fontsize=14)
        current_ax.set_ylabel("Narrow Phase Time (us)" if i == 0 else "", fontsize=14)
        title = "Sparse" if spacing == "0.1" else "Dense"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        current_ax.set_xlim(0, 90000)  # Set consistent x-axis limits
        current_ax.xaxis.set_major_locator(plt.MultipleLocator(20000))
        current_ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # Only show legend on the first subplot
        if i == 0:
            current_ax.legend(fontsize=12, title_fontsize=13)
        else:
            current_ax.get_legend().remove() if current_ax.get_legend() else None
            
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return axes

def plot_timing_overall_vs_resolution(all_data, run_types, resolutions, ax=None):
    """
    Plots TimingOverall vs Resolution for given run_types using seaborn.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (e.g., ["sycl-gpu", "drake-cpu"])
        resolutions: List of resolution values
        ax: Optional matplotlib axis to plot on
    """
    import pandas as pd
    # Prepare data for plotting
    plot_data = []
    for run_type in run_types:
        for res in resolutions:
            timings = all_data[run_type][res]["timing_overall"].get("timings", {})
            # Use corrected timing for sycl-cpu, avg_us for others
            timing_data = timings.get("HydroelasticQuery", {})
            timing = get_corrected_timing(timing_data, run_type)
            plot_data.append({
                "Resolution": res,
                "TimingOverall": timing,
                "RunType": run_type
            })
    df = pd.DataFrame(plot_data)
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Resolution", y="TimingOverall", hue="RunType", marker="o", ax=ax)
    ax.set_xticks(resolutions)
    ax.set_xlabel("Mesh Res. (mm)", fontsize=16)
    ax.set_ylabel("HydroelasticQuery Time (us)", fontsize=16)
    ax.legend(fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    return ax


def plot_broad_narrow_misc_bar(all_data, run_types, resolutions, ax=None):
    """
    Plots a grouped stacked bar plot comparing BroadPhase, NarrowPhase, and Misc times for sycl-gpu and drake-cpu.
    The full bar is HydroelasticQueryTime, with segments for BroadPhase, NarrowPhase, and Misc.
    Only sycl-gpu and drake-cpu are compared.
    Uses correct timing keys for each run_type.
    Args:
        all_data: Nested dict as in spatula.py
        run_types: List of run types (should include 'sycl-gpu' and 'drake-cpu')
        resolutions: List of resolution values
        ax: Optional matplotlib axis to plot on
    """
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.35
    x = list(range(len(resolutions)))
    colors = {
        'BroadPhase': '#1f77b4',
        'NarrowPhase': '#ff7f0e',
        'Misc': '#2ca02c',
    }
    run_type_offsets = {'sycl-gpu': -bar_width/2, 'drake-cpu': bar_width/2}
    legend_handles = {}
    for i, run_type in enumerate(['sycl-gpu', 'drake-cpu']):
        if run_type not in run_types:
            continue
        broad_vals, narrow_vals, misc_vals, total_vals = [], [], [], []
        for res in resolutions:
            timings = all_data[run_type][res]["timing_overall"].get("timings", {})
            hq_data = timings.get("HydroelasticQuery", {})
            hq = get_corrected_timing(hq_data, run_type)
            if run_type == 'sycl-gpu':
                kernel_timing = all_data[run_type][res]["kernel_timing"].get("kernel_timings", {})
                broad_data = kernel_timing.get("transform_and_broad_phase", {})
                narrow_data = kernel_timing.get("compute_contact_polygons", {})
                broad = get_corrected_timing(broad_data, run_type)
                narrow = get_corrected_timing(narrow_data, run_type)
            else:
                broad_data = timings.get("BroadPhase", {})
                narrow_data = timings.get("NarrowPhase", {})
                broad = get_corrected_timing(broad_data, run_type)
                narrow = get_corrected_timing(narrow_data, run_type)
            misc = max(hq - broad - narrow, 0)
            broad_vals.append(broad)
            narrow_vals.append(narrow)
            misc_vals.append(misc)
            total_vals.append(hq)
        xpos = [xi + run_type_offsets[run_type] for xi in x]
        b1 = ax.bar(xpos, broad_vals, bar_width, color=colors['BroadPhase'], label=f'BroadPhase' if run_type=='sycl-gpu' else None, hatch='//' if run_type=='sycl-gpu' else None)
        b2 = ax.bar(xpos, narrow_vals, bar_width, bottom=broad_vals, color=colors['NarrowPhase'], label=f'NarrowPhase' if run_type=='sycl-gpu' else None, hatch='//' if run_type=='sycl-gpu' else None)
        bottoms = [b+n for b, n in zip(broad_vals, narrow_vals)]
        b3 = ax.bar(xpos, misc_vals, bar_width, bottom=bottoms, color=colors['Misc'], label=f'Misc' if run_type=='sycl-gpu' else None, hatch='//' if run_type=='sycl-gpu' else None)
        
        

    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.set_xlabel("Mesh Res. (mm) \n(left: sycl-gpu, right: drake-cpu)", fontsize=16)
    ax.set_ylabel("Time (us)", fontsize=16)
    # Build legend (remove duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, fontsize=12, title_fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    return ax

    