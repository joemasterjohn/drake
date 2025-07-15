import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
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


def calculate_number_of_elements_clutter(obp, sr, problem_size_data):
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

def get_corrected_timing(timing_data, run_type):
    """
    Get corrected timing for sycl by excluding JIT compilation time.
    For sycl-cpu/sycl-gpu: (total_us - max_us) / (calls - 1)
    For others: use avg_us
    """
    if run_type == "sycl-cpu" or run_type == "sycl-gpu":
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
    
    # Create subplots dynamically based on number of spacings
    n_cols = len(spacings)
    if ax is None:
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), sharey=True)
    
    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])
    
    # Define colors for different timing components
    colors = {
        'BroadPhase': '#1f77b4',
        'NarrowPhase': '#ff7f0e',
        'Misc': '#2ca02c',
    }
    
    # Plot for each spacing
    for col_idx, spacing in enumerate(spacings):
        current_ax = axes[col_idx]
        
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
                               label=f'BroadPhase' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            b2 = current_ax.bar(xpos, narrow_vals, bar_width, bottom=broad_vals, color=colors['NarrowPhase'], 
                               label=f'NarrowPhase' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            bottoms = [b + n for b, n in zip(broad_vals, narrow_vals)]
            b3 = current_ax.bar(xpos, misc_vals, bar_width, bottom=bottoms, color=colors['Misc'], 
                               label=f'Misc' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
        
        # Customize the subplot
        # Show fewer tick marks to prevent overlap
        if len(num_gpp) > 20:
            # For many values, show every other tick mark
            tick_indices = list(range(0, len(num_gpp), 2))
            if len(num_gpp) - 1 not in tick_indices:  # Always show the last value
                tick_indices.append(len(num_gpp) - 1)
            tick_positions = [x[i] for i in tick_indices]
            tick_labels = [calculate_actual_objects(num_gpp[i]) for i in tick_indices]
        else:
            tick_positions = x
            tick_labels = [calculate_actual_objects(num_gpp[i]) for i in x]
        
        current_ax.set_xticks(tick_positions)
        current_ax.set_xticklabels(tick_labels)
        current_ax.tick_params(axis='x', rotation=45)
        current_ax.set_xlabel("Total Geometries", fontsize=14)
        current_ax.set_ylabel("Time (us)" if col_idx == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Add subtitle indicating bar grouping only on first subplot
        if col_idx == 0:
            current_ax.text(0.5, -0.18, "(left: sycl-gpu, right: drake-cpu)", 
                           transform=current_ax.transAxes, ha='center', fontsize=12)
        
        # Only show legend on the first subplot
        if col_idx == 0:
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

def plot_broad_narrow_misc_vs_obp(all_data, run_types, objects_per_pile, sphere_resolutions, ax=None):
    """
    Plots a grouped stacked bar plot comparing BroadPhase, NarrowPhase, and Misc times for sycl-gpu and drake-cpu.
    The full bar is HydroelasticQueryTime, with segments for BroadPhase, NarrowPhase, and Misc.
    Uses subplots for different sphere resolutions.
    Args:
        all_data: Nested dict as in object_scaling.py
        run_types: List of run types (should include 'sycl-gpu' and 'drake-cpu')
        objects_per_pile: List of objects per pile  (e.g., ["1", "2", "5", "10", "20"])
        sphere_resolutions: List of sphere resolutions (e.g., ["0.0050", "0.0100", "0.0200", "0.0400"])
        ax: Optional matplotlib axis to plot on
    """
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots dynamically based on number of sphere resolutions
    n_cols = len(sphere_resolutions)
    if ax is None:
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), sharey=True)
    
    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])
    
    # Define colors for different timing components
    colors = {
        'BroadPhase': '#1f77b4',
        'NarrowPhase': '#ff7f0e',
        'Misc': '#2ca02c',
    }
    
    # Plot for each spacing
    for col_idx, sr in enumerate(sphere_resolutions):
        current_ax = axes[col_idx]
        
        bar_width = 0.35
        x = list(range(len(objects_per_pile)))
        run_type_offsets = {'sycl-gpu': -bar_width/2, 'drake-cpu': bar_width/2}
        
        for i, run_type in enumerate(['sycl-gpu', 'drake-cpu']):
            if run_type not in run_types:
                continue
                
            broad_vals, narrow_vals, misc_vals, total_vals = [], [], [], []
            
            for obp in objects_per_pile:
                timings = all_data[run_type][obp][sr]["timing_overall"].get("timings", {})
                hq_data = timings.get("HydroelasticQuery", {})
                hq = get_corrected_timing(hq_data, run_type)
                
                if run_type == 'sycl-gpu':
                    kernel_timing = all_data[run_type][obp][sr]["kernel_timing"].get("kernel_timings", {})
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
                               label=f'BroadPhase' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            b2 = current_ax.bar(xpos, narrow_vals, bar_width, bottom=broad_vals, color=colors['NarrowPhase'], 
                               label=f'NarrowPhase' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            bottoms = [b + n for b, n in zip(broad_vals, narrow_vals)]
            b3 = current_ax.bar(xpos, misc_vals, bar_width, bottom=bottoms, color=colors['Misc'], 
                               label=f'Misc' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
        
        # Customize the subplot
        current_ax.set_xticks(x)
        current_ax.set_xticklabels([calculate_actual_objects_clutter(obp) for obp in objects_per_pile])
        current_ax.set_xlabel("Total geometries", fontsize=14)
        current_ax.set_ylabel("Time (us)" if col_idx == 0 else "", fontsize=14)
        current_ax.set_title(f"Sphere Resolution- {sr}", fontsize=15, fontweight='bold')
        
        # Add subtitle indicating bar grouping only on first subplot
        if col_idx == 0:
            current_ax.text(0.5, -0.18, "(left: sycl-gpu, right: drake-cpu)", 
                           transform=current_ax.transAxes, ha='center', fontsize=12, fontweight='bold')
        
        # Only show legend on the first subplot
        if col_idx == 0:
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
def plot_broad_narrow_misc_vs_num_elements(all_data, run_types, spacings, num_gpp, ax=None):
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
    
    # Create subplots dynamically based on number of spacings
    n_cols = len(spacings)
    if ax is None:
        fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7), sharey=True)
    
    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])
    
    # Define colors for different timing components
    colors = {
        'BroadPhase': '#1f77b4',
        'NarrowPhase': '#ff7f0e',
        'Misc': '#2ca02c',
    }
    
    num_elements = [calculate_number_of_elements_objects_scaling(gpp, all_data[run_types[0]][spacings[0]][gpp]["problem_size"]) for gpp in num_gpp]
    
    # Plot for each spacing
    for col_idx, spacing in enumerate(spacings):
        current_ax = axes[col_idx]
        
        bar_width = 0.35
        x = list(range(len(num_elements)))
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
                               label=f'BroadPhase' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            b2 = current_ax.bar(xpos, narrow_vals, bar_width, bottom=broad_vals, color=colors['NarrowPhase'], 
                               label=f'NarrowPhase' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            bottoms = [b + n for b, n in zip(broad_vals, narrow_vals)]
            b3 = current_ax.bar(xpos, misc_vals, bar_width, bottom=bottoms, color=colors['Misc'], 
                               label=f'Misc' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
        
        # Customize the subplot
        current_ax.set_xticks(x)
        
        # Format x-axis labels to be more readable
        def format_num_elements(num):
            if num >= 1000000:
                return f"{num/1000000:.1f}M"
            elif num >= 1000:
                return f"{num/1000:.0f}K"
            else:
                return str(num)
        
        formatted_labels = [format_num_elements(num) for num in num_elements]
        current_ax.set_xticklabels(formatted_labels, rotation=45, ha='right')
        current_ax.set_xlabel("Number of Elements", fontsize=14)
        current_ax.set_ylabel("Time (us)" if col_idx == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        
        # Add subtitle indicating bar grouping only on first subplot
        if col_idx == 0:
            current_ax.text(0.5, -0.18, "(left: sycl-gpu, right: drake-cpu)", 
                           transform=current_ax.transAxes, ha='center', fontsize=12)
        
        # Only show legend on the first subplot
        if col_idx == 0:
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
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    return axes
def plot_broad_narrow_misc_vs_num_elements_clutter(all_data, run_types, objects_per_pile, sphere_resolutions, ax=None):
    """
    Plots a grouped stacked bar plot comparing BroadPhase, NarrowPhase, and Misc times for sycl-gpu and drake-cpu.
    The full bar is HydroelasticQueryTime, with segments for BroadPhase, NarrowPhase, and Misc.
    Uses subplots for different sphere resolutions.
    Args:
        all_data: Nested dict as in object_scaling.py
        run_types: List of run types (should include 'sycl-gpu' and 'drake-cpu')
        objects_per_pile: List of objects per pile  (e.g., ["1", "2", "5", "10", "20"])
        sphere_resolutions: List of sphere resolutions (e.g., ["0.0050", "0.0100", "0.0200", "0.0400"])
        ax: Optional matplotlib axis to plot on
    """
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create subplots dynamically based on number of sphere resolutions
    n_cols = len(sphere_resolutions)
    if ax is None:
        fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7), sharey=True)
    else:
        # If ax is provided, we assume it's a single axis, so we can't create subplots
        fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7), sharey=True)
    
    # Handle single column case
    if n_cols == 1:
        axes = np.array([axes])
    
    # Define colors for different timing components
    colors = {
        'BroadPhase': '#1f77b4',
        'NarrowPhase': '#ff7f0e',
        'Misc': '#2ca02c',
    }
    

    
    # Plot for each sphere resolution
    for col_idx, sr in enumerate(sphere_resolutions):
        num_elements = [calculate_number_of_elements_clutter(obp, sr, all_data[run_types[0]][obp][sr]["problem_size"]) for obp in objects_per_pile]
        current_ax = axes[col_idx]
        
        bar_width = 0.35
        x = list(range(len(num_elements)))
        run_type_offsets = {'sycl-gpu': -bar_width/2, 'drake-cpu': bar_width/2}
        
        for i, run_type in enumerate(['sycl-gpu', 'drake-cpu']):
            if run_type not in run_types:
                continue
                
            broad_vals, narrow_vals, misc_vals, total_vals = [], [], [], []
            
            for obp in objects_per_pile:
                timings = all_data[run_type][obp][sr]["timing_overall"].get("timings", {})
                hq_data = timings.get("HydroelasticQuery", {})
                hq = get_corrected_timing(hq_data, run_type)
                
                if run_type == 'sycl-gpu':
                    kernel_timing = all_data[run_type][obp][sr]["kernel_timing"].get("kernel_timings", {})
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
                               label=f'BroadPhase' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            b2 = current_ax.bar(xpos, narrow_vals, bar_width, bottom=broad_vals, color=colors['NarrowPhase'], 
                               label=f'NarrowPhase' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
            
            bottoms = [b + n for b, n in zip(broad_vals, narrow_vals)]
            b3 = current_ax.bar(xpos, misc_vals, bar_width, bottom=bottoms, color=colors['Misc'], 
                               label=f'Misc' if col_idx == 0 and run_type == 'sycl-gpu' else None, 
                               hatch='//' if run_type == 'sycl-gpu' else None)
        
        # Customize the subplot
        current_ax.set_xticks(x)
        
        # Format x-axis labels to be more readable
        def format_num_elements(num):
            if num >= 1000000:
                return f"{num/1000000:.1f}M"
            elif num >= 1000:
                return f"{num/1000:.0f}K"
            else:
                return str(num)
        
        formatted_labels = [format_num_elements(num) for num in num_elements]
        current_ax.set_xticklabels(formatted_labels, rotation=45, ha='right')
        current_ax.set_xlabel("Number of Elements", fontsize=14)
        current_ax.set_ylabel("Time (us)" if col_idx == 0 else "", fontsize=14)
        # Set title based on sphere resolution
        current_ax.set_title(f"Sphere Resolution- {sr}", fontsize=15, fontweight='bold')
        
        # Add subtitle indicating bar grouping only on first subplot
        if col_idx == 0:
            current_ax.text(0.5, -0.18, "(left: sycl-gpu, right: drake-cpu)", 
                           transform=current_ax.transAxes, ha='center', fontsize=12)
        
        # Only show legend on the first subplot
        if col_idx == 0:
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
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
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
        # Show fewer tick marks to prevent overlap
        if len(num_gpp) > 6:
            # For many values, show every other tick mark
            tick_indices = list(range(0, len(num_gpp), 2))
            if len(num_gpp) - 1 not in tick_indices:  # Always show the last value
                tick_indices.append(len(num_gpp) - 1)
            tick_values = [int(num_gpp[i]) for i in tick_indices]
        else:
            tick_values = [int(gpp) for gpp in num_gpp]
        
        current_ax.set_xticks(tick_values)
        current_ax.tick_params(axis='x', rotation=45)
        current_ax.set_xlabel("Number of Objects per Group", fontsize=14)
        current_ax.set_ylabel("Narrow Phase Time (us)" if i == 0 else "", fontsize=14)
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
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
        # title = "Sparse" if spacing == "0.1" else "Dense"
        title = ""
        if(spacing == "0.1"):
            title = "Sparse - 0.1"
        elif(spacing == "0.15"):
            title = "Sparse - 0.15"
        elif(spacing == "0.05"):
            title = "Dense - 0.05"
        current_ax.set_title(title, fontsize=15, fontweight='bold')
        current_ax.set_xlim(0, 500000)  # Set consistent x-axis limits
        current_ax.xaxis.set_major_locator(plt.MultipleLocator(100000))
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
