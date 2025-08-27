import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from utils import get_data

def get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, key):
    """
    Get corrected timing for sycl by excluding JIT compilation time.
    For sycl-cpu/sycl-gpu: read the txt file and remove the first timing
    """
    first_time = float(raw_timing_data[key])
    total_time = float(timing_data.get(key, {}).get("total_us", 0))
    return total_time - first_time

def get_gpu_corrected_timing_avg_time(timing_data, raw_timing_data, key):
    """
    Get corrected timing for sycl by excluding JIT compilation time.
    For sycl-cpu/sycl-gpu: read the txt file and remove the first timing
    """
    calls = int(timing_data.get("calls", 1))
    first_time = float(raw_timing_data[key])
    total_time = float(timing_data.get("total_us", 0))
    return (total_time - first_time) / (calls - 1)

def extract_time_split(all_data, what_run_type, demo):
    # Return six components (seconds): solver_time, misc_solver_time, broad_phase_time, narrow_phase_time, memcpy_time, hydro_unaccounted_time
    if(what_run_type == "sycl-gpu"):
        # All time in s
        hydro_total_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("HydroelasticQuery", {}).get("total_us", None)) / 1e6
        total_advance_to_time = float(all_data[demo][what_run_type]["advance_to"].get("advance_to_time", None)) # Already in s
        fcl_broad_phase_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("FCLBroadPhase", {}).get("total_us", None)) / 1e6
        solve_with_guess = all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("solve_with_guess", {}).get("total_us", None)

        # Corrected GPU timings
        timing_data = all_data[demo][what_run_type]["kernel_timing"].get("kernel_timings", {})
        raw_timing_data = all_data[demo][what_run_type]["raw_timing_data"]
        memcpy_host_to_device = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "unpack_transforms") / 1e6
        memcpy_device_to_host = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "device_to_host_memcpy") / 1e6
        broad_phase_time_kernel = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "transform_and_broad_phase") / 1e6
        narrow_phase_time_kernel = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "compute_contact_polygons") / 1e6
        compact_polygon_time = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "compact_polygon_data") / 1e6

        memcpy_time = memcpy_host_to_device + memcpy_device_to_host
        broad_phase_time = broad_phase_time_kernel + fcl_broad_phase_time
        narrow_phase_time = narrow_phase_time_kernel + compact_polygon_time
        hydro_unaccounted_time = hydro_total_time - broad_phase_time - narrow_phase_time - memcpy_time
        if solve_with_guess != None:
            solve_with_guess = float(solve_with_guess) / 1e6
            misc_solver_time = total_advance_to_time - solve_with_guess - hydro_total_time
        else:
            misc_solver_time = 0
            solve_with_guess = total_advance_to_time - hydro_total_time

        return solve_with_guess, misc_solver_time, broad_phase_time, narrow_phase_time, memcpy_time, hydro_unaccounted_time

    elif(what_run_type == "drake-cpu"):
        # All time in s
        hydro_total_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("HydroelasticQuery", {}).get("total_us", None)) / 1e6
        total_advance_to_time = float(all_data[demo][what_run_type]["advance_to"].get("advance_to_time", None)) # Already in s
        broad_phase_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("BroadPhase", {}).get("total_us", None)) / 1e6
        narrow_phase_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("NarrowPhase", {}).get("total_us", None)) / 1e6
        fcl_broad_phase_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("FCLBroadPhase", {}).get("total_us", None)) / 1e6

        broad_phase_time = broad_phase_time + fcl_broad_phase_time
        hydro_unaccounted_time = hydro_total_time - broad_phase_time - narrow_phase_time
        solve_with_guess = all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("solve_with_guess", {}).get("total_us", None)
        if solve_with_guess != None:
            solve_with_guess = float(solve_with_guess) / 1e6
            misc_solver_time = total_advance_to_time - solve_with_guess - hydro_total_time
        else:
            misc_solver_time = 0
            solve_with_guess = total_advance_to_time - hydro_total_time

        return solve_with_guess, misc_solver_time, broad_phase_time, narrow_phase_time, 0, hydro_unaccounted_time
    
def extract_hydro_time(all_data, what_run_type, demo):
    if(what_run_type == "sycl-gpu"):
        # All time in s
        hydro_total_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("HydroelasticQuery", {}).get("total_us", None)) / 1e6
        fcl_broad_phase_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("FCLBroadPhase", {}).get("total_us", None)) / 1e6

        # Get the corrected GPU timing
        timing_data = all_data[demo][what_run_type]["kernel_timing"].get("kernel_timings", {})
        raw_timing_data = all_data[demo][what_run_type]["raw_timing_data"]

        memcpy_host_to_device = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "unpack_transforms") / 1e6
        memcpy_device_to_host = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "device_to_host_memcpy") / 1e6
        broad_phase_time = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "transform_and_broad_phase") / 1e6
        narrow_phase_time = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "compute_contact_polygons") / 1e6
        compact_polygon_time = get_gpu_corrected_timing_total_time(timing_data, raw_timing_data, "compact_polygon_data") / 1e6

        memcpy_time = memcpy_host_to_device + memcpy_device_to_host
        narrow_phase_time = narrow_phase_time + compact_polygon_time
        broad_phase_time = broad_phase_time + fcl_broad_phase_time
        hydro_unaccounted_time = hydro_total_time - broad_phase_time - narrow_phase_time - memcpy_time
        print(f"hydro_unaccounted_time: {hydro_unaccounted_time}")

        return broad_phase_time, narrow_phase_time, memcpy_time, hydro_unaccounted_time
    
    elif(what_run_type == "drake-cpu"):
        # All time in s
        hydro_total_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("HydroelasticQuery", {}).get("total_us", None)) / 1e6
        broad_phase_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("BroadPhase", {}).get("total_us", None)) / 1e6
        narrow_phase_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("NarrowPhase", {}).get("total_us", None)) / 1e6
        fcl_broad_phase_time = float(all_data[demo][what_run_type]["timing_overall"].get("timings", {}).get("FCLBroadPhase", {}).get("total_us", None)) / 1e6

        broad_phase_time = broad_phase_time + fcl_broad_phase_time
        hydro_unaccounted_time = hydro_total_time - broad_phase_time - narrow_phase_time
        print(f"hydro_unaccounted_time: {hydro_unaccounted_time}")

        return broad_phase_time, narrow_phase_time, 0, hydro_unaccounted_time
    
def plot_demos_split(all_data, plot_dir):
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for different timing components
    colors = {
        'Solver': '#1f77b4',
        'Solver Misc': '#d62728',
        'BroadPhase': '#ff7f0e', 
        'NarrowPhase': '#9467bd',
        'Memcpy': '#2ca02c',
        'Hydro Misc': '#8c564b'
    }
    
    # Bar width and positioning
    bar_width = 0.35
    x_positions = list(range(len(demos)))
    
    # Plot for each demo
    for i, demo in enumerate(demos):
        # Extract time splits (absolute seconds)
        gpu_solver_time, gpu_misc_solver_time, gpu_broad_phase_time, gpu_narrow_phase_time, gpu_memcpy_time, gpu_hydro_unaccounted_time = extract_time_split(all_data, "sycl-gpu", demo)
        cpu_solver_time, cpu_misc_solver_time, cpu_broad_phase_time, cpu_narrow_phase_time, cpu_memcpy_time, cpu_hydro_unaccounted_time = extract_time_split(all_data, "drake-cpu", demo)
        
        # Calculate total times
        gpu_total = gpu_solver_time + gpu_misc_solver_time + gpu_broad_phase_time + gpu_narrow_phase_time + gpu_memcpy_time + gpu_hydro_unaccounted_time
        cpu_total = cpu_solver_time + cpu_misc_solver_time + cpu_broad_phase_time + cpu_narrow_phase_time + cpu_memcpy_time + cpu_hydro_unaccounted_time
        
        # Debug prints
        print(f"demo: {demo}")
        print(f"  GPU - solver: {gpu_solver_time:.3f}, misc: {gpu_misc_solver_time:.3f}, broad: {gpu_broad_phase_time:.3f}, narrow: {gpu_narrow_phase_time:.3f}, memcpy: {gpu_memcpy_time:.3f}, hydro_unaccounted: {gpu_hydro_unaccounted_time:.3f}, total: {gpu_total:.3f}")
        print(f"  CPU - solver: {cpu_solver_time:.3f}, misc: {cpu_misc_solver_time:.3f}, broad: {cpu_broad_phase_time:.3f}, narrow: {cpu_narrow_phase_time:.3f}, memcpy: {cpu_memcpy_time:.3f}, hydro_unaccounted: {cpu_hydro_unaccounted_time:.3f}, total: {cpu_total:.3f}")
        
        # Compute the maximum speedup possible if hydro goes to 0 from cpu
        cpu_time_without_hydro = cpu_solver_time + cpu_misc_solver_time
        max_speedup = cpu_total / cpu_time_without_hydro
        print(f"max_speedup: {max_speedup}")
        
        # Calculate fractions (ensure they sum to 1)
        gpu_solver_frac = (gpu_solver_time / gpu_total) if gpu_total > 0 else 0
        gpu_misc_solver_frac = (gpu_misc_solver_time / gpu_total) if gpu_total > 0 else 0
        gpu_broad_frac = (gpu_broad_phase_time / gpu_total) if gpu_total > 0 else 0
        gpu_narrow_frac = (gpu_narrow_phase_time / gpu_total) if gpu_total > 0 else 0
        gpu_memcpy_frac = (gpu_memcpy_time / gpu_total) if gpu_total > 0 else 0
        gpu_hydro_misc_frac = (gpu_hydro_unaccounted_time / gpu_total) if gpu_total > 0 else 0
        
        cpu_solver_frac = (cpu_solver_time / cpu_total) if cpu_total > 0 else 0
        cpu_misc_solver_frac = (cpu_misc_solver_time / cpu_total) if cpu_total > 0 else 0
        cpu_broad_frac = (cpu_broad_phase_time / cpu_total) if cpu_total > 0 else 0
        cpu_narrow_frac = (cpu_narrow_phase_time / cpu_total) if cpu_total > 0 else 0
        cpu_memcpy_frac = (cpu_memcpy_time / cpu_total) if cpu_total > 0 else 0
        cpu_hydro_misc_frac = (cpu_hydro_unaccounted_time / cpu_total) if cpu_total > 0 else 0
        
        # GPU bar (left)
        gpu_x = x_positions[i] - bar_width/2
        
        # Create stacked bars for GPU
        ax.bar(gpu_x, gpu_solver_frac, bar_width, color=colors['Solver'], 
               label='Solver' if i == 0 else None, hatch='//')
        ax.bar(gpu_x, gpu_misc_solver_frac, bar_width, bottom=gpu_solver_frac, 
               color=colors['Solver Misc'], label='Solver Misc' if i == 0 else None, hatch='//')
        ax.bar(gpu_x, gpu_broad_frac, bar_width, 
               bottom=gpu_solver_frac + gpu_misc_solver_frac, 
               color=colors['BroadPhase'], label='BroadPhase' if i == 0 else None, hatch='//')
        ax.bar(gpu_x, gpu_narrow_frac, bar_width, 
               bottom=gpu_solver_frac + gpu_misc_solver_frac + gpu_broad_frac, 
               color=colors['NarrowPhase'], label='NarrowPhase' if i == 0 else None, hatch='//')
        ax.bar(gpu_x, gpu_memcpy_frac, bar_width, 
               bottom=gpu_solver_frac + gpu_misc_solver_frac + gpu_broad_frac + gpu_narrow_frac, 
               color=colors['Memcpy'], label='Memcpy' if i == 0 else None, hatch='//')
        ax.bar(gpu_x, gpu_hydro_misc_frac, bar_width, 
               bottom=gpu_solver_frac + gpu_misc_solver_frac + gpu_broad_frac + gpu_narrow_frac + gpu_memcpy_frac, 
               color=colors['Hydro Misc'], label='Hydro Misc' if i == 0 else None, hatch='//')
        # CPU bar (right)
        cpu_x = x_positions[i] + bar_width/2
        
        # Create stacked bars for CPU
        ax.bar(cpu_x, cpu_solver_frac, bar_width, color=colors['Solver'])
        ax.bar(cpu_x, cpu_misc_solver_frac, bar_width, bottom=cpu_solver_frac, 
               color=colors['Solver Misc'])
        ax.bar(cpu_x, cpu_broad_frac, bar_width, 
               bottom=cpu_solver_frac + cpu_misc_solver_frac, 
               color=colors['BroadPhase'])
        ax.bar(cpu_x, cpu_narrow_frac, bar_width, 
               bottom=cpu_solver_frac + cpu_misc_solver_frac + cpu_broad_frac, 
               color=colors['NarrowPhase'])
        ax.bar(cpu_x, cpu_memcpy_frac, bar_width, 
               bottom=cpu_solver_frac + cpu_misc_solver_frac + cpu_broad_frac + cpu_narrow_frac, 
               color=colors['Memcpy'])
        ax.bar(cpu_x, cpu_hydro_misc_frac, bar_width, 
               bottom=cpu_solver_frac + cpu_misc_solver_frac + cpu_broad_frac + cpu_narrow_frac + cpu_memcpy_frac, 
               color=colors['Hydro Misc'])
    # Customize the plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(demos)
    ax.set_ylabel('Time Fraction', fontsize=14)
    # ax.set_title('Time Split Comparison: GPU vs CPU', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, fontsize=12, title_fontsize=13)
    
    # Add subtitle indicating bar grouping
    ax.text(0.5, -0.15, "(left: sycl-gpu, right: drake-cpu)", 
           transform=ax.transAxes, ha='center', fontsize=12)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{plot_dir}/time_split_comparison_normalized.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {base_dir}/{plot_dir}/time_split_comparison_normalized.png")
    plt.show()
    
def plot_hydro_split_time_normalized(all_data, plot_dir):
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for different timing components
    colors = {
        'BroadPhase': '#ff7f0e',
        'NarrowPhase': '#9467bd',
        'Memcpy': '#2ca02c',
        'Hydro Misc': '#8c564b'
    }

    # Bar width and positioning
    bar_width = 0.35
    x_positions = list(range(len(demos)))

    # Plot for each demo
    for i, demo in enumerate(demos):
        # Extract time splits
        gpu_broad_phase_time, gpu_narrow_phase_time, gpu_memcpy_time, gpu_hydro_unaccounted_time = extract_hydro_time(all_data, "sycl-gpu", demo)
        cpu_broad_phase_time, cpu_narrow_phase_time, cpu_memcpy_time, cpu_hydro_unaccounted_time = extract_hydro_time(all_data, "drake-cpu", demo)

        # Calculate totals
        gpu_total = gpu_broad_phase_time + gpu_narrow_phase_time + gpu_memcpy_time + gpu_hydro_unaccounted_time
        cpu_total = cpu_broad_phase_time + cpu_narrow_phase_time + cpu_memcpy_time + cpu_hydro_unaccounted_time

        # Debug prints
        print(f"demo: {demo}")
        print(f"  GPU - broad: {gpu_broad_phase_time:.3f}, narrow: {gpu_narrow_phase_time:.3f}, memcpy: {gpu_memcpy_time:.3f}, hydro_unaccounted: {gpu_hydro_unaccounted_time:.3f}, total: {gpu_total:.3f}")
        print(f"  CPU - broad: {cpu_broad_phase_time:.3f}, narrow: {cpu_narrow_phase_time:.3f}, memcpy: {cpu_memcpy_time:.3f}, hydro_unaccounted: {cpu_hydro_unaccounted_time:.3f}, total: {cpu_total:.3f}")

        # Fractions
        gpu_broad_frac = (gpu_broad_phase_time / gpu_total) if gpu_total > 0 else 0
        gpu_narrow_frac = (gpu_narrow_phase_time / gpu_total) if gpu_total > 0 else 0
        gpu_memcpy_frac = (gpu_memcpy_time / gpu_total) if gpu_total > 0 else 0
        gpu_hydro_misc_frac = (gpu_hydro_unaccounted_time / gpu_total) if gpu_total > 0 else 0

        cpu_broad_frac = (cpu_broad_phase_time / cpu_total) if cpu_total > 0 else 0
        cpu_narrow_frac = (cpu_narrow_phase_time / cpu_total) if cpu_total > 0 else 0
        cpu_memcpy_frac = (cpu_memcpy_time / cpu_total) if cpu_total > 0 else 0
        cpu_hydro_misc_frac = (cpu_hydro_unaccounted_time / cpu_total) if cpu_total > 0 else 0

        # Positions
        gpu_x = x_positions[i] - bar_width/2
        cpu_x = x_positions[i] + bar_width/2

        # GPU stacked
        ax.bar(gpu_x, gpu_broad_frac, bar_width, color=colors['BroadPhase'], label='BroadPhase' if i == 0 else None, hatch='//')
        ax.bar(gpu_x, gpu_narrow_frac, bar_width, bottom=gpu_broad_frac, color=colors['NarrowPhase'], label='NarrowPhase' if i == 0 else None, hatch='//')
        ax.bar(gpu_x, gpu_memcpy_frac, bar_width, bottom=gpu_broad_frac + gpu_narrow_frac, color=colors['Memcpy'], label='Memcpy' if i == 0 else None, hatch='//')
        ax.bar(gpu_x, gpu_hydro_misc_frac, bar_width, bottom=gpu_broad_frac + gpu_narrow_frac + gpu_memcpy_frac, color=colors['Hydro Misc'], label='Hydro Misc' if i == 0 else None, hatch='//')

        # CPU stacked
        ax.bar(cpu_x, cpu_broad_frac, bar_width, color=colors['BroadPhase'])
        ax.bar(cpu_x, cpu_narrow_frac, bar_width, bottom=cpu_broad_frac, color=colors['NarrowPhase'])
        ax.bar(cpu_x, cpu_memcpy_frac, bar_width, bottom=cpu_broad_frac + cpu_narrow_frac, color=colors['Memcpy'])
        ax.bar(cpu_x, cpu_hydro_misc_frac, bar_width, bottom=cpu_broad_frac + cpu_narrow_frac + cpu_memcpy_frac, color=colors['Hydro Misc'])

    # Customize the plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(demos)
    ax.set_ylabel('Time Fraction', fontsize=14)
    ax.set_ylim(0, 1)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, fontsize=12, title_fontsize=13, loc='upper left')

    # Subtitle
    ax.text(0.5, -0.15, "(left: sycl-gpu, right: drake-cpu)", transform=ax.transAxes, ha='center', fontsize=12)

    # Grid and styling
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{plot_dir}/hydro_split_normalized.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {base_dir}/{plot_dir}/hydro_split_normalized.png")
    plt.show()

def plot_demos_split_single(run_type, all_data, plot_dir):
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors must match combined plot
    colors = {
        'Solver': '#1f77b4',
        'Solver Misc': '#d62728',
        'BroadPhase': '#ff7f0e',
        'NarrowPhase': '#9467bd',
        'Memcpy': '#2ca02c',
        'Hydro Misc': '#8c564b'
    }

    bar_width = 0.5
    x_positions = list(range(len(demos)))
    hatch_pattern = '//' if run_type == 'sycl-gpu' else None

    for i, demo in enumerate(demos):
        solver_time, misc_solver_time, broad_phase_time, narrow_phase_time, memcpy_time, hydro_misc_time = extract_time_split(all_data, run_type, demo)

        total_time = solver_time + misc_solver_time + broad_phase_time + narrow_phase_time + memcpy_time + hydro_misc_time

        # Debug prints
        print(f"[single-{run_type}] demo: {demo}")
        print(f"  {run_type} - solver: {solver_time:.3f}, misc: {misc_solver_time:.3f}, broad: {broad_phase_time:.3f}, narrow: {narrow_phase_time:.3f}, memcpy: {memcpy_time:.3f}, hydro_misc: {hydro_misc_time:.3f}, total: {total_time:.3f}")

        # Fractions
        solver_frac = (solver_time / total_time) if total_time > 0 else 0
        misc_solver_frac = (misc_solver_time / total_time) if total_time > 0 else 0
        broad_frac = (broad_phase_time / total_time) if total_time > 0 else 0
        narrow_frac = (narrow_phase_time / total_time) if total_time > 0 else 0
        memcpy_frac = (memcpy_time / total_time) if total_time > 0 else 0
        hydro_misc_frac = (hydro_misc_time / total_time) if total_time > 0 else 0

        x = x_positions[i]
        ax.bar(x, solver_frac, bar_width, color=colors['Solver'], label='Solver' if i == 0 else None, hatch=hatch_pattern)
        ax.bar(x, misc_solver_frac, bar_width, bottom=solver_frac, color=colors['Solver Misc'], label='Solver Misc' if i == 0 else None, hatch=hatch_pattern)
        ax.bar(x, broad_frac, bar_width, bottom=solver_frac + misc_solver_frac, color=colors['BroadPhase'], label='BroadPhase' if i == 0 else None, hatch=hatch_pattern)
        ax.bar(x, narrow_frac, bar_width, bottom=solver_frac + misc_solver_frac + broad_frac, color=colors['NarrowPhase'], label='NarrowPhase' if i == 0 else None, hatch=hatch_pattern)
        ax.bar(x, memcpy_frac, bar_width, bottom=solver_frac + misc_solver_frac + broad_frac + narrow_frac, color=colors['Memcpy'], label='Memcpy' if i == 0 else None, hatch=hatch_pattern)
        ax.bar(x, hydro_misc_frac, bar_width, bottom=solver_frac + misc_solver_frac + broad_frac + narrow_frac + memcpy_frac, color=colors['Hydro Misc'], label='Hydro Misc' if i == 0 else None, hatch=hatch_pattern)

    # Axes and legend
    ax.set_xticks(x_positions)
    ax.set_xticklabels(demos)
    ax.set_ylabel('Time Fraction', fontsize=14)
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); new_handles = []; new_labels = []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            new_handles.append(h); new_labels.append(l); seen.add(l)
    ax.legend(new_handles, new_labels, fontsize=12, title_fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    outfile = f"{base_dir}/{plot_dir}/time_split_comparison_normalized_{run_type}.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {outfile}")
    plt.show()

def plot_hydro_split_time_normalized_single(run_type, all_data, plot_dir):
    # Set style
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'BroadPhase': '#ff7f0e',
        'NarrowPhase': '#9467bd',
        'Memcpy': '#2ca02c',
        'Hydro Misc': '#8c564b'
    }

    bar_width = 0.5
    x_positions = list(range(len(demos)))
    hatch_pattern = '//' if run_type == 'sycl-gpu' else None

    for i, demo in enumerate(demos):
        broad_phase_time, narrow_phase_time, memcpy_time, hydro_misc_time = extract_hydro_time(all_data, run_type, demo)
        total_time = broad_phase_time + narrow_phase_time + memcpy_time + hydro_misc_time

        # Debug prints
        print(f"[single-hydro-{run_type}] demo: {demo}")
        print(f"  {run_type} - broad: {broad_phase_time:.3f}, narrow: {narrow_phase_time:.3f}, memcpy: {memcpy_time:.3f}, hydro_misc: {hydro_misc_time:.3f}, total: {total_time:.3f}")

        # Fractions
        broad_frac = (broad_phase_time / total_time) if total_time > 0 else 0
        narrow_frac = (narrow_phase_time / total_time) if total_time > 0 else 0
        memcpy_frac = (memcpy_time / total_time) if total_time > 0 else 0
        hydro_misc_frac = (hydro_misc_time / total_time) if total_time > 0 else 0

        x = x_positions[i]
        ax.bar(x, broad_frac, bar_width, color=colors['BroadPhase'], label='BroadPhase' if i == 0 else None, hatch=hatch_pattern)
        ax.bar(x, narrow_frac, bar_width, bottom=broad_frac, color=colors['NarrowPhase'], label='NarrowPhase' if i == 0 else None, hatch=hatch_pattern)
        ax.bar(x, memcpy_frac, bar_width, bottom=broad_frac + narrow_frac, color=colors['Memcpy'], label='Memcpy' if i == 0 else None, hatch=hatch_pattern)
        ax.bar(x, hydro_misc_frac, bar_width, bottom=broad_frac + narrow_frac + memcpy_frac, color=colors['Hydro Misc'], label='Hydro Misc' if i == 0 else None, hatch=hatch_pattern)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(demos)
    ax.set_ylabel('Time Fraction', fontsize=14)
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); new_handles = []; new_labels = []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            new_handles.append(h); new_labels.append(l); seen.add(l)
    ax.legend(new_handles, new_labels, fontsize=12, title_fontsize=13, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    outfile = f"{base_dir}/{plot_dir}/hydro_split_normalized_{run_type}.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {outfile}")
    plt.show()
    
def main():
    # store all data in a nested dictionary: all_data[demo][run_type][data_type]
    all_data = {demo: {run_type: {} for run_type in run_types} for demo in demos}
    for demo in demos:
        for run_type in run_types:
            all_data[demo][run_type] = {}
            json_path, demo_name = json_path_per_demo[demo]
            # Problem size
            json_path_problem_size = f"{base_dir}/{json_path}/{demo_name}_{run_type}_problem_size.json"
            data_problem_size = get_data(json_path_problem_size)
            all_data[demo][run_type]["problem_size"] = data_problem_size
            
            # Timing overall
            json_path_timing_overall = f"{base_dir}/{json_path}/{demo_name}_{run_type}_timing_overall.json"
            data_timing_overall = get_data(json_path_timing_overall)
            all_data[demo][run_type]["timing_overall"] = data_timing_overall
            
            json_path_advance_to = f"{base_dir}/{json_path}/{demo_name}_{run_type}_timing_advance_to.json"
            data_advance_to = get_data(json_path_advance_to)
            all_data[demo][run_type]["advance_to"] = data_advance_to
            
            # Kernel timing
            kernel_keys = ["unpack_transforms", "transform_and_broad_phase", "device_to_host_memcpy", "compute_contact_polygons", "compact_polygon_data"]
            txt_base_path = f"{base_dir}/{json_path}/{demo_name}_{run_type}_timing_"
            if(run_type == "sycl-gpu" or run_type == "sycl-cpu"):
                json_path_kernel_timing = f"{base_dir}/{json_path}/{demo_name}_{run_type}_timing.json"
                data_kernel_timing = get_data(json_path_kernel_timing)
                all_data[demo][run_type]["kernel_timing"] = data_kernel_timing
                all_data[demo][run_type]["raw_timing_data"] = {}
                for key in kernel_keys:
                    txt_path = f"{txt_base_path}{key}.txt"
                    with open(txt_path, "r") as f:
                        third_line = f.readlines()[2]
                        third_line_split_cost = float(third_line.split(" ")[1])
                        all_data[demo][run_type]["raw_timing_data"][key] = third_line_split_cost
    
    
    # Plot normalized timing data
    # For GPU this is split memcpy + hydroelastic query + solver
    # For CPU this is hydroelastic query + solver
    plot_dir = "plots_joint_demo"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    plot_demos_split(all_data, plot_dir)
    plot_hydro_split_time_normalized(all_data, plot_dir)
    plot_demos_split_single('sycl-gpu', all_data, plot_dir)
    plot_demos_split_single('drake-cpu', all_data, plot_dir)
    plot_hydro_split_time_normalized_single('sycl-gpu', all_data, plot_dir)
    plot_hydro_split_time_normalized_single('drake-cpu', all_data, plot_dir)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.getcwd())
    run_types = ["sycl-gpu", "drake-cpu"]
    demos = ["Barret Discrete SAP", "Barret No EC", "Barret With EC","Anzu No EC", "Anzu With EC", "Clutter", "Spatula 50 Envs"]
    # demos = ["Barret Discrete SAP", "Barret No EC", "Barret With EC"]
    # demos = ["Anzu No EC", "Anzu With EC"]
    # demos = ["Clutter 10", "Clutter 50"]
    # demos = ["Spatula 50 Envs"]
    # demos = ["1 Envs", "100 Envs", "800 Envs"]
    json_path_per_demo = {
        "Barret No EC": ("performance_jsons_BarretHand_convex", "barret"),
        "Barret With EC": ("performance_jsons_BarretHand_convex_error_control", "barret"),
        "Barret Discrete SAP": ("performance_jsons_BarretHand", "barret"),
        "Anzu No EC": ("performance_jsons_anzu", "anzu"),
        "Anzu With EC": ("performance_jsons_anzu_error_control", "anzu"),
        "500 Envs": ("performance_jsons_spatula_slip_control_scale_convex_trial", "spatula_slip_control_5_500"),
        "800 Envs": ("performance_jsons_spatula_slip_control_scale_convex", "spatula_slip_control_5_800"),
        "100 Envs": ("performance_jsons_spatula_slip_control_scale_convex", "spatula_slip_control_5_100"),
        "Spatula 50 Envs": ("performance_jsons_spatula_slip_control_scale_convex", "spatula_slip_control_5_50"),
        "Clutter 10": ("performance_jsons_clutter_convex_Aug13", "clutter_10_1.0000_0.0200_3"),
        "Clutter 50": ("performance_jsons_clutter_convex_Aug13", "clutter_50_1.0000_0.0050_3")
    }
    main()