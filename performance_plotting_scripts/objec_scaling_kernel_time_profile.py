import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

def read_timing_data(file_path):
    """Read timing data from txt file and return lists of time steps and timing values."""
    time_steps = []
    timings = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        time_steps.append(int(parts[0]))  # time_step is the first column
                        timings.append(float(parts[1]))   # timing_us is the second column
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return [], []
    return time_steps, timings

def plot_kernel_timing_lines(base_dir, demo_name, spacings, num_gpp, kernels, run_type="sycl-gpu"):
    """Plot line plots for specified kernels, showing timing vs time step for each configuration."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Define y-axis limits for each kernel
    y_limits = {
        "transform_and_broad_phase": 70000,
        "compute_contact_polygons": 2000
    }
    
    # Create subplots for each kernel
    fig, axes = plt.subplots(len(kernels), len(spacings), figsize=(6*len(spacings), 5*len(kernels)))
    if len(kernels) == 1:
        axes = axes.reshape(1, -1)
    if len(spacings) == 1:
        axes = axes.reshape(-1, 1)
    
    for kernel_idx, kernel in enumerate(kernels):
        for spacing_idx, spacing in enumerate(spacings):
            ax = axes[kernel_idx, spacing_idx]
            
            # Plot data for each gpp value
            for gpp in num_gpp:
                # Construct file path
                file_path = f"{base_dir}/performance_jsons_bvh/{demo_name}_{spacing}_{gpp}_{run_type}_timing_{kernel}.txt"
                
                # Read timing data
                time_steps, timings = read_timing_data(file_path)
                
                if timings:
                    ax.plot(time_steps, timings, label=f'GPP {gpp}', linewidth=1, alpha=0.8)
            
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Timing (μs)', fontsize=10)
            ax.set_title(f'{kernel.replace("_", " ").title()} - Spacing {spacing}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set y-axis limit for this kernel
            if kernel in y_limits:
                ax.set_ylim(0, y_limits[kernel])
    
    plt.tight_layout()
    return fig

def main():
    base_dir = os.path.dirname(os.getcwd())
    demo_name = "objects_scaling"
    spacings = ["0.1", "0.05"]
    num_gpp = ["1", "2", "5", "10", "20"]
    kernels = ["transform_and_broad_phase", "compute_contact_polygons"]
    run_type = "sycl-gpu"
    
    # Create plots directory
    plot_dir = "plots_bvh_1s"
    if not os.path.exists(f"{base_dir}/{plot_dir}"):
        os.makedirs(f"{base_dir}/{plot_dir}")
    
    # Plot: Line plots for each configuration
    print("Creating line plots for each configuration...")
    fig = plot_kernel_timing_lines(base_dir, demo_name, spacings, num_gpp, kernels, run_type)
    plt.savefig(f"{base_dir}/{plot_dir}/object_scaling_kernel_timing_lines.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {base_dir}/{plot_dir}/object_scaling_kernel_timing_lines.png")
    plt.close()
    
    print("Line plots completed!")

if __name__ == "__main__":
    main()
