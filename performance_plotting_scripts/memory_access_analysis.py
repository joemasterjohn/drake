import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration: Set to "sparse" or "dense" to analyze the corresponding data
SCENARIO = "dense"  # Change this to "dense" when analyzing dense data


def read_narrow_phase_indices(filename):
    """
    Read narrow phase check indices from file.
    Returns a numpy array of indices.
    """
    indices = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comment lines
                if line.startswith('#') or not line:
                    continue
                indices.append(int(line))
        return np.array(indices)
    except FileNotFoundError:
        print(f"Warning: File {filename} not found")
        return np.array([])


def analyze_memory_access_spread(indices, group_size):
    """
    Analyze memory access spread for given group size.
    
    Args:
        indices: Array of narrow phase check indices
        group_size: Number of checks per group (8 for warp, 32 for work group)
    
    Returns:
        Array of ranges (max - min) for each group
    """
    if len(indices) == 0:
        return np.array([])
    
    # Group indices into chunks of group_size
    num_complete_groups = len(indices) // group_size
    ranges = []
    
    for i in range(num_complete_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group = indices[start_idx:end_idx]
        
        if len(group) > 0:
            range_val = np.max(group) - np.min(group)
            ranges.append(range_val)
    
    return np.array(ranges)


def plot_memory_access_analysis(scenario="sparse"):
    """
    Create plots analyzing memory access patterns for different time steps.
    
    Args:
        scenario: "sparse" or "dense" - determines which data files to read
    """
    # Configuration
    time_steps = [150, 275, 1000]
    base_dir = os.path.dirname(os.getcwd())
    perf_analysis_dir = f"{base_dir}/perf_analysis"
    
    # Group sizes
    warp_size = 8  # 8 checks per warp (32 threads / 4 threads per check)
    work_group_size = 32  # 32 checks per work group (128 threads / 4 threads per check)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Memory Access Spread Analysis - {scenario.title()}\n(Lower spread = better locality)', 
                 fontsize=16, fontweight='bold')
    
    # Colors for different time steps
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Read data and create plots
    all_warp_ranges = []
    all_workgroup_ranges = []
    
    for i, step in enumerate(time_steps):
        filename = f"{perf_analysis_dir}/{scenario}_narrow_phase_check_indices_step_{step}.txt"
        indices = read_narrow_phase_indices(filename)
        
        if len(indices) == 0:
            print(f"No data found for step {step} ({scenario})")
            continue
        
        # Analyze warp-level memory access (8 checks)
        warp_ranges = analyze_memory_access_spread(indices, warp_size)
        all_warp_ranges.append(warp_ranges)
        
        # Analyze work group-level memory access (32 checks)
        workgroup_ranges = analyze_memory_access_spread(indices, work_group_size)
        all_workgroup_ranges.append(workgroup_ranges)
        
        # Plot warp analysis (top row)
        ax_warp = axes[0, i]
        if len(warp_ranges) > 0:
            ax_warp.hist(warp_ranges, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
            ax_warp.set_title(f'Step {step}\nWarp Level (8 checks)', fontsize=12, fontweight='bold')
            ax_warp.set_xlabel('Index Range (max - min)')
            ax_warp.set_ylabel('Frequency')
            ax_warp.grid(True, alpha=0.3)
            
            # Add statistics
            mean_range = np.mean(warp_ranges)
            median_range = np.median(warp_ranges)
            ax_warp.axvline(mean_range, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_range:.0f}')
            ax_warp.axvline(median_range, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_range:.0f}')
            ax_warp.legend(fontsize=10)
            
            print(f"Step {step} ({scenario}) - Warp (8 checks): Mean range = {mean_range:.1f}, Median = {median_range:.1f}")
        
        # Plot work group analysis (bottom row)
        ax_wg = axes[1, i]
        if len(workgroup_ranges) > 0:
            ax_wg.hist(workgroup_ranges, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
            ax_wg.set_title(f'Step {step}\nWork Group Level (32 checks)', fontsize=12, fontweight='bold')
            ax_wg.set_xlabel('Index Range (max - min)')
            ax_wg.set_ylabel('Frequency')
            ax_wg.grid(True, alpha=0.3)
            
            # Add statistics
            mean_range = np.mean(workgroup_ranges)
            median_range = np.median(workgroup_ranges)
            ax_wg.axvline(mean_range, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_range:.0f}')
            ax_wg.axvline(median_range, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_range:.0f}')
            ax_wg.legend(fontsize=10)
            
            print(f"Step {step} ({scenario}) - Work Group (32 checks): Mean range = {mean_range:.1f}, Median = {median_range:.1f}")
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = f"{base_dir}/plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    plt.savefig(f"{plots_dir}/memory_access_spread_analysis_{scenario}.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {plots_dir}/memory_access_spread_analysis_{scenario}.png")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print(f"MEMORY ACCESS LOCALITY ANALYSIS SUMMARY - {scenario.upper()}")
    print("="*80)
    print("Lower index ranges indicate better memory locality.")
    print("Higher ranges suggest scattered memory access patterns.")
    print("\nInterpretation:")
    print("- Warp level (8 checks): Shows locality within 32-thread warps")
    print("- Work group level (32 checks): Shows locality within 128-thread work groups")
    print("="*80)


def create_comparative_analysis(scenario="sparse"):
    """
    Create a comparative analysis showing how memory access patterns change over time.
    
    Args:
        scenario: "sparse" or "dense" - determines which data files to read
    """
    time_steps = [150, 275, 1000]
    base_dir = os.path.dirname(os.getcwd())
    perf_analysis_dir = f"{base_dir}/perf_analysis"
    
    warp_size = 8
    work_group_size = 32
    
    # Collect data for all time steps
    warp_stats = {'steps': [], 'mean': [], 'median': [], 'std': []}
    workgroup_stats = {'steps': [], 'mean': [], 'median': [], 'std': []}
    
    for step in time_steps:
        filename = f"{perf_analysis_dir}/{scenario}_narrow_phase_check_indices_step_{step}.txt"
        indices = read_narrow_phase_indices(filename)
        
        if len(indices) == 0:
            continue
            
        # Warp analysis
        warp_ranges = analyze_memory_access_spread(indices, warp_size)
        if len(warp_ranges) > 0:
            warp_stats['steps'].append(step)
            warp_stats['mean'].append(np.mean(warp_ranges))
            warp_stats['median'].append(np.median(warp_ranges))
            warp_stats['std'].append(np.std(warp_ranges))
        
        # Work group analysis
        workgroup_ranges = analyze_memory_access_spread(indices, work_group_size)
        if len(workgroup_ranges) > 0:
            workgroup_stats['steps'].append(step)
            workgroup_stats['mean'].append(np.mean(workgroup_ranges))
            workgroup_stats['median'].append(np.median(workgroup_ranges))
            workgroup_stats['std'].append(np.std(workgroup_ranges))
    
    # Create line plots showing trends over time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Memory Access Trends Over Time - {scenario.title()}', fontsize=14, fontweight='bold')
    
    # Warp trends
    if warp_stats['steps']:
        ax1.plot(warp_stats['steps'], warp_stats['mean'], 'o-', label='Mean', linewidth=2)
        ax1.plot(warp_stats['steps'], warp_stats['median'], 's-', label='Median', linewidth=2)
        ax1.fill_between(warp_stats['steps'], 
                        np.array(warp_stats['mean']) - np.array(warp_stats['std']),
                        np.array(warp_stats['mean']) + np.array(warp_stats['std']),
                        alpha=0.3, label='±1 Std Dev')
        ax1.set_title('Warp Level Memory Access Spread\n(8 checks per warp)', fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Index Range')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Work group trends
    if workgroup_stats['steps']:
        ax2.plot(workgroup_stats['steps'], workgroup_stats['mean'], 'o-', label='Mean', linewidth=2)
        ax2.plot(workgroup_stats['steps'], workgroup_stats['median'], 's-', label='Median', linewidth=2)
        ax2.fill_between(workgroup_stats['steps'], 
                        np.array(workgroup_stats['mean']) - np.array(workgroup_stats['std']),
                        np.array(workgroup_stats['mean']) + np.array(workgroup_stats['std']),
                        alpha=0.3, label='±1 Std Dev')
        ax2.set_title('Work Group Level Memory Access Spread\n(32 checks per work group)', fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Index Range')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plots_dir = f"{base_dir}/plots"
    plt.savefig(f"{plots_dir}/memory_access_trends_{scenario}.png", dpi=300, bbox_inches='tight')
    print(f"Saved trend analysis to {plots_dir}/memory_access_trends_{scenario}.png")
    plt.show()


def main():
    """
    Main function to run the memory access analysis.
    """
    print(f"Analyzing memory access patterns from narrow phase check indices ({SCENARIO})...")
    print("This analysis helps understand GPU memory locality:")
    print("- Smaller index ranges = better memory coalescing")
    print("- Larger index ranges = scattered memory access")
    print()
    
    # Run the main analysis
    plot_memory_access_analysis(SCENARIO)
    
    # Run comparative analysis
    create_comparative_analysis(SCENARIO)


if __name__ == "__main__":
    main() 