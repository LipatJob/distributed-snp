#!/usr/bin/env python3
"""
Improved visualization generators for SNP benchmark analysis.
Generates both absolute and normalized breakdowns for deeper insights.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# --- Configuration & Style ---
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4) # Slightly larger font for readability
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# logical sort order for consistency across all plots
IMPL_ORDER = [
    'CpuSnp',
    'SparseCudaSnp', 
    'OptimizedCudaSnp',
    'NaiveCudaMpiSnp', 
    'OptimizedCudaMpiSnp_Linear', 
    'OptimizedCudaMpiSnp_Louvain', 
    'OptimizedCudaMpiSnp_RedBlue'
]

# Consistent Color Palette
IMPL_COLORS = {
    'CpuSnp': '#2E86AB',            # Blue
    'SparseCudaSnp': '#F18F01',     # Orange
    'OptimizedCudaSnp': '#A23B72',  # Purple
    'NaiveCudaMpiSnp': '#C73E1D',   # Red
    'OptimizedCudaMpiSnp_Linear': '#6A994E', # Green
    'OptimizedCudaMpiSnp_Louvain': '#BC4B51', # Dark Red
    'OptimizedCudaMpiSnp_RedBlue': '#8B5A3C'  # Brown
}

# Colors for Stacked Components
COMPONENT_COLORS = {
    'Compute': '#1f77b4',       # Muted Blue
    'MPI Comm': '#ff7f0e',      # Safety Orange
    'CUDA Mem': '#2ca02c'       # Cooked Asparagus Green
}

LABELS = {
    'CpuSnp': 'CPU Baseline',
    'SparseCudaSnp': 'Sparse CUDA',
    'OptimizedCudaSnp': 'Optimized CUDA',
    'NaiveCudaMpiSnp': 'Naive MPI+CUDA',
    'OptimizedCudaMpiSnp_Linear': 'Opt MPI (Linear)',
    'OptimizedCudaMpiSnp_Louvain': 'Opt MPI (Louvain)',
    'OptimizedCudaMpiSnp_RedBlue': 'Opt MPI (RedBlue)'
}

# Create a global palette mapping Labels -> Colors
GLOBAL_PALETTE = {LABELS.get(k, k): v for k, v in IMPL_COLORS.items()}

def load_benchmark_data(json_path):
    """Load, parse, and clean benchmark JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    benchmarks = data['benchmarks']
    df = pd.DataFrame(benchmarks)
    
    # Parse name: Implementation/Pattern/Size1/...
    name_parts = df['name'].str.split('/', expand=True)
    df['Implementation'] = name_parts[0]
    df['Pattern'] = name_parts[1]
    
    if len(name_parts.columns) > 2:
        df['Size1'] = name_parts[2]
        try:
            df['Size'] = df['Size1'].astype(int)
        except ValueError:
            pass
    
    # Ensure specific counters exist; fill missing with 0
    # Note: 'Compute_ms' usually tracks Kernel time in your specific benchmark output
    required_cols = ['Compute_ms', 'MPI_Comm_ms', 'CUDA_Memory_ms', 'Throughput_steps/s', 'cpu_time']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    
    # Enforce logical sorting
    df['Implementation'] = pd.Categorical(df['Implementation'], categories=IMPL_ORDER, ordered=True)
    df = df.sort_values(['Size', 'Implementation'])
    
    return df

def throughput_vs_size_lines(df, viz_dir):
    """Line plot of Throughput vs Size (Log-Log)."""
    patterns = sorted(df['Pattern'].unique())
    
    for pattern in patterns:
        pattern_df = df[df['Pattern'] == pattern].copy()
        if len(pattern_df['Size'].unique()) < 2:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use simple names for legend
        pattern_df['Label'] = pattern_df['Implementation'].map(lambda x: LABELS.get(x, x))
        
        sns.lineplot(data=pattern_df, x='Size', y='Throughput_steps/s', hue='Label',
                     style='Label', markers=True, dashes=False, 
                     palette=GLOBAL_PALETTE,
                     linewidth=2.5, markersize=8, ax=ax)
        
        ax.set_xlabel('System Size (Neurons)', fontweight='bold')
        ax.set_ylabel('Throughput (Steps/s)', fontweight='bold')
        ax.set_title(f'Throughput Scaling - {pattern}', fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend(title='Implementation', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, which="major", ls="-", alpha=0.5)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'throughput_vs_size_{pattern}.png')
        plt.close()
        print(f"  ✓ Throughput vs Size ({pattern})")

def speedup_vs_size_bars(df, viz_dir):
    """Grouped Bar plot of Speedup vs CPU Baseline."""
    cpu_data = df[df['Implementation'] == 'CpuSnp']
    if len(cpu_data) == 0:
        print("    ! Skipping speedup plots (No CPU baseline found)")
        return

    patterns = sorted(df['Pattern'].unique())
    implementations = [impl for impl in IMPL_ORDER if impl != 'CpuSnp' and impl in df['Implementation'].unique()]
    
    for pattern in patterns:
        cpu_pattern = cpu_data[cpu_data['Pattern'] == pattern]
        plot_data = []
        pattern_df = df[df['Pattern'] == pattern]
        sizes = sorted(pattern_df['Size'].unique())
        
        for size in sizes:
            cpu_subset = cpu_pattern[cpu_pattern['Size'] == size]
            if len(cpu_subset) == 0: continue
            
            cpu_tput = cpu_subset['Throughput_steps/s'].mean()
            if cpu_tput <= 0: continue
                
            for impl in implementations:
                impl_subset = df[(df['Implementation'] == impl) & 
                                 (df['Pattern'] == pattern) & 
                                 (df['Size'] == size)]
                if len(impl_subset) > 0:
                    speedup = impl_subset['Throughput_steps/s'].mean() / cpu_tput
                    plot_data.append({
                        'Size': str(size),
                        'Label': LABELS.get(impl, impl),
                        'Speedup': speedup,
                        'Original_Impl': impl
                    })
        
        if not plot_data: continue
            
        plot_df = pd.DataFrame(plot_data)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.barplot(data=plot_df, x='Size', y='Speedup', hue='Label',
                    palette=GLOBAL_PALETTE, ax=ax, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('System Size', fontweight='bold')
        ax.set_ylabel('Speedup vs CPU (Log Scale)', fontweight='bold')
        ax.set_title(f'Speedup vs Baseline - {pattern}', fontweight='bold')
        ax.set_yscale('log')
        
        # Reference line at 1x
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='CPU Baseline (1x)')
        
        ax.legend(title='Implementation', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, axis='y', which="both", ls="-", alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'speedup_vs_size_{pattern}.png')
        plt.close()
        print(f"  ✓ Speedup Bars ({pattern})")

def communication_percentage_lines(df, viz_dir):
    """
    Line plot of Communication Overhead % vs Size.
    Generates two views: Linear (Zoomed) and Log scale.
    """
    # Filter for implementations containing "Mpi"
    mpi_impls = [impl for impl in IMPL_ORDER if 'Mpi' in impl and impl in df['Implementation'].unique()]
    if not mpi_impls: return

    patterns = sorted(df['Pattern'].unique())
    
    for pattern in patterns:
        plot_data = []
        for impl in mpi_impls:
            impl_data = df[(df['Implementation'] == impl) & (df['Pattern'] == pattern)].sort_values('Size')
            if len(impl_data) == 0: continue
            
            for _, row in impl_data.iterrows():
                total = row['cpu_time']
                comm = row['MPI_Comm_ms']
                if total > 0:
                    plot_data.append({
                        'Size': row['Size'],
                        'Percentage': (comm / total) * 100,
                        'Label': LABELS.get(impl, impl),
                        'Original_Impl': impl
                    })
        
        if not plot_data: continue
            
        plot_df = pd.DataFrame(plot_data)

        # --- VIEW 1: LINEAR (Zoomed in) ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(data=plot_df, x='Size', y='Percentage', hue='Label',
                     style='Label', markers=True, dashes=False,
                     palette=GLOBAL_PALETTE, linewidth=2.5, markersize=9, ax=ax)
        
        ax.set_xlabel('System Size (Neurons)', fontweight='bold')
        ax.set_ylabel('Comm. Overhead (%)', fontweight='bold')
        ax.set_title(f'Communication Overhead (Linear) - {pattern}', fontweight='bold')
        ax.set_xscale('log', base=2)
        
        # FIX: Remove hard 0-100 limit. Auto-scale + distinct margin
        y_max = plot_df['Percentage'].max()
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.2) 
        
        ax.legend(title='Implementation', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'comm_overhead_linear_{pattern}.png')
        plt.close()

        # --- VIEW 2: LOG SCALE (For order of magnitude gaps) ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(data=plot_df, x='Size', y='Percentage', hue='Label',
                     style='Label', markers=True, dashes=False,
                     palette=GLOBAL_PALETTE, linewidth=2.5, markersize=9, ax=ax)
        
        ax.set_xlabel('System Size (Neurons)', fontweight='bold')
        ax.set_ylabel('Comm. Overhead (% - Log Scale)', fontweight='bold')
        ax.set_title(f'Communication Overhead (Log) - {pattern}', fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log') # Log Y-axis
        
        # Proper grid for log plots
        ax.grid(True, which="major", ls="-", alpha=0.5)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        
        ax.legend(title='Implementation', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'comm_overhead_log_{pattern}.png')
        plt.close()
        print(f"  ✓ Comm Overhead ({pattern})")

def performance_profile_plot(df, viz_dir):
    """Dolan-More Performance Profile."""
    problems = [(s, p) for s in df['Size'].unique() for p in df['Pattern'].unique()]
    implementations = [i for i in IMPL_ORDER if i in df['Implementation'].unique()]
    
    ratios = {impl: [] for impl in implementations}
    
    for size, pattern in problems:
        subset = df[(df['Size'] == size) & (df['Pattern'] == pattern)]
        if len(subset) == 0: continue
            
        max_tput = subset['Throughput_steps/s'].max()
        if max_tput <= 0: continue
            
        for impl in implementations:
            val = subset[subset['Implementation'] == impl]['Throughput_steps/s']
            if len(val) > 0 and val.iloc[0] > 0:
                ratios[impl].append(max_tput / val.iloc[0])
            else:
                ratios[impl].append(np.inf)
            
    plot_data = []
    taus = np.linspace(1, 10, 200)
    
    for impl in implementations:
        impl_ratios = np.array(ratios[impl])
        if len(impl_ratios) == 0: continue
        ys = [np.sum(impl_ratios <= t) / len(impl_ratios) for t in taus]
        for t, y in zip(taus, ys):
            plot_data.append({
                'Tau': t, 'Probability': y, 
                'Label': LABELS.get(impl, impl), 'Original_Impl': impl
            })
            
    plot_df = pd.DataFrame(plot_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=plot_df, x='Tau', y='Probability', hue='Label',
                 palette=GLOBAL_PALETTE, linewidth=2, ax=ax)
        
    ax.set_xlabel('Performance Ratio (τ)', fontweight='bold')
    ax.set_ylabel('Probability (Within τ of Best)', fontweight='bold')
    ax.set_title('Performance Profile (Higher is Better)', fontweight='bold')
    ax.legend(title='Implementation', loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 10)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'performance_profile.png')
    plt.close()
    print("  ✓ Performance Profile")

def compute_breakdown_stacked_bars(df, viz_dir):
    """
    Generates TWO breakdown plots per pattern:
    1. Absolute Time (log scale if necessary, usually linear per subplot)
    2. Normalized (%) Time - CRITICAL for insights on bottlenecks.
    """
    sns.set_theme(style="whitegrid")
    
    # Components to stack
    stack_cols = ['Compute_ms', 'MPI_Comm_ms', 'CUDA_Memory_ms']
    labels_map = {'Compute_ms': 'Compute', 'MPI_Comm_ms': 'MPI Comm', 'CUDA_Memory_ms': 'CUDA Mem'}
    colors = [COMPONENT_COLORS['Compute'], COMPONENT_COLORS['MPI Comm'], COMPONENT_COLORS['CUDA Mem']]
    
    patterns = sorted(df['Pattern'].unique())
    
    for pattern in patterns:
        pattern_df = df[df['Pattern'] == pattern].copy()
        sizes = sorted(pattern_df['Size'].unique())
        
        if not sizes: continue
        
        # --- PLOT 1: ABSOLUTE TIME (Subplots by Size) ---
        fig, axes = plt.subplots(len(sizes), 1, figsize=(10, 5 * len(sizes)), sharey=False)
        if len(sizes) == 1: axes = [axes]
        else: axes = axes.flatten()
            
        for i, size in enumerate(sizes):
            ax = axes[i]
            size_df = pattern_df[pattern_df['Size'] == size].copy()
            # Sort explicitly
            size_df['ShortName'] = size_df['Implementation'].map(lambda x: LABELS.get(x, x))
            
            # Plot
            plot_df = size_df.set_index('ShortName')[stack_cols]
            plot_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.75, edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'Absolute Time Breakdown - Size {size}', fontweight='bold')
            ax.set_ylabel('Time (ms)')
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            ax.grid(axis='y', alpha=0.5, linestyle='--')
            
            # Clean legend
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [labels_map.get(l, l) for l in labels]
            ax.legend(handles, new_labels, title="Component")

        plt.suptitle(f'Execution Time Breakdown - {pattern}', fontweight='bold', y=1.005)
        plt.tight_layout()
        plt.savefig(viz_dir / f'breakdown_absolute_{pattern}.png', bbox_inches='tight')
        plt.close()

        # --- PLOT 2: NORMALIZED (%) TIME (Subplots by Size) ---
        # This is where the real insights come from for differing magnitudes
        fig, axes = plt.subplots(len(sizes), 1, figsize=(10, 5 * len(sizes)), sharey=False)
        if len(sizes) == 1: axes = [axes]
        else: axes = axes.flatten()

        for i, size in enumerate(sizes):
            ax = axes[i]
            size_df = pattern_df[pattern_df['Size'] == size].copy()
            size_df['ShortName'] = size_df['Implementation'].map(lambda x: LABELS.get(x, x))
            
            # Normalize row-wise to 100%
            data_subset = size_df.set_index('ShortName')[stack_cols]
            data_norm = data_subset.div(data_subset.sum(axis=1), axis=0) * 100
            
            data_norm.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.75, edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'Relative Bottleneck Analysis - Size {size}', fontweight='bold')
            ax.set_ylabel('Share of Time (%)')
            ax.set_xlabel('')
            ax.set_ylim(0, 100)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
            # Add percentage labels on bars if space permits
            for c in ax.containers:
                ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', fontsize=9, padding=0)

            handles, labels = ax.get_legend_handles_labels()
            new_labels = [labels_map.get(l, l) for l in labels]
            ax.legend(handles, new_labels, title="Component", bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.suptitle(f'Normalized Bottleneck Analysis - {pattern}', fontweight='bold', y=1.005)
        plt.tight_layout()
        plt.savefig(viz_dir / f'breakdown_normalized_{pattern}.png', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Breakdowns (Absolute & Normalized) ({pattern})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_snp.py <path_to_data.json>")
        sys.exit(1)
        
    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File {json_path} not found")
        sys.exit(1)
        
    viz_dir = json_path.parent / 'viz'
    viz_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {json_path}...")
    df = load_benchmark_data(json_path)
    
    print("Generating insightful visualizations...")
    throughput_vs_size_lines(df, viz_dir)
    speedup_vs_size_bars(df, viz_dir)
    communication_percentage_lines(df, viz_dir)
    performance_profile_plot(df, viz_dir)
    compute_breakdown_stacked_bars(df, viz_dir)
    
    print(f"\n✓ Insights ready! Check folder: {viz_dir}")

if __name__ == "__main__":
    main()