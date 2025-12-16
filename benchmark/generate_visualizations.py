#!/usr/bin/env python3
"""
Visualization generators for SNP benchmark analysis.

Creates publication-quality plots comparing performance across implementations,
sizes, and input patterns.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

# Set style for clean, professional plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color palette for consistent, distinguishable colors
COLORS = {
    'CpuSnp': '#2E86AB',           # Blue
    'SparseCudaSnp': '#F18F01',    # Orange
    'OptimizedCudaSnp': '#A23B72',          # Purple
    'NaiveCudaMpiSnp': '#C73E1D',  # Red
    'OptimizedCudaMpiSnp_Linear': '#6A994E', # Green
    'OptimizedCudaMpiSnp_Louvain': '#BC4B51', # Dark red
    'OptimizedCudaMpiSnp_RedBlue': '#8B5A3C'  # Brown
}

# Implementation labels for clean display
LABELS = {
    'CpuSnp': 'CPU',
    'SparseCudaSnp': 'Sparse CUDA',
    'OptimizedCudaSnp': 'Optimized CUDA',
    'NaiveCudaMpiSnp': 'Naive CUDA+MPI',
    'OptimizedCudaMpiSnp_Linear': 'Optimized CUDA+MPI (Linear)',
    'OptimizedCudaMpiSnp_Louvain': 'Optimized CUDA+MPI (Louvain)',
    'OptimizedCudaMpiSnp_RedBlue': 'Optimized CUDA+MPI (RedBlue)'
}


def throughput_comparison(df, viz_dir):
    """
    Generate throughput comparison charts separated by system size.
    
    Shows steps/second performance across implementations for different
    input patterns at each size scale.
    """
    sizes = sorted(df['Size'].unique())
    patterns = sorted(df['Pattern'].unique())
    
    for size in sizes:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        size_df = df[df['Size'] == size].copy()
        
        # Prepare data for grouped bar chart
        implementations = sorted(size_df['Implementation'].unique())
        x = np.arange(len(patterns))
        width = 0.8 / len(implementations)
        
        for i, impl in enumerate(implementations):
            impl_data = size_df[size_df['Implementation'] == impl]
            throughputs = [
                impl_data[impl_data['Pattern'] == p]['Throughput_steps/s'].values[0]
                if len(impl_data[impl_data['Pattern'] == p]) > 0 else 0
                for p in patterns
            ]
            
            offset = (i - len(implementations) / 2) * width + width / 2
            bars = ax.bar(x + offset, throughputs, width, 
                         label=LABELS.get(impl, impl),
                         color=COLORS.get(impl, None),
                         alpha=0.85)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Input Pattern', fontweight='bold')
        ax.set_ylabel('Throughput (steps/s)', fontweight='bold')
        ax.set_title(f'Throughput Comparison - Size {size}', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(patterns)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'throughput_size_{size}.png')
        plt.close()
        
        print(f"  ✓ Throughput comparison (size {size})")


def communication_overhead(df, viz_dir):
    """
    Analyze communication overhead for MPI implementations.
    
    Creates stacked bar charts showing computation vs communication time
    to identify bandwidth bottlenecks.
    """
    # Filter only MPI implementations
    mpi_impls = [impl for impl in df['Implementation'].unique() 
                 if 'Mpi' in impl or 'MPI' in impl]
    
    if not mpi_impls:
        print("  ⊘ No MPI implementations found for communication analysis")
        return
    
    mpi_df = df[df['Implementation'].isin(mpi_impls)].copy()
    
    # Create figure with subplots for each size
    sizes = sorted(mpi_df['Size'].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 6))
    if len(sizes) == 1:
        axes = [axes]
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        size_df = mpi_df[mpi_df['Size'] == size].copy()
        
        # Group by implementation and pattern
        grouped = size_df.groupby(['Implementation', 'Pattern']).agg({
            'comp_time': 'mean',
            'comm_time': 'mean'
        }).reset_index()
        
        # Create labels combining implementation and pattern
        grouped['label'] = grouped.apply(
            lambda r: f"{LABELS.get(r['Implementation'], r['Implementation'])}\n{r['Pattern']}", 
            axis=1
        )
        
        x = np.arange(len(grouped))
        
        # Stacked bars
        p1 = ax.bar(x, grouped['comp_time'], label='Computation', 
                   color='#2E86AB', alpha=0.85)
        p2 = ax.bar(x, grouped['comm_time'], bottom=grouped['comp_time'],
                   label='Communication', color='#F18F01', alpha=0.85)
        
        # Add percentage labels for communication overhead
        for i, (comp, comm) in enumerate(zip(grouped['comp_time'], grouped['comm_time'])):
            total = comp + comm
            if total > 0:
                pct = (comm / total) * 100
                ax.text(i, total, f'{pct:.1f}%', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Implementation & Pattern', fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title(f'Communication Overhead - Size {size}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['label'], rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'communication_overhead.png')
    plt.close()
    
    print(f"  ✓ Communication overhead analysis")


def scaling_stability(df, viz_dir):
    """
    Analyze how performance scales from small to large workloads.
    
    Shows throughput retention as system size increases, identifying
    which implementations scale most efficiently.
    """
    sizes = sorted(df['Size'].unique())
    
    if len(sizes) < 2:
        print("  ⊘ Need at least 2 sizes for scaling analysis")
        return
    
    # Use first and last size for comparison
    small_size, large_size = sizes[0], sizes[-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate average throughput across all patterns for each implementation
    implementations = sorted(df['Implementation'].unique())
    
    small_throughputs = []
    large_throughputs = []
    impl_labels = []
    
    for impl in implementations:
        small_data = df[(df['Implementation'] == impl) & (df['Size'] == small_size)]
        large_data = df[(df['Implementation'] == impl) & (df['Size'] == large_size)]
        
        if len(small_data) > 0 and len(large_data) > 0:
            small_avg = small_data['Throughput_steps/s'].mean()
            large_avg = large_data['Throughput_steps/s'].mean()
            
            small_throughputs.append(small_avg)
            large_throughputs.append(large_avg)
            impl_labels.append(LABELS.get(impl, impl))
    
    # Plot 1: Paired bar chart
    x = np.arange(len(impl_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, small_throughputs, width, 
                   label=f'Size {small_size}', color='#2E86AB', alpha=0.85)
    bars2 = ax1.bar(x + width/2, large_throughputs, width,
                   label=f'Size {large_size}', color='#A23B72', alpha=0.85)
    
    ax1.set_xlabel('Implementation', fontweight='bold')
    ax1.set_ylabel('Throughput (steps/s)', fontweight='bold')
    ax1.set_title('Throughput: Small vs Large Systems', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(impl_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Scaling efficiency (percentage of small system performance retained)
    scaling_efficiency = [
        (large / small * 100) if small > 0 else 0
        for small, large in zip(small_throughputs, large_throughputs)
    ]
    
    colors = [COLORS.get(impl, '#888888') for impl in implementations if impl in [
        k for k, v in LABELS.items() if v in impl_labels
    ]]
    
    bars = ax2.barh(impl_labels, scaling_efficiency, color=colors, alpha=0.85)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.0f}%',
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel('Scaling Efficiency (%)', fontweight='bold')
    ax2.set_ylabel('Implementation', fontweight='bold')
    ax2.set_title(f'Performance Retention ({small_size} → {large_size})', fontweight='bold')
    ax2.axvline(x=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='100% retention')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'scaling_stability.png')
    plt.close()
    
    print(f"  ✓ Scaling stability analysis")


def pattern_sensitivity(df, viz_dir):
    """
    Examine how input patterns affect performance.
    
    Heatmap showing relative performance across implementations and patterns,
    revealing data-dependent bottlenecks.
    """
    patterns = sorted(df['Pattern'].unique())
    implementations = sorted(df['Implementation'].unique())
    sizes = sorted(df['Size'].unique())
    
    for size in sizes:
        size_df = df[df['Size'] == size].copy()
        
        # Create matrix: rows = implementations, cols = patterns
        matrix = []
        row_labels = []
        
        for impl in implementations:
            impl_data = size_df[size_df['Implementation'] == impl]
            if len(impl_data) == 0:
                continue
                
            row = []
            for pattern in patterns:
                pattern_data = impl_data[impl_data['Pattern'] == pattern]
                if len(pattern_data) > 0:
                    row.append(pattern_data['Throughput_steps/s'].values[0])
                else:
                    row.append(0)
            
            matrix.append(row)
            row_labels.append(LABELS.get(impl, impl))
        
        matrix = np.array(matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(patterns)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(patterns)
        ax.set_yticklabels(row_labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(patterns)):
                text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(f'Pattern Sensitivity - Size {size}\n(Throughput: steps/s)', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Input Pattern', fontweight='bold')
        ax.set_ylabel('Implementation', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Throughput (steps/s)', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'pattern_sensitivity_size_{size}.png')
        plt.close()
        
        print(f"  ✓ Pattern sensitivity heatmap (size {size})")


def speedup_analysis(df, viz_dir):
    """
    Additional visualization: Speedup relative to CPU baseline.
    
    Shows how much faster each implementation is compared to CPU,
    providing context for acceleration gains.
    """
    # Get CPU baseline
    cpu_data = df[df['Implementation'] == 'CpuSnp']
    
    if len(cpu_data) == 0:
        print("  ⊘ No CPU baseline found for speedup analysis")
        return
    
    sizes = sorted(df['Size'].unique())
    patterns = sorted(df['Pattern'].unique())
    
    fig, axes = plt.subplots(1, len(sizes), figsize=(7 * len(sizes), 6))
    if len(sizes) == 1:
        axes = [axes]
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        
        # Get CPU baseline for this size
        cpu_size = cpu_data[cpu_data['Size'] == size]
        cpu_baseline = cpu_size['Throughput_steps/s'].mean()
        
        if cpu_baseline == 0:
            continue
        
        size_df = df[df['Size'] == size].copy()
        
        # Calculate speedups
        implementations = sorted([impl for impl in size_df['Implementation'].unique() 
                                 if impl != 'CpuSnp'])
        
        # Group by implementation and pattern
        x = np.arange(len(patterns))
        width = 0.8 / len(implementations)
        
        for i, impl in enumerate(implementations):
            impl_data = size_df[size_df['Implementation'] == impl]
            speedups = []
            
            for pattern in patterns:
                pattern_data = impl_data[impl_data['Pattern'] == pattern]
                if len(pattern_data) > 0:
                    throughput = pattern_data['Throughput_steps/s'].values[0]
                    # Get CPU baseline for this specific pattern
                    cpu_pattern = cpu_size[cpu_size['Pattern'] == pattern]
                    cpu_tput = cpu_pattern['Throughput_steps/s'].values[0] if len(cpu_pattern) > 0 else cpu_baseline
                    speedup = throughput / cpu_tput if cpu_tput > 0 else 0
                    speedups.append(speedup)
                else:
                    speedups.append(0)
            
            offset = (i - len(implementations) / 2) * width + width / 2
            bars = ax.bar(x + offset, speedups, width,
                         label=LABELS.get(impl, impl),
                         color=COLORS.get(impl, None),
                         alpha=0.85)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}×',
                           ha='center', va='bottom', fontsize=8)
        
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='CPU baseline')
        ax.set_xlabel('Input Pattern', fontweight='bold')
        ax.set_ylabel('Speedup vs CPU', fontweight='bold')
        ax.set_title(f'Speedup Analysis - Size {size}', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(patterns)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'speedup_analysis.png')
    plt.close()
    
    print(f"  ✓ Speedup analysis")


def generate_all_visualizations(df, viz_dir):
    """
    Generate all visualization plots.
    
    Args:
        df: Pandas DataFrame with benchmark results
        viz_dir: Path to output directory for plots
    """
    print("\nGenerating visualizations...")
    
    # Ensure output directory exists
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate each visualization
    throughput_comparison(df, viz_dir)
    communication_overhead(df, viz_dir)
    scaling_stability(df, viz_dir)
    pattern_sensitivity(df, viz_dir)
    speedup_analysis(df, viz_dir)
    
    print(f"\n✓ All visualizations saved to: {viz_dir}")


def parse_reframe_report(report_path):
    """
    Parse ReFrame report.json and associated output files to extract metrics.
    """
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    data = []
    
    for run in report['runs']:
        for testcase in run['testcases']:
            # Extract parameters
            sim_type = testcase['sim_type']
            size = testcase['input_size']
            partitioner = testcase['partitioner']
            output_dir = Path(testcase['outputdir'])
            
            # Map to Implementation name
            if sim_type == 'cpu':
                impl = 'CpuSnp'
            elif sim_type == 'sparse-cuda':
                impl = 'SparseCudaSnp'
            elif sim_type == 'optimized-cuda':
                impl = 'OptimizedCudaSnp'
            elif sim_type == 'naive-cuda-mpi':
                impl = 'NaiveCudaMpiSnp'
            elif sim_type == 'optimized-cuda-mpi':
                if partitioner == 'linear':
                    impl = 'OptimizedCudaMpiSnp_Linear'
                elif partitioner == 'louvain':
                    impl = 'OptimizedCudaMpiSnp_Louvain'
                elif partitioner == 'redblue':
                    impl = 'OptimizedCudaMpiSnp_RedBlue'
                else:
                    impl = f'OptimizedCudaMpiSnp_{partitioner.capitalize()}'
            else:
                impl = sim_type
            
            # Parse output file for metrics
            log_file = output_dir / 'rfm_job.out'
            metrics = {}
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        
                        # Extract metrics using regex
                        loop_time = re.search(r'\[BenchMetric\] LoopTime: (\S+)', content)
                        throughput = re.search(r'\[BenchMetric\] Throughput: (\S+)', content)
                        compute_time = re.search(r'\[BenchMetric\] ComputeTime: (\S+)', content)
                        comm_time = re.search(r'\[BenchMetric\] MPI_CommTime: (\S+)', content)
                        
                        if loop_time: metrics['LoopTime'] = float(loop_time.group(1))
                        if throughput: metrics['Throughput_steps/s'] = float(throughput.group(1))
                        if compute_time: metrics['comp_time'] = float(compute_time.group(1))
                        if comm_time: metrics['comm_time'] = float(comm_time.group(1))
                        else: metrics['comm_time'] = 0.0
                except Exception as e:
                    print(f"Warning: Failed to read {log_file}: {e}")
            
            if 'Throughput_steps/s' in metrics:
                entry = {
                    'Size': int(size),
                    'Pattern': 'ReverseSorted',
                    'Implementation': impl,
                    'Throughput_steps/s': metrics['Throughput_steps/s'] * 1000, # Convert steps/ms to steps/s
                    'comp_time': metrics.get('comp_time', 0),
                    'comm_time': metrics.get('comm_time', 0)
                }
                data.append(entry)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SNP benchmark visualizations')
    parser.add_argument('--report', type=str, default='report.json', help='Path to ReFrame report.json')
    parser.add_argument('--output', type=str, default='visualizations', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not Path(args.report).exists():
        print(f"Error: Report file {args.report} not found.")
        exit(1)
        
    df = parse_reframe_report(args.report)
    
    if df.empty:
        print("No data found in report.")
    else:
        print(f"Loaded {len(df)} benchmark results.")
        generate_all_visualizations(df, args.output)
