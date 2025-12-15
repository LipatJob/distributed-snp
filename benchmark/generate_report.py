#!/usr/bin/env python3
"""
Generate benchmark visualization and reports from JSON output.

Usage:
    python generate_report.py <path_to_benchmark.json>
    
Creates:
    - benchmark/results/{timestamp}/data.json (raw data)
    - benchmark/results/{timestamp}/table.txt (tabular comparison)
    - benchmark/results/{timestamp}/viz/*.png (visualization plots)
"""

import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Configuration: Add new implementations here
IMPLEMENTATIONS = {
    'CpuSnp': {'color': '#1f77b4', 'has_mpi': False, 'label': 'CPU'},
    'CudaSnp': {'color': '#ff7f0e', 'has_mpi': False, 'label': 'CUDA'},
    'SparseCudaSnp': {'color': '#2ca02c', 'has_mpi': False, 'label': 'Sparse CUDA'},
    'NaiveCudaMpiSnp': {'color': '#d62728', 'has_mpi': True, 'label': 'Naive CUDA+MPI'},
    'CudaMpiSnp_Linear': {'color': '#9467bd', 'has_mpi': True, 'label': 'CUDA+MPI (Linear)'},
    'CudaMpiSnp_Louvain': {'color': '#8c564b', 'has_mpi': True, 'label': 'CUDA+MPI (Louvain)'},
    'CudaMpiSnp_RedBlue': {'color': '#e377c2', 'has_mpi': True, 'label': 'CUDA+MPI (RedBlue)'},
}


def load_benchmark_data(json_path):
    """Load and parse benchmark JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    benchmarks = data['benchmarks']
    df = pd.DataFrame(benchmarks)
    
    # Parse benchmark names to extract components
    name_parts = df['name'].str.split('/', expand=True)
    df['Implementation'] = name_parts[0]
    df['Pattern'] = name_parts[1]
    df['Size1'] = name_parts[2]
    df['Size2'] = name_parts[3]
    
    # Convert Size counter to integer if it exists
    if 'Size' in df.columns:
        df['Size'] = df['Size'].astype(int)
    else:
        # Fallback to Size1
        df['Size'] = df['Size1'].astype(int)
    
    # Add computation time (cpu_time - communication for MPI implementations)
    df['comp_time'] = df['cpu_time']
    df['comm_time'] = df.get('MPI_Comm_ms', 0)
    
    # For MPI implementations, separate computation from communication
    mask = df['comm_time'] > 0
    df.loc[mask, 'comp_time'] = df.loc[mask, 'cpu_time'] - df.loc[mask, 'comm_time']
    
    return df, data['context']


def generate_table(df, context):
    """Generate tabular text comparison."""
    lines = []
    lines.append("=" * 100)
    lines.append("BENCHMARK REPORT")
    lines.append("=" * 100)
    lines.append(f"Date: {context['date']}")
    lines.append(f"Host: {context['host_name']}")
    lines.append(f"CPUs: {context['num_cpus']}")
    lines.append(f"Total Benchmarks: {len(df)}")
    lines.append("=" * 100)
    lines.append("")
    
    # Summary statistics by implementation
    lines.append("SUMMARY STATISTICS (All Tests)")
    lines.append("-" * 100)
    
    summary_data = []
    available_implementations = [impl for impl in IMPLEMENTATIONS.keys() 
                                if impl in df['Implementation'].values]
    
    for impl in available_implementations:
        impl_df = df[df['Implementation'] == impl]
        config = IMPLEMENTATIONS[impl]
        
        summary_data.append({
            'Implementation': config['label'],
            'Tests': len(impl_df),
            'Avg Real Time (ms)': f"{impl_df['real_time'].mean():.2f}",
            'Avg CPU Time (ms)': f"{impl_df['cpu_time'].mean():.2f}",
            'Avg Compute (ms)': f"{impl_df['comp_time'].mean():.2f}",
            'Avg Comm (ms)': f"{impl_df['comm_time'].mean():.2f}" if config['has_mpi'] else "N/A",
            'Comm Overhead (%)': f"{(impl_df['comm_time'].mean() / impl_df['cpu_time'].mean() * 100):.2f}" if config['has_mpi'] and impl_df['cpu_time'].mean() > 0 else "N/A",
        })
    
    summary_df = pd.DataFrame(summary_data)
    lines.append(summary_df.to_string(index=False))
    lines.append("")
    lines.append("")
    
    # Detailed results by pattern and size
    patterns = sorted(df['Pattern'].unique())
    sizes = sorted(df['Size'].unique())
    
    for pattern in patterns:
        lines.append(f"PATTERN: {pattern}")
        lines.append("-" * 100)
        
        pattern_df = df[df['Pattern'] == pattern]
        
        for size in sizes:
            size_df = pattern_df[pattern_df['Size'] == size]
            if len(size_df) == 0:
                continue
            
            lines.append(f"\nSize: {size}")
            lines.append("")
            
            detail_data = []
            for impl in available_implementations:
                impl_data = size_df[size_df['Implementation'] == impl]
                if len(impl_data) == 0:
                    continue
                
                row = impl_data.iloc[0]
                config = IMPLEMENTATIONS[impl]
                
                detail_row = {
                    'Implementation': config['label'],
                    'Real Time (ms)': f"{row['real_time']:.2f}",
                    'CPU Time (ms)': f"{row['cpu_time']:.2f}",
                    'Compute (ms)': f"{row['comp_time']:.2f}",
                }
                
                if config['has_mpi']:
                    detail_row['Comm (ms)'] = f"{row['comm_time']:.2f}"
                    detail_row['Comm %'] = f"{(row['comm_time'] / row['cpu_time'] * 100):.2f}" if row['cpu_time'] > 0 else "N/A"
                else:
                    detail_row['Comm (ms)'] = "N/A"
                    detail_row['Comm %'] = "N/A"
                
                if 'Steps' in row and pd.notna(row['Steps']):
                    detail_row['Steps'] = f"{int(row['Steps'])}"
                    
                if 'Throughput_steps/s' in row and pd.notna(row['Throughput_steps/s']):
                    detail_row['Throughput (steps/s)'] = f"{row['Throughput_steps/s']:.2f}"
                
                detail_data.append(detail_row)
            
            detail_df = pd.DataFrame(detail_data)
            lines.append(detail_df.to_string(index=False))
            lines.append("")
        
        lines.append("")
    
    lines.append("=" * 100)
    return "\n".join(lines)


def create_time_comparison_plot(df, output_path):
    """Create time comparison plots."""
    available_implementations = [impl for impl in IMPLEMENTATIONS.keys() 
                                if impl in df['Implementation'].values]
    
    patterns = sorted(df['Pattern'].unique())
    fig, axes = plt.subplots(1, len(patterns), figsize=(6*len(patterns), 6))
    if len(patterns) == 1:
        axes = [axes]
    
    for idx, pattern in enumerate(patterns):
        ax = axes[idx]
        pattern_df = df[df['Pattern'] == pattern]
        
        x_pos = np.arange(len(pattern_df['Size'].unique()))
        width = 0.8 / len(available_implementations)
        
        for i, impl in enumerate(available_implementations):
            impl_df = pattern_df[pattern_df['Implementation'] == impl].sort_values('Size')
            
            if len(impl_df) == 0:
                continue
            
            config = IMPLEMENTATIONS[impl]
            
            if config['has_mpi']:
                # Stack computation and communication time
                comp = impl_df['comp_time'].values
                comm = impl_df['comm_time'].values
                
                ax.bar(x_pos + i*width, comm, width,
                      label=f"{config['label']} (Comm)",
                      color=config['color'], alpha=0.4, hatch='//')
                ax.bar(x_pos + i*width, comp, width, bottom=comm, 
                      label=f"{config['label']} (Comp)",
                      color=config['color'], alpha=0.8)
            else:
                times = impl_df['real_time'].values
                ax.bar(x_pos + i*width, times, width,
                      label=config['label'],
                      color=config['color'], alpha=0.8)
        
        ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'{pattern} Data', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + width * (len(available_implementations) - 1) / 2)
        ax.set_xticklabels(sorted(pattern_df['Size'].unique()))
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_speedup_plot(df, baseline, output_path):
    """Create speedup comparison plot."""
    available_implementations = [impl for impl in IMPLEMENTATIONS.keys() 
                                if impl in df['Implementation'].values]
    
    if baseline not in available_implementations:
        print(f"Warning: Baseline '{baseline}' not found, skipping speedup plot")
        return
    
    if len(available_implementations) < 2:
        print("Warning: Need at least 2 implementations for speedup plot")
        return
    
    patterns = sorted(df['Pattern'].unique())
    fig, axes = plt.subplots(1, len(patterns), figsize=(6*len(patterns), 6))
    if len(patterns) == 1:
        axes = [axes]
    
    for idx, pattern in enumerate(patterns):
        ax = axes[idx]
        pattern_df = df[df['Pattern'] == pattern]
        
        baseline_df = pattern_df[pattern_df['Implementation'] == baseline][['Size', 'real_time']]
        baseline_df = baseline_df.rename(columns={'real_time': 'baseline_time'})
        
        for impl in available_implementations:
            if impl == baseline:
                continue
            
            impl_df = pattern_df[pattern_df['Implementation'] == impl][['Size', 'real_time']]
            if len(impl_df) == 0:
                continue
                
            merged = impl_df.merge(baseline_df, on='Size')
            merged['speedup'] = merged['baseline_time'] / merged['real_time']
            
            config = IMPLEMENTATIONS[impl]
            ax.plot(merged['Size'], merged['speedup'], 
                   marker='o', linewidth=2, markersize=8,
                   label=config['label'], color=config['color'])
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Speedup vs {IMPLEMENTATIONS[baseline]["label"]}', fontsize=12, fontweight='bold')
        ax.set_title(f'{pattern} Data', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_communication_overhead_plot(df, output_path):
    """Create communication overhead plot."""
    mpi_impls = [impl for impl in IMPLEMENTATIONS.keys() 
                 if IMPLEMENTATIONS[impl]['has_mpi'] and impl in df['Implementation'].values]
    
    if not mpi_impls:
        print("Warning: No MPI implementations found, skipping communication overhead plot")
        return
    
    plot_df = df[df['Implementation'].isin(mpi_impls)].copy()
    plot_df['comm_percentage'] = (plot_df['comm_time'] / plot_df['cpu_time']) * 100
    
    patterns = sorted(plot_df['Pattern'].unique())
    fig, axes = plt.subplots(1, len(patterns), figsize=(6*len(patterns), 6))
    if len(patterns) == 1:
        axes = [axes]
    
    for idx, pattern in enumerate(patterns):
        ax = axes[idx]
        pattern_df = plot_df[plot_df['Pattern'] == pattern]
        
        for impl in mpi_impls:
            impl_df = pattern_df[pattern_df['Implementation'] == impl].sort_values('Size')
            if len(impl_df) == 0:
                continue
                
            config = IMPLEMENTATIONS[impl]
            
            ax.plot(impl_df['Size'], impl_df['comm_percentage'],
                   marker='o', linewidth=2, markersize=8,
                   label=config['label'], color=config['color'])
        
        ax.set_xlabel('Problem Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Communication Overhead (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{pattern} Data', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <benchmark_json_file>")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File {json_path} not found")
        sys.exit(1)
    
    # Extract timestamp from filename or use current time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "benchmark_" in json_path.name:
        # Try to extract timestamp from filename
        try:
            timestamp = json_path.stem.split("_", 1)[1]
        except:
            pass
    
    # Create output directory structure
    results_dir = json_path.parent / timestamp
    viz_dir = results_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating report for: {json_path}")
    print(f"Output directory: {results_dir}")
    
    # Load data
    print("Loading benchmark data...")
    df, context = load_benchmark_data(json_path)
    print(f"  Found {len(df)} benchmark results")
    print(f"  Implementations: {sorted(df['Implementation'].unique())}")
    
    # Copy JSON to new location
    import shutil
    data_json_path = results_dir / "data.json"
    shutil.copy(json_path, data_json_path)
    print(f"✓ Saved raw data: {data_json_path}")
    
    # Generate table
    print("Generating tabular report...")
    table_text = generate_table(df, context)
    table_path = results_dir / "table.txt"
    with open(table_path, 'w') as f:
        f.write(table_text)
    print(f"✓ Saved table: {table_path}")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    print("  - Time comparison plot...")
    create_time_comparison_plot(df, viz_dir / "time_comparison.png")
    
    print("  - Speedup analysis plot...")
    # Use first available implementation as baseline, prefer CpuSnp
    available = [impl for impl in IMPLEMENTATIONS.keys() if impl in df['Implementation'].values]
    baseline = 'CpuSnp' if 'CpuSnp' in available else available[0] if available else None
    if baseline:
        create_speedup_plot(df, baseline, viz_dir / "speedup_analysis.png")
    
    print("  - Communication overhead plot...")
    create_communication_overhead_plot(df, viz_dir / "communication_overhead.png")
    
    print(f"✓ Saved visualizations: {viz_dir}")
    
    print("")
    print("=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Results: {results_dir}")
    print(f"  - data.json: Raw benchmark data")
    print(f"  - table.txt: Tabular comparison")
    print(f"  - viz/: Visualization plots")
    print("=" * 80)
    
    # Print table preview
    print("\nTABLE PREVIEW:")
    print("-" * 80)
    lines = table_text.split('\n')
    for line in lines[:30]:  # Show first 30 lines
        print(line)
    if len(lines) > 30:
        print(f"\n... ({len(lines) - 30} more lines) ...")
    print("")


if __name__ == "__main__":
    main()
