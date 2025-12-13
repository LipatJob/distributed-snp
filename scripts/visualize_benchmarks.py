#!/usr/bin/env python3

"""
Visualize benchmark results with interactive plots
Usage: python3 scripts/visualize_benchmarks.py [result_files...]
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import numpy as np

# ============================================================================
# CONFIGURATION - Add new implementations here
# ============================================================================

IMPLEMENTATION_CONFIG = {
    'CpuSnp': {
        'label': 'CPU',
        'color': '#3498db',
        'category': 'baseline',
        'order': 1
    },
    'CudaSnp': {
        'label': 'CUDA',
        'color': '#9b59b6',
        'category': 'gpu',
        'order': 2
    },
    'SparseCudaSnp': {
        'label': 'Sparse CUDA',
        'color': '#8e44ad',
        'category': 'gpu',
        'order': 3
    },
    'NaiveCudaMpiSnp': {
        'label': 'Naive CUDA+MPI',
        'color': '#e74c3c',
        'category': 'distributed',
        'order': 4
    },
    'CudaMpiSnp': {
        'label': 'Optimized CUDA+MPI',
        'color': '#2ecc71',
        'category': 'distributed',
        'order': 5
    }
}

# Default baseline for speedup calculations
BASELINE_IMPL = 'CpuSnp'

# Default distribution for scalability plots
DEFAULT_DISTRIBUTION = 'Random'

# ============================================================================

# ============================================================================

def get_impl_info(impl_name: str, key: str, default=None):
    """Get configuration info for an implementation"""
    return IMPLEMENTATION_CONFIG.get(impl_name, {}).get(key, default)

def load_json_benchmark(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_benchmark_name(name: str) -> Dict[str, Any]:
    """Parse benchmark name to extract implementation, size, max_value, and distribution"""
    # Format: Implementation/Distribution/Size/MaxValue/iterations:N
    parts = name.split('/')
    if len(parts) < 4:
        return {}
    
    try:
        impl = parts[0]
        distribution = parts[1]
        size = int(parts[2])
        max_value = int(parts[3])
        
        return {
            'implementation': impl,
            'size': size,
            'max_value': max_value,
            'distribution': distribution
        }
    except (ValueError, IndexError):
        return {}

def collect_benchmark_data(results: List[Dict[str, Any]]) -> Dict[Tuple, List[Dict]]:
    """Collect and organize benchmark data"""
    grouped = defaultdict(list)
    
    for result in results:
        benchmarks = result.get('benchmarks', [])
        for bench in benchmarks:
            name = bench['name']
            parsed = parse_benchmark_name(name)
            
            if parsed:
                key = (parsed['size'], parsed['max_value'], parsed['distribution'])
                
                # Extract metrics
                data = {
                    'impl': parsed['implementation'],
                    'time': bench['real_time'],
                    'cpu_time': bench['cpu_time'],
                    'comm_time': bench.get('Comm_ms', 0),
                    'compute_time': bench.get('ComputeTime_ms', 0),
                    'num_processes': bench.get('Procs', 1)
                }
                grouped[key].append(data)
    
    return grouped

def plot_execution_time_comparison(grouped_data: Dict, output_dir: Path):
    """Create bar chart comparing execution times across implementations"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get all implementations present in the data, sorted by order
    all_impls = set()
    for benchmarks in grouped_data.values():
        for bench in benchmarks:
            all_impls.add(bench['impl'])
    
    # Sort implementations by configured order
    sorted_impls = sorted(
        [impl for impl in all_impls if impl in IMPLEMENTATION_CONFIG],
        key=lambda x: get_impl_info(x, 'order', 999)
    )
    
    # Prepare data structures
    configs = []
    impl_times = {impl: [] for impl in sorted_impls}
    
    for key in sorted(grouped_data.keys()):
        size, max_val, dist = key
        benchmarks = grouped_data[key]
        
        config_label = f"{size}\n{dist[:8]}"
        configs.append(config_label)
        
        # Collect times for each implementation
        times_for_config = {impl: None for impl in sorted_impls}
        for bench in benchmarks:
            impl = bench['impl']
            if impl in times_for_config:
                times_for_config[impl] = bench['time']
        
        # Append to lists (0 if not found)
        for impl in sorted_impls:
            impl_times[impl].append(times_for_config[impl] if times_for_config[impl] else 0)
    
    # Create grouped bars
    x = np.arange(len(configs))
    num_impls = len(sorted_impls)
    width = 0.8 / num_impls if num_impls > 0 else 0.25
    
    bars_list = []
    for i, impl in enumerate(sorted_impls):
        offset = (i - num_impls/2 + 0.5) * width
        color = get_impl_info(impl, 'color', '#95a5a6')
        label = get_impl_info(impl, 'label', impl)
        bars = ax.bar(x + offset, impl_times[impl], width, label=label, color=color, alpha=0.8)
        bars_list.append(bars)
    
    # Customize plot
    ax.set_xlabel('Configuration (Size / Distribution)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('SNP Sort Execution Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
    
    for bars in bars_list:
        add_labels(bars)
    
    plt.tight_layout()
    output_file = output_dir / 'execution_time_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_speedup_analysis(grouped_data: Dict, output_dir: Path):
    """Create speedup comparison chart"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get all non-baseline implementations
    all_impls = set()
    for benchmarks in grouped_data.values():
        for bench in benchmarks:
            if bench['impl'] != BASELINE_IMPL:
                all_impls.add(bench['impl'])
    
    sorted_impls = sorted(
        [impl for impl in all_impls if impl in IMPLEMENTATION_CONFIG],
        key=lambda x: get_impl_info(x, 'order', 999)
    )
    
    configs = []
    impl_speedups = {impl: [] for impl in sorted_impls}
    
    for key in sorted(grouped_data.keys()):
        size, max_val, dist = key
        benchmarks = grouped_data[key]
        
        config_label = f"{size}\n{dist[:8]}"
        configs.append(config_label)
        
        # Get baseline time
        baseline_time = None
        times_for_config = {}
        for bench in benchmarks:
            if bench['impl'] == BASELINE_IMPL:
                baseline_time = bench['time']
            times_for_config[bench['impl']] = bench['time']
        
        # Calculate speedups
        for impl in sorted_impls:
            if baseline_time and impl in times_for_config and baseline_time > 0 and times_for_config[impl] > 0:
                speedup = baseline_time / times_for_config[impl]
                impl_speedups[impl].append(speedup)
            else:
                impl_speedups[impl].append(0)
    
    x = np.arange(len(configs))
    num_impls = len(sorted_impls)
    width = 0.8 / num_impls if num_impls > 0 else 0.35
    
    bars_list = []
    for i, impl in enumerate(sorted_impls):
        offset = (i - num_impls/2 + 0.5) * width
        color = get_impl_info(impl, 'color', '#95a5a6')
        label = get_impl_info(impl, 'label', impl)
        bars = ax.bar(x + offset, impl_speedups[impl], width, label=label, color=color, alpha=0.8)
        bars_list.append(bars)
    
    # Add baseline at 1.0x
    baseline_label = get_impl_info(BASELINE_IMPL, 'label', BASELINE_IMPL)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, 
               label=f'Baseline ({baseline_label})')
    
    ax.set_xlabel('Configuration (Size / Distribution)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup vs CPU (higher is better)', fontsize=12, fontweight='bold')
    ax.set_title('SNP Sort Speedup Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}x',
                       ha='center', va='bottom', fontsize=8)
    
    for bars in bars_list:
        add_labels(bars)
    
    plt.tight_layout()
    output_file = output_dir / 'speedup_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_comm_vs_compute(grouped_data: Dict, output_dir: Path):
    """Create stacked bar chart showing communication vs compute time"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    configs = []
    naive_comm = []
    naive_comp = []
    opt_comm = []
    opt_comp = []
    
    for key in sorted(grouped_data.keys()):
        size, max_val, dist = key
        benchmarks = grouped_data[key]
        
        config_label = f"{size}\n{dist[:8]}"
        configs.append(config_label)
        
        naive_comm_time = 0
        naive_comp_time = 0
        opt_comm_time = 0
        opt_comp_time = 0
        
        for bench in benchmarks:
            impl = bench['impl']
            comm = bench.get('comm_time', 0)
            comp = bench.get('compute_time', 0)
            
            if impl == 'NaiveCudaMpiSnp':
                naive_comm_time = comm
                naive_comp_time = comp
            elif impl == 'CudaMpiSnp':
                opt_comm_time = comm
                opt_comp_time = comp
        
        naive_comm.append(naive_comm_time)
        naive_comp.append(naive_comp_time)
        opt_comm.append(opt_comm_time)
        opt_comp.append(opt_comp_time)
    
    x = np.arange(len(configs))
    width = 0.35
    
    # Stacked bars for Naive
    p1 = ax.bar(x - width/2, naive_comp, width, label='Naive Compute', color='#3498db', alpha=0.8)
    p2 = ax.bar(x - width/2, naive_comm, width, bottom=naive_comp, label='Naive Comm', color='#e74c3c', alpha=0.8)
    
    # Stacked bars for Optimized
    p3 = ax.bar(x + width/2, opt_comp, width, label='Optimized Compute', color='#1abc9c', alpha=0.8)
    p4 = ax.bar(x + width/2, opt_comm, width, bottom=opt_comp, label='Optimized Comm', color='#f39c12', alpha=0.8)
    
    ax.set_xlabel('Configuration (Size / Distribution)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Communication vs Computation Time Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = output_dir / 'comm_vs_compute.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_scalability(grouped_data: Dict, output_dir: Path):
    """Create scalability chart showing performance vs input size"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group by distribution and track sizes
    by_distribution = defaultdict(lambda: defaultdict(list))
    
    for key, benchmarks in grouped_data.items():
        size, max_val, dist = key
        
        for bench in benchmarks:
            impl = bench['impl']
            time = bench['time']
            by_distribution[dist][impl].append((size, time))
    
    # Plot for the configured default distribution
    dist_to_plot = DEFAULT_DISTRIBUTION
    if dist_to_plot in by_distribution:
        impl_data = by_distribution[dist_to_plot]
        
        # Sort implementations by order
        sorted_impls = sorted(
            [impl for impl in impl_data.keys() if impl in IMPLEMENTATION_CONFIG],
            key=lambda x: get_impl_info(x, 'order', 999)
        )
        
        for impl in sorted_impls:
            data_points = impl_data[impl]
            # Sort by size
            data_points.sort(key=lambda x: x[0])
            sizes = [x[0] for x in data_points]
            times = [x[1] for x in data_points]
            
            color = get_impl_info(impl, 'color', '#95a5a6')
            label = get_impl_info(impl, 'label', impl)
            
            ax.plot(sizes, times, marker='o', linewidth=2, markersize=8, 
                   color=color, label=label, alpha=0.8)
    
    ax.set_xlabel('Input Size (number of elements)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'Scalability Analysis (Distribution: {dist_to_plot})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_file = output_dir / 'scalability_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def generate_summary_report(grouped_data: Dict, output_dir: Path):
    """Generate a text summary report"""
    output_file = output_dir / 'benchmark_summary.txt'
    
    # Get all implementations present in the data
    all_impls = set()
    for benchmarks in grouped_data.values():
        for bench in benchmarks:
            all_impls.add(bench['impl'])
    
    sorted_impls = sorted(
        [impl for impl in all_impls if impl in IMPLEMENTATION_CONFIG],
        key=lambda x: get_impl_info(x, 'order', 999)
    )
    
    # Get non-baseline implementations for speedup stats
    non_baseline_impls = [impl for impl in sorted_impls if impl != BASELINE_IMPL]
    
    with open(output_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("SNP SORT BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 120 + "\n\n")
        
        # Build header
        header = f"{'Config':<30}"
        for impl in sorted_impls:
            label = get_impl_info(impl, 'label', impl)
            header += f" {label:<15}"
        for impl in non_baseline_impls:
            label = get_impl_info(impl, 'label', impl)
            header += f" {label} Spd:<15"[:15]
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 120 + "\n")
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        
        # Collect all speedups for statistics
        impl_speedups = {impl: [] for impl in non_baseline_impls}
        
        for key in sorted(grouped_data.keys()):
            size, max_val, dist = key
            benchmarks = grouped_data[key]
            
            config_name = f"{size}/{dist}"
            
            # Collect times
            times = {}
            for bench in benchmarks:
                impl = bench['impl']
                if impl in sorted_impls:
                    times[impl] = bench['time']
            
            # Build row
            row = f"{config_name:<30}"
            
            # Add times
            for impl in sorted_impls:
                time = times.get(impl)
                time_str = f"{time:.2f} ms" if time else "N/A"
                row += f" {time_str:<15}"
            
            # Add speedups
            baseline_time = times.get(BASELINE_IMPL)
            for impl in non_baseline_impls:
                impl_time = times.get(impl)
                if baseline_time and impl_time and baseline_time > 0 and impl_time > 0:
                    speedup = baseline_time / impl_time
                    impl_speedups[impl].append(speedup)
                    row += f" {speedup:.2f}x{'':<10}"
                else:
                    row += f" {'N/A':<15}"
            
            f.write(row + "\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 120 + "\n\n")
        
        baseline_label = get_impl_info(BASELINE_IMPL, 'label', BASELINE_IMPL)
        f.write(f"Speedup vs {baseline_label}:\n\n")
        
        for impl in non_baseline_impls:
            speedups = impl_speedups[impl]
            if speedups:
                label = get_impl_info(impl, 'label', impl)
                f.write(f"{label}:\n")
                f.write(f"  Average: {np.mean(speedups):.2f}x\n")
                f.write(f"  Median:  {np.median(speedups):.2f}x\n")
                f.write(f"  Min:     {np.min(speedups):.2f}x\n")
                f.write(f"  Max:     {np.max(speedups):.2f}x\n")
                f.write(f"  StdDev:  {np.std(speedups):.2f}\n\n")
        
        # Compare distributed implementations if both exist
        if 'NaiveCudaMpiSnp' in impl_speedups and 'CudaMpiSnp' in impl_speedups:
            naive_speedups = impl_speedups['NaiveCudaMpiSnp']
            opt_speedups = impl_speedups['CudaMpiSnp']
            if naive_speedups and opt_speedups:
                improvement = np.mean(opt_speedups) / np.mean(naive_speedups)
                f.write(f"Optimization Improvement:\n")
                f.write(f"  Optimized CUDA+MPI is {improvement:.2f}x faster than Naive CUDA+MPI on average\n\n")
        
        f.write("=" * 120 + "\n")
    
    print(f"Saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize SNP Sort benchmark results')
    parser.add_argument('files', nargs='+', help='JSON benchmark result files')
    parser.add_argument('-o', '--output', default='benchmark_results', 
                       help='Output directory for plots (default: benchmark_results)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load benchmark results
    print("Loading benchmark results...")
    results = []
    for filepath in args.files:
        try:
            result = load_json_benchmark(filepath)
            results.append(result)
            print(f"  ✓ Loaded: {filepath}")
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")
    
    if not results:
        print("\nNo valid benchmark results found!")
        sys.exit(1)
    
    # Collect and organize data
    print("\nProcessing benchmark data...")
    grouped_data = collect_benchmark_data(results)
    print(f"  Found {len(grouped_data)} unique configurations")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_execution_time_comparison(grouped_data, output_dir)
    plot_speedup_analysis(grouped_data, output_dir)
    plot_comm_vs_compute(grouped_data, output_dir)
    plot_scalability(grouped_data, output_dir)
    
    # Generate text summary
    print("\nGenerating summary report...")
    generate_summary_report(grouped_data, output_dir)
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - execution_time_comparison.png")
    print(f"  - speedup_analysis.png")
    print(f"  - comm_vs_compute.png")
    print(f"  - scalability_analysis.png")
    print(f"  - benchmark_summary.txt")

if __name__ == '__main__':
    main()
