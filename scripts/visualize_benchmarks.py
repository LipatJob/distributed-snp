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

def load_json_benchmark(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_benchmark_name(name: str) -> Dict[str, Any]:
    """Parse benchmark name to extract implementation, size, max_value, and distribution"""
    # Format: Implementation_Size_MaxValue_Distribution
    parts = name.split('/')
    base_name = parts[0] if parts else name
    
    components = base_name.split('_')
    
    # Find the implementation name (everything before the first number)
    impl_parts = []
    remaining_parts = []
    found_number = False
    
    for part in components:
        if not found_number and not part.isdigit():
            impl_parts.append(part)
        else:
            found_number = True
            remaining_parts.append(part)
    
    if len(remaining_parts) >= 3:
        impl = '_'.join(impl_parts)
        size = remaining_parts[0]
        max_val = remaining_parts[1]
        dist = '_'.join(remaining_parts[2:])
        
        try:
            return {
                'implementation': impl,
                'size': int(size),
                'max_value': int(max_val),
                'distribution': dist
            }
        except ValueError:
            return {}
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
                    'comm_time': bench.get('CommTime_ms', 0),
                    'compute_time': bench.get('ComputeTime_ms', 0),
                    'num_processes': bench.get('NumProcesses', 1)
                }
                grouped[key].append(data)
    
    return grouped

def plot_execution_time_comparison(grouped_data: Dict, output_dir: Path):
    """Create bar chart comparing execution times across implementations"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    configs = []
    cpu_times = []
    naive_cuda_mpi_times = []
    cuda_mpi_times = []
    
    for key in sorted(grouped_data.keys()):
        size, max_val, dist = key
        benchmarks = grouped_data[key]
        
        config_label = f"{size}\n{dist[:8]}"
        configs.append(config_label)
        
        cpu_time = None
        naive_time = None
        opt_time = None
        
        for bench in benchmarks:
            impl = bench['impl']
            if 'NaiveCpuSnpSort' in impl or 'CpuSnpSort' in impl:
                cpu_time = bench['time']
            elif 'NaiveCudaMpiSnpSort' in impl:
                naive_time = bench['time']
            elif 'CudaMpiSnpSort' in impl:
                opt_time = bench['time']
        
        cpu_times.append(cpu_time if cpu_time else 0)
        naive_cuda_mpi_times.append(naive_time if naive_time else 0)
        cuda_mpi_times.append(opt_time if opt_time else 0)
    
    # Create grouped bars
    x = np.arange(len(configs))
    width = 0.25
    
    bars1 = ax.bar(x - width, cpu_times, width, label='CPU', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, naive_cuda_mpi_times, width, label='Naive CUDA+MPI', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, cuda_mpi_times, width, label='Optimized CUDA+MPI', color='#2ecc71', alpha=0.8)
    
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
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    output_file = output_dir / 'execution_time_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_speedup_analysis(grouped_data: Dict, output_dir: Path):
    """Create speedup comparison chart"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    configs = []
    naive_speedups = []
    opt_speedups = []
    
    for key in sorted(grouped_data.keys()):
        size, max_val, dist = key
        benchmarks = grouped_data[key]
        
        config_label = f"{size}\n{dist[:8]}"
        configs.append(config_label)
        
        cpu_time = None
        naive_time = None
        opt_time = None
        
        for bench in benchmarks:
            impl = bench['impl']
            if 'NaiveCpuSnpSort' in impl or 'CpuSnpSort' in impl:
                cpu_time = bench['time']
            elif 'NaiveCudaMpiSnpSort' in impl:
                naive_time = bench['time']
            elif 'CudaMpiSnpSort' in impl:
                opt_time = bench['time']
        
        # Calculate speedups (higher is better)
        if cpu_time and naive_time and cpu_time > 0:
            naive_speedups.append(cpu_time / naive_time)
        else:
            naive_speedups.append(0)
        
        if cpu_time and opt_time and cpu_time > 0:
            opt_speedups.append(cpu_time / opt_time)
        else:
            opt_speedups.append(0)
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, naive_speedups, width, label='Naive CUDA+MPI', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, opt_speedups, width, label='Optimized CUDA+MPI', color='#2ecc71', alpha=0.8)
    
    # Add baseline at 1.0x
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (CPU)')
    
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
    
    add_labels(bars1)
    add_labels(bars2)
    
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
            
            if 'NaiveCudaMpiSnpSort' in impl:
                naive_comm_time = comm
                naive_comp_time = comp
            elif 'CudaMpiSnpSort' in impl:
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
    
    # Plot for the most common distribution (e.g., RANDOM)
    dist_to_plot = 'RANDOM'
    if dist_to_plot in by_distribution:
        impl_data = by_distribution[dist_to_plot]
        
        for impl, data_points in impl_data.items():
            # Sort by size
            data_points.sort(key=lambda x: x[0])
            sizes = [x[0] for x in data_points]
            times = [x[1] for x in data_points]
            
            # Choose color and label
            if 'NaiveCudaMpiSnpSort' in impl:
                color = '#e74c3c'
                label = 'Naive CUDA+MPI'
            elif 'CudaMpiSnpSort' in impl:
                color = '#2ecc71'
                label = 'Optimized CUDA+MPI'
            elif 'Cpu' in impl:
                color = '#3498db'
                label = 'CPU'
            else:
                continue
            
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
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("SNP SORT BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        # Detailed results table
        f.write("DETAILED RESULTS\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Config':<30} {'CPU':<12} {'Naive MPI':<12} {'Opt MPI':<12} {'Naive Spdup':<12} {'Opt Spdup':<12}\n")
        f.write("-" * 100 + "\n")
        
        all_naive_speedups = []
        all_opt_speedups = []
        
        for key in sorted(grouped_data.keys()):
            size, max_val, dist = key
            benchmarks = grouped_data[key]
            
            config_name = f"{size}/{dist}"
            
            cpu_time = naive_time = opt_time = None
            
            for bench in benchmarks:
                impl = bench['impl']
                if 'NaiveCpuSnpSort' in impl or 'CpuSnpSort' in impl:
                    cpu_time = bench['time']
                elif 'NaiveCudaMpiSnpSort' in impl:
                    naive_time = bench['time']
                elif 'CudaMpiSnpSort' in impl:
                    opt_time = bench['time']
            
            cpu_str = f"{cpu_time:.2f} ms" if cpu_time else "N/A"
            naive_str = f"{naive_time:.2f} ms" if naive_time else "N/A"
            opt_str = f"{opt_time:.2f} ms" if opt_time else "N/A"
            
            naive_speedup = cpu_time / naive_time if cpu_time and naive_time else None
            opt_speedup = cpu_time / opt_time if cpu_time and opt_time else None
            
            if naive_speedup:
                all_naive_speedups.append(naive_speedup)
            if opt_speedup:
                all_opt_speedups.append(opt_speedup)
            
            naive_sp_str = f"{naive_speedup:.2f}x" if naive_speedup else "N/A"
            opt_sp_str = f"{opt_speedup:.2f}x" if opt_speedup else "N/A"
            
            f.write(f"{config_name:<30} {cpu_str:<12} {naive_str:<12} {opt_str:<12} {naive_sp_str:<12} {opt_sp_str:<12}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 100 + "\n\n")
        
        if all_naive_speedups:
            f.write(f"Naive CUDA+MPI Speedup:\n")
            f.write(f"  Average: {np.mean(all_naive_speedups):.2f}x\n")
            f.write(f"  Median:  {np.median(all_naive_speedups):.2f}x\n")
            f.write(f"  Min:     {np.min(all_naive_speedups):.2f}x\n")
            f.write(f"  Max:     {np.max(all_naive_speedups):.2f}x\n")
            f.write(f"  StdDev:  {np.std(all_naive_speedups):.2f}\n\n")
        
        if all_opt_speedups:
            f.write(f"Optimized CUDA+MPI Speedup:\n")
            f.write(f"  Average: {np.mean(all_opt_speedups):.2f}x\n")
            f.write(f"  Median:  {np.median(all_opt_speedups):.2f}x\n")
            f.write(f"  Min:     {np.min(all_opt_speedups):.2f}x\n")
            f.write(f"  Max:     {np.max(all_opt_speedups):.2f}x\n")
            f.write(f"  StdDev:  {np.std(all_opt_speedups):.2f}\n\n")
        
        if all_naive_speedups and all_opt_speedups:
            improvement = np.mean(all_opt_speedups) / np.mean(all_naive_speedups)
            f.write(f"Optimization Improvement: {improvement:.2f}x\n")
            f.write(f"  (Optimized is {improvement:.2f}x faster than Naive on average)\n\n")
        
        f.write("=" * 100 + "\n")
    
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
