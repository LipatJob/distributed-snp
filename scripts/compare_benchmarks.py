#!/usr/bin/env python3

"""
Compare benchmark results and generate performance analysis
Usage: python3 scripts/compare_benchmarks.py [result_files...]
"""

import json
import sys
import csv
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

def load_json_benchmark(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_benchmark_name(name: str) -> Dict[str, str]:
    """Parse benchmark name to extract implementation, size, max_value, and distribution"""
    # Format: Implementation_Size_MaxValue_Distribution
    parts = name.split('/')
    base_name = parts[0] if parts else name
    
    components = base_name.split('_')
    if len(components) >= 4:
        impl = components[0]
        size = components[1]
        max_val = components[2]
        dist = '_'.join(components[3:])
        
        return {
            'implementation': impl,
            'size': int(size),
            'max_value': int(max_val),
            'distribution': dist
        }
    return {}

def analyze_benchmarks(results: List[Dict[str, Any]]) -> None:
    """Analyze and compare benchmark results"""
    
    # Group benchmarks by configuration
    grouped = defaultdict(list)
    
    for result in results:
        benchmarks = result.get('benchmarks', [])
        for bench in benchmarks:
            name = bench['name']
            parsed = parse_benchmark_name(name)
            
            if parsed:
                key = (parsed['size'], parsed['max_value'], parsed['distribution'])
                grouped[key].append({
                    'impl': parsed['implementation'],
                    'time': bench['real_time'],
                    'cpu_time': bench['cpu_time'],
                    'throughput': bench.get('Throughput(elem/s)', 0)
                })
    
    # Print comparison table
    print("\n" + "="*100)
    print("SNP Sort Benchmark Comparison")
    print("="*100)
    print(f"{'Configuration':<40} {'CPU Time (ms)':<35} {'Speedup vs CPU':<20}")
    print(f"{'Size/MaxVal/Distribution':<40} {'CPU  |  Naive  |  Optimized':<35} {'Naive  |  Optimized':<20}")
    print("-"*100)
    
    # Sort by size for better readability
    for key in sorted(grouped.keys()):
        size, max_val, dist = key
        config_name = f"{size}/{max_val}/{dist}"
        
        benchmarks = grouped[key]
        
        # Find each implementation
        cpu_time = None
        naive_time = None
        opt_time = None
        
        for bench in benchmarks:
            impl = bench['impl']
            time_ms = bench['time']
            
            if 'CpuSnpSort' in impl:
                cpu_time = time_ms
            elif 'NaiveCudaMpiSnpSort' in impl:
                naive_time = time_ms
            elif 'CudaMpiSnpSort' in impl:
                opt_time = time_ms
        
        # Format times
        cpu_str = f"{cpu_time:.2f}" if cpu_time else "N/A"
        naive_str = f"{naive_time:.2f}" if naive_time else "N/A"
        opt_str = f"{opt_time:.2f}" if opt_time else "N/A"
        
        times_str = f"{cpu_str:<7} | {naive_str:<7} | {opt_str:<7}"
        
        # Calculate speedups
        speedup_naive = cpu_time / naive_time if cpu_time and naive_time else None
        speedup_opt = cpu_time / opt_time if cpu_time and opt_time else None
        
        naive_speedup_str = f"{speedup_naive:.2f}x" if speedup_naive else "N/A"
        opt_speedup_str = f"{speedup_opt:.2f}x" if speedup_opt else "N/A"
        
        speedup_str = f"{naive_speedup_str:<7} | {opt_speedup_str:<7}"
        
        print(f"{config_name:<40} {times_str:<35} {speedup_str:<20}")
    
    print("="*100)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("-"*100)
    
    all_cpu_times = []
    all_naive_times = []
    all_opt_times = []
    all_naive_speedups = []
    all_opt_speedups = []
    
    for benchmarks in grouped.values():
        cpu_time = None
        naive_time = None
        opt_time = None
        
        for bench in benchmarks:
            impl = bench['impl']
            time_ms = bench['time']
            
            if 'CpuSnpSort' in impl:
                cpu_time = time_ms
                all_cpu_times.append(time_ms)
            elif 'NaiveCudaMpiSnpSort' in impl:
                naive_time = time_ms
                all_naive_times.append(time_ms)
            elif 'CudaMpiSnpSort' in impl:
                opt_time = time_ms
                all_opt_times.append(time_ms)
        
        if cpu_time and naive_time:
            all_naive_speedups.append(cpu_time / naive_time)
        if cpu_time and opt_time:
            all_opt_speedups.append(cpu_time / opt_time)
    
    if all_naive_speedups:
        avg_naive = sum(all_naive_speedups) / len(all_naive_speedups)
        max_naive = max(all_naive_speedups)
        min_naive = min(all_naive_speedups)
        print(f"Naive CUDA/MPI Speedup:    Avg: {avg_naive:.2f}x  |  Min: {min_naive:.2f}x  |  Max: {max_naive:.2f}x")
    
    if all_opt_speedups:
        avg_opt = sum(all_opt_speedups) / len(all_opt_speedups)
        max_opt = max(all_opt_speedups)
        min_opt = min(all_opt_speedups)
        print(f"Optimized CUDA/MPI Speedup: Avg: {avg_opt:.2f}x  |  Min: {min_opt:.2f}x  |  Max: {max_opt:.2f}x")
    
    print("-"*100)
    
    # Recommendations
    print("\nRecommendations:")
    print("-"*100)
    
    if all_opt_speedups and avg_opt > 1.0:
        print("✓ Optimized CUDA/MPI implementation shows significant speedup over CPU")
        print(f"  → Use for production workloads (avg {avg_opt:.2f}x faster)")
    elif all_opt_speedups and avg_opt <= 1.0:
        print("⚠ Optimized CUDA/MPI implementation is not faster than CPU")
        print("  → Consider using CPU implementation for small datasets")
        print("  → Check GPU overhead and data transfer costs")
    
    if all_naive_speedups and all_opt_speedups:
        improvement = (avg_opt / avg_naive) if avg_naive > 0 else 0
        if improvement > 1.2:
            print(f"✓ Optimizations provide {improvement:.2f}x improvement over naive implementation")
        else:
            print(f"⚠ Optimizations provide minimal improvement ({improvement:.2f}x)")
    
    print("-"*100)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_benchmarks.py <result_file1.json> [result_file2.json ...]")
        print("\nExample:")
        print("  python3 compare_benchmarks.py benchmark_results/*.json")
        sys.exit(1)
    
    results = []
    for filepath in sys.argv[1:]:
        try:
            result = load_json_benchmark(filepath)
            results.append(result)
            print(f"Loaded: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if not results:
        print("No valid benchmark results found")
        sys.exit(1)
    
    analyze_benchmarks(results)

if __name__ == '__main__':
    main()
