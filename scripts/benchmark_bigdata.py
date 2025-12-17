import subprocess
import json
import sys
import os
import time

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def read_result(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    neurons = os.environ.get("NEURONS", "100000")
    hosts = os.environ.get("HOSTS", "localhost,10.0.0.2")
    steps = os.environ.get("STEPS", "10")
    
    # 1. Generate
    print("\n========================================")
    print("Step 1: Generating Dataset")
    print("========================================")
    run_command(f"make bigdata-generate NEURONS={neurons} HOSTS='{hosts}'")
    
    # 2. Run Naive
    print("\n========================================")
    print("Step 2: Running Naive Implementation")
    print("========================================")
    run_command(f"make bigdata-run HOSTS='{hosts}' STEPS={steps} IMPL=naive")
    
    # 3. Run Optimized
    print("\n========================================")
    print("Step 3: Running Optimized Implementation")
    print("========================================")
    run_command(f"make bigdata-run HOSTS='{hosts}' STEPS={steps} IMPL=optimized")
    
    # 4. Compare
    num_nodes = len(hosts.split(','))
    # Note: The filenames now include the implementation
    naive_file = f"bigdata/results/run_naive_{num_nodes}nodes.json"
    opt_file = f"bigdata/results/run_optimized_{num_nodes}nodes.json"
    
    try:
        naive_res = read_result(naive_file)
        opt_res = read_result(opt_file)
        
        print("\n========================================")
        print("Benchmark Results")
        print("========================================")
        print(f"Neurons: {neurons}")
        print(f"Nodes:   {num_nodes} ({hosts})")
        print(f"Steps:   {steps}")
        print("-" * 85)
        print(f"{'Metric':<25} | {'Naive':<15} | {'Optimized':<15} | {'Speedup':<10}")
        print("-" * 85)
        
        metrics = [
            ("Total Time (s)", "total_time_s"),
            ("Compute Time (ms)", "max_compute_time_ms"),
            ("Comm Time (ms)", "max_comm_time_ms"),
            ("State Checksum", "state_checksum")
        ]
        
        for label, key in metrics:
            n_val = naive_res.get(key, 0)
            o_val = opt_res.get(key, 0)
            
            if key == "state_checksum":
                match = "✅" if n_val == o_val else "❌"
                print(f"{label:<25} | {n_val:<15} | {o_val:<15} | {match:<10}")
            else:
                speedup = n_val / o_val if o_val > 0 else 0.0
                print(f"{label:<25} | {n_val:<15.4f} | {o_val:<15.4f} | {speedup:<10.2f}x")
        print("-" * 85)
            
    except Exception as e:
        print(f"Error reading results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
