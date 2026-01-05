# SNP System Performance Analysis - Nsys Profiling Results

## Overview

This Jupyter notebook (`nsys_analysis.ipynb`) provides comprehensive performance analysis of Spiking Neural P (SNP) system implementations using NVIDIA Nsys profiling data. It supports multiple backend implementations (CUDA-only and MPI+CUDA distributed) and generates publication-ready visualizations comparing their performance characteristics.

## Purpose

The notebook analyzes profiling data to:

1. **Quantify GPU acceleration effectiveness** - Compare CPU compute time vs GPU execution time
2. **Measure MPI communication overhead** - Identify bottlenecks in distributed implementations
3. **Evaluate partitioning strategies** - Compare linear, Louvain, and Red-Blue partitioning algorithms
4. **Assess load balance** - Analyze per-rank performance distribution in multi-GPU setups
5. **Identify optimization opportunities** - Highlight CUDA API overhead and kernel performance

## Implementations Analyzed

### CUDA Implementations
- **optimized-cuda**: Single-GPU optimized implementation
- **sparse-cuda**: Sparse matrix optimization for single GPU

### MPI+CUDA Distributed Implementations
- **naive-cuda-mpi-linear**: Basic distributed implementation with linear partitioning
- **naive-cuda-mpi-louvain**: Basic distributed with Louvain graph partitioning
- **naive-cuda-mpi-red-blue**: Basic distributed with Red-Blue Pebbling partitioning
- **optimized-cuda-mpi-linear**: Optimized distributed with linear partitioning
- **optimized-cuda-mpi-louvain**: Optimized distributed with Louvain partitioning
- **optimized-cuda-mpi-red-blue**: Optimized distributed with Red-Blue Pebbling partitioning

## Key Features

### 1. CPU Time Extraction from SQLite
Unlike CSV exports which only contain GPU and MPI metrics, this notebook queries the nsys SQLite databases directly to extract:
- Total execution timeline (start/end timestamps)
- OS Runtime (OSRT) API calls
- CPU compute time calculated as: `Total Execution Time - (CUDA Kernel + CUDA Memory + MPI)`

### 2. Multi-Rank MPI Support
For distributed implementations with multiple MPI ranks:
- **Automatic rank detection**: Identifies all rank files (rank0, rank1, rank2, etc.)
- **Aggregation strategies**: 
  - Sums GPU/CPU work across ranks (total computational work)
  - Takes max wall time (longest-running rank determines overall execution time)
- **Per-rank analysis**: Visualizes load distribution and identifies imbalances

### 4. Publication-Ready Output
- High-DPI (300 DPI) PDF and PNG exports
- LaTeX-formatted tables for direct paper inclusion
- CSV data exports for further analysis

## Visualizations Generated

### Visualization 1: Time Distribution (CPU + GPU + MPI)
**File**: `time_distribution_with_cpu.{pdf,png}`

Stacked bar charts showing the fundamental time breakdown:
- CPU compute time
- CUDA kernel execution
- CUDA memory operations
- MPI communication (for distributed implementations)

Separate subplots for CUDA-only vs MPI+CUDA implementations.

### Visualization 2: CPU vs GPU Comparison
**File**: `cpu_vs_gpu_comparison.{pdf,png}`

Two-panel comparison showing:
- **Panel 1**: Absolute CPU vs GPU execution time (side-by-side bars)
- **Panel 2**: GPU speedup factor and GPU utilization percentage

Highlights the effectiveness of GPU offloading.

### Visualization 3: Per-Rank Load Balance Analysis
**File**: `per_rank_load_balance.{pdf,png}`

Four-panel visualization for multi-rank MPI implementations:
- **Panel 1**: Total execution time per rank (grouped vertical bars)
- **Panel 2**: GPU kernel time per rank (grouped vertical bars)
- **Panel 3**: MPI communication time per rank (grouped vertical bars)
- **Panel 4**: Load balance metrics (Coefficient of Variation and Max/Min ratio)

Each implementation shows sub-bars for individual ranks, making load imbalance immediately visible.

### Visualization 4: Partitioning Strategy Load Balance
**File**: `partitioning_load_balance.{pdf,png}`

Compares the three partitioning strategies (Linear, Louvain, Red-Blue) across four metrics:
- **Panel 1**: Load imbalance (Max/Min execution time ratio)
- **Panel 2**: Load variability (Coefficient of Variation)
- **Panel 3**: GPU workload balance
- **Panel 4**: MPI communication balance

Shows which partitioning strategy achieves best load distribution.

### Visualization 5: Kernel Performance Breakdown
**File**: `kernel_breakdown.{pdf,png}`

Stacked bar chart showing time spent in each CUDA kernel type. Identifies which kernels dominate execution and where optimization efforts should focus.

### Visualization 6: MPI Communication Analysis
**File**: `mpi_communication.{pdf,png}`

Two-panel analysis:
- **Panel 1**: MPI operation breakdown (Isend, Waitall, Allgather, etc.)
- **Panel 2**: Total MPI time comparison across implementations

Color-coded by naive (blue) vs optimized (orange) variants.

### Visualization 7: MPI Message Volume
**File**: `mpi_message_volume.{pdf,png}`

Two metrics:
- **Panel 1**: Total data volume transferred (MB)
- **Panel 2**: Average message size (KB)

Helps correlate communication time with data movement patterns.

### Visualization 8: Partitioning Strategy Comparison
**File**: `partitioning_comparison.{pdf,png}`

Four-panel comprehensive comparison:
- **Panel 1**: Total execution time by strategy
- **Panel 2**: MPI communication time
- **Panel 3**: Message volume
- **Panel 4**: Communication overhead ratio (MPI time / total time %)

Key for justifying partitioning strategy choice in publications.

### Visualization 9: Overall Performance Comparison
**File**: `overall_performance.{pdf,png}`

Horizontal bar charts showing:
- **Panel 1**: Absolute execution time (all implementations)
- **Panel 2**: Speedup relative to slowest implementation

Color-coded by CUDA (green) vs MPI+CUDA (blue) implementations.

### Visualization 10: CUDA API Overhead Analysis
**File**: `cuda_api_overhead.{pdf,png}`

Analyzes synchronization and launch overhead:
- **Panel 1**: API call time breakdown (cudaStreamSynchronize, cudaLaunchKernel, etc.)
- **Panel 2**: Synchronization overhead as percentage of total CUDA API time

Identifies if excessive synchronization is limiting performance.

## Usage Instructions

### 1. Data Preparation

Run nsys profiling on your implementations:

```bash
nsys profile -o snp_profile_<implementation> -f true --stats=true ./your_binary
```

For MPI implementations with multiple ranks:
```bash
mpirun -n 3 nsys profile -o snp_profile_<implementation>_rank%q{OMPI_COMM_WORLD_RANK} \
  -f true --stats=true ./your_binary
```

This generates:
- `*.sqlite` - Timeline databases (queried for CPU time)
- `*_stats_*.csv` - Summary statistics (GPU kernels, MPI operations, etc.)

### 2. Configure Timestamp

Set the `TIMESTAMP` variable in the second code cell to match your profiling run directory:

```python
TIMESTAMP = "20260104_184616"  # Change this to your profiling run
```

### 3. Run All Cells

Execute all cells in order (Kernel → Run All). The notebook will:
- Detect available implementations
- Automatically identify number of ranks per implementation
- Generate all 10 visualizations
- Create summary tables
- Export LaTeX code for tables

### 4. Check Outputs

Visualizations and tables are saved to:
```
results/nsys/{TIMESTAMP}/figures/
```

## Data Requirements

### Directory Structure
```
profiling/results/nsys/{TIMESTAMP}/{implementation}/
├── snp_profile_*_rank0*.sqlite          # Timeline database (rank 0)
├── snp_profile_*_rank1*.sqlite          # Timeline database (rank 1)
├── ...
├── snp_profile_*_rank0*_stats_cuda_api_sum.csv
├── snp_profile_*_rank0*_stats_cuda_gpu_kern_sum.csv
├── snp_profile_*_rank0*_stats_mpi_event_sum.csv
├── snp_profile_*_rank0*_stats_mpi_msg_size_sum.csv
└── ... (additional CSV files for each rank)
```

### Required CSV Files
- `stats_cuda_api_sum.csv` - CUDA API call summary
- `stats_cuda_api_gpu_sum.csv` - GPU-specific CUDA calls
- `stats_cuda_gpu_kern_sum.csv` - Kernel execution statistics
- `stats_mpi_event_sum.csv` - MPI operation breakdown (MPI implementations only)
- `stats_mpi_msg_size_sum.csv` - Message size statistics (MPI implementations only)

## Key Functions

### Data Loading
- `load_csv_safely(impl, file_suffix, rank=None)` - Load CSV for specific implementation and rank
- `load_all_ranks_csv(impl, file_suffix)` - Load CSV files for all ranks
- `get_num_ranks(impl)` - Detect number of MPI ranks
- `get_rank_files(impl, pattern)` - Find all rank files matching pattern

### SQLite Queries
- `query_nsys_db(impl, query, rank=0)` - Execute SQL on nsys database
- `get_execution_timeline(impl, rank=0)` - Get total execution time from timeline
- `get_detailed_time_breakdown(impl, rank=0)` - Per-rank time breakdown
- `get_aggregate_time_breakdown(impl)` - Aggregated multi-rank breakdown

### Analysis Helpers
- `extract_strategy(impl_name)` - Extract partitioning strategy from name
- `extract_variant(impl_name)` - Determine naive vs optimized variant
- `extract_rank_number(filename)` - Parse rank ID from filename

## Load Balance Metrics

### Coefficient of Variation (CV)
```
CV = (Standard Deviation / Mean) × 100%
```
Lower CV indicates better balance. CV < 10% is generally considered well-balanced.

### Load Imbalance Ratio
```
Imbalance = Max Execution Time / Min Execution Time
```
Ratio of 1.0 is perfect balance. Values > 1.2 indicate significant imbalance.

## Performance Interpretation Guide

### Good Performance Indicators
✓ Low CPU time percentage (< 20%) - indicates effective GPU utilization
✓ High GPU kernel time percentage (> 60%) - computation is GPU-bound (good)
✓ Low MPI time percentage (< 15%) - minimal communication overhead
✓ Load imbalance ratio close to 1.0 (< 1.1) - well-distributed work
✓ Low CV (< 10%) across ranks - consistent performance

### Performance Issues to Address
✗ High CPU time (> 30%) - may have serialization bottlenecks
✗ High synchronization overhead - excessive cudaStreamSynchronize calls
✗ High MPI time (> 25%) - communication-bound, consider better partitioning
✗ Large message volumes with high latency - optimize communication patterns
✗ High load imbalance (> 1.2) - poor work distribution

## Partitioning Strategy Selection

Based on load balance analysis:

1. **Linear Partitioning**: Simple but may create imbalance in irregular graphs
2. **Louvain Partitioning**: Graph-based community detection, generally best for irregular graphs
3. **Red-Blue Pebbling**: Memory hierarchy-aware, good for specific access patterns

The notebook automatically identifies which strategy achieves best load balance for your workload.

## Output Files

### Figures (PDF + PNG)
All visualizations saved in both formats for paper submission and presentations.

### Tables
- `performance_summary_with_cpu.csv` - Complete performance metrics table
- `performance_table.tex` - LaTeX-formatted table for paper inclusion

### LaTeX Table Usage
```latex
\input{figures/performance_table.tex}
```

## Customization

### Adding New Implementations
Add to the appropriate list in the configuration cell:
```python
CUDA_IMPLS = ["optimized-cuda", "sparse-cuda", "your-new-cuda-impl"]
MPI_IMPLS = ["optimized-cuda-mpi-louvain", "your-new-mpi-impl"]
```

### Modifying Color Scheme
Edit the color constants in the second code cell:
```python
COLOR_CPU = '#FFA726'        # Change to your preferred color
COLOR_GPU_KERNEL = '#2E7D32'
# ... etc
```

### Filtering Implementations
To analyze only specific implementations, modify the configuration lists before running.

## Technical Notes

### CPU Time Calculation
CPU time is derived from the execution timeline:
```
CPU Time = Total Execution Time - (GPU Kernel + GPU Memory + MPI)
```

This represents:
- Application logic on CPU
- Data preparation and marshaling
- CUDA API overhead (cudaMalloc, cudaMemcpy, etc.)
- OS scheduling and context switches

### Multi-Rank Aggregation
For MPI implementations:
- **Work metrics** (GPU/CPU time) are summed across ranks (total work performed)
- **Wall time** uses max across ranks (slowest rank determines completion)
- This correctly captures both computational load and critical path

### Known Limitations
- CSV exports don't include CPU time (requires SQLite queries)
- Very large timelines (> 1M events) may slow SQLite queries
- Nsys must be run with `--stats=true` to generate CSV files
- MPI implementations must use rank-specific output files (`%q{OMPI_COMM_WORLD_RANK}`)

## Dependencies

```python
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
sqlite3 (built-in)
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn
```

## Development History

This notebook was developed through iterative refinement:

1. **Initial version**: Basic CSV loading and simple bar charts
2. **CPU time addition**: SQLite database queries to extract CPU compute time
3. **Multi-rank support**: Automatic rank detection, aggregation, and per-rank analysis
4. **Load balance visualization**: Changed from line plots to grouped vertical bars for clarity
5. **Color standardization**: Unified color scheme across all visualizations for consistency

## Contributing

To extend this notebook:

1. Add new data loading functions for additional nsys exports
2. Create new visualization cells following the existing pattern
3. Update color constants if adding new metric categories
4. Add new analysis functions for custom metrics

## Citation

If using this notebook for research publications, please cite the SNP system paper:

```bibtex
@article{snp-system-2026,
  title={Distributed Simulation of Spiking Neural P Systems using Multi-GPU Computing},
  author={Your Names Here},
  journal={Journal Name},
  year={2026}
}
```