# SNP Sort Benchmark System

Comprehensive benchmarking system for comparing performance of different SNP-based sorting implementations.

## Overview

This benchmark suite evaluates three SNP sort implementations:

1. **CPU SNP Sort** (`createSnpSort()`)
   - Single-threaded CPU implementation using `NaiveCpuSnpSimulator`
   - Baseline for comparison

2. **Naive CUDA/MPI SNP Sort** (`createNaiveCudaMpiSnpSort()`)
   - GPU-accelerated using `NaiveCudaMpiSnpSimulator`
   - Basic distributed implementation without advanced optimizations

3. **Optimized CUDA/MPI SNP Sort** (`createCudaMpiSnpSort()`)
   - GPU-accelerated using `CudaMpiSnpSimulator`
   - Fully optimized with MPI distribution
   - Expected best performance

## Test Coverage

### Input Sizes
- **Small**: 10, 50, 100 elements
- **Medium**: 500, 1000 elements
- **Large**: 5000+ elements (configurable)

### Input Distributions
- **RANDOM**: Uniformly distributed random integers
- **SORTED**: Already sorted (best case)
- **REVERSE_SORTED**: Sorted descending (worst case)
- **NEARLY_SORTED**: 90% sorted with random swaps
- **FEW_UNIQUE**: Many duplicates (5-10 unique values)
- **UNIFORM**: All elements have same value

### Value Ranges
- Small: 0-10
- Medium: 0-100
- Large: 0-1000

## Quick Start

### Build the Benchmark

```bash
# Configure and build
make clean
make build

# Or rebuild from scratch
make rebuild
```

### Run Benchmarks

#### Local Execution (Single Node)
```bash
# Run all benchmarks locally
make benchmark-sort

# Or use the script directly with options
./scripts/run_benchmark.sh --help
```

#### Distributed Execution (Multiple Nodes)
```bash
# Run benchmarks across distributed nodes
make benchmark-distributed

# Or use the script with custom configuration
./scripts/run_distributed_benchmark.sh -n 4 --filter "CudaMpi.*RANDOM"
```

### Compare Results

```bash
# Compare all benchmark results
make compare-benchmarks

# Or compare specific runs
python3 scripts/compare_benchmarks.py benchmark_results/benchmark_*.json
```

## Usage Examples

### Running Specific Benchmarks

```bash
# Run only RANDOM distribution benchmarks
./scripts/run_benchmark.sh -f ".*RANDOM"

# Run only 100-element benchmarks
./scripts/run_benchmark.sh -f ".*_100_.*"

# Run only optimized CUDA/MPI implementation
./scripts/run_benchmark.sh -f "CudaMpiSnpSort.*"
```

### Output Formats

```bash
# Console output (default)
./scripts/run_benchmark.sh

# JSON format for analysis
./scripts/run_benchmark.sh --json

# CSV format for spreadsheets
./scripts/run_benchmark.sh --csv
```

### Distributed Execution Options

```bash
# Run with 4 MPI processes
./scripts/run_distributed_benchmark.sh -n 4

# Run without hostfile (localhost only)
./scripts/run_distributed_benchmark.sh --no-hostfile -n 2

# Use custom hostfile
./scripts/run_distributed_benchmark.sh --hostfile custom_hosts.txt -n 8

# Run specific benchmarks distributed
./scripts/run_distributed_benchmark.sh -n 4 -f ".*500.*RANDOM" --json
```

## Benchmark Metrics

### Reported Metrics

- **Execution Time**: Wall-clock time in milliseconds
- **CPU Time**: Actual CPU time spent
- **Throughput**: Elements sorted per second
- **Elements**: Number of elements in test
- **MaxValue**: Maximum value in input range

### Calculated Metrics (via compare script)

- **Speedup**: Performance relative to CPU baseline
- **Average Speedup**: Mean speedup across all tests
- **Min/Max Speedup**: Range of speedup values

## Output Structure

### Console Output
```
========================================
SNP Sort Benchmark Suite
========================================

Running CpuSnpSort_10_10_RANDOM
Time: 5.234 ms, Throughput: 1910 elem/s

...

========================================
Benchmark Complete
========================================
```

### JSON Output
Saved to `benchmark_results/benchmark_YYYYMMDD_HHMMSS.json`

### CSV Output
Saved to `benchmark_results/benchmark_YYYYMMDD_HHMMSS.csv`

## Comparison Report

The comparison script (`compare_benchmarks.py`) generates:

### Comparison Table
```
==================================================================================================
SNP Sort Benchmark Comparison
==================================================================================================
Configuration                            CPU Time (ms)                       Speedup vs CPU      
Size/MaxVal/Distribution                 CPU  |  Naive  |  Optimized        Naive  |  Optimized
--------------------------------------------------------------------------------------------------
10/10/RANDOM                             5.23    |  4.12    |  2.89          1.27x  |  1.81x
...
```

### Summary Statistics
- Average speedup for each implementation
- Min/Max speedup observed
- Performance trends

### Recommendations
- When to use each implementation
- Optimization suggestions
- Configuration tips

## Performance Analysis

### Expected Results

1. **Small Inputs (< 100 elements)**
   - CPU may be competitive due to GPU overhead
   - Naive CUDA/MPI shows minimal speedup
   - Optimized CUDA/MPI provides moderate speedup

2. **Medium Inputs (100-1000 elements)**
   - GPU acceleration becomes significant
   - Optimized implementation shows clear advantage
   - Communication overhead still manageable

3. **Large Inputs (> 1000 elements)**
   - Maximum speedup achieved
   - Optimized implementation scales well
   - Network bandwidth may become bottleneck

### Input Distribution Impact

- **SORTED/UNIFORM**: Best case - minimal work
- **RANDOM**: Average case - typical performance
- **REVERSE_SORTED**: Worst case - maximum comparisons
- **FEW_UNIQUE**: Many duplicates - optimization opportunities

## Validation

All benchmarks include:
- **Correctness Verification**: Output is verified as sorted
- **Stability Checks**: No crashes or errors
- **Reproducibility**: Consistent results with same seed

## Advanced Configuration

### Custom Benchmarks

Add new benchmarks in [SortBenchmark.cpp](../benchmark/SortBenchmark.cpp):

```cpp
// Add custom size/distribution
BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 2000, 500, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_2000_500_RANDOM)
    ->Unit(benchmark::kMillisecond);
```

### MPI Configuration

Edit [hostfile.txt](../hostfile.txt) for distributed runs:
```
node1 slots=2
node2 slots=2
node3 slots=4
```

### Google Benchmark Options

Pass additional options to the benchmark executable:

```bash
# Run for minimum 5 seconds per benchmark
./scripts/run_benchmark.sh --benchmark_min_time=5

# Run specific number of iterations
./scripts/run_benchmark.sh --benchmark_repetitions=10

# Display all available benchmarks
build/sort_benchmark --benchmark_list_tests
```

## Troubleshooting

### Common Issues

1. **MPI Errors**
   ```
   Solution: Ensure all nodes have MPI configured correctly
   Check: mpirun --version
   ```

2. **CUDA Errors**
   ```
   Solution: Verify CUDA toolkit installation
   Check: nvidia-smi
   ```

3. **Build Failures**
   ```
   Solution: Clean and rebuild
   make clean && make build
   ```

4. **No Results Found**
   ```
   Solution: Ensure benchmarks have run with JSON/CSV output
   ./scripts/run_benchmark.sh --json
   ```

## Files and Structure

```
benchmark/
├── SortBenchmark.cpp              # Main benchmark implementation
└── sort_benchmark.prompt.md       # This README

scripts/
├── run_benchmark.sh               # Local benchmark runner
├── run_distributed_benchmark.sh   # Distributed benchmark runner
└── compare_benchmarks.py          # Result analysis tool

benchmark_results/                 # Output directory (created automatically)
├── benchmark_*.json               # JSON results
└── benchmark_*.csv                # CSV results

hostfile.txt                       # MPI host configuration
```

## Contributing

To add new benchmarks:

1. Define new test cases in `SortBenchmark.cpp`
2. Update CMakeLists.txt if needed
3. Test locally before distributed runs
4. Document any new distributions or metrics

## Performance Optimization Tips

### For SNP Sort Algorithm
- Performance depends on maximum value in input
- Larger max values require more simulation ticks
- Consider normalizing inputs if possible

### For GPU Acceleration
- Ensure sufficient GPU memory
- Monitor data transfer overhead
- Batch operations when possible

### For Distributed Computing
- Balance load across nodes
- Minimize inter-node communication
- Use fast network interconnects

## References

- [Google Benchmark Documentation](https://github.com/google/benchmark)
- [SNP System Documentation](../docs/snp_explanation.tex)
- [Sort Implementation](../src/sort/SnpSort.cpp)

## License

Same as main project.
