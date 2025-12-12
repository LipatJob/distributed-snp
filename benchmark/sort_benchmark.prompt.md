# Sort Benchmark Implementation Prompt

## Overview
Create a comprehensive benchmarking system to compare the performance of different SNP-based sorting implementations in this distributed computing project.

## Available Sort Implementations

The project has three sort implementations, all using the `ISort` interface:

1. **CPU-based SNP Sort** (`createSnpSort()`)
   - Uses `NaiveCpuSnpSimulator` 
   - Single-threaded CPU implementation
   - Baseline for comparison

2. **CUDA/MPI Optimized SNP Sort** (`createCudaMpiSnpSort()`)
   - Uses `CudaMpiSnpSimulator`
   - GPU-accelerated with MPI distribution
   - Expected best performance

3. **Naive CUDA/MPI SNP Sort** (`createNaiveCudaMpiSnpSort()`)
   - Uses `NaiveCudaMpiSnpSimulator`
   - Basic GPU implementation without optimizations
   - Intermediate performance expected

## Requirements

### 1. Benchmark Framework Setup
- Use Google Benchmark library (or similar high-performance benchmarking tool)
- Implement in `benchmark/SortBenchmark.cpp`
- Support command-line arguments for configuration
- Generate output in multiple formats (console, CSV, JSON)

### 2. Test Cases to Implement

#### Input Size Variations
- Small: 10, 50, 100 elements
- Medium: 500, 1000, 5000 elements  
- Large: 10000, 50000 elements (if memory permits)

#### Input Value Ranges
- Small values: 0-10
- Medium values: 0-100
- Large values: 0-1000

#### Input Distributions
- **Random**: Uniformly distributed random integers
- **Sorted**: Already sorted ascending (best case)
- **Reverse Sorted**: Sorted descending (worst case)
- **Nearly Sorted**: 90% sorted with random swaps
- **Few Unique**: Many duplicates (e.g., only 5-10 unique values)
- **Uniform**: All elements have the same value

### 3. Metrics to Measure

#### Performance Metrics
- **Execution Time**: Wall-clock time for sort operation
- **Throughput**: Elements sorted per second
- **Speedup**: Relative to CPU baseline
- **Efficiency**: Speedup / number of processors

#### Resource Metrics (if feasible)
- Memory usage
- GPU utilization
- Network communication overhead (for MPI)
- Load balancing across nodes

### 4. Output Format

Generate a report including:
- Summary table comparing all implementations
- Performance charts (ASCII or save data for plotting)
- Statistical analysis (mean, median, std dev, min, max)
- Recommendations for when to use each implementation

### 6. Build System Integration

Update `CMakeLists.txt` to:
- Find and link Google Benchmark library
- Create `sort_benchmark` executable
- Support MPI and CUDA if needed
- Add custom target: `make benchmark-sort`

### 7. Execution Scripts

Create scripts for:
- Running benchmarks on single node
- Running distributed benchmarks across multiple nodes
- Comparing results between runs
- Generating visualization plots

### 8. Validation

Ensure benchmarks include:
- Correctness verification (sorted output is actually sorted)
- Stability checks (implementation doesn't crash)
- Consistent results across multiple runs
- Proper MPI initialization/finalization for distributed tests

## Success Criteria

- All three implementations can be benchmarked
- Clear performance differences are measurable
- Results are reproducible
- Easy to add new test cases
- Output is useful for optimization decisions

## Implementation Flexibility

**You are authorized to modify any implementation code as needed for comprehensive benchmarking:**

### Permitted Modifications
- **Add instrumentation**: Insert timing hooks, counters, and profiling code in sort implementations
- **Extend interfaces**: Add methods to `ISort` or `ISnpSimulator` interfaces to expose internal metrics
- **Enhance simulators**: Modify CPU/CUDA/MPI simulators to track:
  - Communication overhead
  - Synchronization time
  - Memory allocation/transfer time
  - Kernel execution time
  - Load balancing statistics
  - Number of neurons processed per node
- **Add debugging outputs**: Include verbose logging modes for understanding bottlenecks
- **Create wrapper classes**: Build instrumented wrappers around existing implementations without breaking them
- **Modify system configuration**: Adjust `SnpSystemConfig` to include performance metadata

### Guidelines
- Maintain backward compatibility when possible
- Document all modifications clearly
- Preserve correctness of sorting algorithm
- Use conditional compilation (`#ifdef BENCHMARK_MODE`) for invasive changes
- Keep instrumentation overhead minimal to avoid skewing results

## Notes

- The SNP sorting algorithm's performance depends on the maximum value in the input (determines simulation ticks)
- MPI implementations may require specific hostfile configuration
- CUDA implementations require GPU availability
- Consider warmup runs to exclude initialization overhead
- **Feel free to refactor or extend any code to get better measurements** - the goal is comprehensive performance analysis
