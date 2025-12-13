# SNP Simulator Profiling Guide

This guide explains how to profile the SNP Simulator implementations using NVIDIA Nsight Systems.

## Overview

The profiling setup includes:

1. **snp_profile** - A dedicated profiling executable that runs SNP simulators on medium-sized arrays
2. **run_profile.sh** - A wrapper script to run the profiler with MPI in a format suitable for Nsight Systems
3. **Make targets** - Easy commands to build and distribute the profiler

## Architecture

### Easy Extensibility

The profiling system is designed to make it trivial to add new simulator implementations:

```cpp
// In src/snp/snp_profile.cpp

void RegisterSimulators() {
    gSimulatorFactories["NaiveCudaMpiSnp"] = createNaiveCudaMpiSimulator;
    gSimulatorFactories["CudaMpiSnp"] = createCudaMpiSimulator;
    // To add a new simulator:
    // gSimulatorFactories["MyNewSimulator"] = createMyNewSimulator;
}
```

Simply add a new factory function to the `gSimulatorFactories` map, and it's automatically available for profiling.

### Medium-Sized Test Arrays

The profiler uses arrays of **1000 elements** by default (configurable), which matches the "Medium" test suite from `SortBenchmark.cpp`:

```cpp
const std::vector<TestConfig> Medium = {
    {"Medium_Sort", 1000, 1000, Distribution::SORTED, 5},
    {"Medium_RevSort", 1000, 1000, Distribution::REVERSE_SORTED, 5},
    {"Medium_Rand", 1000, 1000, Distribution::RANDOM, 5},
};
```

## Usage

### 1. Build the Profiler

```bash
# Build just the profiler
make profile-build

# Or as part of full build
make all
```

The binary will be at `build/snp_profile`.

### 2. Simple Local Profiling

```bash
# Run with default settings (CudaMpiSnp, 1000 elements, 3 iterations)
mpirun -np 2 ./build/snp_profile

# Profile NaiveCudaMpiSnp with 5000 element array
mpirun -np 2 ./build/snp_profile -s NaiveCudaMpiSnp -n 5000

# Run with 4 MPI processes and 10 iterations
mpirun -np 4 ./build/snp_profile -i 10
```

### 3. Using NVIDIA Nsight Systems

#### Via UI:

1. Open NVIDIA Nsight Systems
2. Create a new profile session
3. Set the command to:
   ```
   mpirun -np 2 --hostfile /path/to/hostfile.txt /path/to/build/snp_profile
   ```
4. Configure MPI settings in Nsight Systems UI
5. Click "Start"

#### Via provided script:

```bash
# Build, distribute, and prepare for Nsight Systems
make profile-prepare

# This outputs helpful instructions and copies the profiler to remote nodes

# Then run:
./scripts/run_profile.sh [options]
```

#### Via command line (nsys):

```bash
# Profile with Nsight Systems command-line tool
nsys profile -o profile_results mpirun -np 2 ./build/snp_profile
```

### 4. Distributed Profiling

```bash
# Distribute profiler to all configured nodes
make profile-distribute

# Or prepare everything at once
make profile-prepare

# Then on remote node, run:
./scripts/run_profile.sh -s CudaMpiSnp -n 5000
```

## Profiler Options

The `snp_profile` executable accepts the following options:

```
-s, --simulator <name>   Simulator to profile
                         Options: CudaMpiSnp, NaiveCudaMpiSnp
                         Default: CudaMpiSnp

-n, --size <size>        Array size for sorting
                         Default: 1000

-i, --iterations <iter>  Number of profiling iterations
                         Default: 3

-h, --help               Show help message
```

### Example Commands

```bash
# Profile CudaMpiSnp with default parameters
mpirun -np 2 ./build/snp_profile

# Profile NaiveCudaMpiSnp with larger array
mpirun -np 2 ./build/snp_profile -s NaiveCudaMpiSnp -n 5000

# Compare with 4 processes
mpirun -np 4 ./build/snp_profile

# Run just 1 iteration for quick test
mpirun -np 2 ./build/snp_profile -i 1
```

## run_profile.sh Script

The `scripts/run_profile.sh` script is a convenient wrapper that:

1. Checks for profiler binary existence
2. Generates hostfile if needed
3. Runs MPI profiler with proper configuration
4. Supports all the same options as `snp_profile`

### Usage

```bash
# Make script executable (done automatically by make)
chmod +x scripts/run_profile.sh

# Profile with default settings
./scripts/run_profile.sh

# Profile NaiveCudaMpiSnp with 4 processes
./scripts/run_profile.sh -s NaiveCudaMpiSnp -np 4

# Profile with larger array and more iterations
./scripts/run_profile.sh -n 5000 -i 5 -np 2

# Show available options
./scripts/run_profile.sh --help
```

## Workflow for Distributed Profiling

1. **Setup**:
   ```bash
   make profile-prepare
   ```
   This builds, distributes, and configures everything needed.

2. **On local machine**, use Nsight Systems UI to connect and profile

3. **Or on remote machine**, run:
   ```bash
   cd ~/distributed-snp-new
   ./scripts/run_profile.sh -s CudaMpiSnp -n 5000
   ```

## Adding New Simulators

To profile a new simulator implementation:

1. Create factory function in your simulator's `.cu` file:
   ```cpp
   std::unique_ptr<ISnpSimulator> createMyNewSimulator() {
       return std::make_unique<MyNewSimulator>();
   }
   ```

2. Add declaration in `src/snp/ISnpSimulator.hpp`:
   ```cpp
   std::unique_ptr<ISnpSimulator> createMyNewSimulator();
   ```

3. Register in `src/snp/snp_profile.cpp`:
   ```cpp
   void RegisterSimulators() {
       // ... existing registrations
       gSimulatorFactories["MyNewSimulator"] = createMyNewSimulator;
   }
   ```

4. Rebuild:
   ```bash
   make profile-build
   ```

5. Use:
   ```bash
   mpirun -np 2 ./build/snp_profile -s MyNewSimulator
   ```

## Profiling Tips

### Performance Metrics

The profiler prints a performance report including:
- Simulation execution time
- Communication time (for MPI implementations)
- Compute vs communication breakdown

### Array Sizes

Match your profiling to your use case:

- **Small** (100 elements): Quick profiling, minimal GPU utilization
- **Medium** (1000 elements): **Default**, good balance
- **Large** (5000+ elements): Full GPU utilization, longer runs

### Iteration Count

- Default (3 iterations): Good for variance estimation
- 1 iteration: Quick test run
- 5-10 iterations: Better statistical confidence

### MPI Process Count

- 2 processes: **Default**, good for local testing and small clusters
- 4-8 processes: Typical cluster configuration
- Match to your target deployment

## Files

| File | Purpose |
|------|---------|
| `src/snp/snp_profile.cpp` | Main profiling program source |
| `scripts/run_profile.sh` | MPI wrapper script for profiling |
| `Makefile` | `profile-build`, `profile-distribute`, `profile-prepare` targets |
| `CMakeLists.txt` | Build configuration for `snp_profile` executable |

## Troubleshooting

### Profiler not found

```bash
# Error: snp_profile not found at build/snp_profile
# Solution:
make profile-build
```

### MPI connection issues

```bash
# Check connectivity to nodes
make check-nodes

# Generate/verify hostfile
make generate-hostfile
cat hostfile.txt
```

### Nsight Systems integration issues

1. Ensure Nsight Systems is installed
2. Check that MPI path is correct in Nsight Systems settings
3. Run simple test first:
   ```bash
   nsys profile -o test mpirun -np 2 /bin/echo "Test"
   ```

### Simulator not available

```bash
# Check available simulators
./build/snp_profile --help

# Add new simulator to RegisterSimulators() in snp_profile.cpp
# Rebuild: make profile-build
```

## Performance Analysis Example

```bash
# Collect baseline with CudaMpiSnp
mpirun -np 2 ./build/snp_profile -s CudaMpiSnp -n 1000 -i 5

# Compare with NaiveCudaMpiSnp
mpirun -np 2 ./build/snp_profile -s NaiveCudaMpiSnp -n 1000 -i 5

# Profile with Nsight Systems for detailed metrics
nsys profile -o cudampi_1000 mpirun -np 2 ./build/snp_profile -s CudaMpiSnp -n 1000
nsys profile -o naive_1000 mpirun -np 2 ./build/snp_profile -s NaiveCudaMpiSnp -n 1000

# View results in Nsight Systems UI
# File -> Open -> cudampi_1000.nsys-rep
# File -> Open -> naive_1000.nsys-rep
```

## References

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Benchmark Suite](../benchmark/README.md)
- [SNP Simulator Documentation](snp_explanation.md)
