# Distributed SNP System - Project Overview

## Introduction

This project implements a high-performance computing framework for **Spiking Neural P Systems (SN P Systems)**, a bio-inspired parallel computing model based on membrane computing theory. The system provides multiple implementation backends optimized for different computational environments, from single-CPU execution to distributed multi-GPU clusters.

## What are Spiking Neural P Systems?

Spiking Neural P Systems are distributed parallel computing models inspired by the neurophysiological behavior of neurons communicating via electrical impulses (spikes). An SN P system consists of:

- **Neurons**: Processing units containing spikes (represented by the symbol `a`)
- **Synapses**: Directed connections between neurons forming a graph structure
- **Rules**: Deterministic firing and forgetting rules that govern neuron behavior
- **Global Clock**: Synchronizes parallel execution across all neurons

### Key Characteristics

- **Parallel Execution**: All neurons operate simultaneously in each time step
- **Matrix Representation**: System state and transitions are represented using linear algebra for efficient computation
- **Deterministic Behavior**: Rule execution follows a fixed priority order for predictable results

## Project Architecture

### Core Components

#### 1. **Linear Algebra Module** (`src/linear_algebra/`)

Provides matrix and vector operations with three backend implementations:

- **CPU Backend** ([CpuMatrixOps.cpp](src/linear_algebra/CpuMatrixOps.cpp)): OpenMP-accelerated operations
- **CUDA Backend** ([CudaMatrixOps.cu](src/linear_algebra/CudaMatrixOps.cu)): Single-GPU acceleration
- **MPI+CUDA Backend** ([MpiCudaMatrixOps.cu](src/linear_algebra/MpiCudaMatrixOps.cu)): Distributed multi-GPU computing

#### 2. **SNP Simulator Module** (`src/snp/`)

Implements the core SN P system simulation with multiple backends:

- **NaiveCpuSnpSimulator**: Reference CPU implementation
- **CudaSnpSimulator**: Single-GPU optimized implementation
- **SparseCudaSnpSimulator**: Optimized for sparse system matrices
- **NaiveCudaMpiSnpSimulator**: Basic distributed implementation
- **CudaMpiSnpSimulator**: Optimized distributed multi-GPU implementation

All simulators implement the [ISnpSimulator](src/snp/ISnpSimulator.hpp) interface, providing:
- `loadSystem()`: Initialize the SN P system configuration
- `step()`: Advance simulation by specified time steps
- `getGlobalState()`: Retrieve current neuron spike counts
- `reset()`: Reset to initial configuration

#### 3. **Sorting Module** (`src/sort/`)

Demonstrates practical applications of SN P systems by implementing sorting algorithms using the SNP framework. The [SnpSort](src/sort/SnpSort.cpp) class implements the [ISort](src/sort/ISort.hpp) interface.

#### 4. **System Configuration** (`src/snp/SnpSystemConfig.hpp`)

Defines data structures for constructing SN P systems:
- `SnpRule`: Firing and forgetting rules (threshold, consumption, production, delay)
- `SnpNeuron`: Neuron definition with rules and initial state
- `SnpSynapse`: Connections between neurons
- `SnpSystemConfig`: Complete system specification

## Implementation Backends

### CPU Backend
- Uses OpenMP for multi-threaded parallel execution
- Reference implementation for validation
- Best for small systems or CPU-only environments

### CUDA Backend
- Leverages NVIDIA GPUs for massive parallelism
- Kernel-based matrix operations
- Optimized memory coalescing and shared memory usage
- Ideal for medium to large systems on single-GPU machines

### MPI+CUDA Backend
- Distributes computation across multiple nodes
- Each node utilizes its GPUs via CUDA
- MPI handles inter-node communication
- Supports matrix partitioning for large-scale systems
- Best for very large systems requiring distributed memory

### Sparse Implementation
- Optimized for systems with sparse connectivity
- Uses sparse matrix representations (CSR format)
- Reduces memory footprint and computational overhead
- Ideal for realistic biological networks

## Build System

The project uses CMake with a convenient Makefile wrapper:

### Build Targets
- `make`: Build in Release mode
- `make debug`: Build with debug symbols
- `make test`: Run unit tests
- `make test-distributed`: Run MPI-distributed tests
- `make benchmark`: Run local benchmarks
- `make benchmark-distributed`: Run distributed benchmarks

### Configuration
- **C++ Standard**: C++17
- **CUDA Standard**: C++17
- **Minimum CMake**: 3.18
- **CUDA Architectures**: Configurable (default: 75 for Turing GPUs)

## Testing Infrastructure

Comprehensive test suite using Google Test:

### Test Categories
1. **Linear Algebra Tests** ([tests/linear_algebra/](tests/linear_algebra/))
   - Matrix operations correctness
   - Backend compatibility
   - Numerical precision

2. **SNP Simulator Tests** ([tests/snp/](tests/snp/))
   - System configuration loading
   - Simulation correctness
   - Multi-step execution
   - State management

3. **Sorting Tests** ([tests/sort/](tests/sort/))
   - Sorting correctness
   - Edge cases (empty, single element, duplicates)

### Distributed Testing
- Uses MPI test runner
- Validates multi-node correctness
- Checks communication patterns
- Configured via [hostfile.txt](hostfile.txt)

## Benchmarking Framework

Located in [benchmark/](benchmark/), the framework measures:

- **Execution Time**: Wall-clock time for operations
- **Throughput**: Operations per second
- **Scalability**: Performance across different system sizes
- **Backend Comparison**: CPU vs CUDA vs MPI+CUDA

### Benchmark Tools
- [SortBenchmark.cpp](benchmark/SortBenchmark.cpp): Sorting performance evaluation
- [run_benchmark.sh](scripts/run_benchmark.sh): Local benchmarking script
- [run_distributed_benchmark.sh](scripts/run_distributed_benchmark.sh): Distributed benchmarking
- [visualize_benchmarks.py](scripts/visualize_benchmarks.py): Results visualization
- [compare_benchmarks.py](scripts/compare_benchmarks.py): Performance comparison

Results are stored in [benchmark_results/](benchmark_results/) with JSON format and visualization outputs.

## Mathematical Foundation

The system evolution is governed by the transition equation:

$$C^{(k+1)} = C^{(k)} + St^{(k+1)} \odot (Iv^{(k)} \cdot M_{\Pi} + STv^{(k)})$$

Where:
- $C^{(k)}$: Configuration vector (spike counts at time $k$)
- $St^{(k)}$: Status vector (neuron open/closed state)
- $Iv^{(k)}$: Indicator vector (which rules fire)
- $M_{\Pi}$: Spiking transition matrix
- $STv^{(k)}$: External spike train input
- $\odot$: Hadamard (element-wise) product

This matrix formulation enables efficient parallel computation on GPUs and distributed systems.

## Dependencies

### Required
- **CMake** 3.18+
- **C++ Compiler**: GCC, Clang, or MSVC with C++17 support
- **CUDA Toolkit** 11.0+ (for GPU backends)
- **MPI**: OpenMPI or MPICH (for distributed backends)
- **OpenMP**: For CPU parallelization

### Optional
- **Google Test**: Automatically downloaded by CMake if not found
- **Google Benchmark**: For performance benchmarking

## Documentation

- [snp_explanation.md](snp_explanation.md): Detailed SN P systems theory and formalization
- [cuda.md](cuda.md): CUDA implementation details
- [cuda_mpi.md](cuda_mpi.md): Distributed implementation architecture
- [naive_cuda_mpi.md](naive_cuda_mpi.md): Basic distributed approach
- [prompts/](prompts/): Development prompts and specifications

## Use Cases

### Research Applications
- Modeling neural network dynamics
- Studying parallel computing models
- Algorithm design using membrane computing
- Distributed systems research

### Practical Applications
- Sorting algorithms (demonstrated in [src/sort/](src/sort/))
- Computational biology simulations
- Parallel pattern matching
- Graph algorithms

## Performance Characteristics

### CPU Backend
- **Pros**: Simple, portable, no special hardware
- **Cons**: Limited parallelism, slower for large systems
- **Best for**: Small systems (<1000 neurons)

### CUDA Backend
- **Pros**: Massive parallelism, fast for medium/large systems
- **Cons**: Requires NVIDIA GPU, single-node limitation
- **Best for**: Medium to large systems (1000-100,000 neurons)

### MPI+CUDA Backend
- **Pros**: Scales to very large systems, utilizes cluster resources
- **Cons**: Communication overhead, complex setup
- **Best for**: Very large systems (>100,000 neurons) or multi-GPU clusters

### Sparse Backend
- **Pros**: Reduced memory, faster for sparse graphs
- **Cons**: Additional complexity, overhead for dense systems
- **Best for**: Biologically realistic networks with sparse connectivity

## Getting Started

1. **Clone and Build**:
   ```bash
   git clone <repository-url>
   cd distributed-snp-new
   make
   ```

2. **Run Tests**:
   ```bash
   make test
   ```

3. **Run Demo**:
   ```bash
   make run
   ```

4. **Run Benchmarks**:
   ```bash
   make benchmark
   ```

5. **Distributed Execution**:
   ```bash
   # Edit hostfile.txt for your cluster
   make test-distributed
   make benchmark-distributed
   ```

## Contributing

When adding new features:
1. Implement the appropriate interface ([ISnpSimulator](src/snp/ISnpSimulator.hpp) or [ISort](src/sort/ISort.hpp))
2. Add unit tests in [tests/](tests/)
3. Update documentation
4. Run all tests before submitting
5. Add benchmarks for performance-critical code

## License

[License information not provided in source files]

## References

For theoretical background on Spiking Neural P Systems, see:
- [docs/snp_explanation.md](snp_explanation.md) - Complete formalization
- Academic papers on Membrane Computing and SN P Systems
- Research on parallel computing models

---

**Last Updated**: December 13, 2025
