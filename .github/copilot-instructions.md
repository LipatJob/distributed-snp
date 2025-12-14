# Distributed SNP Project Instructions

## Project Overview
This is a high-performance C++ library for simulating Spiking Neural P (SNP) Systems. It features multiple backends (CPU, CUDA, MPI+CUDA) to support scalable simulations. The core logic relies on matrix operations: $C(k+1) = C(k) + Sp(k) * M$.

## Architecture & Design
- **Core Abstractions**:
  - `ISnpSimulator` (`src/snp/ISnpSimulator.hpp`): Main interface for SNP simulations. Hides backend details.
  - `IMatrixOps` (`src/linear_algebra/MatrixOps.h`): Interface for matrix operations (Multiply, Hadamard, Add).
  - `ISort` (`src/sort/ISort.hpp`): Interface for sorting algorithms implemented using SNP systems.
- **Backends**:
  - **CPU**: OpenMP-accelerated (`CpuMatrixOps.cpp`, `NaiveCpuSnpSimulator.cpp`).
  - **CUDA**: Single-GPU acceleration (`CudaMatrixOps.cu`, `CudaSnpSimulator.cu`).
  - **MPI+CUDA**: Distributed multi-GPU computing (`MpiCudaMatrixOps.cu`, `CudaMpiSnpSimulator.cu`).
  - **Sparse**: Optimized CUDA implementation for sparse matrices (`SparseCudaSnpSimulator.cu`).

## Build & Test Workflow
- **Build System**: CMake is the primary build system (requires 3.18+).
  - Use `make` wrappers for convenience: `make`, `make debug`, `make release`.
  - Direct CMake: `mkdir build && cd build && cmake .. && cmake --build .`
- **Testing**:
  - Google Test is used. Run with `make test` or `ctest --output-on-failure` in the build directory.
  - Tests are located in `tests/`.
- **Benchmarking**:
  - Scripts in `scripts/` (e.g., `run_benchmark.sh`) automate performance testing.
  - Benchmarks are located in `benchmark/`.

## Coding Conventions
- **Language**: C++17 standard.
- **CUDA**:
  - Kernel launches and device memory management should be encapsulated within `.cu` files.
  - Use `checkCudaErrors` or similar error handling macros (if available) for CUDA calls.
- **MPI**:
  - MPI calls should be isolated in MPI-specific implementations (`MpiCudaMatrixOps.cu`, `CudaMpiSnpSimulator.cu`).
  - Ensure proper synchronization (barriers) when necessary.
- **Interfaces**:
  - Always program to interfaces (`ISnpSimulator`, `IMatrixOps`) to ensure backend interchangeability.
  - Use factory functions (e.g., `createMatrixOps`) to instantiate concrete classes.

## Key Files
- `src/main.cpp`: Demo application entry point.
- `src/snp/SnpSystemConfig.hpp`: Configuration structure for SNP systems.
- `src/linear_algebra/MatrixOps.h`: Matrix operations interface and factory.
- `CMakeLists.txt`: Project build configuration and dependency management.
