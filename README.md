# new-snp

Matrix operations library with CPU, CUDA, and MPI+CUDA backends.

## Features

- **CPU Backend**: OpenMP-accelerated matrix operations
- **CUDA Backend**: GPU-accelerated matrix operations
- **MPI+CUDA Backend**: Distributed GPU computing with MPI
- Comprehensive test suite using Google Test

## Requirements

- CMake 3.18 or higher
- C++17 compatible compiler (GCC, Clang, MSVC)
- NVIDIA CUDA Toolkit 11.0 or higher
- OpenMPI or MPICH
- OpenMP support
- Google Test (automatically downloaded if not found)

## Building the Project

### Using Make (Recommended)

```bash
# Build with default settings (Release mode)
make

# Build in Debug mode
make debug

# Build in Release mode  
make release

# Run the demo application
make run

# Run tests
make test

# Clean build artifacts
make clean

# Install libraries and headers
make install

# Show all available targets
make help
```

### Using CMake directly

```bash
# Configure
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j

# Run tests
ctest --output-on-failure

# Install
cmake --install .
```

## Build Options

You can customize the build using environment variables:

```bash
# Set build type
BUILD_TYPE=Debug make

# Set CUDA architectures (based on your GPU)
CUDA_ARCH=75,80,86 make

# Set number of parallel jobs
JOBS=8 make

# Use specific compilers
CXX=clang++ NVCC=nvcc MPICC=mpicxx make
```

### CMake Options

- `BUILD_TESTS`: Enable/disable test building (default: ON)
- `CMAKE_CUDA_ARCHITECTURES`: CUDA compute capabilities to target (default: 75;80;86)

```bash
cmake .. -DBUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES="75;80"
```

## CUDA Architecture Selection

Adjust the CUDA architecture based on your GPU:

- **75**: RTX 20 series (Turing)
- **80**: A100 (Ampere)
- **86**: RTX 30 series (Ampere)
- **89**: RTX 40 series (Ada Lovelace)
- **90**: H100 (Hopper)

Find your GPU architecture: https://developer.nvidia.com/cuda-gpus

## Project Structure

```
new-snp/
├── CMakeLists.txt          # CMake build configuration
├── Makefile                # Convenience wrapper around CMake
├── README.md               # This file
├── src/
│   ├── linear_algebra/     # Matrix operations library
│   │   ├── MatrixOps.h
│   │   ├── CpuMatrixOps.cpp
│   │   ├── CudaMatrixOps.cu
│   │   ├── KernelMatrixOps.cuh
│   │   └── MpiCudaMatrixOps.cu
│   └── snp/               # SNP-specific code
└── tests/
    ├── linear_algebra/     # Linear algebra tests
    │   └── MatrixOps.cpp
    └── snp/               # SNP tests
```

## Running the Demo

A demo application is included that performs matrix multiplication using all backends:

### Local Execution

```bash
# Run with default settings (4 MPI processes)
make run

# Or run directly
mpirun -np 4 ./build/matrix_demo

# Adjust number of processes
mpirun -np 8 ./build/matrix_demo
```

### Distributed Execution

To run across multiple nodes:

```bash
# Configure nodes (edit Makefile or set environment variables)
export NODES="node1 node2 node3 node4"
export REMOTE_USER="username"
export REMOTE_DIR="~/new-snp"

# Check connectivity to all nodes
make check-nodes

# Distribute binary to all nodes
make distribute

# Generate MPI hostfile
make generate-hostfile

# Run on distributed nodes
make run-distributed

# Or do everything in one step
make run-distributed
```

#### Distribution Options

You can customize the distribution using environment variables:

```bash
# Distribute to specific nodes
NODES="gpu-node1 gpu-node2 gpu-node3" make distribute

# Use different username
REMOTE_USER=admin make distribute

# Custom remote directory
REMOTE_DIR=/opt/new-snp make distribute

# Custom hostfile
HOSTFILE=my_hosts.txt make generate-hostfile
```

The demo will:
- Perform matrix multiplication on CPU, CUDA, and MPI+CUDA backends
- Display execution time for each backend
- Verify that all backends produce identical results

## Running Tests

Tests are built automatically and use MPI for distributed testing:

```bash
# Run all tests
make test

# Or using CMake/CTest directly
cd build
ctest --output-on-failure

# Run with different number of MPI processes
mpirun -np 4 ./build/test_matrix_ops
```

## Usage Example

```cpp
#include "MatrixOps.h"

// Create matrix operations with CPU backend
auto ops = createMatrixOps<float>(BackendType::CPU);

// Or use CUDA backend
auto cuda_ops = createMatrixOps<float>(BackendType::CUDA);

// Or use MPI+CUDA backend
auto mpi_ops = createMatrixOps<float>(BackendType::MPI_CUDA);

// Perform matrix multiplication
ops->multiply(A, B, C, m, n, k);
```

## Troubleshooting

### CUDA not found
Ensure CUDA toolkit is installed and `nvcc` is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### MPI not found
Install OpenMPI or MPICH:
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev

# macOS
brew install open-mpi
```

### OpenMP not supported
Ensure your compiler supports OpenMP:
```bash
# GCC includes OpenMP by default
# For Clang on macOS:
brew install libomp
```

## License

[Add your license here]
