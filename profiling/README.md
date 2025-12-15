# SNP System Profiling

This directory contains profiling tools for analyzing performance of different SNP (Spiking Neural P System) implementations using NVIDIA Nsight Systems.

## Overview

The profiling system allows you to:
- Profile CPU, CUDA, and MPI implementations of SNP systems
- Capture detailed performance metrics (CPU utilization, CUDA kernels, MPI communication)
- Compare different implementations side-by-side
- Analyze distributed implementations across multiple nodes
- Identify performance bottlenecks in sorting operations

## Requirements

### Software Dependencies

1. **NVIDIA Nsight Systems** - For profiling
   ```bash
   # Install on Ubuntu/Debian
   sudo apt install nsight-systems-cli
   
   # Or download from NVIDIA:
   # https://developer.nvidia.com/nsight-systems
   ```

2. **CUDA Toolkit** - For GPU implementations
   ```bash
   sudo apt install nvidia-cuda-toolkit
   ```

3. **MPI** - For distributed implementations
   ```bash
   sudo apt install mpich
   ```

### Hardware Requirements

- **For CPU profiling**: Any modern CPU
- **For CUDA profiling**: NVIDIA GPU with compute capability 7.5+
- **For MPI profiling**: Multiple nodes with network connectivity (or single node with oversubscription)

## Quick Start

### 1. Build the Profiling Tool

```bash
make build
```

This will compile the `snp_profile` executable in `build/snp_profile`.

### 2. Profile Locally

Profile all implementations on a single machine:

```bash
make profile
```

Profile specific implementations:

```bash
make profile-cpu          # CPU implementation only
make profile-cuda         # CUDA implementation only
make profile-mpi          # MPI implementations (requires 2+ processes)
```

### 3. Profile Across Distributed Nodes

```bash
# Setup hostfile first
make generate-hostfile

# Profile MPI implementations across nodes
make profile-distributed
```

### 4. View Results

Profiling results are saved as `.nsys-rep` files in `profiling/results/`.

```bash
# View graphically with Nsight Systems GUI
nsys-ui profiling/results/snp_profile_cuda_20231213_120000.nsys-rep

# Generate text report
nsys stats profiling/results/snp_profile_cuda_20231213_120000.nsys-rep
```

## Profiling Targets

### Makefile Targets

| Target | Description | MPI Processes |
|--------|-------------|---------------|
| `make profile` | Profile all implementations locally | 1 (or 2 for MPI) |
| `make profile-cpu` | Profile CPU implementation | 1 |
| `make profile-cuda` | Profile CUDA implementation | 1 |
| `make profile-mpi` | Profile MPI implementations | 2 (default) |
| `make profile-all` | Profile all with various configs | Variable |
| `make profile-distributed` | Profile across nodes | 2+ (from hostfile) |

### Available Implementations

The profiling tool supports the following SNP implementations:

1. **NaiveCpuSnp** - Simple CPU-based implementation
   - Single process
   - Good baseline for comparison
   - No GPU required

2. **CudaSnp** - CUDA-accelerated implementation
   - Single process
   - Requires NVIDIA GPU
   - GPU kernel profiling

3. **SparseCudaSnp** - Sparse matrix CUDA implementation
   - Single process
   - Requires NVIDIA GPU
   - Optimized for sparse SNP systems

4. **NaiveCudaMpiSnp** - Distributed CUDA+MPI (naive)
   - Multiple processes (2+)
   - Requires NVIDIA GPU on each node
   - MPI communication profiling

5. **CudaMpiSnp** - Optimized distributed CUDA+MPI
   - Multiple processes (2+)
   - Requires NVIDIA GPU on each node
   - Advanced MPI communication patterns

## Advanced Usage

### Custom Profiling Script

You can use the profiling script directly for more control:

```bash
# Profile specific implementation
./scripts/run_snp_profiling.sh -i cuda

# Profile with custom number of MPI processes
./scripts/run_snp_profiling.sh -i mpi -n 4

# Profile with custom nsys options
./scripts/run_snp_profiling.sh -i all --nsys-opts "--trace=cuda,mpi,nvtx"

# Profile without hostfile (localhost only)
./scripts/run_snp_profiling.sh -i mpi -n 2 --no-hostfile

# Custom output prefix
./scripts/run_snp_profiling.sh -i cuda -o my_custom_profile
```

### Script Options

```
Usage: ./scripts/run_snp_profiling.sh [options]

Options:
  -i, --implementation NAME  Implementation to profile (default: all)
                            Options: cpu, cuda, sparse-cuda, naive-cuda-mpi,
                                     cuda-mpi, mpi, all
  -n, --num-procs N         Number of MPI processes (default: 2)
  --hostfile FILE           Path to MPI hostfile (default: ./hostfile.txt)
  --no-hostfile             Don't use hostfile, run all on localhost
  --nsys-opts "OPTIONS"     Additional nsys options (default: none)
  -o, --output PREFIX       Output file prefix (default: snp_profile)
  -h, --help                Show help message
```

## Profiling Configuration

### Test Array Size

The profiling tool uses **medium-sized arrays** (1000 elements) by default, matching the benchmark suite:

- Array Size: 1000 elements
- Max Value: 1000
- Distribution: Random

This provides a good balance between execution time and meaningful profiling data.

### Adding New Implementations

To add a new SNP implementation to profiling:

1. **Add factory function** in `src/profiling/SnpProfile.cpp`:
   ```cpp
   std::unique_ptr<ISort> createMyNewSnpSort();
   ```

2. **Add to profileConfigs** vector:
   ```cpp
   if (targetImpl.empty() || targetImpl == "my-new" || targetImpl == "all") {
       profileConfigs.push_back({
           "MyNewSnp",
           createMyNewSnpSort,
           MEDIUM_SIZE,
           MEDIUM_MAX,
           false  // true if requires MPI
       });
   }
   ```

3. **Update profiling script** `scripts/run_snp_profiling.sh`:
   ```bash
   my-new)
       run_profiling "my-new"
       ;;
   ```

4. **Rebuild**:
   ```bash
   make build
   ```

## Understanding Profiling Output

### Nsight Systems Report

The `.nsys-rep` files contain:

- **CPU Timeline**: CPU thread activity, system calls
- **CUDA Timeline**: GPU kernel launches, memory transfers
- **MPI Timeline**: MPI communication calls (Send, Recv, Barrier, etc.)
- **NVTX Markers**: Custom annotations (if used)

### Key Metrics to Analyze

1. **Computation Time**
   - CUDA kernel execution time
   - CPU execution time
   - Ratio of compute to communication

2. **Communication Overhead** (for MPI implementations)
   - MPI_Send/MPI_Recv time
   - MPI_Barrier synchronization
   - Network transfer time

3. **GPU Utilization**
   - Kernel occupancy
   - Memory bandwidth utilization
   - Warp efficiency

4. **Load Balancing** (for distributed)
   - Time differences between ranks
   - Idle time waiting for communication

## Distributed Profiling

### Setup for Multiple Nodes

1. **Create/Update hostfile**:
   ```bash
   # Edit hostfile.txt
   localhost slots=2
   10.0.0.2 slots=2
   10.0.0.3 slots=2
   ```

2. **Ensure SSH access** to all nodes without password:
   ```bash
   ssh-copy-id user@10.0.0.2
   ssh-copy-id user@10.0.0.3
   ```

3. **Run distributed profiling**:
   ```bash
   make profile-distributed NUM_PROCS=6
   ```

### Binary Distribution

For distributed implementations, the profiling script automatically:
1. Copies the `snp_profile` executable to all remote nodes
2. Creates necessary directories on remote nodes
3. Runs MPI with the specified hostfile

This ensures all nodes have the correct binary before profiling starts.

## Troubleshooting

### nsys command not found

Install NVIDIA Nsight Systems:
```bash
sudo apt install nsight-systems-cli
```

### MPI Error: "slots" issue

Update your hostfile with slot specifications:
```
localhost slots=2
```

### CUDA Error: No GPU found

Ensure:
1. NVIDIA GPU is present: `nvidia-smi`
2. CUDA drivers are installed
3. Running on a node with GPU (for distributed)

### Profiling file too large

Reduce traced events:
```bash
./scripts/run_snp_profiling.sh -i cuda --nsys-opts "--trace=cuda"
```

### MPI distributed profiling hangs

Check:
1. Network connectivity: `ping 10.0.0.2`
2. SSH access: `ssh 10.0.0.2 echo "test"`
3. MPI installation on all nodes
4. Firewall settings

## Results Directory Structure

```
profiling/results/
├── snp_profile_cpu_20231213_120000.nsys-rep
├── snp_profile_cuda_20231213_120100.nsys-rep
├── snp_profile_sparse-cuda_20231213_120200.nsys-rep
├── snp_profile_naive-cuda-mpi_20231213_120300.nsys-rep
└── snp_profile_cuda-mpi_20231213_120400.nsys-rep
```

Each `.nsys-rep` file contains the complete profiling data for that implementation run.

## Integration with Development Workflow

### Typical Workflow

1. **Develop** a new SNP implementation
2. **Benchmark** it using `make benchmark-sort`
3. **Profile** it using `make profile-cuda` (or appropriate target)
4. **Analyze** bottlenecks in Nsight Systems GUI
5. **Optimize** based on profiling insights
6. **Repeat** steps 2-5

### Comparing Implementations

To compare two implementations:

1. Profile both:
   ```bash
   make profile-cpu
   make profile-cuda
   ```

2. Open both in Nsight Systems:
   ```bash
   nsys-ui profiling/results/snp_profile_cpu_*.nsys-rep profiling/results/snp_profile_cuda_*.nsys-rep
   ```

3. Compare side-by-side in the GUI

## References

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [MPI Performance Analysis](https://www.mcs.anl.gov/research/projects/perfvis/)
- [CUDA Profiling Best Practices](https://docs.nvidia.com/cuda/profiler-users-guide/)

## Support

For issues or questions about profiling:
1. Check this README
2. Review profiling script help: `./scripts/run_snp_profiling.sh --help`
3. Check NVIDIA Nsight Systems documentation
