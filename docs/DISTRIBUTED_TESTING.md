# Distributed Testing Guide

This guide explains how to run tests across multiple hosts using MPI for the distributed matrix operations library.

## Overview

The test suite includes tests for three implementations:
- **CPU**: Multi-threaded CPU implementation using OpenMP
- **CUDA**: Single-GPU CUDA implementation
- **MPI+CUDA**: Distributed multi-GPU implementation using MPI for communication

## Prerequisites

### On All Nodes
1. **CUDA Toolkit** (12.x or compatible)
2. **MPI** (OpenMPI or MPICH)
3. **C++ Compiler** with C++17 support
4. **Network connectivity** between nodes

### SSH Configuration
Ensure passwordless SSH is configured between nodes:
```bash
# Generate SSH key (if not already done)
ssh-keygen -t rsa -b 4096

# Copy to remote nodes
ssh-copy-id user@10.0.0.2
```

## Running Tests

### Local Tests (Single Node)
Run tests on the local machine only:
```bash
make test
```

### Distributed Tests (Multiple Nodes)

#### Option 1: Using Make Targets (Recommended)
```bash
# Deploy and run tests across all configured nodes
make test-distributed
```

This will:
1. Build the test executable
2. Deploy it to remote nodes (currently: 10.0.0.2)
3. Deploy required libraries
4. Run tests with MPI across all nodes

#### Option 2: Manual Deployment
```bash
# Build tests
make build

# Deploy to remote hosts
./scripts/deploy_tests.sh

# Run distributed tests
./scripts/run_distributed_tests.sh
```

#### Option 3: Direct MPI Command
```bash
# After building and deploying
mpirun -np 2 \
       --host localhost,10.0.0.2 \
       --allow-run-as-root \
       --oversubscribe \
       -x LD_LIBRARY_PATH \
       /home/shared/distributed-snp-new/build/test_matrix_ops
```

## Test Distribution

The tests are distributed as follows:

### MPI Process Distribution
- **Process 0** (localhost): Runs CPU, CUDA, and MPI+CUDA tests
- **Process 1** (10.0.0.2): Participates in MPI+CUDA distributed operations

### Test Categories

1. **CPU Tests** - Run on Process 0 only
   - Basic correctness
   - Edge cases (1x1 matrices, zeros)
   - Different matrix sizes

2. **CUDA Tests** - Run on Process 0 only
   - GPU kernel correctness
   - Memory transfer validation
   - Large matrix operations

3. **MPI+CUDA Tests** - Run across all processes
   - Distributed matrix multiplication
   - Distributed element-wise operations
   - Cross-process communication validation
   - Multi-GPU coordination

## Configuration

### Hosts Configuration
Edit the host list in [`CMakeLists.txt`](CMakeLists.txt#L125):
```cmake
add_test(NAME MatrixOpsTest 
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 
                 --host localhost,10.0.0.2
                 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_matrix_ops> 
                 ${MPIEXEC_POSTFLAGS} --allow-run-as-root --oversubscribe)
```

### Remote Host Setup
Edit [`scripts/deploy_tests.sh`](scripts/deploy_tests.sh):
```bash
REMOTE_HOST="10.0.0.2"
REMOTE_USER="${REMOTE_USER:-$(whoami)}"
```

### Number of Processes
Adjust in [`scripts/run_distributed_tests.sh`](scripts/run_distributed_tests.sh):
```bash
NUM_PROCS=2
HOSTS="localhost,10.0.0.2"
```

## Test Output

### Successful Run
```
[==========] Running 30 tests from 6 test suites.
[----------] Global test environment set-up.
[----------] 10 tests from MatrixMultiplyTest/AllBackends
[ RUN      ] MatrixMultiplyTest/AllBackends.BasicMultiplication/CPU
[       OK ] MatrixMultiplyTest/AllBackends.BasicMultiplication/CPU (5 ms)
...
[==========] 30 tests from 6 test suites ran. (1234 ms total)
[  PASSED  ] 30 tests.
```

### MPI Process Information
Each MPI process prints its rank during initialization:
- Process 0 validates results and reports test outcomes
- Process 1 participates in computation but doesn't report to stdout

## Troubleshooting

### "unable to access or execute" Error
The test executable is not available on the remote host:
```bash
# Solution: Deploy tests
make deploy-tests
```

### "No slots available" Error
Too many processes requested:
```bash
# Add --oversubscribe flag to mpirun command
# Or reduce number of processes in configuration
```

### SSH Connection Issues
```bash
# Check connectivity
ssh user@10.0.0.2

# Verify SSH keys
ssh-add -l

# Test MPI connectivity
mpirun -np 2 --host localhost,10.0.0.2 hostname
```

### CUDA Device Issues
```bash
# Check GPU availability on all nodes
ssh 10.0.0.2 nvidia-smi

# Verify CUDA runtime library path
ssh 10.0.0.2 "echo \$LD_LIBRARY_PATH"
```

### Library Path Issues
Ensure LD_LIBRARY_PATH is exported to remote processes:
```bash
mpirun ... -x LD_LIBRARY_PATH ...
```

## Advanced: Scaling to More Nodes

To add more nodes:

1. **Update CMakeLists.txt**:
```cmake
add_test(NAME MatrixOpsTest 
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                 --host localhost,10.0.0.2,10.0.0.3,10.0.0.4
                 ...)
```

2. **Update deployment script**:
```bash
# In scripts/deploy_tests.sh, add loop for multiple hosts
for host in 10.0.0.2 10.0.0.3 10.0.0.4; do
    ssh ${REMOTE_USER}@${host} "mkdir -p ${BUILD_DIR}"
    scp ${BUILD_DIR}/test_matrix_ops ${REMOTE_USER}@${host}:${BUILD_DIR}/
done
```

3. **Update MpiCudaMatrixOps**:
   - Ensure matrix dimensions are divisible by number of processes
   - For M processes, M must divide matrix rows evenly

## Performance Notes

- **Network Latency**: MPI communication adds overhead; larger matrices benefit more from distribution
- **GPU Memory**: Each process needs sufficient GPU memory for its partition
- **Load Balancing**: Current implementation uses 1D row decomposition; ensure even distribution
- **Bandwidth**: Network bandwidth between nodes affects MPI+CUDA performance

## Monitoring

### During Test Execution
```bash
# On each node, monitor GPU usage
watch -n 1 nvidia-smi

# Monitor MPI processes
ps aux | grep test_matrix_ops

# Check network traffic
iftop -i eth0
```

### Test Timing
GoogleTest provides timing information for each test. Use this to identify bottlenecks:
- CPU tests: Should complete in < 10ms for small matrices
- CUDA tests: Include GPU transfer overhead (5-20ms typical)
- MPI+CUDA tests: Include network communication (20-100ms depending on network)

## See Also

- [Main README](../README.md) - Project overview
- [MatrixOps.h](../src/linear_algebra/MatrixOps.h) - Interface documentation
- [MpiCudaMatrixOps.cu](../src/linear_algebra/MpiCudaMatrixOps.cu) - Distributed implementation
