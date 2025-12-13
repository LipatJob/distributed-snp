# Role
Act as a Senior High-Performance Computing (HPC) Engineer specializing in performance optimization for Hybrid MPI+CUDA architectures.

# Task
Optimize the existing `CudaMpiSnpSimulator` implementation to achieve maximum performance on multi-node clusters with NVIDIA T4 GPUs. The current implementation is functional but has significant performance bottlenecks that need to be addressed.

# Context
You are working with a distributed Spiking Neural P (SNP) System simulator that processes **millions of neurons** across multiple compute nodes. The current implementation successfully handles correctness but requires deep optimization across all layers of the memory hierarchy and communication subsystems.

**Reference Materials Provided:**
- `CudaMpiSnpSimulator.cu` - Current working implementation
- `snp_explanation.md` - Mathematical formulation of SNP systems
- `cuda_mpi_snp.prompt.md` - Original implementation requirements
- `SnpSimulatorTest.cpp` - Tests that the SNP Simulator must satisfy

# Current Performance Bottlenecks Identified

## 1. **Memory Transfer Inefficiencies**
- **Problem:** Currently using standard `cudaMemcpy` for all host-device transfers during communication phases
- **Impact:** Significant latency due to synchronous blocking transfers
- **Target Areas:**
  - Export buffer downloads (Device → Host)
  - Import buffer uploads (Host → Device)
  - MPI send/receive buffer management

## 2. **Communication Overhead**
- **Problem:** Non-overlapped computation and communication; all-to-all exchange pattern may be suboptimal
- **Impact:** Idle GPU time during MPI operations; excessive barrier synchronization
- **Target Areas:**
  - MPI_Isend/Irecv usage patterns
  - Potential for CUDA-aware MPI
  - Computation-communication overlap opportunities

## 3. **GPU Memory Access Patterns**
- **Problem:** Potential uncoalesced memory accesses; atomicAdd contention in spike propagation
- **Impact:** Low memory bandwidth utilization; serialization in atomic operations
- **Target Areas:**
  - `kPropagateLocal` kernel memory access patterns
  - `kApplyImports` atomic operations
  - Structure-of-Arrays (SoA) layout optimization

## 4. **Kernel Launch Overhead**
- **Problem:** Multiple sequential kernel launches with full device synchronization between each
- **Impact:** GPU underutilization; excessive kernel launch latency
- **Target Areas:**
  - Kernel fusion opportunities
  - CUDA streams for concurrent execution
  - Reduction of `cudaDeviceSynchronize()` calls

# Optimization Requirements

1. **Implement Pinned Memory for MPI Buffers**
   - Replace `std::vector` with `cudaMallocHost()` pinned allocations
   - Enable faster PCIe transfers for communication buffers
   - Measure and report transfer time improvements

2. **Optimize Kernel Memory Access**
   - Analyze and fix coalescing issues in synapse kernels
   - Consider shared memory for frequently accessed data
   - Reduce atomic contention through better algorithms (e.g., local reduction before global atomic)

3. **Overlap Computation and Communication**
   - Use CUDA streams to pipeline operations
   - Start MPI transfers while kernels are still executing
   - Hide communication latency behind computation

4. **Memory Hierarchy Optimization**
   - Strategic use of constant memory for read-only data
   - Texture memory for sparse access patterns
   - L2 cache optimization through access patterns

# Implementation Guidelines

## Code Quality Requirements
- **Maintain correctness:** All optimizations must preserve simulation accuracy
- **Documentation:** Comment all non-obvious optimizations with rationale

## Specific Technical Constraints
- **Interface Preservation:** Do not modify `ISnpSimulator.hpp` interface
- **Backward Compatibility:** Code must work on T4 GPUs (Compute Capability 7.5)
- **Error Handling:** Maintain robust error checking with graceful degradation

# Deliverables

## 1. Optimized Implementation
- Modified `CudaMpiSnpSimulator.cu` with all optimizations applied

# Additional Considerations
- Consider algorithmic improvements (e.g., sparse representations if applicable)