# Role
Act as a Senior High-Performance Computing (HPC) Engineer specializing in Hybrid MPI+CUDA architectures.

# Task
Implement the `CudaMpiSnpSimulator` class in C++. This class must facilitate a distributed simulation of Spiking Neural P (SNP) Systems across multiple nodes, each equipped with an NVIDIA T4 GPU.

# Context & Constraints
1.  **Architecture:** The implementation must use a hybrid model: MPI for inter-node communication and CUDA for intra-node parallel processing.
2.  **Performance Goals:** The input size is massive (millions of neurons). You must prioritize performance over code brevity. Specifically, you must implement latency-hiding techniques for minimizing communication overhead between:
    * **Process <-> Process** 
    * **Processor <-> GPU**
    * **GPU <-> Global Memory**
3.  **Interface Strictness:** You must strictly adhere to the `ISnpSimulator.hpp` interface. Do not modify the header files or the signature of existing methods. You may add private helper methods or internal classes/structs within the implementation file.
4.  **Reference Material:**
    * I have provided `snp_explanation.tex` below for the mathematical rules of the system.
    * Use `NaiveCpuSnpSimulator.cpp` to understand the sequential logic correctness.
    * Use `CudaSnpSimulator.cu` (single-node implementation) as a baseline for the CUDA kernels. Adapt these kernels for distributed processing.
5. **Edge Cases:** Handle edge cases such as:
    * Nodes with zero neurons (should gracefully skip computation).
    * Variety of input sizes which may be too small or may not evenly divide across nodes
