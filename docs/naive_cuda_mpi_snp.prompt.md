# Role
Act as a Senior High-Performance Computing (HPC) Engineer specializing in Hybrid MPI+CUDA architectures.

# Task
Implement the `NaiveCudaMpiSnpSimulator` class in C++. This class must facilitate a distributed simulation of Spiking Neural P (SNP) Systems across multiple nodes, each equipped with an NVIDIA T4 GPU.

# Context & Constraints
1.  **Architecture:** The implementation must use a hybrid model: MPI for inter-node communication and CUDA for intra-node parallel processing.
2.  **Goals:** The goal is to create a straightforward, easy-to-understand implementation of the SNP simulation using CUDA and MPI. Prioritize code clarity and correctness over performance optimizations. Specifically, ensure proper data transfer and communication between:
    * **Process <-> Process**
    * **Processor <-> GPU**
    * **GPU <-> Global Memory**
3.  **Interface Strictness:** You must strictly adhere to the `ISnpSimulator.hpp` interface. Do not modify the header files or the signature of existing methods. You may add private helper methods or internal classes/structs within the implementation file. Also update the factory methods in the interface, CMakeLists, tests, and benchmarks to include this new class.
4.  **Reference Material:**
    * I have provided `snp_explanation.tex` below for the mathematical rules of the system.
    * Use `NaiveCpuSnpSimulator.cpp` to understand the sequential logic correctness.
    * Use `CudaSnpSimulator.cu` (single-node implementation) as a baseline for the CUDA kernels. You must adapt these kernels to work in a multi-node MPI environment.
5. **Edge Cases:** Handle edge cases such as:
    * Nodes with zero neurons (should gracefully skip computation).
    * Variety of input sizes which may be too small or may not evenly divide across nodes
