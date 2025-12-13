# Role
Act as a Senior High-Performance Computing (HPC) Engineer specializing in Hybrid CUDA architectures.

# Task
Implement the `SparseCudaSnpSimulator` class in C++. This class should extend the existing `ISnpSimulator` interface to provide a parallel implementation of the Spiking Neural P System (SNP) using CUDA for GPU acceleration. The implementation must use the Sparse representation of the SNP model to optimize memory usage and computational efficiency as Mentioned in sparse_snp_paper.tex

# Context & Constraints
1.  **Architecture:** The implementation must leverage CUDA for parallel computation on GPUs. Ensure that the code is optimized for performance, taking advantage of CUDA's capabilities for handling sparse data structures.
2.  **Goals:** The primary goal is to implement the SNP simulation using a sparse representation to reduce memory overhead and improve computation speed. The implementation should be capable of handling large-scale SNP models efficiently. Make sure to keep the implementation simple and readable but prioritize efficiency
3.  **Interface Strictness:** You must strictly adhere to the `ISnpSimulator.hpp` interface. Do not modify the header files or the signature of existing methods. You may add private helper methods or internal classes/structs within the implementation file. Also update the factory methods in the interface, CMakeLists, tests, and benchmarks to include this new class.
4.  **Reference Material:**
    * I have provided `snp_explanation.tex` below for the mathematical rules of the system.
    * Use `NaiveCpuSnpSimulator.cpp` to understand the sequential logic correctness.
    * Use `sparse_snp_paper.tex` to understand the sparse representation and optimizations.
    * Refer to `sparse_snp_model.cu` to understand how to implement sparse data structures in CUDA.
    * There is also `CudaSnpSimulator.cu` which implements a non-sparse version of the SNP simulation using CUDA. You can use this as a baseline for your implementation.