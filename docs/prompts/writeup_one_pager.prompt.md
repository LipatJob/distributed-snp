# Role
Act as a Senior Software Architect and Technical Writer.

# Task
Generate a concise, one-page executive summary of the "Distributed SNP" codebase found in the current workspace. This summary is intended for a technical audience (developers and researchers) who need a quick understanding of the project's capabilities, architecture, and implementation details without reading the full documentation.

# Context & Requirements
1.  **Project Overview**: Briefly explain that this is a high-performance C++ library for simulating Spiking Neural P (SNP) Systems using matrix operations.
2.  **Key Features**:
    *   **Hybrid Parallelism**: Mention the use of MPI for distributed computing and CUDA for GPU acceleration.
    *   **Backends**: List the supported simulation backends (CPU, Naive CUDA, Distributed MPI+CUDA, Sparse CUDA).
    *   **Core Logic**: Briefly touch upon the matrix-algebraic formulation ($C(k+1) = C(k) + Sp(k) \times M$).
3.  **Architecture**:
    *   Highlight the modular design using interfaces (`ISnpSimulator`, `IMatrixOps`) and the Factory pattern.
    *   Mention the separation of concerns between the simulation logic and the linear algebra engine.
4.  **Codebase Structure**:
    *   `src/snp`: Core simulation logic and partitioners.
    *   `src/linear_algebra`: Matrix operation implementations (CPU, CUDA, MPI).
    *   `src/sort`: Example application (sorting using SNP systems).
5.  **Build & Test**: Mention CMake, Google Test, and the benchmarking suite.

# Constraints
*   **Length**: Strictly limited to one page (approximately 400-500 words).
*   **Tone**: Professional, technical, and concise.
*   **Format**: Use Markdown with clear headings and bullet points.
