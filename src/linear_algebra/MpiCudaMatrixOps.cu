/**
 * @file MpiCudaMatrixOps.cu
 * @brief Distributed implementation using MPI for communication and CUDA for computation.
 */

#include "MatrixOps.h"
#include "KernelMatrixOps.cuh" // Reusing the shared kernels
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>

template <typename T>
class MpiCudaMatrixOps : public IMatrixOps<T> {
    int rank;
    int size;

public:
    MpiCudaMatrixOps() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // --- Multi-GPU Setup ---
        // If a node has multiple GPUs, assign them round-robin based on rank.
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        if (num_devices > 0) {
            cudaSetDevice(rank % num_devices);
        }
    }

    // ========================================================================
    // Matrix Multiplication: C = A * B
    // Strategy: 1D Row Decomposition
    // 1. Scatter rows of A (M/size rows per rank)
    // 2. Broadcast entire B (Every rank needs full B)
    // 3. Compute partial C (M/size rows) on GPU
    // 4. Gather C rows back to Root
    // ========================================================================
    void multiply(const T* A, const T* B, T* C, size_t m, size_t n, size_t k) override {
        // Validation: For this simple implementation, assume M divides evenly.
        if (m % size != 0) {
            if (rank == 0) std::cerr << "Error: M (" << m << ") must be divisible by MPI size (" << size << ")" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t rows_local = m / size;
        size_t size_A_loc = rows_local * k;
        size_t size_B     = k * n;
        size_t size_C_loc = rows_local * n;

        // 1. Host Memory Prep
        std::vector<T> loc_A(size_A_loc);
        std::vector<T> loc_B(size_B);
        std::vector<T> loc_C(size_C_loc);

        // 2. Distribute Data
        // Scatter A: Root sends chunks of A to everyone (including itself)
        MPI_Scatter(A, size_A_loc * sizeof(T), MPI_BYTE, 
                    loc_A.data(), size_A_loc * sizeof(T), MPI_BYTE, 
                    0, MPI_COMM_WORLD);

        // Broadcast B: Root sends full B to everyone
        if (rank == 0) {
            std::copy(B, B + size_B, loc_B.begin());
        }
        MPI_Bcast(loc_B.data(), size_B * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);

        // 3. GPU Computation (Local)
        T *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size_A_loc * sizeof(T));
        cudaMalloc(&d_B, size_B * sizeof(T));
        cudaMalloc(&d_C, size_C_loc * sizeof(T));

        cudaMemcpy(d_A, loc_A.data(), size_A_loc * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, loc_B.data(), size_B * sizeof(T), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (rows_local + block.y - 1) / block.y);
        
        // Launch Shared Kernel
        kMultiply<<<grid, block>>>(d_A, d_B, d_C, rows_local, n, k);
        cudaDeviceSynchronize();

        cudaMemcpy(loc_C.data(), d_C, size_C_loc * sizeof(T), cudaMemcpyDeviceToHost);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

        // 4. Gather Result
        MPI_Gather(loc_C.data(), size_C_loc * sizeof(T), MPI_BYTE,
                   C, size_C_loc * sizeof(T), MPI_BYTE,
                   0, MPI_COMM_WORLD);
    }

    // ========================================================================
    // Hadamard Product: C = A ⊙ B
    // Strategy: 1D Linear Decomposition
    // Since this is element-wise, we just split the total array size evenly.
    // ========================================================================
    void hadamard(const T* A, const T* B, T* C, size_t m, size_t n) override {
        launchDistributedElementWise(A, B, C, m * n, 0); // 0 = Hadamard
    }

    // ========================================================================
    // Matrix Addition: C = A + B
    // Strategy: 1D Linear Decomposition
    // ========================================================================
    void add(const T* A, const T* B, T* C, size_t m, size_t n) override {
        launchDistributedElementWise(A, B, C, m * n, 1); // 1 = Add
    }

    std::string getBackendName() const override { return "MPI + CUDA (Distributed)"; }

private:
    /**
     * @brief Helper for element-wise operations (Add/Hadamard)
     * Handles Scatter -> GPU Compute -> Gather
     */
    void launchDistributedElementWise(const T* A, const T* B, T* C, size_t total_elements, int opType) {
        if (total_elements % size != 0) {
            // For production code, you would handle the remainder (padding or variable counts using MPI_Scatterv)
            if (rank == 0) std::cerr << "Warning: Total elements not divisible by MPI size. Truncation may occur in this demo." << std::endl;
        }

        size_t chunk_size = total_elements / size;
        size_t bytes = chunk_size * sizeof(T);

        // Host Buffers
        std::vector<T> loc_A(chunk_size);
        std::vector<T> loc_B(chunk_size);
        std::vector<T> loc_C(chunk_size);

        // 1. Scatter A and B
        MPI_Scatter(A, bytes, MPI_BYTE, loc_A.data(), bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Scatter(B, bytes, MPI_BYTE, loc_B.data(), bytes, MPI_BYTE, 0, MPI_COMM_WORLD);

        // 2. GPU Compute
        T *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        cudaMemcpy(d_A, loc_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, loc_B.data(), bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (chunk_size + threads - 1) / threads;

        if (opType == 0) {
            kHadamard<<<blocks, threads>>>(d_A, d_B, d_C, chunk_size);
        } else {
            kAdd<<<blocks, threads>>>(d_A, d_B, d_C, chunk_size);
        }
        cudaDeviceSynchronize();

        cudaMemcpy(loc_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

        // 3. Gather C
        MPI_Gather(loc_C.data(), bytes, MPI_BYTE, C, bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
};

// =========================================================================
// Builder Function
// =========================================================================
template <typename T>
std::unique_ptr<IMatrixOps<T>> makeMpiCudaOps() {
    return std::make_unique<MpiCudaMatrixOps<T>>();
}

// =========================================================================
// Explicit Instantiations
// =========================================================================
template std::unique_ptr<IMatrixOps<float>> makeMpiCudaOps<float>();
template std::unique_ptr<IMatrixOps<double>> makeMpiCudaOps<double>();