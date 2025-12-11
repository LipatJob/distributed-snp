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
    // Strategy: 1D Row Decomposition with Uneven Distribution
    // 1. Scatter rows of A (base_rows + extra for some ranks) using Scatterv
    // 2. Broadcast entire B (Every rank needs full B)
    // 3. Compute partial C (variable rows) on GPU
    // 4. Gather C rows back to Root using Gatherv
    // ========================================================================
    void multiply(const T* A, const T* B, T* C, size_t m, size_t n, size_t k) override {
        // Calculate distribution: base_rows for all, +1 for first remainder ranks
        size_t base_rows = m / size;
        size_t remainder = m % size;
        size_t rows_local = base_rows + (rank < remainder ? 1 : 0);
        
        size_t size_A_loc = rows_local * k;
        size_t size_B     = k * n;
        size_t size_C_loc = rows_local * n;

        // 1. Host Memory Prep
        std::vector<T> loc_A(size_A_loc);
        std::vector<T> loc_B(size_B);
        std::vector<T> loc_C(size_C_loc);

        // 2. Distribute Data using Scatterv
        std::vector<int> sendcounts_A, displs_A, sendcounts_C, displs_C;
        if (rank == 0) {
            sendcounts_A.resize(size);
            displs_A.resize(size);
            sendcounts_C.resize(size);
            displs_C.resize(size);
            
            for (int i = 0; i < size; i++) {
                size_t rows_i = base_rows + (i < remainder ? 1 : 0);
                sendcounts_A[i] = rows_i * k * sizeof(T);
                sendcounts_C[i] = rows_i * n * sizeof(T);
                displs_A[i] = (i > 0) ? displs_A[i-1] + sendcounts_A[i-1] : 0;
                displs_C[i] = (i > 0) ? displs_C[i-1] + sendcounts_C[i-1] : 0;
            }
        }
        
        // Scatter A: Root sends variable-sized chunks to everyone
        MPI_Scatterv(A, sendcounts_A.data(), displs_A.data(), MPI_BYTE,
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

        // 4. Gather Result using Gatherv
        MPI_Gatherv(loc_C.data(), size_C_loc * sizeof(T), MPI_BYTE,
                    C, sendcounts_C.data(), displs_C.data(), MPI_BYTE,
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
     * Handles Scatterv -> GPU Compute -> Gatherv with uneven distribution
     */
    void launchDistributedElementWise(const T* A, const T* B, T* C, size_t total_elements, int opType) {
        // Calculate distribution: base_chunk for all, +1 for first remainder ranks
        size_t base_chunk = total_elements / size;
        size_t remainder = total_elements % size;
        size_t chunk_size = base_chunk + (rank < remainder ? 1 : 0);
        size_t bytes = chunk_size * sizeof(T);

        // Host Buffers
        std::vector<T> loc_A(chunk_size);
        std::vector<T> loc_B(chunk_size);
        std::vector<T> loc_C(chunk_size);

        // Prepare counts and displacements for Scatterv/Gatherv
        std::vector<int> sendcounts, displs;
        if (rank == 0) {
            sendcounts.resize(size);
            displs.resize(size);
            
            for (int i = 0; i < size; i++) {
                size_t chunk_i = base_chunk + (i < remainder ? 1 : 0);
                sendcounts[i] = chunk_i * sizeof(T);
                displs[i] = (i > 0) ? displs[i-1] + sendcounts[i-1] : 0;
            }
        }

        // 1. Scatter A and B using Scatterv
        MPI_Scatterv(A, sendcounts.data(), displs.data(), MPI_BYTE,
                     loc_A.data(), bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(B, sendcounts.data(), displs.data(), MPI_BYTE,
                     loc_B.data(), bytes, MPI_BYTE, 0, MPI_COMM_WORLD);

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

        // 3. Gather C using Gatherv
        MPI_Gatherv(loc_C.data(), bytes, MPI_BYTE,
                    C, sendcounts.data(), displs.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);
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