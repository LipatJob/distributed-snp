#include "MatrixOps.h"
#include "KernelMatrixOps.cuh" // Include the shared kernels
#include <cuda_runtime.h>
#include <iostream>

template <typename T>
class CudaMatrixOps : public IMatrixOps<T> {
public:
    void multiply(const T* A, const T* B, T* C, size_t m, size_t n, size_t k) override {
        T *d_A, *d_B, *d_C;
        size_t bA = m * k * sizeof(T);
        size_t bB = k * n * sizeof(T);
        size_t bC = m * n * sizeof(T);

        cudaMalloc(&d_A, bA); cudaMalloc(&d_B, bB); cudaMalloc(&d_C, bC);
        cudaMemcpy(d_A, A, bA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, bB, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (m + 15) / 16);
        
        // Use the kernel from KernelMatrixOps.cuh
        kMultiply<<<grid, block>>>(d_A, d_B, d_C, m, n, k);

        cudaMemcpy(C, d_C, bC, cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    void hadamard(const T* A, const T* B, T* C, size_t m, size_t n) override {
        launchElementWise(A, B, C, m * n, 0);
    }

    void add(const T* A, const T* B, T* C, size_t m, size_t n) override {
        launchElementWise(A, B, C, m * n, 1);
    }

    std::string getBackendName() const override { return "CUDA (Single GPU)"; }

private:
    void launchElementWise(const T* A, const T* B, T* C, size_t size, int op) {
        T *d_A, *d_B, *d_C;
        size_t bytes = size * sizeof(T);
        cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
        cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (size + 255) / 256;

        if (op == 0) kHadamard<<<blocks, threads>>>(d_A, d_B, d_C, size);
        else         kAdd<<<blocks, threads>>>(d_A, d_B, d_C, size);

        cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
};

// Builder definition
template <typename T>
std::unique_ptr<IMatrixOps<T>> makeCudaOps() {
    return std::make_unique<CudaMatrixOps<T>>();
}

// Explicit Instantiations
template std::unique_ptr<IMatrixOps<float>> makeCudaOps<float>();
template std::unique_ptr<IMatrixOps<double>> makeCudaOps<double>();