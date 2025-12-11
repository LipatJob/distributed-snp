#include "MatrixOps.h"
#include <omp.h>

template <typename T>
class CpuMatrixOps : public IMatrixOps<T> {
public:
    void multiply(const T* A, const T* B, T* C, size_t m, size_t n, size_t k) override {
        #pragma omp parallel for collapse(2)
        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < n; ++col) {
                T sum = 0;
                for (size_t i = 0; i < k; ++i) {
                    sum += A[row * k + i] * B[i * n + col];
                }
                C[row * n + col] = sum;
            }
        }
    }

    void hadamard(const T* A, const T* B, T* C, size_t m, size_t n) override {
        size_t len = m * n;
        #pragma omp parallel for
        for (size_t i = 0; i < len; ++i) C[i] = A[i] * B[i];
    }

    void add(const T* A, const T* B, T* C, size_t m, size_t n) override {
        size_t len = m * n;
        #pragma omp parallel for
        for (size_t i = 0; i < len; ++i) C[i] = A[i] + B[i];
    }

    std::string getBackendName() const override { return "CPU (OpenMP)"; }
};

// Builder Implementation
template <typename T>
std::unique_ptr<IMatrixOps<T>> makeCpuOps() {
    return std::make_unique<CpuMatrixOps<T>>();
}

// Explicit Instantiation (Required for templates in separate files)
template std::unique_ptr<IMatrixOps<float>> makeCpuOps<float>();
template std::unique_ptr<IMatrixOps<double>> makeCpuOps<double>();