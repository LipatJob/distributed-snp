#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// Enumeration for available backends
enum class BackendType { CPU, CUDA, MPI_CUDA };

// Abstract Interface
template <typename T> class IMatrixOps {
public:
  virtual ~IMatrixOps() = default;

  virtual void multiply(const T *A, const T *B, T *C, size_t m, size_t n,
                        size_t k) = 0;

  virtual void hadamard(const T *A, const T *B, T *C, size_t m, size_t n) = 0;

  virtual void add(const T *A, const T *B, T *C, size_t m, size_t n) = 0;

  virtual std::string getBackendName() const = 0;
};

// =========================================================================
// Backend Builders
// These allow us to implement the classes in separate .cpp/.cu files
// while keeping the main factory logic simple.
// =========================================================================

template <typename T> std::unique_ptr<IMatrixOps<T>> makeCpuOps();

template <typename T> std::unique_ptr<IMatrixOps<T>> makeCudaOps();

template <typename T> std::unique_ptr<IMatrixOps<T>> makeMpiCudaOps();

// Main Factory
template <typename T>
std::unique_ptr<IMatrixOps<T>> createMatrixOps(BackendType type) {
  switch (type) {
  case BackendType::CPU:
    return makeCpuOps<T>();
  case BackendType::CUDA:
    return makeCudaOps<T>();
  case BackendType::MPI_CUDA:
    return makeMpiCudaOps<T>();
  default:
    return nullptr;
  }
}

#endif // MATRIX_OPS_H