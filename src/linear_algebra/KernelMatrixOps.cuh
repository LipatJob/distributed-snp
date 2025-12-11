#ifndef KERNEL_MATRIX_OPS_CUH
#define KERNEL_MATRIX_OPS_CUH

#include <cuda_runtime.h>

/**
 * @brief Standard Matrix Multiplication Kernel
 * C = A * B
 */
template <typename T>
__global__ void kMultiply(const T *A, const T *B, T *C, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    T sum = 0;
    for (int i = 0; i < k; ++i) {
      sum += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
  }
}

/**
 * @brief Hadamard (Element-wise) Product Kernel
 * C = A ⊙ B
 */
template <typename T>
__global__ void kHadamard(const T *A, const T *B, T *C, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    C[idx] = A[idx] * B[idx];
}

/**
 * @brief Element-wise Addition Kernel
 * C = A + B
 */
template <typename T>
__global__ void kAdd(const T *A, const T *B, T *C, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    C[idx] = A[idx] + B[idx];
}

#endif // KERNEL_MATRIX_OPS_CUH