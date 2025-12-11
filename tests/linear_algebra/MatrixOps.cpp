/**
 * @file MatrixOps.cpp
 * @brief Comprehensive test suite for all MatrixOps implementations
 * 
 * Tests CPU, CUDA, and MPI+CUDA backends with:
 * - Basic correctness tests
 * - Edge cases (1x1, 0 values)
 * - Different matrix sizes
 * - Floating point precision checks
 */

#include "MatrixOps.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

// ============================================================================
// Test Fixture and Helper Functions
// ============================================================================

constexpr double EPSILON = 1e-5;

/**
 * @brief Helper function to compare floating point arrays with tolerance
 */
template <typename T>
bool arraysEqual(const T* a, const T* b, size_t size, T epsilon = EPSILON) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": " 
                      << a[i] << " != " << b[i] 
                      << " (diff: " << std::abs(a[i] - b[i]) << ")" << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * @brief Initialize matrix with sequential values for testing
 */
template <typename T>
void initMatrix(T* matrix, size_t rows, size_t cols, T startValue = 1.0) {
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = startValue + static_cast<T>(i);
    }
}

/**
 * @brief Initialize identity matrix
 */
template <typename T>
void initIdentityMatrix(T* matrix, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            matrix[i * size + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// ============================================================================
// Test Suite: Matrix Multiplication
// ============================================================================

class MatrixMultiplyTest : public ::testing::TestWithParam<BackendType> {
protected:
    void SetUp() override {
        backend = GetParam();
    }
    
    BackendType backend;
};

TEST_P(MatrixMultiplyTest, SmallMatrixMultiply) {
    auto ops = createMatrixOps<double>(backend);
    ASSERT_NE(ops, nullptr) << "Failed to create MatrixOps for backend";

    // Test 2x3 * 3x2 = 2x2
    // A = [1 2 3]    B = [7  8]
    //     [4 5 6]        [9 10]
    //                    [11 12]
    // C = [1*7+2*9+3*11  1*8+2*10+3*12]  = [58  64]
    //     [4*7+5*9+6*11  4*8+5*10+6*12]    [139 154]
    
    double A[] = {1, 2, 3, 4, 5, 6};
    double B[] = {7, 8, 9, 10, 11, 12};
    double C[4] = {0};
    double expected[] = {58, 64, 139, 154};
    
    ops->multiply(A, B, C, 2, 2, 3);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C, expected, 4)) 
            << "Backend: " << ops->getBackendName();
    }
}

TEST_P(MatrixMultiplyTest, IdentityMatrixMultiply) {
    auto ops = createMatrixOps<double>(backend);
    ASSERT_NE(ops, nullptr);

    size_t size = 4;
    std::vector<double> A(size * size);
    std::vector<double> I(size * size);
    std::vector<double> C(size * size, 0);
    
    initMatrix(A.data(), size, size);
    initIdentityMatrix(I.data(), size);
    
    // A * I should equal A
    ops->multiply(A.data(), I.data(), C.data(), size, size, size);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C.data(), A.data(), size * size))
            << "Backend: " << ops->getBackendName();
    }
}

TEST_P(MatrixMultiplyTest, LargerMatrixMultiply) {
    auto ops = createMatrixOps<float>(backend);
    ASSERT_NE(ops, nullptr);

    // Test with 8x8 matrices (divisible by 4 for MPI)
    size_t m = 8, n = 8, k = 8;
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C(m * n, 0);
    
    // Initialize with simple pattern
    for (size_t i = 0; i < m * k; ++i) A[i] = static_cast<float>(i % 10);
    for (size_t i = 0; i < k * n; ++i) B[i] = static_cast<float>(i % 10);
    
    // Compute reference on rank 0
    std::vector<float> reference(m * n, 0);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0;
                for (size_t p = 0; p < k; ++p) {
                    sum += A[i * k + p] * B[p * n + j];
                }
                reference[i * n + j] = sum;
            }
        }
    }
    
    ops->multiply(A.data(), B.data(), C.data(), m, n, k);
    
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C.data(), reference.data(), m * n, 1e-4f))
            << "Backend: " << ops->getBackendName();
    }
}

TEST_P(MatrixMultiplyTest, SingleElementMultiply) {
    auto ops = createMatrixOps<double>(backend);
    ASSERT_NE(ops, nullptr);

    // 1x1 * 1x1 = 1x1
    double A[] = {3.5};
    double B[] = {2.0};
    double C[] = {0.0};
    double expected[] = {7.0};
    
    ops->multiply(A, B, C, 1, 1, 1);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_NEAR(C[0], expected[0], EPSILON)
            << "Backend: " << ops->getBackendName();
    }
}

// ============================================================================
// Test Suite: Hadamard Product (Element-wise Multiplication)
// ============================================================================

class HadamardTest : public ::testing::TestWithParam<BackendType> {
protected:
    void SetUp() override {
        backend = GetParam();
    }
    
    BackendType backend;
};

TEST_P(HadamardTest, BasicHadamard) {
    auto ops = createMatrixOps<double>(backend);
    ASSERT_NE(ops, nullptr);

    // [1 2 3] ⊙ [2 2 2] = [2 4 6]
    // [4 5 6]   [2 2 2]   [8 10 12]
    double A[] = {1, 2, 3, 4, 5, 6};
    double B[] = {2, 2, 2, 2, 2, 2};
    double C[6] = {0};
    double expected[] = {2, 4, 6, 8, 10, 12};
    
    ops->hadamard(A, B, C, 2, 3);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C, expected, 6))
            << "Backend: " << ops->getBackendName();
    }
}

TEST_P(HadamardTest, HadamardWithZeros) {
    auto ops = createMatrixOps<float>(backend);
    ASSERT_NE(ops, nullptr);

    size_t m = 4, n = 4; // 16 elements, divisible by 4
    std::vector<float> A(m * n);
    std::vector<float> B(m * n, 0); // All zeros
    std::vector<float> C(m * n, 99); // Initialize with non-zero
    
    initMatrix(A.data(), m, n, 1.0f);
    
    ops->hadamard(A.data(), B.data(), C.data(), m, n);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        for (size_t i = 0; i < m * n; ++i) {
            EXPECT_NEAR(C[i], 0.0f, EPSILON)
                << "Element " << i << " should be zero. Backend: " << ops->getBackendName();
        }
    }
}

TEST_P(HadamardTest, HadamardLargeMatrix) {
    auto ops = createMatrixOps<double>(backend);
    ASSERT_NE(ops, nullptr);

    size_t m = 16, n = 8; // 128 elements, divisible by 4
    std::vector<double> A(m * n);
    std::vector<double> B(m * n);
    std::vector<double> C(m * n, 0);
    std::vector<double> expected(m * n);
    
    for (size_t i = 0; i < m * n; ++i) {
        A[i] = static_cast<double>(i) * 0.5;
        B[i] = 2.0;
        expected[i] = A[i] * B[i];
    }
    
    ops->hadamard(A.data(), B.data(), C.data(), m, n);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C.data(), expected.data(), m * n))
            << "Backend: " << ops->getBackendName();
    }
}

TEST_P(HadamardTest, HadamardIdentity) {
    auto ops = createMatrixOps<float>(backend);
    ASSERT_NE(ops, nullptr);

    size_t m = 4, n = 4;
    std::vector<float> A(m * n);
    std::vector<float> ones(m * n, 1.0f);
    std::vector<float> C(m * n, 0);
    
    initMatrix(A.data(), m, n, 1.0f);
    
    // A ⊙ 1 = A
    ops->hadamard(A.data(), ones.data(), C.data(), m, n);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C.data(), A.data(), m * n, 1e-5f))
            << "Backend: " << ops->getBackendName();
    }
}

// ============================================================================
// Test Suite: Matrix Addition
// ============================================================================

class AdditionTest : public ::testing::TestWithParam<BackendType> {
protected:
    void SetUp() override {
        backend = GetParam();
    }
    
    BackendType backend;
};

TEST_P(AdditionTest, BasicAddition) {
    auto ops = createMatrixOps<double>(backend);
    ASSERT_NE(ops, nullptr);

    // [1 2] + [5 6] = [6 8]
    // [3 4]   [7 8]   [10 12]
    double A[] = {1, 2, 3, 4};
    double B[] = {5, 6, 7, 8};
    double C[4] = {0};
    double expected[] = {6, 8, 10, 12};
    
    ops->add(A, B, C, 2, 2);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C, expected, 4))
            << "Backend: " << ops->getBackendName();
    }
}

TEST_P(AdditionTest, AdditionWithZeros) {
    auto ops = createMatrixOps<float>(backend);
    ASSERT_NE(ops, nullptr);

    size_t m = 4, n = 4;
    std::vector<float> A(m * n);
    std::vector<float> zeros(m * n, 0);
    std::vector<float> C(m * n, 0);
    
    initMatrix(A.data(), m, n, 1.0f);
    
    // A + 0 = A
    ops->add(A.data(), zeros.data(), C.data(), m, n);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C.data(), A.data(), m * n, 1e-5f))
            << "Backend: " << ops->getBackendName();
    }
}

TEST_P(AdditionTest, AdditionLargeMatrix) {
    auto ops = createMatrixOps<double>(backend);
    ASSERT_NE(ops, nullptr);

    size_t m = 16, n = 8;
    std::vector<double> A(m * n);
    std::vector<double> B(m * n);
    std::vector<double> C(m * n, 0);
    std::vector<double> expected(m * n);
    
    for (size_t i = 0; i < m * n; ++i) {
        A[i] = static_cast<double>(i) * 1.5;
        B[i] = static_cast<double>(i) * 0.5;
        expected[i] = A[i] + B[i];
    }
    
    ops->add(A.data(), B.data(), C.data(), m, n);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C.data(), expected.data(), m * n))
            << "Backend: " << ops->getBackendName();
    }
}

TEST_P(AdditionTest, AdditionNegativeValues) {
    auto ops = createMatrixOps<double>(backend);
    ASSERT_NE(ops, nullptr);

    // Test with negative values
    double A[] = {-1, -2, -3, -4};
    double B[] = {1, 2, 3, 4};
    double C[4] = {0};
    double expected[] = {0, 0, 0, 0};
    
    ops->add(A, B, C, 2, 2);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C, expected, 4))
            << "Backend: " << ops->getBackendName();
    }
}

TEST_P(AdditionTest, AdditionCommutative) {
    auto ops = createMatrixOps<float>(backend);
    ASSERT_NE(ops, nullptr);

    size_t m = 4, n = 4;
    std::vector<float> A(m * n);
    std::vector<float> B(m * n);
    std::vector<float> C1(m * n, 0);
    std::vector<float> C2(m * n, 0);
    
    initMatrix(A.data(), m, n, 1.0f);
    initMatrix(B.data(), m, n, 10.0f);
    
    // Test commutativity: A + B = B + A
    ops->add(A.data(), B.data(), C1.data(), m, n);
    ops->add(B.data(), A.data(), C2.data(), m, n);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C1.data(), C2.data(), m * n, 1e-5f))
            << "Addition should be commutative. Backend: " << ops->getBackendName();
    }
}

// ============================================================================
// Test Suite: Backend Verification
// ============================================================================

TEST(BackendTest, AllBackendsAvailable) {
    auto cpu_ops = createMatrixOps<double>(BackendType::CPU);
    auto cuda_ops = createMatrixOps<double>(BackendType::CUDA);
    auto mpi_cuda_ops = createMatrixOps<double>(BackendType::MPI_CUDA);
    
    EXPECT_NE(cpu_ops, nullptr) << "CPU backend should be available";
    EXPECT_NE(cuda_ops, nullptr) << "CUDA backend should be available";
    EXPECT_NE(mpi_cuda_ops, nullptr) << "MPI+CUDA backend should be available";
}

TEST(BackendTest, BackendNames) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        auto cpu_ops = createMatrixOps<double>(BackendType::CPU);
        auto cuda_ops = createMatrixOps<double>(BackendType::CUDA);
        auto mpi_cuda_ops = createMatrixOps<double>(BackendType::MPI_CUDA);
        
        EXPECT_EQ(cpu_ops->getBackendName(), "CPU (OpenMP)");
        EXPECT_EQ(cuda_ops->getBackendName(), "CUDA (Single GPU)");
        EXPECT_EQ(mpi_cuda_ops->getBackendName(), "MPI + CUDA (Distributed)");
    }
}

// ============================================================================
// Test Suite: Cross-Backend Consistency
// ============================================================================

class CrossBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_ops = createMatrixOps<double>(BackendType::CPU);
        cuda_ops = createMatrixOps<double>(BackendType::CUDA);
        mpi_cuda_ops = createMatrixOps<double>(BackendType::MPI_CUDA);
    }
    
    std::unique_ptr<IMatrixOps<double>> cpu_ops;
    std::unique_ptr<IMatrixOps<double>> cuda_ops;
    std::unique_ptr<IMatrixOps<double>> mpi_cuda_ops;
};

TEST_F(CrossBackendTest, MultiplyConsistency) {
    size_t m = 8, n = 8, k = 8;
    std::vector<double> A(m * k);
    std::vector<double> B(k * n);
    std::vector<double> C_cpu(m * n, 0);
    std::vector<double> C_cuda(m * n, 0);
    std::vector<double> C_mpi(m * n, 0);
    
    initMatrix(A.data(), m, k);
    initMatrix(B.data(), k, n, 10.0);
    
    cpu_ops->multiply(A.data(), B.data(), C_cpu.data(), m, n, k);
    cuda_ops->multiply(A.data(), B.data(), C_cuda.data(), m, n, k);
    mpi_cuda_ops->multiply(A.data(), B.data(), C_mpi.data(), m, n, k);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C_cpu.data(), C_cuda.data(), m * n))
            << "CPU and CUDA results should match";
        EXPECT_TRUE(arraysEqual(C_cpu.data(), C_mpi.data(), m * n))
            << "CPU and MPI+CUDA results should match";
    }
}

TEST_F(CrossBackendTest, HadamardConsistency) {
    size_t m = 16, n = 8;
    std::vector<double> A(m * n);
    std::vector<double> B(m * n);
    std::vector<double> C_cpu(m * n, 0);
    std::vector<double> C_cuda(m * n, 0);
    std::vector<double> C_mpi(m * n, 0);
    
    initMatrix(A.data(), m, n);
    initMatrix(B.data(), m, n, 5.0);
    
    cpu_ops->hadamard(A.data(), B.data(), C_cpu.data(), m, n);
    cuda_ops->hadamard(A.data(), B.data(), C_cuda.data(), m, n);
    mpi_cuda_ops->hadamard(A.data(), B.data(), C_mpi.data(), m, n);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C_cpu.data(), C_cuda.data(), m * n))
            << "CPU and CUDA results should match";
        EXPECT_TRUE(arraysEqual(C_cpu.data(), C_mpi.data(), m * n))
            << "CPU and MPI+CUDA results should match";
    }
}

TEST_F(CrossBackendTest, AdditionConsistency) {
    size_t m = 16, n = 8;
    std::vector<double> A(m * n);
    std::vector<double> B(m * n);
    std::vector<double> C_cpu(m * n, 0);
    std::vector<double> C_cuda(m * n, 0);
    std::vector<double> C_mpi(m * n, 0);
    
    initMatrix(A.data(), m, n);
    initMatrix(B.data(), m, n, 20.0);
    
    cpu_ops->add(A.data(), B.data(), C_cpu.data(), m, n);
    cuda_ops->add(A.data(), B.data(), C_cuda.data(), m, n);
    mpi_cuda_ops->add(A.data(), B.data(), C_mpi.data(), m, n);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_TRUE(arraysEqual(C_cpu.data(), C_cuda.data(), m * n))
            << "CPU and CUDA results should match";
        EXPECT_TRUE(arraysEqual(C_cpu.data(), C_mpi.data(), m * n))
            << "CPU and MPI+CUDA results should match";
    }
}

// ============================================================================
// Parameterized Test Instantiation
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    MatrixMultiplyTest,
    ::testing::Values(BackendType::CPU, BackendType::CUDA, BackendType::MPI_CUDA),
    [](const ::testing::TestParamInfo<BackendType>& info) {
        switch (info.param) {
            case BackendType::CPU: return "CPU";
            case BackendType::CUDA: return "CUDA";
            case BackendType::MPI_CUDA: return "MPI_CUDA";
            default: return "Unknown";
        }
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    HadamardTest,
    ::testing::Values(BackendType::CPU, BackendType::CUDA, BackendType::MPI_CUDA),
    [](const ::testing::TestParamInfo<BackendType>& info) {
        switch (info.param) {
            case BackendType::CPU: return "CPU";
            case BackendType::CUDA: return "CUDA";
            case BackendType::MPI_CUDA: return "MPI_CUDA";
            default: return "Unknown";
        }
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    AdditionTest,
    ::testing::Values(BackendType::CPU, BackendType::CUDA, BackendType::MPI_CUDA),
    [](const ::testing::TestParamInfo<BackendType>& info) {
        switch (info.param) {
            case BackendType::CPU: return "CPU";
            case BackendType::CUDA: return "CUDA";
            case BackendType::MPI_CUDA: return "MPI_CUDA";
            default: return "Unknown";
        }
    }
);

// ============================================================================
// MPI Initialization/Finalization
// ============================================================================

int main(int argc, char** argv) {
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);

    // 2. Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Optional: Add a listener to print test results only from Rank 0
    // This prevents cluttered output where every rank prints "[  PASSED  ]"
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // 3. Run Tests
    int result = RUN_ALL_TESTS();

    // 4. Finalize MPI
    MPI_Finalize();

    return result;
}