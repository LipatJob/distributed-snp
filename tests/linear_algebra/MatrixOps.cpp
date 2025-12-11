#include <gtest/gtest.h>
#include "MatrixOps.h"
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Global MPI Rank/Size for use in tests
int g_rank;
int g_size;

// ============================================================================
// Helper: Reference Implementations (The "Golden" Logic)
// ============================================================================
template <typename T>
void refMatMul(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, 
               size_t m, size_t n, size_t k) {
    for(size_t row = 0; row < m; ++row) {
        for(size_t col = 0; col < n; ++col) {
            T sum = 0;
            for(size_t i = 0; i < k; ++i) {
                sum += A[row * k + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

template <typename T>
void refAdd(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C) {
    for(size_t i=0; i<A.size(); ++i) C[i] = A[i] + B[i];
}

// ============================================================================
// Helper: Validation
// ============================================================================
template <typename T>
void verifyResults(const std::vector<T>& computed, const std::vector<T>& expected, double tolerance = 1e-4) {
    ASSERT_EQ(computed.size(), expected.size()) << "Size mismatch";
    for(size_t i = 0; i < computed.size(); ++i) {
        EXPECT_NEAR(computed[i], expected[i], tolerance) 
            << "Mismatch at index " << i;
    }
}

// ============================================================================
// Test Fixture: Parameterized for BackendType
// ============================================================================
class MatrixOpsTest : public ::testing::TestWithParam<BackendType> {
protected:
    void SetUp() override {
        // Dimensions small enough for unit testing
        M = 128; N = 128; K = 128;
        
        // Ensure M is divisible by MPI size for the MPI implementation constraint
        if (M % g_size != 0) {
            // Adjust M to be compatible if running on weird rank counts
            M = ((M + g_size - 1) / g_size) * g_size;
        }

        sizeA = M * K;
        sizeB = K * N;
        sizeC = M * N;
    }

    // Helper to generate data deterministically across ranks
    void generateData(std::vector<float>& v, size_t size, float val_start) {
        v.resize(size);
        for(size_t i=0; i<size; ++i) v[i] = val_start + (i % 10);
    }

    size_t M, N, K;
    size_t sizeA, sizeB, sizeC;
};

// ============================================================================
// Test Case: Matrix Multiplication
// ============================================================================
TEST_P(MatrixOpsTest, MultipliesCorrectly) {
    BackendType type = GetParam();
    
    // 1. Prepare Data
    // We generate data on ALL ranks to simplify logic, but MPI backend only reads Rank 0.
    std::vector<float> h_A, h_B, h_C(sizeC, 0.0f);
    std::vector<float> ref_C(sizeC, 0.0f);

    generateData(h_A, sizeA, 1.0f);
    generateData(h_B, sizeB, 0.5f);

    // 2. Compute Reference (Only needed on verification rank)
    // For MPI backend, only Rank 0 checks. For others, everyone checks.
    bool should_verify = (type != BackendType::MPI_CUDA) || (g_rank == 0);

    if (should_verify) {
        refMatMul(h_A, h_B, ref_C, M, N, K);
    }

    // 3. Run Implementation
    auto ops = createMatrixOps<float>(type);
    
    // MPI Backend Contract: A and B pointers valid on Root. C valid on Root after call.
    // Non-MPI Backends: Pointers valid everywhere.
    // Since we allocated on all ranks above, .data() is valid everywhere.
    ops->multiply(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    // 4. Verify
    if (should_verify) {
        verifyResults(h_C, ref_C, 1e-3);
    }
}

// ============================================================================
// Test Case: Addition
// ============================================================================
TEST_P(MatrixOpsTest, AddsCorrectly) {
    BackendType type = GetParam();
    
    std::vector<float> h_A, h_B, h_C(sizeC, 0.0f), ref_C(sizeC, 0.0f);
    generateData(h_A, sizeC, 1.0f);
    generateData(h_B, sizeC, 2.0f);

    bool should_verify = (type != BackendType::MPI_CUDA) || (g_rank == 0);

    if(should_verify) refAdd(h_A, h_B, ref_C);

    auto ops = createMatrixOps<float>(type);
    ops->add(h_A.data(), h_B.data(), h_C.data(), M, N); // M*N elements

    if(should_verify) verifyResults(h_C, ref_C);
}

// ============================================================================
// Instantiate Tests
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    MatrixBackendTests,
    MatrixOpsTest,
    ::testing::Values(
        BackendType::CPU,
        BackendType::CUDA,
        BackendType::MPI_CUDA
    ),
    [](const testing::TestParamInfo<MatrixOpsTest::ParamType>& info) {
        switch(info.param) {
            case BackendType::CPU: return "CPU";
            case BackendType::CUDA: return "CUDA";
            case BackendType::MPI_CUDA: return "MPI_CUDA";
            default: return "Unknown";
        }
    }
);

// ============================================================================
// Custom Main (Initialize MPI)
// ============================================================================
int main(int argc, char **argv) {
    // 1. Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    // 2. Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // 3. Modify GTest output listeners so only Rank 0 prints details
    // (Optional, but keeps console clean)
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (g_rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    // 4. Run Tests
    int result = RUN_ALL_TESTS();

    // 5. Finalize MPI
    MPI_Finalize();
    return result;
}