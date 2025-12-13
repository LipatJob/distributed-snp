#include <mpi.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "ISort.hpp"

enum class SortBackend {
    NAIVE_CPU_SNP_SORT,
    CUDA_SNP_SORT,
    CUDA_MPI_SNP_SORT,
    NAIVE_CUDA_MPI_SNP_SORT
};

// Factory wrapper for different backends
std::unique_ptr<ISort> createSorter(SortBackend backend) {
    switch (backend) {
        case SortBackend::NAIVE_CPU_SNP_SORT:
            return createNaiveCpuSnpSort();
        case SortBackend::CUDA_SNP_SORT:
            return createCudaSnpSort();
        case SortBackend::CUDA_MPI_SNP_SORT:
            return createCudaMpiSnpSort();
        default:
            return nullptr;
    }
}

// Test fixture for SnpSort tests
class SnpSortTest : public ::testing::TestWithParam<SortBackend> {
protected:
    std::unique_ptr<ISort> sorter;
    
    void SetUp() override {
        sorter = createSorter(GetParam());
    }
};

// Test sorting an empty array
TEST_P(SnpSortTest, SortEmptyArray) {
    int data[] = {};
    sorter->sort(data, 0);
    SUCCEED();
}

// Test sorting a single element
TEST_P(SnpSortTest, SortSingleElement) {
    int data[] = {42};
    sorter->sort(data, 1);
    EXPECT_EQ(data[0], 42);
}

// Test sorting two elements
TEST_P(SnpSortTest, SortTwoElements) {
    int data[] = {5, 3};
    sorter->sort(data, 2);
    EXPECT_EQ(data[0], 3);
    EXPECT_EQ(data[1], 5);
}

// Test sorting already sorted array
TEST_P(SnpSortTest, SortAlreadySorted) {
    int data[] = {1, 2, 3, 4, 5};
    size_t size = 5;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], i + 1);
    }
}

// Test sorting reverse sorted array
TEST_P(SnpSortTest, SortReverseSorted) {
    int data[] = {5, 4, 3, 2, 1};
    size_t size = 5;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], i + 1);
    }
}

// Test sorting array with duplicates
TEST_P(SnpSortTest, SortWithDuplicates) {
    int data[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    size_t size = 10;
    
    sorter->sort(data, size);
    
    // Verify sorted order
    for (size_t i = 1; i < size; ++i) {
        EXPECT_LE(data[i-1], data[i]);
    }
}

// Test sorting small random array
TEST_P(SnpSortTest, SortSmallRandom) {
    int data[] = {7, 2, 9, 1, 5, 3};
    int expected[] = {1, 2, 3, 5, 7, 9};
    size_t size = 6;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], expected[i]);
    }
}

// Test sorting array with all same values
TEST_P(SnpSortTest, SortAllSame) {
    int data[] = {5, 5, 5, 5, 5};
    size_t size = 5;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], 5);
    }
}

// Test sorting array with zeros
TEST_P(SnpSortTest, SortWithZeros) {
    int data[] = {3, 0, 2, 0, 1};
    int expected[] = {0, 0, 1, 2, 3};
    size_t size = 5;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], expected[i]);
    }
}

// Performance test with various array sizes from 20 to 100
TEST_P(SnpSortTest, SortVariousSizes) {
    for (size_t size = 20; size <= 100; ++size) {
        std::vector<int> data(size);
        
        // Fill with descending values
        for (size_t i = 0; i < size; ++i) {
            data[i] = size - i;
        }
        
        sorter->sort(data.data(), size);
        
        // Verify sorted
        for (size_t i = 0; i < size; ++i) {
            EXPECT_EQ(data[i], i + 1) << "Failed at size " << size << ", index " << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    SnpSortTest,
    ::testing::Values(
        SortBackend::NAIVE_CPU_SNP_SORT,
        SortBackend::CUDA_SNP_SORT,
        SortBackend::CUDA_MPI_SNP_SORT
    ),
    [](const ::testing::TestParamInfo<SortBackend>& info) {
        switch (info.param) {
            case SortBackend::NAIVE_CPU_SNP_SORT: return "NaiveCpuSnpSort";
            case SortBackend::CUDA_SNP_SORT: return "CudaSnpSort";
            case SortBackend::CUDA_MPI_SNP_SORT: return "CudaMpiSnpSort";
            default: return "Unknown";
        }
    }
);


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