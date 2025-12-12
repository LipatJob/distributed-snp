#include <mpi.h>
#include <gtest/gtest.h>
#include "ISort.hpp"

// Test fixture for SnpSort tests
class SnpSortTest : public ::testing::Test {
protected:
    std::unique_ptr<ISort> sorter;
    
    void SetUp() override {
        sorter = createCudaMpiSnpSort();
    }
};

// Test sorting an empty array
TEST_F(SnpSortTest, SortEmptyArray) {
    int data[] = {};
    sorter->sort(data, 0);
    SUCCEED();
}

// Test sorting a single element
TEST_F(SnpSortTest, SortSingleElement) {
    int data[] = {42};
    sorter->sort(data, 1);
    EXPECT_EQ(data[0], 42);
}

// Test sorting two elements
TEST_F(SnpSortTest, SortTwoElements) {
    int data[] = {5, 3};
    sorter->sort(data, 2);
    EXPECT_EQ(data[0], 3);
    EXPECT_EQ(data[1], 5);
}

// Test sorting already sorted array
TEST_F(SnpSortTest, SortAlreadySorted) {
    int data[] = {1, 2, 3, 4, 5};
    size_t size = 5;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], i + 1);
    }
}

// Test sorting reverse sorted array
TEST_F(SnpSortTest, SortReverseSorted) {
    int data[] = {5, 4, 3, 2, 1};
    size_t size = 5;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], i + 1);
    }
}

// Test sorting array with duplicates
TEST_F(SnpSortTest, SortWithDuplicates) {
    int data[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    size_t size = 10;
    
    sorter->sort(data, size);
    
    // Verify sorted order
    for (size_t i = 1; i < size; ++i) {
        EXPECT_LE(data[i-1], data[i]);
    }
}

// Test sorting small random array
TEST_F(SnpSortTest, SortSmallRandom) {
    int data[] = {7, 2, 9, 1, 5, 3};
    int expected[] = {1, 2, 3, 5, 7, 9};
    size_t size = 6;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], expected[i]);
    }
}

// Test sorting array with all same values
TEST_F(SnpSortTest, SortAllSame) {
    int data[] = {5, 5, 5, 5, 5};
    size_t size = 5;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], 5);
    }
}

// Test sorting array with zeros
TEST_F(SnpSortTest, SortWithZeros) {
    int data[] = {3, 0, 2, 0, 1};
    int expected[] = {0, 0, 1, 2, 3};
    size_t size = 5;
    
    sorter->sort(data, size);
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], expected[i]);
    }
}

// Performance test with medium-sized array
TEST_F(SnpSortTest, SortMediumArray) {
    const size_t size = 20;
    std::vector<int> data(size);
    
    // Fill with descending values
    for (size_t i = 0; i < size; ++i) {
        data[i] = size - i;
    }
    
    sorter->sort(data.data(), size);
    
    // Verify sorted
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], i + 1);
    }
}

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