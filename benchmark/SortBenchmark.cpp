#include "../src/sort/ISort.hpp"
#include <benchmark/benchmark.h>
#include <mpi.h>
#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include <cmath>

// ============================================================================
// Helper Functions for Test Data Generation
// ============================================================================

enum class Distribution {
    RANDOM,
    SORTED,
    REVERSE_SORTED,
    NEARLY_SORTED,
    FEW_UNIQUE,
    UNIFORM
};

std::vector<int> generateTestData(size_t size, int maxValue, Distribution dist, unsigned seed = 42) {
    std::vector<int> data(size);
    std::mt19937 rng(seed);
    
    switch (dist) {
        case Distribution::RANDOM: {
            std::uniform_int_distribution<int> distrib(0, maxValue);
            for (size_t i = 0; i < size; ++i) {
                data[i] = distrib(rng);
            }
            break;
        }
        
        case Distribution::SORTED: {
            // Generate sorted data (best case)
            std::uniform_int_distribution<int> distrib(0, maxValue);
            for (size_t i = 0; i < size; ++i) {
                data[i] = distrib(rng);
            }
            std::sort(data.begin(), data.end());
            break;
        }
        
        case Distribution::REVERSE_SORTED: {
            // Generate reverse sorted data (worst case)
            std::uniform_int_distribution<int> distrib(0, maxValue);
            for (size_t i = 0; i < size; ++i) {
                data[i] = distrib(rng);
            }
            std::sort(data.begin(), data.end(), std::greater<int>());
            break;
        }
        
        case Distribution::NEARLY_SORTED: {
            // Generate 90% sorted data with random swaps
            std::uniform_int_distribution<int> distrib(0, maxValue);
            for (size_t i = 0; i < size; ++i) {
                data[i] = distrib(rng);
            }
            std::sort(data.begin(), data.end());
            
            // Swap 10% of elements randomly
            size_t numSwaps = size / 10;
            std::uniform_int_distribution<size_t> indexDist(0, size - 1);
            for (size_t i = 0; i < numSwaps; ++i) {
                size_t idx1 = indexDist(rng);
                size_t idx2 = indexDist(rng);
                std::swap(data[idx1], data[idx2]);
            }
            break;
        }
        
        case Distribution::FEW_UNIQUE: {
            // Only 5-10 unique values
            int numUnique = std::min(10, maxValue + 1);
            std::vector<int> uniqueValues(numUnique);
            std::uniform_int_distribution<int> distrib(0, maxValue);
            for (int i = 0; i < numUnique; ++i) {
                uniqueValues[i] = distrib(rng);
            }
            
            std::uniform_int_distribution<int> selectDist(0, numUnique - 1);
            for (size_t i = 0; i < size; ++i) {
                data[i] = uniqueValues[selectDist(rng)];
            }
            break;
        }
        
        case Distribution::UNIFORM: {
            // All elements have the same value
            std::uniform_int_distribution<int> distrib(0, maxValue);
            int value = distrib(rng);
            std::fill(data.begin(), data.end(), value);
            break;
        }
    }
    
    return data;
}

bool isSorted(const std::vector<int>& data) {
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] < data[i - 1]) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Benchmark Fixture
// ============================================================================

class SortBenchmarkFixture : public benchmark::Fixture {
protected:
    int rank;
    int world_size;
    
public:
    void SetUp(const ::benchmark::State& state) override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }
    
    void TearDown(const ::benchmark::State& state) override {
        // Cleanup if needed
    }
};

// ============================================================================
// Benchmark Macros for Different Implementations
// ============================================================================

#define BENCHMARK_SORT_IMPL(ImplName, FactoryFunc, Size, MaxVal, Dist) \
    BENCHMARK_DEFINE_F(SortBenchmarkFixture, ImplName##_##Size##_##MaxVal##_##Dist)(benchmark::State& state) { \
        int rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        \
        for (auto _ : state) { \
            state.PauseTiming(); \
            auto data = generateTestData(Size, MaxVal, Distribution::Dist); \
            auto sorter = FactoryFunc(); \
            state.ResumeTiming(); \
            \
            sorter->sort(data.data(), data.size()); \
            \
            state.PauseTiming(); \
            if (rank == 0 && !isSorted(data)) { \
                state.SkipWithError("Sort failed: output not sorted"); \
            } \
            state.ResumeTiming(); \
        } \
        \
        if (rank == 0) { \
            state.SetItemsProcessed(state.iterations() * Size); \
            state.SetBytesProcessed(state.iterations() * Size * sizeof(int)); \
            state.counters["Elements"] = Size; \
            state.counters["MaxValue"] = MaxVal; \
            state.counters["Throughput(elem/s)"] = benchmark::Counter( \
                state.iterations() * Size, \
                benchmark::Counter::kIsRate \
            ); \
        } \
    }

// ============================================================================
// CPU-Based SNP Sort Benchmarks
// ============================================================================

// Small inputs
BENCHMARK_SORT_IMPL(CpuSnpSort, createSnpSort, 10, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createSnpSort, 10, 10, SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createSnpSort, 10, 10, REVERSE_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createSnpSort, 50, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createSnpSort, 100, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createSnpSort, 100, 100, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createSnpSort, 100, 100, FEW_UNIQUE)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createSnpSort, 100, 100, UNIFORM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond);

// ============================================================================
// Naive CUDA/MPI SNP Sort Benchmarks
// ============================================================================

// Small inputs
BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 10, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 10, 10, SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 10, 10, REVERSE_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 50, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 100, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 100, 100, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 100, 100, FEW_UNIQUE)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 100, 100, UNIFORM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond);

// ============================================================================
// Optimized CUDA/MPI SNP Sort Benchmarks
// ============================================================================

// Small inputs
BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 10, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 10, 10, SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 10, 10, REVERSE_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 50, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 100, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 100, 100, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 100, 100, FEW_UNIQUE)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 100, 100, UNIFORM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond);

// Medium inputs (only for distributed implementations)
BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 500, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_500_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 500, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_500_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSort, 1000, 1000, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_1000_1000_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSort, 1000, 1000, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_1000_1000_RANDOM)->Unit(benchmark::kMillisecond);

// ============================================================================
// Main Function with MPI Support
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Only rank 0 should print benchmark output
    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "SNP Sort Benchmark Suite" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    
    // Initialize benchmark
    ::benchmark::Initialize(&argc, argv);
    
    // Run benchmarks
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        MPI_Finalize();
        return 1;
    }
    
    ::benchmark::RunSpecifiedBenchmarks();
    
    // Cleanup
    ::benchmark::Shutdown();
    
    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Benchmark Complete" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
