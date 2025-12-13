#include "../src/sort/ISort.hpp"
#include "../src/snp/ISnpSimulator.hpp"
#include "../src/snp/SnpSystemConfig.hpp"
#include <benchmark/benchmark.h>
#include <mpi.h>
#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include <cmath>
#include <sstream>
#include <map>
#include <iostream>

// Forward declare simulator factory functions (from ISnpSimulator.hpp)
std::unique_ptr<ISnpSimulator> createNaiveCpuSimulator();
std::unique_ptr<ISnpSimulator> createCudaSimulator();
std::unique_ptr<ISnpSimulator> createCudaMpiSimulator();

// Forward declare sorter factory functions (from ISort.hpp)
std::unique_ptr<ISort> createNaiveCpuSnpSort();
std::unique_ptr<ISort> createCudaSnpSort();
std::unique_ptr<ISort> createNaiveCudaMpiSnpSort();
std::unique_ptr<ISort> createCudaMpiSnpSort();

// Wrapper functions to create sorters for benchmark macro
inline std::unique_ptr<ISort> createNaiveCpuSnpSorter() {
    return createNaiveCpuSnpSort();
}

inline std::unique_ptr<ISort> createCudaSnpSorter() {
    return createCudaSnpSort();
}

inline std::unique_ptr<ISort> createNaiveCudaMpiSnpSorter() {
    return createNaiveCudaMpiSnpSort();
}

inline std::unique_ptr<ISort> createCudaMpiSnpSorter() {
    return createCudaMpiSnpSort();
}

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
    std::unique_ptr<ISort> sorter;
    std::vector<int> testData;
    int maxVal;
    
public:
    void SetUp(const ::benchmark::State& state) override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        
        // Note: test data and sorter will be set up in the benchmark itself
        // because we need different data for each iteration
    }
    
    void TearDown(const ::benchmark::State& state) override {
        sorter.reset();
    }
};

// ============================================================================
// Benchmark Macros for Different Implementations
// ============================================================================

#define BENCHMARK_SORT_IMPL(ImplName, FactoryFunc, Size, MaxVal, Dist) \
    BENCHMARK_DEFINE_F(SortBenchmarkFixture, ImplName##_##Size##_##MaxVal##_##Dist)(benchmark::State& state) { \
        /* Setup phase - not timed */ \
        maxVal = MaxVal; \
        \
        /* Each iteration gets fresh state */ \
        for (auto _ : state) { \
            /* Pause timing for setup */ \
            state.PauseTiming(); \
            \
            /* Generate fresh test data for this iteration */ \
            testData = generateTestData(Size, MaxVal, Distribution::Dist, 42 + state.iterations()); \
            \
            /* Create sorter instance */ \
            sorter = FactoryFunc(); \
            \
            /* Load data and build SNP system (not timed) */ \
            sorter->load(testData.data(), testData.size()); \
            \
            /* All MPI ranks must synchronize before timing */ \
            MPI_Barrier(MPI_COMM_WORLD); \
            \
            /* Resume timing for actual computation */ \
            state.ResumeTiming(); \
            \
            /* Execute the sort (timed) */ \
            std::vector<int> result = sorter->execute(); \
            \
            /* Stop timing before validation */ \
            state.PauseTiming(); \
            \
            /* Verify the result is sorted (only on rank 0) */ \
            if (rank == 0) { \
                if (!isSorted(result)) { \
                    state.SkipWithError("Output is not sorted!"); \
                    break; \
                } \
            } \
            \
            /* Extract and report communication time from performance report */ \
            std::string perfReport = sorter->getPerformanceReport(); \
            double commTime = extractCommTime(perfReport); \
            double computeTime = extractComputeTime(perfReport); \
            \
            if (rank == 0 && commTime > 0.0) { \
                state.counters["CommTime_ms"] = benchmark::Counter(commTime, benchmark::Counter::kAvgIterations); \
            } \
            if (rank == 0 && computeTime > 0.0) { \
                state.counters["ComputeTime_ms"] = benchmark::Counter(computeTime, benchmark::Counter::kAvgIterations); \
            } \
            \
            /* Clean up sorter for next iteration */ \
            sorter.reset(); \
            \
            state.ResumeTiming(); \
        } \
        \
        /* Report problem size */ \
        if (rank == 0) { \
            state.counters["InputSize"] = benchmark::Counter(Size, benchmark::Counter::kDefaults); \
            state.counters["MaxValue"] = benchmark::Counter(MaxVal, benchmark::Counter::kDefaults); \
            state.counters["NumProcesses"] = benchmark::Counter(world_size, benchmark::Counter::kDefaults); \
        } \
    }

// ============================================================================
// Helper functions to extract metrics from performance report
// ============================================================================

double extractCommTime(const std::string& report) {
    // Parse communication time from the performance report
    // Expected format includes "Comm Time: X ms" or similar
    size_t pos = report.find("Comm Time");
    if (pos == std::string::npos) {
        pos = report.find("Comm Time");
    }
    if (pos == std::string::npos) return 0.0;
    
    // Look for the number after the word
    size_t numStart = report.find_first_of("0123456789.", pos);
    if (numStart == std::string::npos) return 0.0;
    
    size_t numEnd = report.find_first_not_of("0123456789.", numStart);
    std::string numStr = report.substr(numStart, numEnd - numStart);
    
    try {
        return std::stod(numStr);
    } catch (...) {
        return 0.0;
    }
}

double extractComputeTime(const std::string& report) {
    // Parse compute time from the performance report
    // Expected format includes "Compute: X ms" or similar
    size_t pos = report.find("Compute");
    if (pos == std::string::npos) {
        pos = report.find("Compute");
    }
    if (pos == std::string::npos) return 0.0;
    
    // Look for the number after the word
    size_t numStart = report.find_first_of("0123456789.", pos);
    if (numStart == std::string::npos) return 0.0;
    
    size_t numEnd = report.find_first_not_of("0123456789.", numStart);
    std::string numStr = report.substr(numStart, numEnd - numStart);
    
    try {
        return std::stod(numStr);
    } catch (...) {
        return 0.0;
    }
}

// ============================================================================
// CPU-Based SNP Sort Benchmarks
// ============================================================================

// Small inputs
// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 10, 10, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 10, 10, SORTED)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 10, 10, REVERSE_SORTED)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 50, 10, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 100, 100, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 100, 100, NEARLY_SORTED)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 100, 100, FEW_UNIQUE)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 100, 100, UNIFORM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond);

// // Medium inputs
// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 200, 200, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_200_200_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 500, 500, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_500_500_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 1000, 1000, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_1000_1000_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CpuSnpSort, createNaiveCpuSnpSimulator, 1000, 1000, NEARLY_SORTED)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_1000_1000_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

// ============================================================================
// CUDA SNP Sort Benchmarks
// ============================================================================

// Small inputs
// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 10, 10, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 10, 10, SORTED)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 10, 10, REVERSE_SORTED)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 50, 10, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 100, 100, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 100, 100, NEARLY_SORTED)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 100, 100, FEW_UNIQUE)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 100, 100, UNIFORM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond);

// // Medium inputs
// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 200, 200, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_200_200_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 500, 500, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_500_500_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 1000, 1000, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_1000_1000_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 1000, 1000, NEARLY_SORTED)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_1000_1000_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

// // Large inputs
// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 2000, 2000, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_2000_2000_RANDOM)->Unit(benchmark::kMillisecond);

// BENCHMARK_SORT_IMPL(CudaSnpSort, createCudaSnpSimulator, 5000, 5000, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaSnpSort_5000_5000_RANDOM)->Unit(benchmark::kMillisecond);


// ============================================================================
// Naive CUDA/MPI SNP Sort Benchmarks
// ============================================================================

// Small inputs
// NOTE: ->Iterations(1) is CRITICAL for MPI benchmarks to prevent deadlock
// All ranks must run the same number of iterations to stay synchronized
BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 10, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 10, 10, SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 10, 10, REVERSE_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 50, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 100, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 100, 100, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 100, 100, FEW_UNIQUE)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 100, 100, UNIFORM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond)->Iterations(1);

// Medium inputs
BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 500, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_500_100_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 1000, 1000, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_1000_1000_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 1000, 1000, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_1000_1000_NEARLY_SORTED)->Unit(benchmark::kMillisecond)->Iterations(1);

// Large inputs
BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSorter, 2000, 2000, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_2000_2000_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(1);

// ============================================================================
// Optimized CUDA/MPI SNP Sort Benchmarks
// ============================================================================

// Small inputs
// NOTE: ->Iterations(1) is CRITICAL for MPI benchmarks to prevent deadlock
// All ranks must run the same number of iterations to stay synchronized
BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 10, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 10, 10, SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 10, 10, REVERSE_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 50, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(10);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 100, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 100, 100, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 100, 100, FEW_UNIQUE)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 100, 100, UNIFORM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond)->Iterations(1);

// Medium inputs

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 500, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_500_100_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 1000, 1000, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_1000_1000_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 1000, 1000, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_1000_1000_NEARLY_SORTED)->Unit(benchmark::kMillisecond)->Iterations(1);

// Large inputs
BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSorter, 2000, 2000, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_2000_2000_RANDOM)->Unit(benchmark::kMillisecond)->Iterations(1);

// BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 5000, 5000, RANDOM)
// BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_5000_5000_RANDOM)->Unit(benchmark::kMillisecond);

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
    
    // CRITICAL FIX: Remove --benchmark_out argument for non-root ranks
    // to prevent multiple processes writing to the same file
    std::vector<char*> filtered_argv;
    for (int i = 0; i < argc; ++i) {
        std::string arg(argv[i]);
        // Skip --benchmark_out and --benchmark_format args on non-root ranks
        if (rank != 0 && (arg.find("--benchmark_out") == 0 || arg.find("--benchmark_format") == 0)) {
            continue;
        }
        filtered_argv.push_back(argv[i]);
    }
    int filtered_argc = filtered_argv.size();
    
    // Initialize benchmark with filtered arguments
    ::benchmark::Initialize(&filtered_argc, filtered_argv.data());
    
    // Run benchmarks
    if (::benchmark::ReportUnrecognizedArguments(filtered_argc, filtered_argv.data())) {
        MPI_Finalize();
        return 1;
    }
    
    // CRITICAL: All MPI ranks must run benchmarks together to avoid deadlock
    // in collective operations (MPI_Allgatherv, MPI_Barrier, etc.)
    // Only rank 0 should output results to avoid duplicate reporting
    if (rank == 0) {
        ::benchmark::RunSpecifiedBenchmarks();
    } else {
        // Non-root ranks: run benchmarks but suppress console output
        // We create a null reporter that discards all output
        class NullReporter : public ::benchmark::BenchmarkReporter {
        public:
            bool ReportContext(const Context&) override { return true; }
            void ReportRuns(const std::vector<Run>&) override {}
            void Finalize() override {}
        };
        
        NullReporter null_reporter;
        ::benchmark::RunSpecifiedBenchmarks(&null_reporter);
    }
    
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
