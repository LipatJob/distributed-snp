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

// Forward declare simulator factory functions (from ISnpSimulator.hpp)
std::unique_ptr<ISnpSimulator> createNaiveCpuSimulator();
std::unique_ptr<ISnpSimulator> createNaiveCudaMpiSimulator();
std::unique_ptr<ISnpSimulator> createCudaMpiSimulator();

// Wrapper functions to create simulators for benchmark macro
inline std::unique_ptr<ISnpSimulator> createCpuSnpSimulator() {
    return createNaiveCpuSimulator();
}

inline std::unique_ptr<ISnpSimulator> createCudaMpiSnpSimulator() {
    return createCudaMpiSimulator();
}

inline std::unique_ptr<ISnpSimulator> createNaiveCudaMpiSnpSimulator() {
    return createNaiveCudaMpiSimulator();
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
// Helper class to build SNP system for sorting (extracted from SnpSort.cpp)
// ============================================================================

SnpSystemConfig buildSortingSystem(const std::vector<int>& inputNumbers) {
    int N = inputNumbers.size();
    
    // Layout: [ Inputs (0..N-1) | Sorters (N..2N-1) | Outputs (2N..3N-1) ]
    int startInput = 0;
    int startSorter = N;
    int startOutput = 2 * N;
    int totalNeurons = 3 * N;

    SnpSystemBuilder builder;
    
    // Create all neurons
    for (int i = 0; i < totalNeurons; ++i) {
        builder.addNeuron(i, 0);
    }

    // Build Topology
    // Inputs -> All Sorters
    for(int i = 0; i < N; ++i) {
        for(int s = 0; s < N; ++s) {
            builder.addSynapse(startInput + i, startSorter + s);
        }
    }

    // Sorters -> Outputs
    for(int s = 0; s < N; ++s) {
        for(int o = s; o < N; ++o) {
            builder.addSynapse(startSorter + s, startOutput + o);
        }
    }

    // Define Rules
    // Input Streams (Neurons 0 to N-1)
    for(int i = 0; i < N; ++i) {
        int neuronId = startInput + i;
        int spikes = inputNumbers[i];
        
        builder.addNeuron(neuronId, spikes);
        builder.addRule(neuronId, 1, 1, 1, 0, 1);
    }

    // Sorters (Neurons N to 2N-1)
    for(int s = 0; s < N; ++s) {
        int sorterID = startSorter + s;
        int target = N - s;

        // 1. Forget High (Higher priority)
        for(int k = N; k > target; --k) {
            builder.addRule(sorterID, k, k, 0, 0, k);
        }
        
        // 2. Fire Exact (at target count)
        builder.addRule(sorterID, target, target, 1, 0, target);

        // 3. Forget Low (Lower priority)
        for(int k = target - 1; k >= 1; --k) {
            builder.addRule(sorterID, k, k, 0, 0, k);
        }
    }

    // Mark output neurons
    SnpSystemConfig config = builder.build();
    for(int o = 0; o < N; ++o) {
        config.neurons[startOutput + o].is_output = true;
    }
    
    return config;
}

// ============================================================================
// Benchmark Fixture
// ============================================================================

class SortBenchmarkFixture : public benchmark::Fixture {
protected:
    int rank;
    int world_size;
    std::unique_ptr<ISnpSimulator> simulator;
    SnpSystemConfig config;
    std::vector<int> testData;
    int maxVal;
    
public:
    void SetUp(const ::benchmark::State& state) override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        
        // Note: test data and simulator will be set up in the benchmark itself
        // because we need different data for each iteration
    }
    
    void TearDown(const ::benchmark::State& state) override {
        simulator.reset();
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
        double totalCommTime = 0.0; \
        double totalComputeTime = 0.0; \
        int iterationCount = 0; \
        \
        for (auto _ : state) { \
            state.PauseTiming(); \
            \
            /* Generate test data */ \
            testData = generateTestData(Size, MaxVal, Distribution::Dist); \
            \
            /* Calculate max value for simulation ticks */ \
            maxVal = 0; \
            for(int n : testData) { \
                if(n > maxVal) maxVal = n; \
            } \
            \
            /* Build SNP system configuration (setup - not timed) */ \
            config = buildSortingSystem(testData); \
            \
            /* Create simulator (setup - not timed) */ \
            simulator = FactoryFunc(); \
            \
            /* Load system into simulator (setup - not timed) */ \
            if (!simulator->loadSystem(config)) { \
                state.SkipWithError("Failed to load SNP system"); \
                break; \
            } \
            \
            /* Reset state before timing starts */ \
            simulator->reset(); \
            \
            state.ResumeTiming(); \
            \
            /* TIMED SECTION: Only the simulation execution */ \
            int ticks = maxVal + 3; \
            simulator->step(ticks); \
            \
            state.PauseTiming(); \
            \
            /* Extract results and verify correctness */ \
            std::vector<int> localState = simulator->getLocalState(); \
            std::vector<int> result; \
            int N = Size; \
            int startOutput = 2 * N; \
            for(int o = 0; o < N; ++o) { \
                int outputIdx = startOutput + o; \
                if (outputIdx < static_cast<int>(localState.size())) { \
                    result.push_back(localState[outputIdx]); \
                } else { \
                    result.push_back(0); \
                } \
            } \
            \
            if (rank == 0 && !isSorted(result)) { \
                state.SkipWithError("Sort failed: output not sorted"); \
            } \
            \
            /* Extract communication metrics from performance report */ \
            std::string perfReport = simulator->getPerformanceReport(); \
            double commTime = extractCommTime(perfReport); \
            double computeTime = extractComputeTime(perfReport); \
            totalCommTime += commTime; \
            totalComputeTime += computeTime; \
            iterationCount++; \
            \
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
            if (iterationCount > 0) { \
                state.counters["AvgCommTime(ms)"] = totalCommTime / iterationCount; \
                state.counters["AvgComputeTime(ms)"] = totalComputeTime / iterationCount; \
                state.counters["CommRatio(%)"] = (totalCommTime / (totalCommTime + totalComputeTime)) * 100.0; \
            } \
        } \
    }

// ============================================================================
// Helper functions to extract metrics from performance report
// ============================================================================

double extractCommTime(const std::string& report) {
    // Parse communication time from the performance report
    // Expected format includes "Communication: X ms" or similar
    size_t pos = report.find("Communication");
    if (pos == std::string::npos) {
        pos = report.find("communication");
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
        pos = report.find("compute");
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
BENCHMARK_SORT_IMPL(CpuSnpSort, createCpuSnpSimulator, 10, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createCpuSnpSimulator, 10, 10, SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createCpuSnpSimulator, 10, 10, REVERSE_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createCpuSnpSimulator, 50, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createCpuSnpSimulator, 100, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createCpuSnpSimulator, 100, 100, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createCpuSnpSimulator, 100, 100, FEW_UNIQUE)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CpuSnpSort, createCpuSnpSimulator, 100, 100, UNIFORM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CpuSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond);

// ============================================================================
// Naive CUDA/MPI SNP Sort Benchmarks
// ============================================================================

// Small inputs
BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 10, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 10, 10, SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 10, 10, REVERSE_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 50, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 100, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 100, 100, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 100, 100, FEW_UNIQUE)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 100, 100, UNIFORM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond);

// ============================================================================
// Optimized CUDA/MPI SNP Sort Benchmarks
// ============================================================================

// Small inputs
BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 10, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_10_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 10, 10, SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_10_10_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 10, 10, REVERSE_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_10_10_REVERSE_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 50, 10, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_50_10_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 100, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 100, 100, NEARLY_SORTED)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_NEARLY_SORTED)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 100, 100, FEW_UNIQUE)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_FEW_UNIQUE)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 100, 100, UNIFORM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_100_100_UNIFORM)->Unit(benchmark::kMillisecond);

// Medium inputs (only for distributed implementations)
BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 500, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_500_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 500, 100, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, CudaMpiSnpSort_500_100_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(NaiveCudaMpiSnpSort, createNaiveCudaMpiSnpSimulator, 1000, 1000, RANDOM)
BENCHMARK_REGISTER_F(SortBenchmarkFixture, NaiveCudaMpiSnpSort_1000_1000_RANDOM)->Unit(benchmark::kMillisecond);

BENCHMARK_SORT_IMPL(CudaMpiSnpSort, createCudaMpiSnpSimulator, 1000, 1000, RANDOM)
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
