#include "../src/sort/ISort.hpp"
#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include "IPartitioner.hpp"
#include "PerformanceMetrics.hpp"
#include <benchmark/benchmark.h>
#include <mpi.h>
#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include <map>
#include <iostream>
#include <functional>
#include <string>

// ============================================================================
// 1. Factory Wrappers (Add new simulators here)
// ============================================================================

// Forward declarations
// (Declarations are now in ISort.hpp)

// Type alias for cleaner code
using SorterFactory = std::function<std::unique_ptr<ISort>()>;

// ============================================================================
// 2. Data Generation & Utils
// ============================================================================

namespace BenchUtils {

    enum class Distribution {
        RANDOM, SORTED, REVERSE_SORTED, NEARLY_SORTED, FEW_UNIQUE, UNIFORM
    };

    struct TestConfig {
        std::string name;
        size_t size;
        int maxVal;
        Distribution dist;
        int iterations = 0; // 0 = default, 1 = forced (needed for heavy MPI)
    };

    std::string DistToString(Distribution d) {
        switch(d) {
            case Distribution::RANDOM: return "Random";
            case Distribution::SORTED: return "Sorted";
            case Distribution::REVERSE_SORTED: return "Reverse";
            case Distribution::NEARLY_SORTED: return "NearlySorted";
            case Distribution::FEW_UNIQUE: return "FewUnique";
            case Distribution::UNIFORM: return "Uniform";
            default: return "Unknown";
        }
    }

    std::vector<int> GenerateData(size_t size, int maxValue, Distribution dist, unsigned seed) {
        std::vector<int> data(size);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> valDist(0, maxValue);

        // (Keeping generation logic compact for brevity - insert your full logic here)
        switch (dist) {
            case Distribution::SORTED:
                for(auto& x : data) x = valDist(rng);
                std::sort(data.begin(), data.end());
                break;
            case Distribution::REVERSE_SORTED:
                for(auto& x : data) x = valDist(rng);
                std::sort(data.begin(), data.end(), std::greater<int>());
                break;
            case Distribution::UNIFORM:
                std::fill(data.begin(), data.end(), maxValue);
                break;
            case Distribution::NEARLY_SORTED:
                for(auto& x : data) x = valDist(rng);
                std::sort(data.begin(), data.end());
                for (size_t i = 0; i < size / 10; ++i) {
                    size_t idx1 = rng() % size;
                    size_t idx2 = rng() % size;
                    std::swap(data[idx1], data[idx2]);
                }
                break;
            case Distribution::FEW_UNIQUE: {
                int uniqueCount = std::max(2, maxValue / 10);
                std::vector<int> uniqueValues(uniqueCount);
                for (auto& x : uniqueValues) x = valDist(rng) % maxValue;
                for (auto& x : data) x = uniqueValues[rng() % uniqueCount];
                break;
            }
            case Distribution::RANDOM: {
                for(auto& x : data) x = valDist(rng);
                break;
            }
            default: // Random and others
                for(auto& x : data) x = valDist(rng);
                break;
        }
        return data;
    }

    bool IsSorted(const std::vector<int>& data) {
        return std::is_sorted(data.begin(), data.end());
    }
}

// ============================================================================
// 3. Unified Benchmark Fixture
// ============================================================================

class SortFixture {
protected:
    int rank;
    int world_size;
    
public:
    void SetUp(const ::benchmark::State& state) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    // The Core Benchmark Logic
    void RunTest(benchmark::State& state, SorterFactory factory, BenchUtils::TestConfig config) {
        for (auto _ : state) {
            state.PauseTiming();
            
            // 1. Generate Data
            auto data = BenchUtils::GenerateData(config.size, config.maxVal, config.dist, 42 + state.iterations());
            
            // 2. Create Sorter
            auto sorter = factory();
            sorter->load(data.data(), data.size());
            
            // 3. Sync before start
            MPI_Barrier(MPI_COMM_WORLD);
            
            state.ResumeTiming();
            
            // 4. Execute
            auto result = sorter->execute();
            
            state.PauseTiming();

            // 5. Verify (Rank 0 only)
            if (rank == 0 && !BenchUtils::IsSorted(result)) {
                state.SkipWithError("Output is not sorted!");
            }

            // 6. Extract Structured Metrics
            PerformanceMetrics metrics = sorter->getPerformanceMetrics();
            
            if (rank == 0) {
                // Report detailed metrics as custom counters
                if (metrics.mpi.has_data()) {
                    state.counters["MPI_Comm_ms"] = benchmark::Counter(
                        metrics.mpi.communication_time_ms, 
                        benchmark::Counter::kAvgIterations
                    );
                    state.counters["MPI_Messages"] = benchmark::Counter(
                        metrics.mpi.total_messages_sent, 
                        benchmark::Counter::kAvgIterations
                    );
                    state.counters["MPI_Bytes"] = benchmark::Counter(
                        metrics.mpi.total_bytes_sent, 
                        benchmark::Counter::kAvgIterations
                    );
                    state.counters["Comm_Pct"] = benchmark::Counter(
                        metrics.communication_percentage(),
                        benchmark::Counter::kAvgIterations
                    );
                }
                
                if (metrics.cuda.has_data()) {
                    state.counters["CUDA_Kernel_ms"] = benchmark::Counter(
                        metrics.cuda.kernel_time_ms,
                        benchmark::Counter::kAvgIterations
                    );
                    state.counters["CUDA_Memory_ms"] = benchmark::Counter(
                        metrics.cuda.memory_transfer_time_ms,
                        benchmark::Counter::kAvgIterations
                    );
                    state.counters["CUDA_Mem_MB"] = benchmark::Counter(
                        metrics.cuda.device_memory_allocated / (1024.0 * 1024.0),
                        benchmark::Counter::kAvgIterations
                    );
                }
                
                state.counters["Compute_ms"] = benchmark::Counter(
                    metrics.compute_time_ms,
                    benchmark::Counter::kAvgIterations
                );
                state.counters["Steps"] = benchmark::Counter(
                    metrics.steps_executed,
                    benchmark::Counter::kAvgIterations
                );
                state.counters["Throughput_steps/s"] = benchmark::Counter(
                    metrics.throughput_steps_per_second(),
                    benchmark::Counter::kAvgIterations
                );
            }
            
            // Cleanup
            sorter.reset();
            state.ResumeTiming();
        }

        if (rank == 0) {
            state.counters["Size"] = config.size;
            state.counters["Procs"] = world_size;
        }
    }

    void TearDown(const ::benchmark::State& state) {
        // Optional cleanup can go here. 
        // Currently empty, but must exist to satisfy the function call.
    }
};

// ============================================================================
// 4. Registration System (The "Easy to Add" Part)
// ============================================================================

// Helper to register a specific simulator with a list of configurations
void RegisterSimulator(std::string name, SorterFactory factory, const std::vector<BenchUtils::TestConfig>& configs) {
    for (const auto& cfg : configs) {
        std::string testName = name + "/" + BenchUtils::DistToString(cfg.dist) + "/" + std::to_string(cfg.size) + "/" + std::to_string(cfg.maxVal);
        
        auto* b = benchmark::RegisterBenchmark(testName.c_str(), 
            [factory, cfg](benchmark::State& st) {
                SortFixture fixture;
                fixture.SetUp(st);
                fixture.RunTest(st, factory, cfg);
                fixture.TearDown(st);
            });
            
        b->Unit(benchmark::kMillisecond);
        if (cfg.iterations > 0) {
            b->Iterations(cfg.iterations);
        }
    }
}

// ============================================================================
// 5. Test Suites (Selectable Groups)
// ============================================================================

namespace Suites {
    using namespace BenchUtils;

    const int tinySize = 16;
    const std::vector<TestConfig> Tiny = {
        {"Tiny_Sort", tinySize, tinySize, Distribution::SORTED, 64},
        {"Tiny_RevSort", tinySize, tinySize, Distribution::REVERSE_SORTED, 64},
        {"Tiny_Rand", tinySize, tinySize, Distribution::RANDOM, 64},
    };

    const int smallSize = 256;
    const std::vector<TestConfig> Small = {
        {"Small_Sort", smallSize, smallSize, Distribution::SORTED, 32},
        {"Small_RevSort", smallSize, smallSize, Distribution::REVERSE_SORTED, 32},
        {"Small_Rand", smallSize, smallSize, Distribution::RANDOM, 32},
    };

    const int mediumSize = 2048;
    const std::vector<TestConfig> Medium = {
        {"Medium_Sort", mediumSize, mediumSize, Distribution::SORTED, 16},
        {"Medium_RevSort", mediumSize, mediumSize, Distribution::REVERSE_SORTED, 16},
        {"Medium_Rand", mediumSize, mediumSize, Distribution::RANDOM, 16},
    };

    const int largeSize = 8192;
    const std::vector<TestConfig> Large = {
        {"Large_Sort", largeSize, largeSize, Distribution::SORTED, 4},
        {"Large_RevSort", largeSize, largeSize, Distribution::REVERSE_SORTED, 4},
        {"Large_Rand", largeSize, largeSize, Distribution::RANDOM, 4},
    };
    
    // Combine vectors helper
    std::vector<TestConfig> All() {
        std::vector<TestConfig> all = Small;
        all.insert(all.end(), Medium.begin(), Medium.end());
        all.insert(all.end(), Large.begin(), Large.end());
        return all;
    }
}

// ============================================================================
// 6. Main
// ============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // --- REGISTRATION PHASE ---
    // This is where you select which simulators run "by default" or add new ones.
    
    // 1. CPU
    RegisterSimulator("CpuSnp", createNaiveCpuSnpSort, Suites::Small);

    // 2. Sparse CUDA
    RegisterSimulator("SparseCudaSnp", createSparseCudaSnpSort, Suites::Small);
    RegisterSimulator("SparseCudaSnp", createSparseCudaSnpSort, Suites::Medium);

    // 3. CUDA
    RegisterSimulator("OptimizedCudaSnp", createOptimizedCudaSnpSort, Suites::Small);
    RegisterSimulator("OptimizedCudaSnp", createOptimizedCudaSnpSort, Suites::Medium);

    // 4. Naive CUDA/MPI
    RegisterSimulator("NaiveCudaMpiSnp", []()
                      { return createNaiveCudaMpiSnpSort(); }, Suites::Small);
    RegisterSimulator("NaiveCudaMpiSnp", []()
                      { return createNaiveCudaMpiSnpSort(); }, Suites::Medium);

    // 5. CUDA/MPI (Linear - Default)
    RegisterSimulator("OptimizedCudaMpiSnp_Linear", []()
                      { return createOptimizedCudaMpiSnpSort(PartitionerType::LINEAR); }, Suites::Small);
    RegisterSimulator("OptimizedCudaMpiSnp_Linear", []()
                      { return createOptimizedCudaMpiSnpSort(PartitionerType::LINEAR); }, Suites::Medium);

    // 5b. CUDA/MPI (Louvain)
    RegisterSimulator("OptimizedCudaMpiSnp_Louvain", []()
                      { return createOptimizedCudaMpiSnpSort(PartitionerType::LOUVAIN); }, Suites::Small);
    RegisterSimulator("OptimizedCudaMpiSnp_Louvain", []()
                      { return createOptimizedCudaMpiSnpSort(PartitionerType::LOUVAIN); }, Suites::Medium);

    // 5c. CUDA/MPI (Red-Blue)
    RegisterSimulator("OptimizedCudaMpiSnp_RedBlue", []()
                      { return createOptimizedCudaMpiSnpSort(PartitionerType::RED_BLUE_BFS); }, Suites::Small);
    RegisterSimulator("OptimizedCudaMpiSnp_RedBlue", []()
                      { return createOptimizedCudaMpiSnpSort(PartitionerType::RED_BLUE_BFS); }, Suites::Medium);

    // --- EXECUTION PHASE ---
    // Only rank 0 initializes benchmark with args to handle output file writing
    if (rank == 0) {
        ::benchmark::Initialize(&argc, argv);
        ::benchmark::RunSpecifiedBenchmarks();
    } else {
        // Non-root ranks: minimal args to prevent file output
        int argc_minimal = 1;
        char* argv_minimal[] = {argv[0]};
        ::benchmark::Initialize(&argc_minimal, argv_minimal);
        
        class NullReporter : public ::benchmark::BenchmarkReporter {
            bool ReportContext(const Context&) override { return true; }
            void ReportRuns(const std::vector<Run>&) override {}
            void Finalize() override {}
        };
        NullReporter null_reporter;
        ::benchmark::RunSpecifiedBenchmarks(&null_reporter);
    }

    MPI_Finalize();
    return 0;
}