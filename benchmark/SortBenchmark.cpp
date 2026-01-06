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
            case Distribution::RANDOM: return "random";
            case Distribution::SORTED: return "sorted";
            case Distribution::REVERSE_SORTED: return "reverse-sorted";
            case Distribution::NEARLY_SORTED: return "nearly-sorted";
            case Distribution::FEW_UNIQUE: return "few-unique";
            case Distribution::UNIFORM: return "uniform";
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
    const int runs = 16;

    // 1. Distributions Suite: Fixed size, various patterns
    const int distSize = 1024;
    const std::vector<TestConfig> Distributions = {
        TestConfig{"dist-random", distSize, distSize, Distribution::REVERSE_SORTED, runs},
        TestConfig{"dist-sorted", distSize, distSize, Distribution::SORTED, runs},
        TestConfig{"dist-reverse", distSize, distSize, Distribution::REVERSE_SORTED, runs},
        TestConfig{"dist-nearly", distSize, distSize, Distribution::NEARLY_SORTED, runs},
        TestConfig{"dist-few-unique", distSize, distSize, Distribution::FEW_UNIQUE, runs},
        TestConfig{"dist-uniform", distSize, distSize, Distribution::UNIFORM, runs}
    };

    // 2. Scaling Suite: Powers of two
    const std::vector<TestConfig> Scaling = [] {
        std::vector<TestConfig> c;
        for (int s = 4; s <= 2048; s *= 2) {
            c.push_back(TestConfig{"scaling", (size_t)s, s, Distribution::REVERSE_SORTED, runs});
        }
        return c;
    }();
}

// ============================================================================
// 6. Main
// ============================================================================

namespace {
    std::vector<std::string> SplitString(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // --- REGISTRIES ---
    
    std::map<std::string, SorterFactory> simRegistry = {
        {"cpu", createNaiveCpuSnpSort},

        // CUDA
        {"optimized-cuda", createOptimizedCudaSnpSort},
        {"sparse-cuda", createSparseCudaSnpSort},

        // Naive MPI
        {"naive-cuda-mpi:linear", []() { return createNaiveCudaMpiSnpSort(PartitionerType::LINEAR); }},
        {"naive-cuda-mpi:louvain", []() { return createNaiveCudaMpiSnpSort(PartitionerType::LOUVAIN); }},
        {"naive-cuda-mpi:red-blue", []() { return createNaiveCudaMpiSnpSort(PartitionerType::RED_BLUE_BFS); }},
        
        // Optimized MPI
        {"optimized-cuda-mpi:linear", []() { return createOptimizedCudaMpiSnpSort(PartitionerType::LINEAR); }},
        {"optimized-cuda-mpi:louvain", []() { return createOptimizedCudaMpiSnpSort(PartitionerType::LOUVAIN); }},
        {"optimized-cuda-mpi:red-blue", []() { return createOptimizedCudaMpiSnpSort(PartitionerType::RED_BLUE_BFS); }}
    };

    std::map<std::string, std::vector<BenchUtils::TestConfig>> suiteRegistry = {
        {"distributions", Suites::Distributions},
        {"scaling", Suites::Scaling}
    };

    // --- ARGUMENT PARSING ---

    std::vector<std::string> requestedImpls;
    std::vector<std::string> requestedSuites;
    std::vector<char*> benchArgvList;
    benchArgvList.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--impls=") == 0) {
            requestedImpls = SplitString(arg.substr(8), ',');
        } else if (arg.find("--suites=") == 0) {
            requestedSuites = SplitString(arg.substr(9), ',');
        } else {
            benchArgvList.push_back(argv[i]);
        }
    }

    // Default fallbacks if no specific args provided
    if (requestedImpls.empty() && requestedSuites.empty()) {
        requestedImpls = {
            "cpu", 
            "optimized-cuda", 
            "optimized-cuda-mpi:linear"
        };
        requestedSuites = {"scaling"};
    } else {
        if (requestedImpls.empty()) requestedImpls = {"cpu"}; 
        if (requestedSuites.empty()) requestedSuites = {"scaling"};
    }

    // --- REGISTRATION PHASE ---

    for (const auto& implName : requestedImpls) {
        if (simRegistry.count(implName)) {
            for (const auto& suiteName : requestedSuites) {
                if (suiteRegistry.count(suiteName)) {
                    RegisterSimulator(implName, simRegistry[implName], suiteRegistry[suiteName]);
                } else if (rank == 0) {
                    std::cerr << "Warning: Unknown suite '" << suiteName << "' ignored.\n";
                }
            }
        } else if (rank == 0) {
            std::cerr << "Warning: Unknown simulator '" << implName << "' ignored.\n";
        }
    }

    // Prepared args for Google Benchmark
    int benchArgc = static_cast<int>(benchArgvList.size());
    char** benchArgv = benchArgvList.data();

    // --- EXECUTION PHASE ---
    // Only rank 0 initializes benchmark with args to handle output file writing
    if (rank == 0) {
        ::benchmark::Initialize(&benchArgc, benchArgv);
        ::benchmark::RunSpecifiedBenchmarks();
    } else {
        // Non-root ranks: minimal args to prevent file output
        // We pass the filtered args, assuming --benchmark_... flags are handled by rank 0 mostly or don't affect file IO directly in Initialize in a hazardous way for MPI workers,
        // BUT the original code used minimal args for workers. Let's respect that safety.
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