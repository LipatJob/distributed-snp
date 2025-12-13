#include "../src/sort/ISort.hpp"
#include "../src/snp/ISnpSimulator.hpp"
#include "../src/snp/SnpSystemConfig.hpp"
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
std::unique_ptr<ISort> createNaiveCpuSnpSort();
std::unique_ptr<ISort> createCudaSnpSort();
std::unique_ptr<ISort> createSparseCudaSnpSort();
std::unique_ptr<ISort> createNaiveCudaMpiSnpSort();
std::unique_ptr<ISort> createCudaMpiSnpSort();

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
                std::fill(data.begin(), data.end(), valDist(rng));
                break;
            default: // Random and others
                for(auto& x : data) x = valDist(rng);
                break;
        }
        return data;
    }

    bool IsSorted(const std::vector<int>& data) {
        return std::is_sorted(data.begin(), data.end());
    }
    
    // Helper to parse metrics
    double ExtractMetric(const std::string& report, const std::string& key) {
        size_t pos = report.find(key);
        if (pos == std::string::npos) return 0.0;
        size_t numStart = report.find_first_of("0123456789.", pos);
        if (numStart == std::string::npos) return 0.0;
        return std::stod(report.substr(numStart));
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

            // 6. Report Metrics
            std::string report = sorter->getPerformanceReport();
            double commTime = BenchUtils::ExtractMetric(report, "Comm Time");
            
            if (rank == 0 && commTime > 0.0) {
                state.counters["Comm_ms"] = benchmark::Counter(commTime, benchmark::Counter::kAvgIterations);
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
        
        // Register the benchmark dynamically
        auto* b = benchmark::RegisterBenchmark(testName.c_str(), 
            [factory, cfg](benchmark::State& st) {
                SortFixture fixture;
                fixture.SetUp(st);
                fixture.RunTest(st, factory, cfg);
                fixture.TearDown(st);
            });
            
        // Apply configuration specifics
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

    const std::vector<TestConfig> Small = {
        {"Small_Rand", 100, 100, Distribution::RANDOM, 10},
        {"Small_Sort", 100, 100, Distribution::SORTED, 10},
    };

    const std::vector<TestConfig> Medium = {
        {"Med_Rand",  1000, 1000, Distribution::RANDOM, 1},
        {"Med_Near",  1000, 1000, Distribution::NEARLY_SORTED, 1},
    };

    const std::vector<TestConfig> Large = {
        {"Lrg_Rand",  5000, 5000, Distribution::RANDOM, 1},
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
    RegisterSimulator("CpuSnp", createNaiveCpuSnpSort, Suites::Medium);

    // 2. CUDA
    RegisterSimulator("CudaSnp", createCudaSnpSort, Suites::Small);
    RegisterSimulator("CudaSnp", createCudaSnpSort, Suites::Medium);

    // 3. Sparse CUDA
    RegisterSimulator("SparseCudaSnp", createSparseCudaSnpSort, Suites::Small);
    RegisterSimulator("SparseCudaSnp", createSparseCudaSnpSort, Suites::Medium);

    // 4. Naive CUDA/MPI
    RegisterSimulator("NaiveCudaMpiSnp", createNaiveCudaMpiSnpSort, Suites::Small);
    RegisterSimulator("NaiveCudaMpiSnp", createNaiveCudaMpiSnpSort, Suites::Medium);

    // 5. CUDA/MPI
    RegisterSimulator("CudaMpiSnp", createCudaMpiSnpSort, Suites::Small);
    RegisterSimulator("CudaMpiSnp", createCudaMpiSnpSort, Suites::Medium);


    // --- EXECUTION PHASE ---
    if (rank == 0) {
        std::cout << "SNP Benchmark Suite Initialized." << std::endl;
        std::cout << "Usage: ./bench --benchmark_filter=<Regex>" << std::endl;
        std::cout << "Examples:" << std::endl; 
        std::cout << "  ./bench --benchmark_filter=\"Cuda\"     (Run only CUDA tests)" << std::endl;
        std::cout << "  ./bench --benchmark_filter=\"Small\"    (Run only small inputs)" << std::endl;
    }

    // Only rank 0 initializes benchmark arguments to handle output file writing
    if (rank == 0) {
        ::benchmark::Initialize(&argc, argv);
        ::benchmark::RunSpecifiedBenchmarks();
    } else {
        // Non-root ranks: Initialize with minimal args (no output file arguments)
        // This prevents non-root ranks from attempting to write to output files
        int argc_minimal = 1;
        char* argv_minimal[] = {argv[0]};
        ::benchmark::Initialize(&argc_minimal, argv_minimal);
        
        // Use a null reporter to suppress all output
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