#include "../sort/ISort.hpp"
#include "../snp/ISnpSimulator.hpp"
#include "../snp/SnpSystemConfig.hpp"
#include <mpi.h>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <functional>
#include <string>
#include <iomanip>
#include <cuda_profiler_api.h>


// ============================================================================
// SNP Implementations Factory Functions (Add new simulators here)
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
// Profiling Configuration
// ============================================================================

struct ProfileConfig {
    std::string name;
    SorterFactory factory;
    size_t arraySize;
    int maxValue;
    bool isDistributed;  // Requires MPI with multiple processes
};

// ============================================================================
// Data Generation Utilities
// ============================================================================

std::vector<int> generateRandomData(size_t size, int maxValue, unsigned seed = 42) {
    std::vector<int> data(size);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, maxValue);
    
    for (auto& val : data) {
        val = dist(rng);
    }
    
    return data;
}

bool isSorted(const std::vector<int>& data) {
    return std::is_sorted(data.begin(), data.end());
}

// ============================================================================
// Profiling Executor
// ============================================================================

void profileImplementation(const ProfileConfig& config, int rank, int worldSize) {
    // Validate that we have enough processes for distributed implementations
    if (config.isDistributed && worldSize < 2) {
        if (rank == 0) {
            std::cout << "[SKIP] " << config.name 
                      << " requires at least 2 MPI processes (current: " << worldSize << ")\n";
        }
        return;
    }
    
    if (rank == 0) {
        std::cout << "\n=======================================================\n";
        std::cout << "Profiling: " << config.name << "\n";
        std::cout << "Array Size: " << config.arraySize << "\n";
        std::cout << "Max Value: " << config.maxValue << "\n";
        std::cout << "MPI Processes: " << worldSize << "\n";
        std::cout << "=======================================================\n";
    }
    
    // Generate test data
    auto data = generateRandomData(config.arraySize, config.maxValue, 42);
    
    // Create sorter instance
    auto sorter = config.factory();
    
    // Load data (preparation phase - not profiled by nsys)
    sorter->load(data.data(), data.size());
    
    if (rank == 0) {
        std::cout << "Starting execution...\n";
    }
    
    cudaProfilerStart();

    // Execute sorting (THIS IS THE SECTION PROFILED BY NSYS)
    auto result = sorter->execute();

    cudaProfilerStop();
    
    // Verify results on rank 0
    if (rank == 0) {
        std::cout << "Execution completed.\n";
        
        if (isSorted(result)) {
            std::cout << "✓ Result is correctly sorted\n";
        } else {
            std::cerr << "✗ ERROR: Result is NOT sorted!\n";
        }
        
        // Print performance report
        std::string perfReport = sorter->getPerformanceReport();
        std::cout << "\nPerformance Report:\n" << perfReport << "\n";
    }
}

// ============================================================================
// Main - Profile Runner
// ============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    
    // ========================================================================
    // Configuration: Define implementations to profile
    // ========================================================================
    // 
    // To add a new implementation:
    // 1. Add factory function forward declaration at the top
    // 2. Add entry to the profileConfigs vector below
    // 3. Set isDistributed = true if it requires MPI > 1 process
    // ========================================================================
    
    std::vector<ProfileConfig> profileConfigs;
    
    // Parse command line for specific implementation
    std::string targetImpl = "";
    if (argc > 1) {
        targetImpl = argv[1];
    }
    
    // Define medium-sized array (similar to benchmark suite)
    const size_t MEDIUM_SIZE = 128;
    const int MEDIUM_MAX = 128;
    
    // CPU Implementation
    if (targetImpl.empty() || targetImpl == "cpu" || targetImpl == "all") {
        profileConfigs.push_back({
            "NaiveCpuSnp",
            createNaiveCpuSnpSort,
            MEDIUM_SIZE,
            MEDIUM_MAX,
            false  // Single process
        });
    }
    
    // CUDA Implementation
    if (targetImpl.empty() || targetImpl == "cuda" || targetImpl == "all") {
        profileConfigs.push_back({
            "CudaSnp",
            createCudaSnpSort,
            MEDIUM_SIZE,
            MEDIUM_MAX,
            false  // Single process
        });
    }
    
    // Sparse CUDA Implementation
    if (targetImpl.empty() || targetImpl == "sparse-cuda" || targetImpl == "all") {
        profileConfigs.push_back({
            "SparseCudaSnp",
            createSparseCudaSnpSort,
            MEDIUM_SIZE,
            MEDIUM_MAX,
            false  // Single process
        });
    }
    
    // Naive CUDA+MPI Implementation
    if (targetImpl.empty() || targetImpl == "naive-cuda-mpi" || targetImpl == "mpi" || targetImpl == "all") {
        profileConfigs.push_back({
            "NaiveCudaMpiSnp",
            createNaiveCudaMpiSnpSort,
            MEDIUM_SIZE,
            MEDIUM_MAX,
            true  // Requires MPI
        });
    }
    
    // CUDA+MPI Implementation
    if (targetImpl.empty() || targetImpl == "cuda-mpi" || targetImpl == "mpi" || targetImpl == "all") {
        profileConfigs.push_back({
            "CudaMpiSnp",
            createCudaMpiSnpSort,
            MEDIUM_SIZE,
            MEDIUM_MAX,
            true  // Requires MPI
        });
    }
    
    // ========================================================================
    // Execute Profiling
    // ========================================================================
    
    if (rank == 0) {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════╗\n";
        std::cout << "║       SNP System Profiling Suite                      ║\n";
        std::cout << "║       NVIDIA Nsight Systems CLI                       ║\n";
        std::cout << "╚════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        std::cout << "MPI Configuration: " << worldSize << " process(es)\n";
        std::cout << "Implementations to profile: " << profileConfigs.size() << "\n";
        
        if (profileConfigs.empty()) {
            std::cout << "\nNo implementations matched filter: '" << targetImpl << "'\n";
            std::cout << "\nAvailable options:\n";
            std::cout << "  cpu           - Profile CPU implementation only\n";
            std::cout << "  cuda          - Profile CUDA implementation only\n";
            std::cout << "  sparse-cuda   - Profile Sparse CUDA implementation only\n";
            std::cout << "  naive-cuda-mpi- Profile Naive CUDA+MPI implementation only\n";
            std::cout << "  cuda-mpi      - Profile CUDA+MPI implementation only\n";
            std::cout << "  mpi           - Profile all MPI implementations\n";
            std::cout << "  all           - Profile all implementations\n";
            std::cout << "  (no arg)      - Profile all implementations\n\n";
        }
    }
    
    for (const auto& config : profileConfigs) {
        profileImplementation(config, rank, worldSize);
    }
    
    if (rank == 0) {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════╗\n";
        std::cout << "║       Profiling Complete                               ║\n";
        std::cout << "╚════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
    }
    
    MPI_Finalize();
    return 0;
}
