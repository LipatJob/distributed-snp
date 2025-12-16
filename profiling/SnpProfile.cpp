#include "../sort/SnpSort.cpp"
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

// No forward declarations needed - using simulator factories from ISnpSimulator.hpp

// Type alias for cleaner code
using SimulatorFactory = std::function<std::unique_ptr<ISnpSimulator>()>;

// ============================================================================
// Profiling Configuration
// ============================================================================

struct ProfileConfig {
    std::string name;
    SimulatorFactory factory;
    size_t arraySize;
    int maxValue;
    bool isDistributed;  // Requires MPI with multiple processes
    int steps;           // Number of steps to run (0 = run to completion)
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
        if (config.steps > 0) {
            std::cout << "Steps: " << config.steps << "\n";
        } else {
            std::cout << "Steps: max (run to completion)\n";
        }
        std::cout << "=======================================================\n";
    }
    
    // Generate test data
    auto data = generateRandomData(config.arraySize, config.maxValue, 42);
    
    // Create simulator instance from factory
    auto simulator = config.factory();
    
    // Keep a raw pointer to simulator for direct step() calls
    ISnpSimulator* simPtr = simulator.get();
    
    // Create sorter with simulator (sorter takes ownership)
    auto sorter = std::make_unique<SnpSort>(std::move(simulator));
    
    // Load data (preparation phase - not profiled by nsys)
    sorter->load(data.data(), data.size());
    
    if (rank == 0) {
        std::cout << "Starting execution...\n";
    }
    
    
    std::vector<int> results;
    // Execute sorting (THIS IS THE SECTION PROFILED)
    if (config.steps > 0) {
        if (rank == 0) {
            std::cout << "Executing for " << config.steps << " steps...\n";
        }
        cudaProfilerStart();
        simPtr->step(config.steps);
        cudaProfilerStop();
    } else {
        if (rank == 0) {
            std::cout << "Executing to completion...\n";
        }
        cudaProfilerStart();
        results = sorter->execute();
        cudaProfilerStop();
    }

    // Report on rank 0
    if (rank == 0) {
        std::cout << "Execution completed.\n";

        if (isSorted(results)) {
            if (rank == 0) {
                std::cout << "Result: Array is sorted correctly.\n";
            }
        } else {
            if (rank == 0) {
                std::cerr << "Result: Array is NOT sorted correctly!\n";
            }
        }
        
        // Print performance report
        std::string perfReport = simPtr->getPerformanceReport();
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
    int numSteps = 0;  // 0 means run to completion
    PartitionerType partitionerType = PartitionerType::LINEAR; // Default partitioner
    size_t arraySize = 2048;  // Default array size
    
    if (argc > 1) {
        targetImpl = argv[1];
    }
    if (argc > 2) {
        numSteps = std::atoi(argv[2]);
    }
    if (argc > 3) {
        // Parse partitioner type from command line
        partitionerType = IPartitioner::parsePartitionerType(argv[3], PartitionerType::LINEAR);
        if (rank == 0) {
            std::cout << "Using partitioner: " << IPartitioner::getPartitionerName(partitionerType) << "\n";
        }
    }
    if (argc > 4) {
        arraySize = std::atoi(argv[4]);
        if (rank == 0) {
            std::cout << "Using array size: " << arraySize << "\n";
        }
    }
    
    // Define array parameters
    const size_t MEDIUM_SIZE = arraySize;
    const int MEDIUM_MAX = arraySize;
    
    // CPU Implementation
    if (targetImpl.empty() || targetImpl == "cpu" || targetImpl == "all") {
        profileConfigs.push_back({
            "NaiveCpuSnp",
            createNaiveCpuSimulator,
            MEDIUM_SIZE,
            MEDIUM_MAX,
            false,  // Single process
            numSteps
        });
    }
    
    // Sparse CUDA Implementation
    if (targetImpl.empty() || targetImpl == "sparse-cuda" || targetImpl == "all") {
        profileConfigs.push_back({
            "SparseCudaSnp",
            createSparseCudaSimulator,
            MEDIUM_SIZE,
            MEDIUM_MAX,
            false,  // Single process
            numSteps
        });
    }

        // Optimized CUDA Implementation
    if (targetImpl.empty() || targetImpl == "optimized-cuda" || targetImpl == "all") {
        profileConfigs.push_back({
            "OptimizedCudaSnp",
            createOptimizedCudaSimulator,
            MEDIUM_SIZE,
            MEDIUM_MAX,
            false,  // Single process
            numSteps
        });
    }
    
    // Naive CUDA+MPI Implementation
    if (targetImpl.empty() || targetImpl == "naive-cuda-mpi" || targetImpl == "mpi" || targetImpl == "all") {
        profileConfigs.push_back({
            "NaiveCudaMpiSnp",
            [partitionerType]() { return createNaiveCudaMpiSimulator(partitionerType); },
            MEDIUM_SIZE,
            MEDIUM_MAX,
            true,  // Requires MPI
            numSteps
        });
    }
    
    // Optimized CUDA+MPI Implementation
    if (targetImpl.empty() || targetImpl == "optimized-cuda-mpi" || targetImpl == "mpi" || targetImpl == "all") {
        profileConfigs.push_back({
            "OptimizedCudaMpiSnp",
            [partitionerType]() { return createOptimizedCudaMpiSimulator(partitionerType); },
            MEDIUM_SIZE,
            MEDIUM_MAX,
            true,  // Requires MPI
            numSteps
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
            std::cout << "\nUsage: " << argv[0] << " [implementation] [steps] [partitioner] [array_size]\n";
            std::cout << "\nAvailable implementations:\n";
            std::cout << "  cpu           - Profile CPU implementation only\n";
            std::cout << "  optimized-cuda          - Profile CUDA implementation only\n";
            std::cout << "  sparse-cuda   - Profile Sparse CUDA implementation only\n";
            std::cout << "  naive-cuda-mpi- Profile Naive CUDA+MPI implementation only\n";
            std::cout << "  optimized-cuda-mpi      - Profile CUDA+MPI implementation only\n";
            std::cout << "  mpi           - Profile all MPI implementations\n";
            std::cout << "  all           - Profile all implementations\n";
            std::cout << "  (no arg)      - Profile all implementations\n";
            std::cout << "\nPartitioner options (for MPI implementations):\n";
            std::cout << "  linear        - Linear (block) partitioning (default)\n";
            std::cout << "  louvain       - Louvain community detection\n";
            std::cout << "  red-blue      - Red-Blue pebbling (BFS)\n";
            std::cout << "\nArray size (default: 2048):\n";
            std::cout << "  N             - Array size (positive integer)\n\n";
            std::cout << "\nExamples:\n";
            std::cout << "  " << argv[0] << " cuda-mpi 100 louvain 4096  # Profile CUDA+MPI with Louvain for 100 steps, array size 4096\n";
            std::cout << "  " << argv[0] << " mpi 0 red-blue 1024        # Profile all MPI with Red-Blue to completion, array size 1024\n";
            std::cout << "  " << argv[0] << " cuda 0 linear 8192         # Profile CUDA to completion, array size 8192\n\n";
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
