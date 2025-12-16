/* SortApp.cpp */
#include "../src/sort/ISort.hpp"
#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include "IPartitioner.hpp"
#include "PerformanceMetrics.hpp"
#include <mpi.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <memory>
#include <cstring>

// --- Minimal Data Generation Utils (Simplified from your code) ---
// (Assume BenchUtils::GenerateData and IsSorted exist here as per your original file)
#include "BenchUtils.hpp" // extracted your Utils to a header for cleanliness

// --- Factory Logic ---
std::unique_ptr<ISort> createSimulator(const std::string& type, const std::string& partitioner) {
    if (type == "cpu") return createNaiveCpuSnpSort();
    if (type == "sparse-cuda") return createSparseCudaSnpSort();
    if (type == "optimized-cuda") return createOptimizedCudaSnpSort(); // Assuming this mapping
    if (type == "naive-cuda-mpi") return createNaiveCudaMpiSnpSort();
    if (type == "optimized-cuda-mpi") {
        PartitionerType pType = PartitionerType::LINEAR;
        if (partitioner == "louvain") pType = PartitionerType::LOUVAIN;
        else if (partitioner == "redblue") pType = PartitionerType::RED_BLUE_BFS;
        return createOptimizedCudaMpiSnpSort(pType);
    }
    throw std::runtime_error("Unknown simulator type: " + type);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. Parse Arguments (Simple manual parsing)
    std::string type = "cpu";
    std::string partitioner = "linear";
    int size = 256;
    int maxVal = 100;
    std::string distStr = "Sorted";
    int iterations = 10;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--type") == 0) type = argv[++i];
        else if (strcmp(argv[i], "--part") == 0) partitioner = argv[++i];
        else if (strcmp(argv[i], "--size") == 0) size = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--iter") == 0) iterations = std::stoi(argv[++i]);
    }

    // 2. Setup (Excluded from timing)
    auto dist = BenchUtils::Distribution::REVERSE_SORTED; // Simplified for demo
    auto data = BenchUtils::GenerateData(size, maxVal, dist, 42);
    auto sorter = createSimulator(type, partitioner);
    sorter->load(data.data(), data.size());

    // Warmup (optional but recommended for CUDA)
    // sorter->execute(); 
    // sorter->reset();

    // 3. The Benchmark Loop
    if (rank == 0) std::cout << "Starting Benchmark: " << type << " Size=" << size << std::endl;

    double total_compute_time = 0.0;

    for (int i = 0; i < iterations; ++i) {
        // Prepare/Reset logic if needed
        MPI_Barrier(MPI_COMM_WORLD); // Strict sync before start

        auto start = std::chrono::high_resolution_clock::now();
        
        // --- CRITICAL SECTION ---
        auto result = sorter->execute();
        // ------------------------

        // Ensure all ranks are done before stopping clock
        MPI_Barrier(MPI_COMM_WORLD); 
        auto end = std::chrono::high_resolution_clock::now();
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        total_compute_time += elapsed.count();

        // Verification (Rank 0 only, on last run)
        if (i == iterations - 1 && rank == 0) {
            if (!BenchUtils::IsSorted(result)) {
                std::cout << "[BenchMetric] Verification: FAILED" << std::endl;
            } else {
                std::cout << "[BenchMetric] Verification: PASSED" << std::endl;
            }
        }
        
        // Reset sorter state for next iteration if necessary
        // sorter->reset(); 
    }

    // 4. Reporting (ReFrame scrapes this)
    if (rank == 0) {
        double avg_time = total_compute_time / iterations;
        PerformanceMetrics metrics = sorter->getPerformanceMetrics();

        // The "Magic" lines ReFrame looks for:
        std::cout << "[BenchMetric] LoopTime: " << avg_time << " ms" << std::endl;
        std::cout << "[BenchMetric] Throughput: " << 1.0 / avg_time << " steps/ms" << std::endl;
        std::cout << "[BenchMetric] ComputeTime: " << metrics.compute_time_ms << " ms" << std::endl;
        std::cout << "[BenchMetric] Steps: " << metrics.steps_executed << std::endl;
        
        if (metrics.algorithm.has_data()) {
            std::cout << "[BenchMetric] Neurons: " << metrics.algorithm.num_neurons << std::endl;
            std::cout << "[BenchMetric] Synapses: " << metrics.algorithm.num_synapses << std::endl;
        }

        if (metrics.cuda.has_data()) {
             std::cout << "[BenchMetric] CudaKernelTime: " << metrics.cuda.kernel_time_ms  << " ms" << std::endl;
             std::cout << "[BenchMetric] CudaMemTime: " << metrics.cuda.memory_transfer_time_ms  << " ms" << std::endl;
        }
        
        if (metrics.mpi.has_data()) {
             std::cout << "[BenchMetric] MPI_Bytes: " << metrics.mpi.total_bytes_sent << std::endl;
             std::cout << "[BenchMetric] MPI_CommTime: " << metrics.mpi.communication_time_ms  << " ms" << std::endl;
             std::cout << "[BenchMetric] MPI_Msgs: " << metrics.mpi.total_messages_sent << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}