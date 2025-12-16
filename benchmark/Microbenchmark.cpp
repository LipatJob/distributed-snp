/**
 * @file microbenchmark.cpp
 * @brief Microbenchmark tool for SNP sorting implementations
 * 
 * Demonstrates the new structured PerformanceMetrics system.
 * Usage: ./microbenchmark -n <size> -i <implementation>
 */

#include "ISort.hpp"
#include "PerformanceMetrics.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <string>
#include <cstring>
#include <iomanip>

#include <mpi.h>

// Implementation factory
std::unique_ptr<ISort> createSorter(const std::string& impl) {
    if (impl == "cpu") {
        return createNaiveCpuSnpSort();
    } else if (impl == "sparse-cuda") {
        return createSparseCudaSnpSort();
    } else if (impl == "optimized-cuda") {
        return createOptimizedCudaSnpSort();
    } else if (impl == "naive-cuda-mpi") {
        return createNaiveCudaMpiSnpSort();
    } else if (impl == "optimized-cuda-mpi") {
        return createOptimizedCudaMpiSnpSort();
    } else if (impl == "optimized-cuda-mpi-louvain") {
        return createOptimizedCudaMpiSnpSort(PartitionerType::LOUVAIN);
    } else if (impl == "optimized-cuda-mpi-redblue") {
        return createOptimizedCudaMpiSnpSort(PartitionerType::RED_BLUE_BFS);
    } else {
        std::cerr << "Unknown implementation: " << impl << "\n";
        return nullptr;
    }
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -n <size>  Input size (default: 100)\n"
              << "  -i <impl>  Implementation (default: cpu)\n"
              << "  -m <max>   Maximum value (default: size)\n"
              << "  -s <seed>  Random seed (default: 42)\n"
              << "  -c         Export CSV format\n"
              << "  -h         Show help\n\n"
              << "Implementations:\n"
              << "  cpu, sparse-cuda, optimized-cuda, naive-cuda-mpi,\n"
              << "  optimized-cuda-mpi, optimized-cuda-mpi-louvain, optimized-cuda-mpi-redblue\n\n"
              << "Example: " << prog << " -n 1000 -i optimized-cuda\n";
}

int main(int argc, char** argv) {
    int rank = 0;
    int world_size = 1;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Parse command-line arguments
    int input_size = 100;
    std::string implementation = "cpu";
    int max_value = -1;
    unsigned seed = 42;
    bool csv_format = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            input_size = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            implementation = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            max_value = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0) {
            csv_format = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::cerr << "Unknown option: " << argv[i] << "\n";
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (max_value < 0) max_value = input_size;
    
    // Print header (rank 0 only)
    if (rank == 0) {
        std::cout << "══════════════════════════════════════════════════════\n"
                  << "  SNP Microbenchmark\n"
                  << "══════════════════════════════════════════════════════\n"
                  << "  Implementation: " << implementation << "\n"
                  << "  Input Size: " << input_size << "\n"
                  << "  MPI Ranks: " << world_size << "\n\n";
    }
    
    // Generate random data
    std::vector<int> data(input_size);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, max_value);
    for (auto& val : data) {
        val = dist(rng);
    }
    
    // Create sorter
    auto sorter = createSorter(implementation);
    if (!sorter) {
        if (rank == 0) {
            std::cerr << "Failed to create sorter for implementation: " << implementation << "\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Run sorting
    sorter->load(data.data(), data.size());
    std::vector<int> result = sorter->execute();
    
    // Verify correctness (rank 0 only)
    bool success = true;
    if (rank == 0) {
        if (!std::is_sorted(result.begin(), result.end())) {
            std::cerr << "✗ ERROR: Result is not sorted!\n";
            success = false;
        } else {
            std::cout << "✓ Sort verification passed\n\n";
        }
    }
    
    // Get and display metrics
    PerformanceMetrics metrics = sorter->getPerformanceMetrics();
    
    if (rank == 0) {
        if (csv_format) {
            std::cout << "\n" << PerformanceMetrics::csvHeader() << "\n"
                     << metrics.toCSV() << "\n";
        } else {
            std::cout << "══════════════════════════════════════════════════════\n"
                     << metrics.toReport(implementation + " Sorter")
                     << "══════════════════════════════════════════════════════\n";
            
            // Additional analysis
            if (metrics.steps_executed > 0 || metrics.cuda.has_data() || metrics.mpi.has_data()) {
                std::cout << "\nPerformance Analysis:\n";
                
                if (metrics.steps_executed > 0) {
                    std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                             << metrics.throughput_steps_per_second() << " steps/sec\n";
                }
                
                if (metrics.cuda.has_data()) {
                    double kernel_pct = metrics.cuda.kernel_time_ms / metrics.total_time_ms * 100.0;
                    std::cout << "  GPU Kernel Time: " << std::setprecision(1) << kernel_pct << "%\n";
                }
                
                if (metrics.mpi.has_data()) {
                    std::cout << "  Communication Overhead: " 
                             << std::setprecision(1) << metrics.communication_percentage() << "%\n";
                }
            }
            std::cout << "\n";
        }
    }
    
    MPI_Finalize();
    
    return success ? 0 : 1;
}
