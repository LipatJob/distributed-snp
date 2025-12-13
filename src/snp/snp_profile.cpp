#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include <mpi.h>
#include <memory>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <chrono>

// Forward declarations - these are defined in their respective .cu files
extern std::unique_ptr<ISnpSimulator> createNaiveCudaMpiSimulator();
extern std::unique_ptr<ISnpSimulator> createCudaMpiSimulator();

/**
 * @brief Profile SNP Simulators for NVIDIA Nsight Systems
 * 
 * This program is designed to profile SnpSimulator implementations by:
 * 1. Creating an array to sort (medium-sized, like in benchmarks)
 * 2. Building an SNP system configuration
 * 3. Running the simulator and timing it
 * 
 * It's designed to be run via Nsight Systems UI with MPI:
 *   nsys launch -c "mpirun -np 2 ./snp_profile -s CudaMpiSnp"
 * 
 * Add new simulators by:
 * 1. Adding a forward declaration
 * 2. Adding a factory function to gSimulatorFactories
 */

// ============================================================================
// Simulator Factory System
// ============================================================================

using SimulatorFactory = std::function<std::unique_ptr<ISnpSimulator>()>;

// Maps simulator name to factory function
static std::map<std::string, SimulatorFactory> gSimulatorFactories;

void RegisterSimulators() {
    gSimulatorFactories["NaiveCudaMpiSnp"] = createNaiveCudaMpiSimulator;
    gSimulatorFactories["CudaMpiSnp"] = createCudaMpiSimulator;
}

// ============================================================================
// Sorting System Generation
// ============================================================================

/**
 * @brief Build a simple SNP system configuration for sorting
 * 
 * This creates an SNP system that can sort an array of values.
 * Based on SnpSort.cpp structure.
 */
class SortingSystemBuilder {
public:
    static SnpSystemConfig BuildForArray(const std::vector<int>& data) {
        SnpSystemConfig config;
        
        if (data.empty()) {
            return config;
        }

        size_t N = data.size();

        // Find min and max to normalize values
        int minVal = data[0];
        int maxVal = data[0];
        for (int val : data) {
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }

        // Offset to make values non-negative
        int offset = (minVal < 0) ? -minVal : 0;

        // Create a simple SNP system for sorting
        // Input neurons (count = N)
        // Working neurons (count = 2*N)
        // Output neurons (count = N)
        // Total: 4*N neurons

        // Initialize basic configuration using the builder
        SnpSystemBuilder builder;
        
        // Add input neurons with initial spike values
        for (size_t i = 0; i < N; ++i) {
            builder.addNeuron(i, data[i] + offset);
        }

        // Add working neurons
        for (size_t i = 0; i < 2 * N; ++i) {
            builder.addNeuron(N + i, 0);
        }

        // Add output neurons
        for (size_t i = 0; i < N; ++i) {
            builder.addNeuron(3 * N + i, 0);
        }

        // Add some basic rules for profiling
        for (size_t i = 0; i < N; ++i) {
            builder.addRule(i, 1, 1, 1, 0);
        }

        return builder.build();
    }
};

// ============================================================================
// Data Generation
// ============================================================================

std::vector<int> GenerateMediumArray(size_t size = 1000, unsigned seed = 42) {
    std::vector<int> data(size);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 10000);
    
    for (auto& val : data) {
        val = dist(rng);
    }
    
    return data;
}

// ============================================================================
// Profiling Harness
// ============================================================================

class ProfileHarness {
private:
    int rank;
    int world_size;
    std::unique_ptr<ISnpSimulator> simulator;
    SnpSystemConfig config;

public:
    ProfileHarness(int r, int ws) : rank(r), world_size(ws) {}

    bool Initialize(const std::string& simulatorName) {
        auto it = gSimulatorFactories.find(simulatorName);
        if (it == gSimulatorFactories.end()) {
            if (rank == 0) {
                std::cerr << "ERROR: Unknown simulator: " << simulatorName << std::endl;
                std::cerr << "Available simulators: ";
                for (const auto& name : gSimulatorFactories) {
                    std::cerr << name.first << " ";
                }
                std::cerr << std::endl;
            }
            return false;
        }

        simulator = it->second();
        if (!simulator) {
            if (rank == 0) {
                std::cerr << "ERROR: Failed to create simulator: " << simulatorName << std::endl;
            }
            return false;
        }

        return true;
    }

    void Profile(size_t arraySize = 1000, int iterations = 3) {
        if (!simulator) {
            if (rank == 0) {
                std::cerr << "ERROR: Simulator not initialized" << std::endl;
            }
            return;
        }

        if (rank == 0) {
            std::cout << "========================================" << std::endl;
            std::cout << "SNP Profiling Harness" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Array Size: " << arraySize << std::endl;
            std::cout << "Iterations: " << iterations << std::endl;
            std::cout << "MPI Processes: " << world_size << std::endl;
            std::cout << "========================================" << std::endl;
        }

        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);

        // Run profiling iterations
        for (int iter = 0; iter < iterations; ++iter) {
            if (rank == 0) {
                std::cout << "\n[Iteration " << (iter + 1) << "/" << iterations << "]" << std::endl;
            }

            // Generate test data
            auto data = GenerateMediumArray(arraySize, 42 + iter);

            // Build SNP system for this array
            config = SortingSystemBuilder::BuildForArray(data);

            if (!simulator->loadSystem(config)) {
                if (rank == 0) {
                    std::cerr << "ERROR: Failed to load system" << std::endl;
                }
                return;
            }

            // Synchronize before profiling
            MPI_Barrier(MPI_COMM_WORLD);

            // Execute simulation (this is what we're profiling)
            if (rank == 0) {
                std::cout << "  Running simulation..." << std::endl;
            }

            int ticks = static_cast<int>(arraySize) + 10;
            simulator->step(ticks);

            // Synchronize after simulation
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                std::cout << "  Simulation complete" << std::endl;
            }
        }

        // Get and report performance metrics
        if (rank == 0) {
            std::string report = simulator->getPerformanceReport();
            std::cout << "\n========================================" << std::endl;
            std::cout << "Performance Report:" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << report << std::endl;
        }
    }
};

// ============================================================================
// Main
// ============================================================================

void PrintUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " [-s <simulator_name>] [-n <array_size>] [-i <iterations>]" << std::endl;
    std::cerr << "\nOptions:" << std::endl;
    std::cerr << "  -s <simulator_name>   Simulator to profile (default: CudaMpiSnp)" << std::endl;
    std::cerr << "  -n <array_size>       Array size to sort (default: 1000)" << std::endl;
    std::cerr << "  -i <iterations>       Number of iterations (default: 3)" << std::endl;
    std::cerr << "\nAvailable simulators:" << std::endl;
    for (const auto& name : gSimulatorFactories) {
        std::cerr << "  - " << name.first << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Register available simulators
    RegisterSimulators();

    // Parse command-line arguments
    std::string simulatorName = "CudaMpiSnp";
    size_t arraySize = 1000;
    int iterations = 3;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-s" && i + 1 < argc) {
            simulatorName = argv[++i];
        } else if (arg == "-n" && i + 1 < argc) {
            arraySize = std::stoul(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            if (rank == 0) PrintUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::cerr << "Unknown option: " << arg << std::endl;
                PrintUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Create and run profiler
    ProfileHarness profiler(rank, world_size);

    if (!profiler.Initialize(simulatorName)) {
        MPI_Finalize();
        return 1;
    }

    profiler.Profile(arraySize, iterations);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
