#include "ISnpSimulator.hpp"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " <descriptor_file> <steps> [impl] [partitioner]" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string descriptor_file = argv[1];
    int steps = std::stoi(argv[2]);
    std::string impl = (argc > 3) ? argv[3] : "optimized";
    std::string partitioner = (argc > 4) ? argv[4] : "linear";

    fs::path desc_path(descriptor_file);
    fs::path dir = desc_path.parent_path();
    std::string part_filename = "partition_" + std::to_string(rank) + ".dat";
    fs::path part_path = dir / part_filename;

    // Resolve Partitioner Type
    PartitionerType pType = PartitionerType::LINEAR;
    if (partitioner == "linear") pType = PartitionerType::LINEAR;
    else if (partitioner == "louvain") pType = PartitionerType::LOUVAIN;
    else if (partitioner == "redblue") pType = PartitionerType::RED_BLUE_BFS;
    else {
        if (rank == 0) std::cerr << "Warning: Unknown partitioner '" << partitioner << "', defaulting to linear." << std::endl;
    }

    // Create Simulator
    std::unique_ptr<ISnpSimulator> sim;
    if (impl == "optimized") {
        sim = createOptimizedCudaMpiSimulator(pType);
    } else if (impl == "naive") {
        sim = createNaiveCudaMpiSimulator(pType);
    } else {
        if (rank == 0) std::cerr << "Error: Unknown implementation '" << impl << "'. Options: optimized, naive" << std::endl;
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "Running with Implementation: " << impl << ", Partitioner: " << partitioner << std::endl;
        if (pType != PartitionerType::LINEAR) {
             std::cout << "Note: Partitioner selection is passed to factory but may be ignored for pre-split data." << std::endl;
        }
    }

    // Load
    if (!sim->loadPresplitSystem(part_path.string())) {
        std::cerr << "Rank " << rank << ": Failed to load system from " << part_path << std::endl;
        if (impl == "naive") {
             std::cerr << "Note: 'naive' implementation does not support pre-split Big Data files." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Run
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    sim->step(steps);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Metrics
    auto metrics = sim->getPerformanceMetrics();
    
    // 1. Reduce Max Times (Batch 2 doubles)
    double local_times[2] = { metrics.mpi.communication_time_ms, metrics.cuda.kernel_time_ms };
    double global_max_times[2] = {0, 0};
    MPI_Reduce(local_times, global_max_times, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    double max_comm = global_max_times[0];
    double max_comp = global_max_times[1];
    
    // 2. Reduce Sum Counts (Batch 3 long longs)
    long long local_counts[3] = { 
        (long long)metrics.algorithm.local_neurons, 
        (long long)metrics.algorithm.total_rules, 
        (long long)metrics.algorithm.num_synapses 
    };
    long long global_counts[3] = {0, 0, 0};
    MPI_Reduce(local_counts, global_counts, 3, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    long long global_n_sum = global_counts[0];
    long long global_r_sum = global_counts[1];
    long long global_s_sum = global_counts[2];
    
    long long local_n = local_counts[0];
    long long local_r = local_counts[1];
    long long local_s = local_counts[2];

    // Validation: Get Global State Checksum
    std::vector<int> global_state = sim->getGlobalState();
    long long state_sum = 0;
    for(int val : global_state) state_sum += val;
    
    // Only Rank 0 prints
    if (rank == 0) {
        std::cout << "{\n";
        std::cout << "  \"nodes\": " << size << ",\n";
        std::cout << "  \"steps\": " << steps << ",\n";
        std::cout << "  \"global_neurons\": " << global_n_sum << ",\n";
        std::cout << "  \"global_rules\": " << global_r_sum << ",\n";
        std::cout << "  \"global_synapses\": " << global_s_sum << ",\n";
        std::cout << "  \"local_neurons\": " << local_n << ",\n";
        std::cout << "  \"local_rules\": " << local_r << ",\n";
        std::cout << "  \"local_synapses\": " << local_s << ",\n";
        std::cout << "  \"total_time_s\": " << elapsed.count() << ",\n";
        std::cout << "  \"max_comm_time_ms\": " << max_comm << ",\n";
        std::cout << "  \"max_compute_time_ms\": " << max_comp << ",\n";
        std::cout << "  \"state_checksum\": " << state_sum << "\n";
        std::cout << "}" << std::endl;
        
        // Also save to file
        fs::create_directories("bigdata/results");
        std::string log_filename = "bigdata/results/run_" + impl + "_" + std::to_string(size) + "_" + std::to_string(global_n_sum) + "nodes.json";
        std::ofstream out(log_filename);
        out << "{\n";
        out << "  \"nodes\": " << size << ",\n";
        out << "  \"steps\": " << steps << ",\n";
        out << "  \"global_neurons\": " << global_n_sum << ",\n";
        out << "  \"global_rules\": " << global_r_sum << ",\n";
        out << "  \"global_synapses\": " << global_s_sum << ",\n";
        out << "  \"local_neurons\": " << local_n << ",\n";
        out << "  \"local_rules\": " << local_r << ",\n";
        out << "  \"local_synapses\": " << local_s << ",\n";
        out << "  \"total_time_s\": " << elapsed.count() << ",\n";
        out << "  \"max_comm_time_ms\": " << max_comm << ",\n";
        out << "  \"max_compute_time_ms\": " << max_comp << ",\n";
        out << "  \"state_checksum\": " << state_sum << "\n";
        out << "}" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
