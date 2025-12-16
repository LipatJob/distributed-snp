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
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " <descriptor_file> <steps>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string descriptor_file = argv[1];
    int steps = std::stoi(argv[2]);

    fs::path desc_path(descriptor_file);
    fs::path dir = desc_path.parent_path();
    std::string part_filename = "partition_" + std::to_string(rank) + ".dat";
    fs::path part_path = dir / part_filename;

    // Create Simulator
    auto sim = createOptimizedCudaMpiSimulator();

    // Load
    if (!sim->loadPresplitSystem(part_path.string())) {
        std::cerr << "Rank " << rank << ": Failed to load system from " << part_path << std::endl;
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
    
    double max_comm = 0;
    double max_comp = 0;
    
    MPI_Reduce(&metrics.mpi.communication_time_ms, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&metrics.cuda.kernel_time_ms, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "{\n";
        std::cout << "  \"nodes\": " << size << ",\n";
        std::cout << "  \"steps\": " << steps << ",\n";
        std::cout << "  \"total_time_s\": " << elapsed.count() << ",\n";
        std::cout << "  \"max_comm_time_ms\": " << max_comm << ",\n";
        std::cout << "  \"max_compute_time_ms\": " << max_comp << "\n";
        std::cout << "}" << std::endl;
        
        // Also save to file
        // Use a unique name based on nodes/timestamp?
        // Or just append to a log?
        // The prompt says "Logs must be saved to a predictable location under benchmark/ or output/."
        // And "Write logs in JSON or CSV".
        
        fs::create_directories("bigdata/results");
        std::string log_filename = "bigdata/results/run_" + std::to_string(size) + "nodes.json";
        std::ofstream out(log_filename);
        out << "{\n";
        out << "  \"nodes\": " << size << ",\n";
        out << "  \"steps\": " << steps << ",\n";
        out << "  \"total_time_s\": " << elapsed.count() << ",\n";
        out << "  \"max_comm_time_ms\": " << max_comm << ",\n";
        out << "  \"max_compute_time_ms\": " << max_comp << "\n";
        out << "}" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
