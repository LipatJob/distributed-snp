#pragma once

#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <iomanip>

/**
 * @brief Structured performance metrics for SNP simulators
 * 
 * This provides a standardized way to collect and report performance data
 * across different simulator implementations (CPU, CUDA, MPI, etc.)
 */
struct PerformanceMetrics {
    // === Core Metrics (All Implementations) ===
    int steps_executed = 0;
    double total_time_ms = 0.0;
    double compute_time_ms = 0.0;
    
    // === CUDA-Specific Metrics ===
    struct CudaMetrics {
        double kernel_time_ms = 0.0;
        double memory_transfer_time_ms = 0.0;
        double host_to_device_time_ms = 0.0;
        double device_to_host_time_ms = 0.0;
        
        // Kernel breakdown (optional)
        std::map<std::string, double> kernel_times;
        
        // Memory statistics
        size_t device_memory_allocated = 0;
        size_t peak_device_memory = 0;
        
        bool has_data() const {
            return kernel_time_ms > 0.0 || memory_transfer_time_ms > 0.0;
        }
    } cuda;
    
    // === MPI-Specific Metrics ===
    struct MpiMetrics {
        double communication_time_ms = 0.0;
        double synchronization_time_ms = 0.0;
        
        // Message statistics
        size_t total_messages_sent = 0;
        size_t total_messages_received = 0;
        size_t total_bytes_sent = 0;
        size_t total_bytes_received = 0;
        
        // Per-rank breakdown (optional)
        int rank = -1;
        int world_size = 0;
        
        // Communication patterns
        std::map<int, size_t> messages_per_rank;  // rank -> message count
        std::map<int, size_t> bytes_per_rank;      // rank -> bytes transferred
        
        bool has_data() const {
            return communication_time_ms > 0.0 || total_messages_sent > 0;
        }
    } mpi;
    
    // === Algorithm-Specific Metrics ===
    struct AlgorithmMetrics {
        // System configuration
        int num_neurons = 0;
        int num_synapses = 0;
        int total_rules = 0;
        
        // Sparse matrix metrics (for SparseCudaSnpSimulator)
        int max_out_degree = 0;
        double sparsity = 0.0;
        
        // Partitioning metrics (for distributed implementations)
        std::string partitioner_type;
        int local_neurons = 0;
        int local_synapses = 0;
        int cross_rank_synapses = 0;
        
        bool has_data() const {
            return num_neurons > 0;
        }
    } algorithm;
    
    // === Derived Metrics ===
    
    double avg_time_per_step() const {
        return steps_executed > 0 ? total_time_ms / steps_executed : 0.0;
    }
    
    double compute_percentage() const {
        return total_time_ms > 0 ? (compute_time_ms / total_time_ms) * 100.0 : 0.0;
    }
    
    double communication_percentage() const {
        return total_time_ms > 0 ? (mpi.communication_time_ms / total_time_ms) * 100.0 : 0.0;
    }
    
    double throughput_steps_per_second() const {
        return total_time_ms > 0 ? (steps_executed * 1000.0) / total_time_ms : 0.0;
    }
    
    // === Formatting ===
    
    /**
     * @brief Generate a human-readable report
     * @param implementation_name Name of the simulator (e.g., "Naive CPU", "CUDA")
     */
    std::string toReport(const std::string& implementation_name = "SNP Simulator") const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        
        ss << "=== " << implementation_name << " Performance Report ===\n";
        
        // Core metrics
        ss << "\n[Core Metrics]\n";
        ss << "  Steps Executed: " << steps_executed << "\n";
        ss << "  Total Time: " << total_time_ms << " ms\n";
        ss << "  Compute Time: " << compute_time_ms << " ms";
        if (total_time_ms > 0) {
            ss << " (" << compute_percentage() << "%)";
        }
        ss << "\n";
        
        if (steps_executed > 0) {
            ss << "  Average Time/Step: " << avg_time_per_step() << " ms\n";
            ss << "  Throughput: " << throughput_steps_per_second() << " steps/sec\n";
        }
        
        // Algorithm metrics
        if (algorithm.has_data()) {
            ss << "\n[System Configuration]\n";
            ss << "  Neurons: " << algorithm.num_neurons << "\n";
            if (algorithm.num_synapses > 0) {
                ss << "  Synapses: " << algorithm.num_synapses << "\n";
            }
            if (algorithm.total_rules > 0) {
                ss << "  Total Rules: " << algorithm.total_rules << "\n";
            }
            if (algorithm.max_out_degree > 0) {
                ss << "  Max Out-Degree: " << algorithm.max_out_degree << "\n";
            }
            if (algorithm.sparsity > 0.0) {
                ss << "  Sparsity: " << (algorithm.sparsity * 100.0) << "%\n";
            }
            if (!algorithm.partitioner_type.empty()) {
                ss << "  Partitioner: " << algorithm.partitioner_type << "\n";
                if (algorithm.local_neurons > 0) {
                    ss << "  Local Neurons: " << algorithm.local_neurons << "\n";
                }
                if (algorithm.cross_rank_synapses > 0) {
                    ss << "  Cross-Rank Synapses: " << algorithm.cross_rank_synapses << "\n";
                }
            }
        }
        
        // CUDA metrics
        if (cuda.has_data()) {
            ss << "\n[CUDA Metrics]\n";
            ss << "  Kernel Time: " << cuda.kernel_time_ms << " ms";
            if (total_time_ms > 0) {
                ss << " (" << (cuda.kernel_time_ms / total_time_ms * 100.0) << "%)";
            }
            ss << "\n";
            
            if (cuda.memory_transfer_time_ms > 0.0) {
                ss << "  Memory Transfer: " << cuda.memory_transfer_time_ms << " ms";
                if (total_time_ms > 0) {
                    ss << " (" << (cuda.memory_transfer_time_ms / total_time_ms * 100.0) << "%)";
                }
                ss << "\n";
                
                if (cuda.host_to_device_time_ms > 0.0) {
                    ss << "    Host->Device: " << cuda.host_to_device_time_ms << " ms\n";
                }
                if (cuda.device_to_host_time_ms > 0.0) {
                    ss << "    Device->Host: " << cuda.device_to_host_time_ms << " ms\n";
                }
            }
            
            if (!cuda.kernel_times.empty()) {
                ss << "  Kernel Breakdown:\n";
                for (const auto& [name, time] : cuda.kernel_times) {
                    ss << "    " << name << ": " << time << " ms\n";
                }
            }
            
            if (cuda.device_memory_allocated > 0) {
                ss << "  Device Memory: " << (cuda.device_memory_allocated / 1024.0 / 1024.0) << " MB\n";
            }
        }
        
        // MPI metrics
        if (mpi.has_data()) {
            ss << "\n[MPI Metrics]\n";
            if (mpi.rank >= 0) {
                ss << "  Rank: " << mpi.rank << " / " << mpi.world_size << "\n";
            }
            ss << "  Communication Time: " << mpi.communication_time_ms << " ms";
            if (total_time_ms > 0) {
                ss << " (" << communication_percentage() << "%)";
            }
            ss << "\n";
            
            if (mpi.synchronization_time_ms > 0.0) {
                ss << "  Synchronization Time: " << mpi.synchronization_time_ms << " ms\n";
            }
            
            if (mpi.total_messages_sent > 0) {
                ss << "  Messages Sent: " << mpi.total_messages_sent << "\n";
                ss << "  Bytes Sent: " << mpi.total_bytes_sent;
                if (mpi.total_messages_sent > 0) {
                    ss << " (avg " << (mpi.total_bytes_sent / mpi.total_messages_sent) << " bytes/msg)";
                }
                ss << "\n";
            }
            
            if (mpi.total_messages_received > 0) {
                ss << "  Messages Received: " << mpi.total_messages_received << "\n";
                ss << "  Bytes Received: " << mpi.total_bytes_received;
                if (mpi.total_messages_received > 0) {
                    ss << " (avg " << (mpi.total_bytes_received / mpi.total_messages_received) << " bytes/msg)";
                }
                ss << "\n";
            }
            
            if (!mpi.messages_per_rank.empty()) {
                ss << "  Communication Pattern:\n";
                for (const auto& [rank, count] : mpi.messages_per_rank) {
                    ss << "    Rank " << rank << ": " << count << " messages";
                    if (mpi.bytes_per_rank.count(rank)) {
                        ss << ", " << mpi.bytes_per_rank.at(rank) << " bytes";
                    }
                    ss << "\n";
                }
            }
        }
        
        return ss.str();
    }
    
    /**
     * @brief Export metrics as CSV row (for benchmarking)
     */
    std::string toCSV() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        
        ss << steps_executed << ","
           << total_time_ms << ","
           << compute_time_ms << ","
           << avg_time_per_step() << ","
           << throughput_steps_per_second() << ","
           << algorithm.num_neurons << ","
           << algorithm.num_synapses << ","
           << cuda.kernel_time_ms << ","
           << cuda.memory_transfer_time_ms << ","
           << mpi.communication_time_ms << ","
           << mpi.total_messages_sent << ","
           << mpi.total_bytes_sent;
        
        return ss.str();
    }
    
    /**
     * @brief Get CSV header for toCSV()
     */
    static std::string csvHeader() {
        return "steps,total_time_ms,compute_time_ms,avg_time_per_step_ms,throughput_steps_per_sec,"
               "neurons,synapses,cuda_kernel_ms,cuda_memory_ms,mpi_comm_ms,mpi_messages,mpi_bytes";
    }
};
