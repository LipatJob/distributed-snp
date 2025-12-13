#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include "IPartitioner.hpp"
#include "LinearPartitioner.hpp"
#include "LouvainPartitioner.hpp"
#include "RedBluePartitioner.hpp"
#include "SnpSystemPermuter.hpp"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <memory>

// --- Macros & Constants ---

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, -1); \
        } \
    } while(0)

#define MPI_CHECK(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, -1); \
        } \
    } while(0)

constexpr int BLOCK_SIZE = 256;

namespace {

// --- Device Structures (Structure of Arrays) ---

// Holds state only for LOCAL neurons owned by this rank
struct LocalNeuronData {
    int* current_spikes;     // C(k)
    int* initial_spikes;     // C(0)
    bool* is_open;           // Status vector St(k)
    int* delay_timer;        // Remaining delay
    int* pending_emission;   // Spikes waiting for delay to expire
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&current_spikes, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&initial_spikes, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&is_open, n * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&delay_timer, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&pending_emission, n * sizeof(int)));
    }

    void free() {
        if (count == 0) return;
        cudaFree(current_spikes);
        cudaFree(initial_spikes);
        cudaFree(is_open);
        cudaFree(delay_timer);
        cudaFree(pending_emission);
    }
};

// Holds rules associated with LOCAL neurons
struct LocalRuleData {
    int* neuron_local_idx;   // Index relative to local partition (0 to local_count-1)
    int* threshold;
    int* consumed;
    int* produced;
    int* delay;
    
    // CSR-like indexing for rules per neuron
    int* rule_start_idx;     // Size: local_neuron_count
    int* rule_count;         // Size: local_neuron_count
    
    int total_rules_count;
    int local_neuron_count;

    void allocate(int n_neurons, int n_rules) {
        local_neuron_count = n_neurons;
        total_rules_count = n_rules;
        if (n_neurons == 0) return;

        CUDA_CHECK(cudaMalloc(&rule_start_idx, n_neurons * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rule_count, n_neurons * sizeof(int)));

        if (n_rules > 0) {
            CUDA_CHECK(cudaMalloc(&neuron_local_idx, n_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&threshold, n_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&consumed, n_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&produced, n_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&delay, n_rules * sizeof(int)));
        }
    }

    void free() {
        if (local_neuron_count == 0) return;
        cudaFree(rule_start_idx);
        cudaFree(rule_count);
        if (total_rules_count > 0) {
            cudaFree(neuron_local_idx);
            cudaFree(threshold);
            cudaFree(consumed);
            cudaFree(produced);
            cudaFree(delay);
        }
    }
};

// Fully Replicated Synapse List (Optimization: All ranks have all synapses)
struct GlobalSynapseData {
    int* source_global_id;
    int* dest_global_id;
    int* weight;
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&source_global_id, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dest_global_id, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weight, n * sizeof(int)));
    }

    void free() {
        if (count == 0) return;
        cudaFree(source_global_id);
        cudaFree(dest_global_id);
        cudaFree(weight);
    }
};

// --- Kernels ---

/**
 * @brief Step 1: Update Status and Calculate Production (Local Only)
 * * Each thread handles one local neuron.
 * 1. Updates delay timers.
 * 2. Checks if neuron opens (delay == 0).
 * 3. Applies rules (Deterministic: first valid rule).
 * 4. Writes total output to `local_production_out`.
 */
__global__ void kLocalComputeAndProduce(
    LocalNeuronData neurons,
    LocalRuleData rules,
    int* local_production_out // Output: Size [local_neuron_count]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;

    // Reset production for this step
    local_production_out[idx] = 0;

    // 1. Update Delays
    if (neurons.delay_timer[idx] > 0) {
        neurons.delay_timer[idx]--;
        if (neurons.delay_timer[idx] == 0) {
            neurons.is_open[idx] = true;
            // If we had pending emissions from a previous delayed rule, they emit NOW.
            local_production_out[idx] += neurons.pending_emission[idx];
            neurons.pending_emission[idx] = 0;
        }
    }

    // 2. If Open, Check Rules
    if (neurons.is_open[idx]) {
        int current_spikes = neurons.current_spikes[idx];
        int r_start = rules.rule_start_idx[idx];
        int r_count = rules.rule_count[idx];

        for (int i = 0; i < r_count; ++i) {
            int r_ptr = r_start + i;
            if (current_spikes >= rules.threshold[r_ptr]) {
                // Rule Applies
                neurons.current_spikes[idx] -= rules.consumed[r_ptr];
                
                int p = rules.produced[r_ptr];
                int d = rules.delay[r_ptr];

                if (d > 0) {
                    // Delayed production
                    neurons.is_open[idx] = false;
                    neurons.delay_timer[idx] = d;
                    neurons.pending_emission[idx] = p;
                } else {
                    // Immediate production
                    local_production_out[idx] += p;
                }
                
                // Deterministic: fire only first applicable rule
                break; 
            }
        }
    }
}

/**
 * @brief Step 2: Distribute Spikes via Synapses (Global -> Local)
 * * Iterates over ALL synapses.
 * If synapse.source fired (checked via global_production buffer) AND synapse.dest is on this rank:
 * Add spikes to the local neuron.
 */
__global__ void kDistributeGlobalSpikes(
    GlobalSynapseData synapses,
    LocalNeuronData neurons,
    const int* __restrict__ global_production_buffer, // Input: Size [total_neurons]
    int my_rank_start_id,
    int my_rank_end_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.count) return;

    int src_global = synapses.source_global_id[idx];
    int dest_global = synapses.dest_global_id[idx];

    // Check 1: Did the source emit spikes this turn?
    int spikes_emitted = global_production_buffer[src_global];

    if (spikes_emitted > 0) {
        // Check 2: Is the destination owned by this rank?
        if (dest_global >= my_rank_start_id && dest_global < my_rank_end_id) {
            
            // Map global ID to local offset
            int dest_local_idx = dest_global - my_rank_start_id;
            
            // [cite_start] Only open neurons receive spikes [cite: 34]
            if (neurons.is_open[dest_local_idx]) {
                int weight = synapses.weight[idx];
                atomicAdd(&neurons.current_spikes[dest_local_idx], spikes_emitted * weight);
            }
        }
    }
}

__global__ void kResetLocalNeurons(LocalNeuronData neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;
    
    neurons.current_spikes[idx] = neurons.initial_spikes[idx];
    neurons.is_open[idx] = true;
    neurons.delay_timer[idx] = 0;
    neurons.pending_emission[idx] = 0;
}

} // anonymous namespace

// --- Main Class Implementation ---

class NaiveCudaMpiSnpSimulator : public ISnpSimulator {
private:
    // MPI Context
    int mpi_rank;
    int mpi_size;
    int global_num_neurons;
    int my_start_id;
    int my_end_id;
    int my_neuron_count;

    // Local Data (Device)
    LocalNeuronData d_local_neurons;
    LocalRuleData d_local_rules;
    int* d_local_production; // Output of phase 1
    
    // Global Data (Device)
    GlobalSynapseData d_synapses;
    int* d_global_production; // Input for phase 2 (replicated)

    // Host Buffers for MPI
    std::vector<int> h_local_production;
    std::vector<int> h_global_production;

    // MPI Gatherv helpers
    std::vector<int> mpi_recv_counts;
    std::vector<int> mpi_displs;

    // Partitioning & Permutation
    std::unique_ptr<IPartitioner> partitioner;
    std::vector<int> new_to_old_map;

    // Metrics
    double total_time_ms = 0;
    double mpi_time_ms = 0;
    double compute_time_ms = 0;

public:
    NaiveCudaMpiSnpSimulator() {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        // Default to Linear (Naive) Partitioning
        partitioner = std::make_unique<LinearPartitioner>();
    }

    // Allow switching partitioner strategy
    void setPartitioner(std::unique_ptr<IPartitioner> p) {
        partitioner = std::move(p);
    }

    ~NaiveCudaMpiSnpSimulator() {
        d_local_neurons.free();
        d_local_rules.free();
        d_synapses.free();
        if (my_neuron_count > 0) cudaFree(d_local_production);
        if (global_num_neurons > 0) cudaFree(d_global_production);
    }

    bool loadSystem(const SnpSystemConfig& original_config) override {
        // 1. Partition & Permute
        if (!partitioner) {
            partitioner = std::make_unique<LinearPartitioner>();
        }

        if (mpi_rank == 0) {
            std::cout << "Naive Simulator Partitioning using: " << IPartitioner::getPartitionerName(partitioner->getType()) << std::endl;
        }

        auto partition = partitioner->partition(original_config, mpi_size);
        auto perm_result = SnpSystemPermuter::permute(original_config, partition, mpi_size);
        
        // Store mapping for output
        new_to_old_map = perm_result.new_to_old;
        
        // Use the new config
        const SnpSystemConfig& config = perm_result.config;
        global_num_neurons = config.neurons.size();

        // 2. Set Local Range from Permutation Result
        // The permuter guarantees that partitions are contiguous in the new ID space
        // Rank i gets [partition_offsets[i], partition_offsets[i] + partition_counts[i])
        my_start_id = perm_result.partition_offsets[mpi_rank];
        my_neuron_count = perm_result.partition_counts[mpi_rank];
        my_end_id = my_start_id + my_neuron_count;

        // Prepare MPI Allgatherv arrays
        mpi_recv_counts = perm_result.partition_counts;
        
        // Compute displacements
        mpi_displs.resize(mpi_size);
        mpi_displs[0] = 0;
        for(int i=1; i<mpi_size; i++) {
            mpi_displs[i] = mpi_displs[i-1] + mpi_recv_counts[i-1];
        }

        // 3. Allocate Device Memory
        d_local_neurons.allocate(my_neuron_count);
        d_synapses.allocate(config.synapses.size());
        
        if (my_neuron_count > 0) {
            CUDA_CHECK(cudaMalloc(&d_local_production, my_neuron_count * sizeof(int)));
        }
        if (global_num_neurons > 0) {
            CUDA_CHECK(cudaMalloc(&d_global_production, global_num_neurons * sizeof(int)));
        }

        // 4. Upload Local Neuron Data
        uploadLocalNeurons(config);
        
        // 5. Upload Rules (Flattened for Local Neurons)
        uploadLocalRules(config);

        // 6. Upload ALL Synapses (Replication strategy)
        uploadGlobalSynapses(config);

        // Allocate Host buffers
        h_local_production.resize(my_neuron_count);
        h_global_production.resize(global_num_neurons);

        return true;
    }

    void step(int steps = 1) override {
        for (int k = 0; k < steps; ++k) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // --- Phase 1: Local Compute ---
            if (my_neuron_count > 0) {
                int grid = (my_neuron_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kLocalComputeAndProduce<<<grid, BLOCK_SIZE>>>(
                    d_local_neurons, d_local_rules, d_local_production
                );
                CUDA_CHECK(cudaGetLastError());
                
                // Copy local production to Host for MPI
                CUDA_CHECK(cudaMemcpy(h_local_production.data(), d_local_production, 
                           my_neuron_count * sizeof(int), cudaMemcpyDeviceToHost));
            }

            // Sync Compute
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();

            // --- Phase 2: Communication (MPI) ---
            // Exchange production vectors. Result: h_global_production has data from ALL ranks.
            // Even if my_neuron_count is 0, we participate in the collective.
            MPI_Allgatherv(
                h_local_production.data(), my_neuron_count, MPI_INT,
                h_global_production.data(), mpi_recv_counts.data(), mpi_displs.data(), MPI_INT,
                MPI_COMM_WORLD
            );

            auto t2 = std::chrono::high_resolution_clock::now();

            // Copy Global Production back to Device
            if (global_num_neurons > 0) {
                CUDA_CHECK(cudaMemcpy(d_global_production, h_global_production.data(), 
                           global_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
            }

            // --- Phase 3: Global Distribution (on GPU) ---
            // Iterate synapses. If source fired (check d_global_production) and dest is mine, update mine.
            if (d_synapses.count > 0) {
                int grid = (d_synapses.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kDistributeGlobalSpikes<<<grid, BLOCK_SIZE>>>(
                    d_synapses, d_local_neurons, d_global_production,
                    my_start_id, my_end_id
                );
                CUDA_CHECK(cudaGetLastError());
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            auto t3 = std::chrono::high_resolution_clock::now();

            // Timing
            compute_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count(); // P1
            compute_time_ms += std::chrono::duration<double, std::milli>(t3 - t2).count(); // P3
            mpi_time_ms     += std::chrono::duration<double, std::milli>(t2 - t1).count(); // P2
        }
    }

    std::vector<int> getGlobalState() const override {
        // 1. Download Local State
        std::vector<int> local_state(my_neuron_count);
        if (my_neuron_count > 0) {
            CUDA_CHECK(cudaMemcpy(local_state.data(), d_local_neurons.current_spikes, 
                       my_neuron_count * sizeof(int), cudaMemcpyDeviceToHost));
        }

        // 2. Gather at Root (Rank 0)
        std::vector<int> global_state(global_num_neurons);

        // We can reuse mpi_recv_counts/displs calculated in loadSystem
        // Note: const_cast is safe here because MPI doesn't modify send buffer
        MPI_Gatherv(
            local_state.data(), my_neuron_count, MPI_INT,
            global_state.data(), const_cast<int*>(mpi_recv_counts.data()), 
            const_cast<int*>(mpi_displs.data()), MPI_INT,
            0, MPI_COMM_WORLD
        );

        // 3. Broadcast result from root to all ranks
        MPI_Bcast(global_state.data(), global_num_neurons, MPI_INT, 0, MPI_COMM_WORLD);

        // 4. Restore Original Order (if permuted)
        if (!new_to_old_map.empty()) {
            std::vector<int> restored_state(global_num_neurons);
            for (int i = 0; i < global_num_neurons; ++i) {
                restored_state[new_to_old_map[i]] = global_state[i];
            }
            return restored_state;
        }

        return global_state;
    }

    void reset() override {
        if (my_neuron_count > 0) {
            int grid = (my_neuron_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kResetLocalNeurons<<<grid, BLOCK_SIZE>>>(d_local_neurons);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        compute_time_ms = 0;
        mpi_time_ms = 0;
    }

    std::string getPerformanceReport() const override {
        // Collect avg times across ranks
        double avg_compute, avg_mpi;
        MPI_Reduce(&compute_time_ms, &avg_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mpi_time_ms, &avg_mpi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            avg_compute /= mpi_size;
            avg_mpi /= mpi_size;
            
            std::ostringstream ss;
            ss << "=== Naive CUDA+MPI Hybrid Simulator Report ===\n";
            ss << "Nodes: " << mpi_size << "\n";
            ss << "Avg Compute Time (CUDA): " << avg_compute << " ms\n";
            ss << "Avg Comm Time (MPI):    " << avg_mpi << " ms\n";
            ss << "Ratio (Comm/Comp):      " << (avg_mpi / (avg_compute + 1e-9)) << "\n";
            return ss.str();
        }
        return "";
    }

private:
    // --- Helper Methods ---

    void uploadLocalNeurons(const SnpSystemConfig& config) {
        if (my_neuron_count == 0) return;

        std::vector<int> spikes(my_neuron_count);
        std::vector<int> init(my_neuron_count);
        std::vector<char> open(my_neuron_count, true);
        std::vector<int> zeros(my_neuron_count, 0);

        for (int i = 0; i < my_neuron_count; ++i) {
            spikes[i] = config.neurons[my_start_id + i].initial_spikes;
            init[i] = spikes[i];
        }

        CUDA_CHECK(cudaMemcpy(d_local_neurons.current_spikes, spikes.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_neurons.initial_spikes, init.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_neurons.is_open, open.data(), my_neuron_count * sizeof(bool), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_neurons.delay_timer, zeros.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_neurons.pending_emission, zeros.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
    }

    void uploadLocalRules(const SnpSystemConfig& config) {
        if (my_neuron_count == 0) return;

        std::vector<int> h_threshold, h_consumed, h_produced, h_delay;
        std::vector<int> h_start(my_neuron_count), h_count(my_neuron_count);

        int current_idx = 0;
        for (int i = 0; i < my_neuron_count; ++i) {
            int global_id = my_start_id + i;
            const auto& rules = config.neurons[global_id].rules;
            
            h_start[i] = current_idx;
            h_count[i] = rules.size();
            
            for (const auto& r : rules) {
                h_threshold.push_back(r.input_threshold);
                h_consumed.push_back(r.spikes_consumed);
                h_produced.push_back(r.spikes_produced);
                h_delay.push_back(r.delay);
                current_idx++;
            }
        }
        
        d_local_rules.allocate(my_neuron_count, current_idx);
        
        CUDA_CHECK(cudaMemcpy(d_local_rules.rule_start_idx, h_start.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_rules.rule_count, h_count.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        
        if (current_idx > 0) {
            CUDA_CHECK(cudaMemcpy(d_local_rules.threshold, h_threshold.data(), current_idx * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_rules.consumed, h_consumed.data(), current_idx * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_rules.produced, h_produced.data(), current_idx * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_rules.delay, h_delay.data(), current_idx * sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    void uploadGlobalSynapses(const SnpSystemConfig& config) {
        size_t n = config.synapses.size();
        if (n == 0) return;

        std::vector<int> src(n), dst(n), w(n);
        for (size_t i = 0; i < n; ++i) {
            src[i] = config.synapses[i].source_id;
            dst[i] = config.synapses[i].dest_id;
            w[i]   = config.synapses[i].weight;
        }

        CUDA_CHECK(cudaMemcpy(d_synapses.source_global_id, src.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_synapses.dest_global_id, dst.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_synapses.weight, w.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    }
};

// Factory Implementation
std::unique_ptr<ISnpSimulator> createNaiveCudaMpiSimulator() {
    return std::make_unique<NaiveCudaMpiSnpSimulator>();
}