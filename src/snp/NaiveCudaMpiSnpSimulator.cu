#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include "SnpSystemPermuter.hpp"
#include "PerformanceMetrics.hpp"
#include "IPartitioner.hpp"
#include "LinearPartitioner.hpp"
#include "LouvainPartitioner.hpp"
#include "RedBluePartitioner.hpp"
#include "../../bigdata/BigDataCommon.hpp"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <memory>
#include <fstream>
#include <map>
#include <tuple>

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
    int steps_executed = 0;

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

    bool loadPresplitSystem(const std::string& partition_file) override {
        std::ifstream in(partition_file, std::ios::binary);
        if (!in) {
            std::cerr << "Rank " << mpi_rank << ": Failed to open " << partition_file << std::endl;
            return false;
        }

        bigdata::PartitionHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (header.magic != bigdata::PARTITION_MAGIC) {
            std::cerr << "Rank " << mpi_rank << ": Invalid magic number" << std::endl;
            return false;
        }

        my_neuron_count = header.num_local_neurons;

        // 1. Exchange counts to build global map
        std::vector<int> rank_counts(mpi_size);
        MPI_Allgather(&my_neuron_count, 1, MPI_INT, rank_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        mpi_recv_counts = rank_counts;
        mpi_displs.resize(mpi_size);
        mpi_displs[0] = 0;
        for (int i = 1; i < mpi_size; ++i) {
            mpi_displs[i] = mpi_displs[i-1] + mpi_recv_counts[i-1];
        }
        global_num_neurons = mpi_displs.back() + mpi_recv_counts.back();
        my_start_id = mpi_displs[mpi_rank];
        my_end_id = my_start_id + my_neuron_count;

        // 2. Allocate Device Memory
        d_local_neurons.allocate(my_neuron_count);
        if (my_neuron_count > 0) {
            CUDA_CHECK(cudaMalloc(&d_local_production, my_neuron_count * sizeof(int)));
        }
        if (global_num_neurons > 0) {
            CUDA_CHECK(cudaMalloc(&d_global_production, global_num_neurons * sizeof(int)));
        }
        // Host buffers
        h_local_production.resize(my_neuron_count);
        h_global_production.resize(global_num_neurons);

        // 3. Read Neurons & Rules
        std::vector<int> h_config(my_neuron_count);
        std::vector<int> h_rule_start(my_neuron_count);
        std::vector<int> h_rule_count(my_neuron_count);
        
        // Temp storage for rules (flattened)
        std::vector<int> r_thresh, r_cons, r_prod, r_delay;
        
        int current_rule_idx = 0;
        for (int i = 0; i < my_neuron_count; ++i) {
            int32_t id, init_spikes, num_rules;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            in.read(reinterpret_cast<char*>(&init_spikes), sizeof(init_spikes));
            in.read(reinterpret_cast<char*>(&num_rules), sizeof(num_rules));
            
            h_config[i] = init_spikes;
            h_rule_start[i] = current_rule_idx;
            h_rule_count[i] = num_rules;

            for (int k = 0; k < num_rules; ++k) {
                bigdata::RuleData rd;
                in.read(reinterpret_cast<char*>(&rd), sizeof(rd));
                r_thresh.push_back(rd.input_threshold);
                r_cons.push_back(rd.spikes_consumed);
                r_prod.push_back(rd.spikes_produced);
                r_delay.push_back(rd.delay);
                current_rule_idx++;
            }
        }

        // Upload Neurons
        if (my_neuron_count > 0) {
            std::vector<int> zeros(my_neuron_count, 0);
            std::vector<char> open(my_neuron_count, true);
            CUDA_CHECK(cudaMemcpy(d_local_neurons.current_spikes, h_config.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_neurons.initial_spikes, h_config.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_neurons.is_open, open.data(), my_neuron_count * sizeof(char), cudaMemcpyHostToDevice)); // bool/char size match?
            CUDA_CHECK(cudaMemcpy(d_local_neurons.delay_timer, zeros.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_neurons.pending_emission, zeros.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        }

        // Upload Rules
        d_local_rules.allocate(my_neuron_count, current_rule_idx);
        CUDA_CHECK(cudaMemcpy(d_local_rules.rule_start_idx, h_rule_start.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_rules.rule_count, h_rule_count.data(), my_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        if (current_rule_idx > 0) {
            CUDA_CHECK(cudaMemcpy(d_local_rules.threshold, r_thresh.data(), current_rule_idx * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_rules.consumed, r_cons.data(), current_rule_idx * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_rules.produced, r_prod.data(), current_rule_idx * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_rules.delay, r_delay.data(), current_rule_idx * sizeof(int), cudaMemcpyHostToDevice));
        }

        // 4. Read Synapses & Prepare for Global Exchange
        // We need to reconstruct the GLOBAL synapse list.
        // Strategy: Gather ALL Local, Export, and Import definitions from ALL ranks.
        
        std::vector<int> local_syn_buf; // [src_global, dst_global, weight, ...]
        std::vector<int> export_syn_buf; // [src_global, target_rank, export_idx, weight, ...]
        std::vector<int> import_syn_buf; // [source_rank, export_idx, dst_global, ...]

        // Read Local Synapses
        for (uint64_t i = 0; i < header.num_local_synapses; ++i) {
            bigdata::LocalSynapseData sd;
            in.read(reinterpret_cast<char*>(&sd), sizeof(sd));
            local_syn_buf.push_back(my_start_id + sd.source_local_idx);
            local_syn_buf.push_back(my_start_id + sd.dest_local_idx);
            local_syn_buf.push_back(sd.weight);
        }

        // Read Export Groups
        for (uint64_t i = 0; i < header.num_export_groups; ++i) {
            bigdata::ExportGroupHeader gh;
            in.read(reinterpret_cast<char*>(&gh), sizeof(gh));
            for (uint64_t k = 0; k < gh.num_synapses; ++k) {
                bigdata::ExportSynapseData sd;
                in.read(reinterpret_cast<char*>(&sd), sizeof(sd));
                export_syn_buf.push_back(my_start_id + sd.source_local_idx);
                export_syn_buf.push_back(gh.target_rank);
                export_syn_buf.push_back(k); // export_idx
                export_syn_buf.push_back(sd.weight);
            }
        }

        // Read Import Groups
        for (uint64_t i = 0; i < header.num_import_groups; ++i) {
            bigdata::ImportGroupHeader gh;
            in.read(reinterpret_cast<char*>(&gh), sizeof(gh));
            for (uint64_t k = 0; k < gh.num_synapses; ++k) {
                bigdata::ImportSynapseData sd;
                in.read(reinterpret_cast<char*>(&sd), sizeof(sd));
                import_syn_buf.push_back(gh.source_rank);
                import_syn_buf.push_back(sd.export_index);
                import_syn_buf.push_back(my_start_id + sd.dest_local_idx);
            }
        }

        // 5. Global Exchange (Allgatherv)
        auto gather_vector = [&](const std::vector<int>& local_vec) {
            int local_size = local_vec.size();
            std::vector<int> sizes(mpi_size);
            MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
            
            std::vector<int> displs(mpi_size);
            displs[0] = 0;
            for(int i=1; i<mpi_size; i++) displs[i] = displs[i-1] + sizes[i-1];
            
            int total_size = displs.back() + sizes.back();
            std::vector<int> global_vec(total_size);
            
            MPI_Allgatherv(local_vec.data(), local_size, MPI_INT, 
                           global_vec.data(), sizes.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
            return global_vec;
        };

        std::vector<int> all_local_syns = gather_vector(local_syn_buf);
        std::vector<int> all_export_syns = gather_vector(export_syn_buf);
        std::vector<int> all_import_syns = gather_vector(import_syn_buf);

        // 6. Reconstruct Global Synapse List
        std::vector<int> final_src, final_dst, final_w;

        // Add Local Synapses
        for (size_t i = 0; i < all_local_syns.size(); i += 3) {
            final_src.push_back(all_local_syns[i]);
            final_dst.push_back(all_local_syns[i+1]);
            final_w.push_back(all_local_syns[i+2]);
        }

        // Map Imports: (SourceRank, TargetRank, ExportIdx) -> GlobalDst
        // Note: Import data is [SourceRank, ExportIdx, GlobalDst]
        // But we need to know WHICH rank provided this import data to know TargetRank.
        // We can infer TargetRank from the displs/sizes used in gather_vector, OR we can just include TargetRank in the buffer.
        // Let's re-do the gather logic slightly to include "MyRank" in the buffer? 
        // Or just iterate using the sizes/displs.
        
        // Re-calculate sizes/displs for imports
        {
            int local_size = import_syn_buf.size();
            std::vector<int> sizes(mpi_size);
            MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
            std::vector<int> displs(mpi_size);
            displs[0] = 0;
            for(int i=1; i<mpi_size; i++) displs[i] = displs[i-1] + sizes[i-1];

            // Map: Key = {SourceRank, TargetRank, ExportIdx} -> Value = GlobalDst
            std::map<std::tuple<int, int, int>, int> import_map;

            for (int r = 0; r < mpi_size; ++r) {
                int start = displs[r];
                int count = sizes[r];
                for (int k = 0; k < count; k += 3) {
                    int src_rank = all_import_syns[start + k];
                    int exp_idx = all_import_syns[start + k + 1];
                    int dst_global = all_import_syns[start + k + 2];
                    int tgt_rank = r; // The rank that provided this data
                    import_map[{src_rank, tgt_rank, exp_idx}] = dst_global;
                }
            }

            // Match Exports
            // Export data: [src_global, target_rank, export_idx, weight]
            // We don't need to know who sent the export data, just the content.
            for (size_t i = 0; i < all_export_syns.size(); i += 4) {
                int src_global = all_export_syns[i];
                int tgt_rank = all_export_syns[i+1];
                int exp_idx = all_export_syns[i+2];
                int weight = all_export_syns[i+3];
                
                // Find Source Rank? No, we need Source Rank to look up in import_map.
                // The Export data doesn't explicitly say "I am from Rank X".
                // But we need it for the key {SourceRank, TargetRank, ExportIdx}.
                // So we DO need to iterate by rank for exports too.
            }
        }
        
        // Correct approach: Iterate by rank for Exports too.
        {
             // Recalculate sizes/displs for imports (needed for map construction)
            int local_imp_size = import_syn_buf.size();
            std::vector<int> imp_sizes(mpi_size);
            MPI_Allgather(&local_imp_size, 1, MPI_INT, imp_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
            std::vector<int> imp_displs(mpi_size);
            imp_displs[0] = 0;
            for(int i=1; i<mpi_size; i++) imp_displs[i] = imp_displs[i-1] + imp_sizes[i-1];

            std::map<std::tuple<int, int, int>, int> import_map;
            for (int r = 0; r < mpi_size; ++r) {
                int start = imp_displs[r];
                int count = imp_sizes[r];
                for (int k = 0; k < count; k += 3) {
                    int src_rank = all_import_syns[start + k];
                    int exp_idx = all_import_syns[start + k + 1];
                    int dst_global = all_import_syns[start + k + 2];
                    import_map[{src_rank, r, exp_idx}] = dst_global;
                }
            }

            // Recalculate sizes/displs for exports
            int local_exp_size = export_syn_buf.size();
            std::vector<int> exp_sizes(mpi_size);
            MPI_Allgather(&local_exp_size, 1, MPI_INT, exp_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
            std::vector<int> exp_displs(mpi_size);
            exp_displs[0] = 0;
            for(int i=1; i<mpi_size; i++) exp_displs[i] = exp_displs[i-1] + exp_sizes[i-1];

            for (int r = 0; r < mpi_size; ++r) {
                int start = exp_displs[r];
                int count = exp_sizes[r];
                for (int k = 0; k < count; k += 4) {
                    int src_global = all_export_syns[start + k];
                    int tgt_rank = all_export_syns[start + k + 1];
                    int exp_idx = all_export_syns[start + k + 2];
                    int weight = all_export_syns[start + k + 3];
                    
                    // Lookup Dst
                    if (import_map.count({r, tgt_rank, exp_idx})) {
                        int dst_global = import_map[{r, tgt_rank, exp_idx}];
                        final_src.push_back(src_global);
                        final_dst.push_back(dst_global);
                        final_w.push_back(weight);
                    } else {
                        // Should not happen if data is consistent
                        if (mpi_rank == 0) std::cerr << "Warning: Unmatched export from Rank " << r << " to " << tgt_rank << " idx " << exp_idx << std::endl;
                    }
                }
            }
        }

        // 7. Upload Synapses
        size_t n_syn = final_src.size();
        
        if (mpi_rank == 0) {
             std::cout << "Rank " << mpi_rank << ": Loaded " << my_neuron_count << " neurons, " 
                  << current_rule_idx << " rules, " << n_syn << " synapses." << std::endl;
        }

        d_synapses.allocate(n_syn);
        if (n_syn > 0) {
            CUDA_CHECK(cudaMemcpy(d_synapses.source_global_id, final_src.data(), n_syn * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_synapses.dest_global_id, final_dst.data(), n_syn * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_synapses.weight, final_w.data(), n_syn * sizeof(int), cudaMemcpyHostToDevice));
        }

        return true;
    }

    bool loadSystem(const SnpSystemConfig& original_config) override {
        // 1. Partition & Permute
        if (!partitioner) {
            partitioner = std::make_unique<LinearPartitioner>();
        }

        std::vector<int> partition;
        if (mpi_rank == 0)
        {
            partition = partitioner->partition(original_config, mpi_size);
        }

        // Broadcast partition to all ranks to ensure consistency
        int n_neurons = original_config.neurons.size();
        if (mpi_rank != 0)
        {
            partition.resize(n_neurons);
        }
        MPI_Bcast(partition.data(), n_neurons, MPI_INT, 0, MPI_COMM_WORLD);
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
        auto step_start = std::chrono::high_resolution_clock::now();
        
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
            steps_executed++;
        }
        
        auto step_end = std::chrono::high_resolution_clock::now();
        total_time_ms += std::chrono::duration<double, std::milli>(step_end - step_start).count();
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

    PerformanceMetrics getPerformanceMetrics() const override {
        PerformanceMetrics metrics;
        
        // Core metrics
        metrics.steps_executed = steps_executed;
        metrics.total_time_ms = total_time_ms;
        metrics.compute_time_ms = compute_time_ms;
        metrics.cuda.kernel_time_ms = compute_time_ms; // Populate kernel time for reporting
        
        // MPI metrics
        metrics.mpi.communication_time_ms = mpi_time_ms;
        metrics.mpi.rank = mpi_rank;
        metrics.mpi.world_size = mpi_size;
        
        // Algorithm metrics
        metrics.algorithm.num_neurons = global_num_neurons;
        metrics.algorithm.local_neurons = my_neuron_count;
        metrics.algorithm.total_rules = d_local_rules.total_rules_count;
        // For Naive, we replicate all synapses. 
        // To avoid double counting in global sum, we only report synapses on Rank 0.
        // Or we report 0 and let the user know Naive doesn't partition synapses.
        // However, BigDataRun.cpp sums them up.
        // Let's report the total count on Rank 0, and 0 on others.
        if (mpi_rank == 0) {
            metrics.algorithm.num_synapses = d_synapses.count;
        } else {
            metrics.algorithm.num_synapses = 0;
        }
        metrics.algorithm.local_synapses = 0; // Naive doesn't have "local" synapses concept really
        metrics.algorithm.partitioner_type = "Naive CUDA+MPI";
        
        return metrics;
    }

    std::string getPerformanceReport() const override {
        // Collect avg times across ranks
        double avg_compute, avg_mpi;
        MPI_Reduce(&compute_time_ms, &avg_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mpi_time_ms, &avg_mpi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            avg_compute /= mpi_size;
            avg_mpi /= mpi_size;
            
            PerformanceMetrics metrics;
            metrics.total_time_ms = avg_compute + avg_mpi;
            metrics.compute_time_ms = avg_compute;
            metrics.mpi.communication_time_ms = avg_mpi;
            metrics.mpi.world_size = mpi_size;
            metrics.algorithm.num_neurons = global_num_neurons;
            metrics.algorithm.partitioner_type = "Naive CUDA+MPI";
            
            std::ostringstream ss;
            ss << metrics.toReport("Naive CUDA+MPI Hybrid Simulator");
            ss << "\n[Distribution]\n";
            ss << "  Compute/Communication Ratio: " 
               << (avg_compute / (avg_mpi + 1e-9)) << "\n";
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
std::unique_ptr<ISnpSimulator> createNaiveCudaMpiSimulator(PartitionerType partitionerType)
{
    auto sim = std::make_unique<NaiveCudaMpiSnpSimulator>();
    std::unique_ptr<IPartitioner> partitioner;
    switch (partitionerType)
    {
    case PartitionerType::LINEAR:
        partitioner = std::make_unique<LinearPartitioner>();
        break;
    case PartitionerType::LOUVAIN:
        partitioner = std::make_unique<LouvainPartitioner>();
        break;
    case PartitionerType::RED_BLUE_BFS:
        partitioner = std::make_unique<RedBluePartitioner>();
        break;
    default:
        partitioner = std::make_unique<LinearPartitioner>();
        break;
    }
    sim->setPartitioner(std::move(partitioner));
    return sim;
}
