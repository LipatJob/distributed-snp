#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include "PerformanceMetrics.hpp"
#include "IPartitioner.hpp"
#include "LinearPartitioner.hpp"
#include "LouvainPartitioner.hpp"
#include "RedBluePartitioner.hpp"
#ifdef ENABLE_METIS
#include "MetisPartitioner.hpp"
#endif
#include "SnpSystemPermuter.hpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <map>
#include <chrono>
#include <sstream>

// --- Macros ---
#define BLOCK_SIZE 128
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

// --- Device Structures (Adapted from SparseCudaSnpSimulator) ---

struct DeviceRuleVector {
    int* input_threshold;
    int* spikes_consumed;
    int* spikes_produced;
    int* delay; 
    int total_rules;

    void allocate(int n) {
        total_rules = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&input_threshold, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_consumed, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_produced, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&delay, n * sizeof(int)));
    }

    void deallocate() {
        if (total_rules == 0) return;
        cudaFree(input_threshold); cudaFree(spikes_consumed);
        cudaFree(spikes_produced); cudaFree(delay);
    }
};

struct DeviceNeuronRuleMap {
    int* rule_start_idx;
    int* rule_count;

    void allocate(int n) {
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&rule_start_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rule_count, n * sizeof(int)));
    }

    void deallocate() {
        cudaFree(rule_start_idx); cudaFree(rule_count);
    }
};

// Local Synapse Matrix (Column-Major: Z x N_local)
struct DeviceSynapseMatrix {
    int* matrix; // Dest IDs (Local)
    int* weights;
    int max_out_degree;
    int num_neurons;

    void allocate(int neurons, int z) {
        num_neurons = neurons;
        max_out_degree = z;
        if (neurons * z == 0) return;
        CUDA_CHECK(cudaMalloc(&matrix, neurons * z * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weights, neurons * z * sizeof(int)));
    }

    void deallocate() {
        if (num_neurons * max_out_degree == 0) return;
        cudaFree(matrix); cudaFree(weights);
    }
};

struct DeviceState {
    int* config_vector;
    int* delay_vector;
    int* spiking_vector;    // Index of active rule
    int* pending_emission;
    int* initial_config;

    void allocate(int n) {
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&config_vector, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&delay_vector, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spiking_vector, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&pending_emission, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&initial_config, n * sizeof(int)));
    }

    void deallocate() {
        cudaFree(config_vector); cudaFree(delay_vector);
        cudaFree(spiking_vector); cudaFree(pending_emission);
        cudaFree(initial_config);
    }
};

// --- MPI Structures ---

// Export Map: Maps (LocalNeuron) -> (ExportBufferIndex)
// We use a simple list approach for exports since they are sparse after partitioning
struct DeviceExportList {
    int* source_neuron_idx; // Which local neuron fires
    int* export_buf_idx;    // Where in the export buffer to write
    int* weight;            // Synapse weight
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&source_neuron_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&export_buf_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weight, n * sizeof(int)));
    }

    void deallocate() {
        if (count == 0) return;
        cudaFree(source_neuron_idx); cudaFree(export_buf_idx); cudaFree(weight);
    }
};

struct DeviceImportMap {
    int* import_buf_idx;
    int* dest_neuron_idx;
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&import_buf_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dest_neuron_idx, n * sizeof(int)));
    }

    void deallocate() {
        if (count == 0) return;
        cudaFree(import_buf_idx); cudaFree(dest_neuron_idx);
    }
};

// --- Kernels ---

// 1. Calculate Spiking Vector (Same as SparseCudaSnpSimulator)
__global__ void k_calc_spiking_vector_mpi(
    int num_neurons,
    const int* __restrict__ config,
    const int* __restrict__ delay,
    int* __restrict__ spiking_vector,
    const int* __restrict__ rule_start,
    const int* __restrict__ rule_count,
    const int* __restrict__ r_threshold
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= num_neurons) return;

    spiking_vector[nid] = -1;

    if (delay[nid] == 0) {
        int current_spikes = config[nid];
        int start = rule_start[nid];
        int count = rule_count[nid];

        for (int i = 0; i < count; ++i) {
            int rid = start + i;
            if (current_spikes >= r_threshold[rid]) {
                spiking_vector[nid] = rid;
                break; 
            }
        }
    }
}

// 2. Local Step: Consume, Delay, Propagate Local
__global__ void k_step_local_mpi(
    int num_neurons,
    int max_out_degree,
    int* config,
    int* delay_vector,
    int* pending_emission,
    const int* __restrict__ spiking_vector,
    const int* __restrict__ synapse_matrix,
    const int* __restrict__ synapse_weights,
    const int* __restrict__ r_consumed,
    const int* __restrict__ r_produced,
    const int* __restrict__ r_delay
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= num_neurons) return;

    int active_rule_idx = spiking_vector[nid];

    // Logic:
    // If firing: Consume spikes, Set Delay.
    // If delay > 0: produce = 0 (stored in pending if needed, but here we handle immediate)
    // Wait, if delay > 0, we don't produce yet.
    
    // We need to handle:
    // A. Rule Fired Now
    // B. Pending Emission from previous step (delay just finished)
    
    // In SparseCudaSnpSimulator, pending_emission stores spikes waiting for delay.
    // k_step_compressed handles both.
    
    int produced_now = 0;

    if (active_rule_idx >= 0 && delay_vector[nid] == 0) {
        int consumed = r_consumed[active_rule_idx];
        int produced = r_produced[active_rule_idx];
        int d = r_delay[active_rule_idx];

        config[nid] -= consumed;

        if (d > 0) {
            delay_vector[nid] = d;
            pending_emission[nid] = produced;
        } else {
            produced_now = produced;
        }
    } else if (delay_vector[nid] > 0) {
        delay_vector[nid]--;
        if (delay_vector[nid] == 0) {
            produced_now = pending_emission[nid];
            pending_emission[nid] = 0;
        }
    }

    // Propagate Local
    if (produced_now > 0) {
        // We also need to store this 'produced_now' somewhere for the Export kernel?
        // Or we can just re-calculate it or store it in pending_emission temporarily?
        // Let's store it in pending_emission with a flag? No.
        // Let's use pending_emission to signal "spikes to send this turn".
        // But pending_emission is for FUTURE sends.
        
        // Hack: We can use the 'spiking_vector' or a temp buffer to store "spikes produced this turn".
        // But 'spiking_vector' only stores rule index.
        
        // Let's just propagate locally here.
        for (int i = 0; i < max_out_degree; ++i) {
            int dest_nid = synapse_matrix[nid * max_out_degree + i];
            if (dest_nid >= 0) {
                int weight = synapse_weights[nid * max_out_degree + i];
                atomicAdd(&config[dest_nid], produced_now * weight);
            } else {
                break;
            }
        }
        
        // For Export: We need to know if this neuron fired.
        // We can check (spiking_vector >= 0 && delay == 0) OR (delay just became 0).
        // This is complex to replicate in a second kernel without state.
        // Solution: Store 'produced_now' in a temporary buffer or reuse 'pending_emission' carefully.
        // Actually, let's use a separate 'current_production' buffer for MPI.
    }
    
    // We need to output 'produced_now' to memory so the Export kernel can see it.
    // Let's repurpose 'pending_emission' to mean "Spikes waiting to be sent OR just sent".
    // No, that confuses the logic.
    
    // Let's add a field to DeviceState: 'current_step_production'.
}

// Revised Kernel 2: Step Local + Record Production
__global__ void k_step_local_record_mpi(
    int num_neurons,
    int max_out_degree,
    int* config,
    int* delay_vector,
    int* pending_emission,
    int* current_production, // Output: Spikes produced this step
    const int* __restrict__ spiking_vector,
    const int* __restrict__ synapse_matrix,
    const int* __restrict__ synapse_weights,
    const int* __restrict__ r_consumed,
    const int* __restrict__ r_produced,
    const int* __restrict__ r_delay
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= num_neurons) return;

    int active_rule_idx = spiking_vector[nid];
    int produced_now = 0;

    // 1. Handle Rule Firing
    if (active_rule_idx >= 0 && delay_vector[nid] == 0) {
        int consumed = r_consumed[active_rule_idx];
        int produced = r_produced[active_rule_idx];
        int d = r_delay[active_rule_idx];

        config[nid] -= consumed;

        if (d > 0) {
            delay_vector[nid] = d;
            pending_emission[nid] = produced;
        } else {
            produced_now = produced;
        }
    } 
    // 2. Handle Delay Expiration
    else if (delay_vector[nid] > 0) {
        delay_vector[nid]--;
        if (delay_vector[nid] == 0) {
            produced_now = pending_emission[nid];
            pending_emission[nid] = 0;
        }
    }

    // 3. Store Production (for Export Kernel)
    current_production[nid] = produced_now;

    // 4. Propagate Local
    if (produced_now > 0) {
        for (int i = 0; i < max_out_degree; ++i) {
            int dest_nid = synapse_matrix[nid * max_out_degree + i];
            if (dest_nid >= 0) {
                int weight = synapse_weights[nid * max_out_degree + i];
                atomicAdd(&config[dest_nid], produced_now * weight);
            } else {
                break;
            }
        }
    }
}

// 3. Fill Export Buffer
__global__ void k_fill_export_buffer(
    int count,
    const int* __restrict__ source_neuron_idx,
    const int* __restrict__ export_buf_idx,
    const int* __restrict__ weight,
    const int* __restrict__ current_production,
    int* export_buffer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int src = source_neuron_idx[idx];
    int produced = current_production[src];

    if (produced > 0) {
        int buf_idx = export_buf_idx[idx];
        int w = weight[idx];
        // Multiple synapses might target the same export buffer slot (if multiple neurons target same remote neuron)
        // But usually export buffer is (Rank, RemoteNeuron).
        // If multiple local neurons target the same remote neuron, we must atomicAdd.
        atomicAdd(&export_buffer[buf_idx], produced * w);
    }
}

// 4. Apply Imports
__global__ void k_apply_imports(
    int count,
    const int* __restrict__ import_buf_idx,
    const int* __restrict__ dest_neuron_idx,
    const int* __restrict__ import_buffer,
    int* config
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int buf_idx = import_buf_idx[idx];
    int spikes = import_buffer[buf_idx];

    if (spikes > 0) {
        int dest = dest_neuron_idx[idx];
        atomicAdd(&config[dest], spikes);
    }
}

// --- Simulator Class ---

class SparseCudaMpiSnpSimulator : public ISnpSimulator {
private:
    // MPI
    int mpi_rank, mpi_size;
    std::unique_ptr<IPartitioner> partitioner;

    // Host Config
    int num_local_neurons = 0;
    int max_local_out_degree = 0;

    // Device Memory
    DeviceState d_state;
    int* d_current_production = nullptr; // Extra buffer
    DeviceRuleVector d_rv;
    DeviceNeuronRuleMap d_map;
    DeviceSynapseMatrix d_sy; // Local synapses
    DeviceExportList d_export;
    DeviceImportMap d_import;

    // Communication Buffers
    int* d_export_buffer = nullptr;
    int* d_import_buffer = nullptr;
    int* h_export_buffer = nullptr;
    int* h_import_buffer = nullptr;
    
    std::vector<int> send_counts;
    std::vector<int> recv_counts;
    std::vector<int> send_displs;
    std::vector<int> recv_displs;
    int total_send_size = 0;
    int total_recv_size = 0;

    // Global State Helpers
    std::vector<int> new_to_old_map;
    std::vector<int> global_neuron_counts;
    std::vector<int> global_neuron_displs;

    // Metrics
    double total_compute_time = 0.0;
    int steps_executed = 0;

public:
    SparseCudaMpiSnpSimulator() {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        // Default Partitioner
        #ifdef ENABLE_METIS
            partitioner = std::make_unique<MetisPartitioner>();
        #else
            partitioner = std::make_unique<LinearPartitioner>();
        #endif
    }

    ~SparseCudaMpiSnpSimulator() {
        d_state.deallocate();
        if (d_current_production) cudaFree(d_current_production);
        d_rv.deallocate();
        d_map.deallocate();
        d_sy.deallocate();
        d_export.deallocate();
        d_import.deallocate();
        
        if (d_export_buffer) cudaFree(d_export_buffer);
        if (d_import_buffer) cudaFree(d_import_buffer);
        if (h_export_buffer) cudaFreeHost(h_export_buffer);
        if (h_import_buffer) cudaFreeHost(h_import_buffer);
    }

    void setPartitioner(std::unique_ptr<IPartitioner> p) {
        partitioner = std::move(p);
    }

    bool loadSystem(const SnpSystemConfig& config) override {
        // 1. Partition
        std::vector<int> partition = partitioner->partition(config, mpi_size);
        
        // 2. Permute (Global -> Local)
        auto perm_result = SnpSystemPermuter::permute(config, partition, mpi_size);
        const auto& local_config = perm_result.config; // This is the FULL config but reordered? 
        // No, SnpSystemPermuter returns the full config reordered.
        // We need to extract OUR slice.
        
        int my_start = perm_result.partition_offsets[mpi_rank];
        int my_count = perm_result.partition_counts[mpi_rank];
        num_local_neurons = my_count;

        // Store global info for getGlobalState
        new_to_old_map = perm_result.new_to_old;
        global_neuron_counts = perm_result.partition_counts;
        global_neuron_displs = perm_result.partition_offsets;

        // 3. Build Local Data Structures
        // We need to iterate ONLY our neurons [my_start, my_start + my_count)
        // But wait, the 'local_config' has neurons 0..N.
        // Our neurons are at indices my_start...
        
        // Let's extract local rules and synapses.
        
        // A. Rules
        std::vector<int> h_threshold, h_consumed, h_produced, h_delay;
        std::vector<int> h_rule_start(num_local_neurons), h_rule_count(num_local_neurons);
        int current_rule_idx = 0;

        for (int i = 0; i < num_local_neurons; ++i) {
            int global_idx = my_start + i;
            const auto& neuron = local_config.neurons[global_idx];
            
            h_rule_start[i] = current_rule_idx;
            h_rule_count[i] = neuron.rules.size();
            
            for (const auto& r : neuron.rules) {
                h_threshold.push_back(r.input_threshold);
                h_consumed.push_back(r.spikes_consumed);
                h_produced.push_back(r.spikes_produced);
                h_delay.push_back(r.delay);
                current_rule_idx++;
            }
        }

        d_rv.allocate(h_threshold.size());
        CUDA_CHECK(cudaMemcpy(d_rv.input_threshold, h_threshold.data(), h_threshold.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rv.spikes_consumed, h_consumed.data(), h_consumed.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rv.spikes_produced, h_produced.data(), h_produced.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rv.delay, h_delay.data(), h_delay.size() * sizeof(int), cudaMemcpyHostToDevice));

        d_map.allocate(num_local_neurons);
        CUDA_CHECK(cudaMemcpy(d_map.rule_start_idx, h_rule_start.data(), num_local_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_map.rule_count, h_rule_count.data(), num_local_neurons * sizeof(int), cudaMemcpyHostToDevice));

        // B. Synapses (Local & Remote)
        // We need to scan synapses originating from our neurons.
        std::vector<std::vector<int>> local_adj(num_local_neurons);
        std::vector<std::vector<int>> local_weights(num_local_neurons);
        
        // Export Lists
        std::vector<int> h_exp_src, h_exp_buf_idx, h_exp_weight;
        
        // Map (Rank, RemoteNeuronGlobalID) -> ExportBufferIndex
        // We need to agree on an ordering for the export buffer.
        // Standard approach: 
        // 1. Identify all (Rank, RemoteID) pairs we send to.
        // 2. Sort them to create a canonical order.
        // 3. Assign indices.
        // 4. Exchange counts with other ranks so they know what to receive.
        
        // Step 1: Collect all outgoing edges
        struct Edge { int src_local; int dst_global; int weight; };
        std::vector<Edge> outgoing_edges;
        
        for (const auto& syn : local_config.synapses) {
            // Check if source is in our range
            if (syn.source_id >= my_start && syn.source_id < my_start + my_count) {
                int src_local = syn.source_id - my_start;
                
                // Check destination
                if (syn.dest_id >= my_start && syn.dest_id < my_start + my_count) {
                    // Local-to-Local
                    int dst_local = syn.dest_id - my_start;
                    local_adj[src_local].push_back(dst_local);
                    local_weights[src_local].push_back(syn.weight);
                } else {
                    // Local-to-Remote
                    outgoing_edges.push_back({src_local, syn.dest_id, syn.weight});
                }
            }
        }

        // Step 2: Build Local Synapse Matrix
        max_local_out_degree = 0;
        for (const auto& adj : local_adj) {
            if (adj.size() > max_local_out_degree) max_local_out_degree = adj.size();
        }
        
        d_sy.allocate(num_local_neurons, max_local_out_degree);
        std::vector<int> h_sy_matrix(num_local_neurons * max_local_out_degree, -1);
        std::vector<int> h_sy_weights(num_local_neurons * max_local_out_degree, 0);
        
        for (int i = 0; i < num_local_neurons; ++i) {
            for (size_t j = 0; j < local_adj[i].size(); ++j) {
                // Column-Major: index = row + col * num_rows = i + j * num_neurons
                // Wait, SparseCudaSnpSimulator uses: nid * max_out_degree + i (Row-Major within flattened?)
                // Let's check SparseCudaSnpSimulator.cu:
                // "synapse_matrix[nid * max_out_degree + i]" -> This is Row-Major (Row=nid).
                // The comment said "Column-Major" but the code used Row-Major indexing.
                // I will stick to the code: Row-Major.
                int idx = i * max_local_out_degree + j;
                h_sy_matrix[idx] = local_adj[i][j];
                h_sy_weights[idx] = local_weights[i][j];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_sy.matrix, h_sy_matrix.data(), h_sy_matrix.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sy.weights, h_sy_weights.data(), h_sy_weights.size() * sizeof(int), cudaMemcpyHostToDevice));

        // Step 3: Build Export/Import Maps
        // We need to know which rank owns which global ID.
        // perm_result.partition_offsets gives us this.
        auto get_rank = [&](int global_id) {
            for (int r = 0; r < mpi_size; ++r) {
                int start = perm_result.partition_offsets[r];
                int count = perm_result.partition_counts[r];
                if (global_id >= start && global_id < start + count) return r;
            }
            return -1;
        };

        // Group outgoing edges by Dest Rank
        std::vector<std::vector<int>> exports_per_rank(mpi_size); // Stores DestGlobalID
        for (const auto& edge : outgoing_edges) {
            int r = get_rank(edge.dst_global);
            if (r != -1 && r != mpi_rank) {
                exports_per_rank[r].push_back(edge.dst_global);
            }
        }

        // Uniqueify and sort exports per rank to establish buffer order
        for (auto& vec : exports_per_rank) {
            std::sort(vec.begin(), vec.end());
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
        }

        // Calculate Send Counts and Displacements
        send_counts.assign(mpi_size, 0);
        send_displs.assign(mpi_size, 0);
        total_send_size = 0;
        for (int r = 0; r < mpi_size; ++r) {
            send_counts[r] = exports_per_rank[r].size();
            send_displs[r] = total_send_size;
            total_send_size += send_counts[r];
        }

        // Exchange counts to determine Recv Counts
        recv_counts.resize(mpi_size);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        recv_displs.assign(mpi_size, 0);
        total_recv_size = 0;
        for (int r = 0; r < mpi_size; ++r) {
            recv_displs[r] = total_recv_size;
            total_recv_size += recv_counts[r];
        }

        // Build Export List (Device)
        // We need to map each outgoing edge to a buffer index.
        // Buffer index for rank r is: send_displs[r] + index_in_exports_per_rank[r]
        for (const auto& edge : outgoing_edges) {
            int r = get_rank(edge.dst_global);
            if (r == -1 || r == mpi_rank) continue;

            // Find index in the unique list
            auto it = std::lower_bound(exports_per_rank[r].begin(), exports_per_rank[r].end(), edge.dst_global);
            int offset = std::distance(exports_per_rank[r].begin(), it);
            int buf_idx = send_displs[r] + offset;

            h_exp_src.push_back(edge.src_local);
            h_exp_buf_idx.push_back(buf_idx);
            h_exp_weight.push_back(edge.weight);
        }

        d_export.allocate(h_exp_src.size());
        if (!h_exp_src.empty()) {
            CUDA_CHECK(cudaMemcpy(d_export.source_neuron_idx, h_exp_src.data(), h_exp_src.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_export.export_buf_idx, h_exp_buf_idx.data(), h_exp_buf_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_export.weight, h_exp_weight.data(), h_exp_weight.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        // Build Import Map
        // We need to know what we are receiving.
        // We send "DestGlobalID" to the target rank so they know how to map the buffer to their local neurons.
        // But MPI_Alltoall only sends counts. We need to exchange the actual metadata (DestGlobalIDs).
        
        std::vector<int> send_meta(total_send_size);
        for (int r = 0; r < mpi_size; ++r) {
            for (size_t i = 0; i < exports_per_rank[r].size(); ++i) {
                send_meta[send_displs[r] + i] = exports_per_rank[r][i];
            }
        }

        std::vector<int> recv_meta(total_recv_size);
        MPI_Alltoallv(send_meta.data(), send_counts.data(), send_displs.data(), MPI_INT,
                      recv_meta.data(), recv_counts.data(), recv_displs.data(), MPI_INT, MPI_COMM_WORLD);

        // Now build Import Map from recv_meta
        // recv_meta contains GlobalIDs that we own. We need to map them to LocalIDs.
        std::vector<int> h_imp_buf_idx, h_imp_dest;
        for (int i = 0; i < total_recv_size; ++i) {
            int global_id = recv_meta[i];
            int local_id = global_id - my_start;
            h_imp_buf_idx.push_back(i);
            h_imp_dest.push_back(local_id);
        }

        d_import.allocate(h_imp_buf_idx.size());
        if (!h_imp_buf_idx.empty()) {
            CUDA_CHECK(cudaMemcpy(d_import.import_buf_idx, h_imp_buf_idx.data(), h_imp_buf_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_import.dest_neuron_idx, h_imp_dest.data(), h_imp_dest.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        // 4. Allocate State & Buffers
        d_state.allocate(num_local_neurons);
        CUDA_CHECK(cudaMalloc(&d_current_production, num_local_neurons * sizeof(int)));
        
        if (total_send_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_export_buffer, total_send_size * sizeof(int)));
            CUDA_CHECK(cudaMallocHost(&h_export_buffer, total_send_size * sizeof(int)));
        }
        if (total_recv_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_import_buffer, total_recv_size * sizeof(int)));
            CUDA_CHECK(cudaMallocHost(&h_import_buffer, total_recv_size * sizeof(int)));
        }

        // 5. Initialize State
        std::vector<int> h_initial_config(num_local_neurons);
        for(int i=0; i<num_local_neurons; ++i) {
            h_initial_config[i] = local_config.neurons[my_start + i].initial_spikes;
        }
        CUDA_CHECK(cudaMemcpy(d_state.initial_config, h_initial_config.data(), num_local_neurons * sizeof(int), cudaMemcpyHostToDevice));
        
        reset();
        return true;
    }

    void reset() override {
        if (num_local_neurons == 0) return;
        int threads = BLOCK_SIZE;
        int blocks = (num_local_neurons + threads - 1) / threads;
        
        // Simple reset kernel (inline or reuse from SparseCudaSnpSimulator)
        // For brevity, I'll just use cudaMemcpy for config and cudaMemset for others
        CUDA_CHECK(cudaMemcpy(d_state.config_vector, d_state.initial_config, num_local_neurons * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_state.delay_vector, 0, num_local_neurons * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_state.spiking_vector, -1, num_local_neurons * sizeof(int))); // -1 = 0xFF
        CUDA_CHECK(cudaMemset(d_state.pending_emission, 0, num_local_neurons * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_current_production, 0, num_local_neurons * sizeof(int)));
        
        steps_executed = 0;
        total_compute_time = 0;
    }

    void step(int num_steps) override {
        int threads = BLOCK_SIZE;
        int blocks = (num_local_neurons + threads - 1) / threads;

        for (int s = 0; s < num_steps; ++s) {
            auto start = std::chrono::high_resolution_clock::now();

            // 1. Calc Spiking Vector
            k_calc_spiking_vector_mpi<<<blocks, threads>>>(
                num_local_neurons,
                d_state.config_vector,
                d_state.delay_vector,
                d_state.spiking_vector,
                d_map.rule_start_idx,
                d_map.rule_count,
                d_rv.input_threshold
            );

            // 2. Step Local & Record Production
            k_step_local_record_mpi<<<blocks, threads>>>(
                num_local_neurons,
                max_local_out_degree,
                d_state.config_vector,
                d_state.delay_vector,
                d_state.pending_emission,
                d_current_production,
                d_state.spiking_vector,
                d_sy.matrix,
                d_sy.weights,
                d_rv.spikes_consumed,
                d_rv.spikes_produced,
                d_rv.delay
            );

            // 3. Fill Export Buffer
            if (d_export.count > 0) {
                CUDA_CHECK(cudaMemset(d_export_buffer, 0, total_send_size * sizeof(int)));
                int exp_blocks = (d_export.count + threads - 1) / threads;
                k_fill_export_buffer<<<exp_blocks, threads>>>(
                    d_export.count,
                    d_export.source_neuron_idx,
                    d_export.export_buf_idx,
                    d_export.weight,
                    d_current_production,
                    d_export_buffer
                );
            }

            // 4. MPI Exchange
            if (total_send_size > 0) {
                CUDA_CHECK(cudaMemcpy(h_export_buffer, d_export_buffer, total_send_size * sizeof(int), cudaMemcpyDeviceToHost));
            }
            
            // Sync before MPI
            cudaDeviceSynchronize();

            MPI_Alltoallv(h_export_buffer, send_counts.data(), send_displs.data(), MPI_INT,
                          h_import_buffer, recv_counts.data(), recv_displs.data(), MPI_INT, MPI_COMM_WORLD);

            if (total_recv_size > 0) {
                CUDA_CHECK(cudaMemcpy(d_import_buffer, h_import_buffer, total_recv_size * sizeof(int), cudaMemcpyHostToDevice));
                
                // 5. Apply Imports
                int imp_blocks = (d_import.count + threads - 1) / threads;
                k_apply_imports<<<imp_blocks, threads>>>(
                    d_import.count,
                    d_import.import_buf_idx,
                    d_import.dest_neuron_idx,
                    d_import_buffer,
                    d_state.config_vector
                );
            }

            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            total_compute_time += std::chrono::duration<double>(end - start).count();
            steps_executed++;
        }
    }

    std::vector<int> getGlobalState() const override {
        // 1. Download Local State
        std::vector<int> local_state(num_local_neurons);
        if (num_local_neurons > 0) {
            CUDA_CHECK(cudaMemcpy(local_state.data(), d_state.config_vector, 
                       num_local_neurons * sizeof(int), cudaMemcpyDeviceToHost));
        }

        // 2. Gather at Root (Rank 0)
        // We need total size
        int total_neurons = 0;
        if (!global_neuron_counts.empty()) {
            total_neurons = std::accumulate(global_neuron_counts.begin(), global_neuron_counts.end(), 0);
        }
        
        std::vector<int> global_state(total_neurons);

        // Note: const_cast is safe here because MPI doesn't modify send buffer
        MPI_Gatherv(
            local_state.data(), num_local_neurons, MPI_INT,
            global_state.data(), const_cast<int*>(global_neuron_counts.data()), 
            const_cast<int*>(global_neuron_displs.data()), MPI_INT,
            0, MPI_COMM_WORLD
        );

        // 3. Broadcast result from root to all ranks
        MPI_Bcast(global_state.data(), total_neurons, MPI_INT, 0, MPI_COMM_WORLD);

        // 4. Restore Original Order (if permuted)
        if (!new_to_old_map.empty()) {
            std::vector<int> restored_state(total_neurons);
            for (int i = 0; i < total_neurons; ++i) {
                restored_state[new_to_old_map[i]] = global_state[i];
            }
            return restored_state;
        }
        
        return global_state;
    }

    PerformanceMetrics getPerformanceMetrics() const override {
        PerformanceMetrics metrics;
        metrics.steps_executed = steps_executed;
        metrics.total_time_ms = total_compute_time * 1000.0;
        metrics.compute_time_ms = total_compute_time * 1000.0;
        
        // MPI metrics (basic)
        metrics.mpi.rank = mpi_rank;
        metrics.mpi.world_size = mpi_size;
        
        return metrics;
    }

    std::string getPerformanceReport() const override {
        std::stringstream ss;
        ss << "SparseCudaMpiSnpSimulator Report (Rank " << mpi_rank << "):\n";
        ss << "  Steps: " << steps_executed << "\n";
        ss << "  Compute Time: " << total_compute_time * 1000.0 << " ms\n";
        return ss.str();
    }
};

std::unique_ptr<ISnpSimulator> createSparseCudaMpiSimulator(PartitionerType partitionerType) {
    auto sim = std::make_unique<SparseCudaMpiSnpSimulator>();
    
    std::unique_ptr<IPartitioner> p;
    switch (partitionerType) {
        case PartitionerType::LINEAR:
            p = std::make_unique<LinearPartitioner>();
            break;
        case PartitionerType::LOUVAIN:
            p = std::make_unique<LouvainPartitioner>();
            break;
        case PartitionerType::RED_BLUE_BFS:
            p = std::make_unique<RedBluePartitioner>();
            break;
#ifdef ENABLE_METIS
        case PartitionerType::METIS:
            p = std::make_unique<MetisPartitioner>();
            break;
#endif
        default:
            p = std::make_unique<LinearPartitioner>();
            break;
    }
    sim->setPartitioner(std::move(p));
    return sim;
}
