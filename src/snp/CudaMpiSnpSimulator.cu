#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <iostream>
#include <sstream>
#include <numeric>
#include <cstring>
#include <chrono>

// --- Error Handling Macros ---

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

#define MPI_CHECK(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

constexpr int BLOCK_SIZE = 64;

namespace {

// --- Device Structures ---

// Local Neuron State (SoA)
struct DeviceNeuronData {
    int* configuration;       // Current spike count
    int* initial_config;      // For reset
    char* is_open;            // Status (using char for byte alignment)
    int* delay_timer;         // Timer
    int* pending_emission;    // Spikes waiting for delay to expire
    int* spike_production;    // Immediate production from rules
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&configuration, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&initial_config, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&is_open, n * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&delay_timer, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&pending_emission, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spike_production, n * sizeof(int)));
    }

    void free() {
        if (count == 0) return;
        cudaFree(configuration); cudaFree(initial_config);
        cudaFree(is_open); cudaFree(delay_timer);
        cudaFree(pending_emission); cudaFree(spike_production);
    }
};

// Rule Data (SoA)
struct DeviceRuleData {
    int* neuron_id = nullptr;
    int* input_threshold = nullptr;
    int* spikes_consumed = nullptr;
    int* spikes_produced = nullptr;
    int* delay = nullptr;
    int* rule_start_idx = nullptr;      
    int* rule_count = nullptr;          
    int count = 0;       // Total rules
    int n_neurons = 0;   // Track neurons for freeing

    void allocate(int total_rules, int num_neurons) {
        count = total_rules;
        n_neurons = num_neurons; // Store this for free()

        // 1. Allocate Rule Arrays (Only if rules exist)
        if (total_rules > 0) {
            CUDA_CHECK(cudaMalloc(&neuron_id, total_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&input_threshold, total_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&spikes_consumed, total_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&spikes_produced, total_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&delay, total_rules * sizeof(int)));
        }

        // 2. Allocate Neuron Maps (ALWAYS if neurons exist, even if rules don't)
        if (num_neurons > 0) {
            CUDA_CHECK(cudaMalloc(&rule_start_idx, num_neurons * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&rule_count, num_neurons * sizeof(int)));
        }
    }

    void free() {
        // cudaFree is safe on nullptr, so we can remove the 'if count==0' checks
        // that were preventing cleanup of partial allocations.
        cudaFree(neuron_id);
        cudaFree(input_threshold);
        cudaFree(spikes_consumed);
        cudaFree(spikes_produced);
        cudaFree(delay);
        cudaFree(rule_start_idx);
        cudaFree(rule_count);
        
        // Reset pointers to prevent double-free
        neuron_id = nullptr; input_threshold = nullptr; 
        spikes_consumed = nullptr; spikes_produced = nullptr; 
        delay = nullptr; rule_start_idx = nullptr; rule_count = nullptr;
    }
};
// Synapse Data (SoA) - Purely Local
struct DeviceLocalSynapseData {
    int* source_idx; // Local Index
    int* dest_idx;   // Local Index
    int* weight;
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&source_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dest_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weight, n * sizeof(int)));
    }

    void free() {
        if (count == 0) return;
        cudaFree(source_idx); cudaFree(dest_idx); cudaFree(weight);
    }
};

// Outgoing Synapse Data (SoA) - Local Source -> Export Buffer
struct DeviceExportSynapseData {
    int* source_idx;      // Local Index
    int* export_buf_idx;  // Index in the contiguous export buffer
    int* weight;
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&source_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&export_buf_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weight, n * sizeof(int)));
    }

    void free() {
        if (count == 0) return;
        cudaFree(source_idx); cudaFree(export_buf_idx); cudaFree(weight);
    }
};

// Import Mapping (SoA) - Import Buffer -> Local Dest
struct DeviceImportMapData {
    int* import_buf_idx; // Index in the contiguous import buffer
    int* dest_idx;       // Local Index
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&import_buf_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dest_idx, n * sizeof(int)));
    }

    void free() {
        if (count == 0) return;
        cudaFree(import_buf_idx); cudaFree(dest_idx);
    }
};

// --- CUDA Kernels ---

// 1. Update delays and open neurons
__global__ void kUpdateStatus(DeviceNeuronData neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;

    if (neurons.delay_timer[idx] > 0) {
        neurons.delay_timer[idx]--;
        if (neurons.delay_timer[idx] == 0) {
            neurons.is_open[idx] = 1;
        }
    }
}

// 2. Select and Fire Rules (Same logic as single-node, adapted for SoA)
__global__ void kSelectAndFire(DeviceNeuronData neurons, DeviceRuleData rules) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;

    if (!neurons.is_open[idx]) return;

    int current_spikes = neurons.configuration[idx];
    int r_start = rules.rule_start_idx[idx];
    int r_count = rules.rule_count[idx];

    // Linear search for first applicable rule (Deterministic)
    for (int i = 0; i < r_count; ++i) {
        int r_idx = r_start + i;
        if (current_spikes >= rules.input_threshold[r_idx]) {
            // Apply Rule
            neurons.configuration[idx] -= rules.spikes_consumed[r_idx];
            int produced = rules.spikes_produced[r_idx];
            int delay = rules.delay[r_idx];

            if (delay > 0) {
                neurons.is_open[idx] = 0;
                neurons.delay_timer[idx] = delay;
                neurons.pending_emission[idx] = produced;
            } else {
                neurons.spike_production[idx] = produced;
            }
            break; // Only one rule fires per step
        }
    }
}

// 3. Propagate Spikes (Local Only)
// Handles both immediate production and pending emissions that just unlocked
__global__ void kPropagateLocal(DeviceNeuronData neurons, DeviceLocalSynapseData synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.count) return;

    int src = synapses.source_idx[idx];
    int dst = synapses.dest_idx[idx];
    int w = synapses.weight[idx];

    int spikes_to_send = 0;

    // Case A: Immediate production
    spikes_to_send += neurons.spike_production[src] * w;

    // Case B: Pending emission (only if neuron is open and has pending)
    // Note: kUpdateStatus runs before this, so if delay hit 0, is_open is true
    if (neurons.is_open[src] && neurons.pending_emission[src] > 0) {
        spikes_to_send += neurons.pending_emission[src] * w;
    }

    if (spikes_to_send > 0 && neurons.is_open[dst]) {
        atomicAdd(&neurons.configuration[dst], spikes_to_send);
    }
}

// 4. Populate Export Buffer (For Remote Targets)
__global__ void kPopulateExport(DeviceNeuronData neurons, DeviceExportSynapseData synapses, int* export_buffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.count) return;

    int src = synapses.source_idx[idx];
    int buf_idx = synapses.export_buf_idx[idx];
    int w = synapses.weight[idx];

    int spikes_to_send = 0;
    spikes_to_send += neurons.spike_production[src] * w;

    if (neurons.is_open[src] && neurons.pending_emission[src] > 0) {
        spikes_to_send += neurons.pending_emission[src] * w;
    }

    if (spikes_to_send > 0) {
        atomicAdd(&export_buffer[buf_idx], spikes_to_send);
    }
}

// 5. Apply Imported Spikes (From Other Ranks)
__global__ void kApplyImports(DeviceNeuronData neurons, DeviceImportMapData imports, int* import_buffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= imports.count) return;

    int buf_idx = imports.import_buf_idx[idx];
    int dst = imports.dest_idx[idx];
    
    int spikes = import_buffer[buf_idx];

    if (spikes > 0 && neurons.is_open[dst]) {
        // Atomic because multiple remote nodes might target the same local neuron
        // (Though the map is usually 1-to-1 per import buffer slot, this is safer)
        atomicAdd(&neurons.configuration[dst], spikes);
    }
}

// 6. Cleanup (Clear production and pending emissions)
__global__ void kCleanup(DeviceNeuronData neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;

    neurons.spike_production[idx] = 0;
    // Only clear pending if it was emitted (neuron is open)
    if (neurons.is_open[idx]) {
        neurons.pending_emission[idx] = 0;
    }
}

__global__ void kResetNeurons(DeviceNeuronData neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;
    neurons.configuration[idx] = neurons.initial_config[idx];
    neurons.is_open[idx] = 1;
    neurons.delay_timer[idx] = 0;
    neurons.pending_emission[idx] = 0;
    neurons.spike_production[idx] = 0;
}

} // namespace


// --- Main Implementation Class ---

class CudaMpiSnpSimulator : public ISnpSimulator {
private:
    int mpi_rank, mpi_size;
    
    // Partitioning
    int global_num_neurons;
    int local_start_idx;
    int local_end_idx;
    int local_num_neurons;

    // Device Data
    DeviceNeuronData d_neurons;
    DeviceRuleData d_rules;
    DeviceLocalSynapseData d_local_synapses;
    DeviceExportSynapseData d_export_synapses;
    DeviceImportMapData d_import_map;

    // Communication Buffers (MPI + Device)
    // We organize buffers by remote rank.
    struct RankCommData {
        // Host Buffers
        std::vector<int> h_send_buf;
        std::vector<int> h_recv_buf;
        // Offsets in the monolithic Device Export/Import buffers
        int export_offset;
        int export_count;
        int import_offset;
        int import_count;
    };
    std::vector<RankCommData> comm_map; // Index = Rank ID

    int* d_export_buffer = nullptr;
    int* d_import_buffer = nullptr;
    int total_export_size = 0;
    int total_import_size = 0;

    // Performance Metrics
    double compute_time = 0.0;
    double comm_time = 0.0;
    int steps = 0;

public:
    CudaMpiSnpSimulator() {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }

    ~CudaMpiSnpSimulator() {
        d_neurons.free();
        d_rules.free();
        d_local_synapses.free();
        d_export_synapses.free();
        d_import_map.free();
        if (d_export_buffer) cudaFree(d_export_buffer);
        if (d_import_buffer) cudaFree(d_import_buffer);
    }

    bool loadSystem(const SnpSystemConfig& config) override {
        global_num_neurons = config.neurons.size();

        // 1. Calculate Partitioning (Block Distribution)
        int base = global_num_neurons / mpi_size;
        int rem = global_num_neurons % mpi_size;
        
        if (mpi_rank < rem) {
            local_num_neurons = base + 1;
            local_start_idx = mpi_rank * local_num_neurons;
        } else {
            local_num_neurons = base;
            local_start_idx = rem * (base + 1) + (mpi_rank - rem) * base;
        }
        local_end_idx = local_start_idx + local_num_neurons;

        // 2. Prepare Local Neurons
        d_neurons.allocate(local_num_neurons);
        std::vector<int> h_initial(local_num_neurons);
        std::vector<char> h_open(local_num_neurons, 1);
        std::vector<int> h_zeros(local_num_neurons, 0);

        for (int i = 0; i < local_num_neurons; ++i) {
            h_initial[i] = config.neurons[local_start_idx + i].initial_spikes;
        }

        CUDA_CHECK(cudaMemcpy(d_neurons.configuration, h_initial.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.initial_config, h_initial.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.is_open, h_open.data(), local_num_neurons * sizeof(char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.delay_timer, h_zeros.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.pending_emission, h_zeros.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.spike_production, h_zeros.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));

        // 3. Upload Rules (Flattened)
        prepareRules(config);

        // 4. Analyze Topology & Prepare Synapses (The Complex Part)
        prepareTopology(config);

        return true;
    }

    void step(int num_steps = 1) override {
        int gridSize = (local_num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int synapseGridSize = (d_local_synapses.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int exportGridSize = (d_export_synapses.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int importGridSize = (d_import_map.count + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Ensure grid sizes are at least 1 to avoid launch errors
        gridSize = std::max(1, gridSize);
        synapseGridSize = std::max(1, synapseGridSize);
        exportGridSize = std::max(1, exportGridSize);
        importGridSize = std::max(1, importGridSize);

        for (int s = 0; s < num_steps; ++s) {
            auto t1 = std::chrono::high_resolution_clock::now();

            // --- Phase 1: Compute (Device) ---
            if (local_num_neurons > 0) {
                // Clear export buffer before accumulating
                if (total_export_size > 0) {
                    CUDA_CHECK(cudaMemset(d_export_buffer, 0, total_export_size * sizeof(int)));
                }

                kUpdateStatus<<<gridSize, BLOCK_SIZE>>>(d_neurons);
                kSelectAndFire<<<gridSize, BLOCK_SIZE>>>(d_neurons, d_rules);
                
                if (d_local_synapses.count > 0) {
                    kPropagateLocal<<<synapseGridSize, BLOCK_SIZE>>>(d_neurons, d_local_synapses);
                }
                
                if (d_export_synapses.count > 0) {
                    kPopulateExport<<<exportGridSize, BLOCK_SIZE>>>(d_neurons, d_export_synapses, d_export_buffer);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            auto t2 = std::chrono::high_resolution_clock::now();
            compute_time += std::chrono::duration<double, std::milli>(t2 - t1).count();

            // --- Phase 2: Communication (Hybrid) ---
            // 1. Download Export Buffers
            if (total_export_size > 0) {
                // In a production system, we would use pinned memory or CUDA-aware MPI. 
                // For robustness here, we copy to host vectors first.
                // Note: We copy the whole monolithic buffer, then scatter to MPI buffers
                std::vector<int> h_all_exports(total_export_size);
                CUDA_CHECK(cudaMemcpy(h_all_exports.data(), d_export_buffer, total_export_size * sizeof(int), cudaMemcpyDeviceToHost));
                
                // Pack into specific send buffers
                for (int r = 0; r < mpi_size; ++r) {
                    if (r == mpi_rank) continue;
                    if (comm_map[r].export_count > 0) {
                        std::memcpy(comm_map[r].h_send_buf.data(), 
                                    &h_all_exports[comm_map[r].export_offset], 
                                    comm_map[r].export_count * sizeof(int));
                    }
                }
            }

            // 2. MPI Exchange
            std::vector<MPI_Request> requests;
            for (int r = 0; r < mpi_size; ++r) {
                if (r == mpi_rank) continue;
                
                // Send
                if (comm_map[r].export_count > 0) {
                    MPI_Request req;
                    MPI_Isend(comm_map[r].h_send_buf.data(), comm_map[r].export_count, MPI_INT, r, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }

                // Recv
                if (comm_map[r].import_count > 0) {
                    MPI_Request req;
                    MPI_Irecv(comm_map[r].h_recv_buf.data(), comm_map[r].import_count, MPI_INT, r, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
            }
            
            if (!requests.empty()) {
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            }

            // 3. Upload Import Buffers
            if (total_import_size > 0) {
                std::vector<int> h_all_imports(total_import_size);
                
                // Gather from MPI buffers
                for (int r = 0; r < mpi_size; ++r) {
                    if (r == mpi_rank) continue;
                    if (comm_map[r].import_count > 0) {
                        std::memcpy(&h_all_imports[comm_map[r].import_offset], 
                                    comm_map[r].h_recv_buf.data(), 
                                    comm_map[r].import_count * sizeof(int));
                    }
                }
                
                CUDA_CHECK(cudaMemcpy(d_import_buffer, h_all_imports.data(), total_import_size * sizeof(int), cudaMemcpyHostToDevice));
            }

            auto t3 = std::chrono::high_resolution_clock::now();
            comm_time += std::chrono::duration<double, std::milli>(t3 - t2).count();

            // --- Phase 3: Apply Imports & Cleanup (Device) ---
            if (local_num_neurons > 0) {
                if (d_import_map.count > 0) {
                    kApplyImports<<<importGridSize, BLOCK_SIZE>>>(d_neurons, d_import_map, d_import_buffer);
                }
                kCleanup<<<gridSize, BLOCK_SIZE>>>(d_neurons);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            auto t4 = std::chrono::high_resolution_clock::now();
            compute_time += std::chrono::duration<double, std::milli>(t4 - t3).count();
            
            steps++;
        }
    }

    std::vector<int> getGlobalState() const override {
        // 1. Retrieve local state
        std::vector<int> local_state(local_num_neurons);
        if (local_num_neurons > 0) {
            CUDA_CHECK(cudaMemcpy(local_state.data(), d_neurons.configuration, local_num_neurons * sizeof(int), cudaMemcpyDeviceToHost));
        }

        // 2. Gather sizes for Allgatherv
        std::vector<int> recv_counts(mpi_size);
        int local_n = local_num_neurons;
        MPI_Allgather(&local_n, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // 3. Calculate displacements
        std::vector<int> displs(mpi_size);
        displs[0] = 0;
        for (int i = 1; i < mpi_size; ++i) {
            displs[i] = displs[i-1] + recv_counts[i-1];
        }
        int total_neurons = displs.back() + recv_counts.back();

        // 4. Gather Data
        std::vector<int> global_state(total_neurons);
        MPI_Allgatherv(local_state.data(), local_n, MPI_INT, 
                       global_state.data(), recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        return global_state;
    }

    void reset() override {
        int gridSize = (local_num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (gridSize > 0) {
            kResetNeurons<<<gridSize, BLOCK_SIZE>>>(d_neurons);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        compute_time = 0;
        comm_time = 0;
        steps = 0;
    }

    std::string getPerformanceReport() const override {
        std::ostringstream ss;
        ss << "=== MPI+CUDA Rank " << mpi_rank << " Report ===\n";
        ss << "Neurons Owned: " << local_num_neurons << "\n";
        ss << "Compute Time: " << compute_time << " ms\n";
        ss << "Comm Time:    " << comm_time << " ms\n";
        ss << "Total Steps:  " << steps << "\n";
        return ss.str();
    }

private:
void prepareRules(const SnpSystemConfig& config) {
        std::vector<int> h_nid, h_thresh, h_cons, h_prod, h_delay;
        std::vector<int> h_start(local_num_neurons, 0); // Initialize to 0
        std::vector<int> h_count(local_num_neurons, 0); // Initialize to 0

        int rule_cursor = 0;
        for (int i = 0; i < local_num_neurons; ++i) {
            int global_id = local_start_idx + i;
            const auto& neuron = config.neurons[global_id];
            
            h_start[i] = rule_cursor;
            h_count[i] = neuron.rules.size();
            
            for (const auto& r : neuron.rules) {
                h_nid.push_back(i); // Local ID
                h_thresh.push_back(r.input_threshold);
                h_cons.push_back(r.spikes_consumed);
                h_prod.push_back(r.spikes_produced);
                h_delay.push_back(r.delay);
                rule_cursor++;
            }
        }

        // Allocate using the fixed logic
        d_rules.allocate(rule_cursor, local_num_neurons);

        // Upload Rule Arrays (Only if rules exist)
        if (rule_cursor > 0) {
            CUDA_CHECK(cudaMemcpy(d_rules.neuron_id, h_nid.data(), rule_cursor * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.input_threshold, h_thresh.data(), rule_cursor * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_consumed, h_cons.data(), rule_cursor * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_produced, h_prod.data(), rule_cursor * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.delay, h_delay.data(), rule_cursor * sizeof(int), cudaMemcpyHostToDevice));
        }

        // Upload Neuron Maps (ALWAYS if neurons exist)
        if (local_num_neurons > 0) {
            CUDA_CHECK(cudaMemcpy(d_rules.rule_start_idx, h_start.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.rule_count, h_count.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    void prepareTopology(const SnpSystemConfig& config) {
        // --- 1. Identify Ranges for all ranks ---
        std::vector<std::pair<int, int>> rank_ranges(mpi_size);
        int base = global_num_neurons / mpi_size;
        int rem = global_num_neurons % mpi_size;
        for (int r = 0; r < mpi_size; ++r) {
            int n = (r < rem) ? base + 1 : base;
            int start = (r < rem) ? r * n : rem * (base + 1) + (r - rem) * base;
            rank_ranges[r] = {start, start + n};
        }

        // --- 2. Classify Synapses ---
        std::vector<int> loc_src, loc_dst, loc_w;
        std::map<int, std::vector<std::pair<int, int>>> export_map; // TargetRank -> vector<{source_local_id, global_dest_id * weight}>
        // Note: For export map, to handle weights correctly in the buffer logic, 
        // we actually need to aggregate (spikes * weight) at the destination.
        // HOWEVER, to keep buffer simple (just spike counts), we should apply weight at SENDER if possible.
        // BUT weight is per synapse. If N1->N2(w=2) and N3->N2(w=1), we can't sum spikes from N1 and N3 simply.
        // SOLUTION: The Export Buffer represents "Spikes Arriving at Unique Destination D from Local Rank".
        // The sender accumulates `production * weight` into the buffer slot for D.
        
        // We need to know the Unique Remote Destinations per rank to build the fixed buffers.
        std::map<int, std::set<int>> remote_targets_per_rank; // Rank -> Set of Global Dest IDs

        for (const auto& syn : config.synapses) {
            bool src_is_local = (syn.source_id >= local_start_idx && syn.source_id < local_end_idx);
            
            if (src_is_local) {
                bool dst_is_local = (syn.dest_id >= local_start_idx && syn.dest_id < local_end_idx);
                int local_src = syn.source_id - local_start_idx;
                
                if (dst_is_local) {
                    loc_src.push_back(local_src);
                    loc_dst.push_back(syn.dest_id - local_start_idx);
                    loc_w.push_back(syn.weight);
                } else {
                    // Find target rank
                    int target_rank = -1;
                    for(int r=0; r<mpi_size; ++r) {
                        if (syn.dest_id >= rank_ranges[r].first && syn.dest_id < rank_ranges[r].second) {
                            target_rank = r;
                            break;
                        }
                    }
                    if (target_rank != -1) {
                        remote_targets_per_rank[target_rank].insert(syn.dest_id);
                    }
                }
            }
        }

        // --- 3. Build Communication Maps (The Handshake) ---
        // We need to agree on the order of neurons in the buffers.
        // We use the natural sort order of Global IDs.
        
        comm_map.resize(mpi_size);
        
        // Setup Export Buffers (What I send)
        std::vector<int> exp_src, exp_buf_idx, exp_w;
        int current_export_offset = 0;

        for (int r = 0; r < mpi_size; ++r) {
            if (r == mpi_rank) continue;

            // My targets in Rank R
            std::vector<int> targets(remote_targets_per_rank[r].begin(), remote_targets_per_rank[r].end());
            
            comm_map[r].export_offset = current_export_offset;
            comm_map[r].export_count = targets.size();
            comm_map[r].h_send_buf.resize(targets.size());
            
            // Map global dest ID -> Index in this rank's specific export chunk
            std::map<int, int> dest_to_chunk_idx;
            for(size_t i=0; i<targets.size(); ++i) {
                dest_to_chunk_idx[targets[i]] = current_export_offset + i;
            }
            current_export_offset += targets.size();

            // Re-scan synapses to build the GPU Export Synapse list
            for (const auto& syn : config.synapses) {
                if (syn.source_id >= local_start_idx && syn.source_id < local_end_idx) { // I am source
                    if (dest_to_chunk_idx.count(syn.dest_id)) {
                        exp_src.push_back(syn.source_id - local_start_idx);
                        exp_buf_idx.push_back(dest_to_chunk_idx[syn.dest_id]);
                        exp_w.push_back(syn.weight);
                    }
                }
            }
        }
        total_export_size = current_export_offset;

        // Setup Import Buffers (What I receive)
        // I need to know what Rank R is sending me.
        // Rank R sends me data for neurons IT touches in MY range.
        // This requires "Global Knowledge" of the config, which we have.
        
        std::vector<int> imp_buf_idx, imp_dst;
        int current_import_offset = 0;

        for (int r = 0; r < mpi_size; ++r) {
            if (r == mpi_rank) continue;

            // Find unique neurons in ME that Rank R touches
            std::set<int> incoming_targets;
            for (const auto& syn : config.synapses) {
                // If source is in Rank R AND dest is in ME
                if (syn.source_id >= rank_ranges[r].first && syn.source_id < rank_ranges[r].second) {
                    if (syn.dest_id >= local_start_idx && syn.dest_id < local_end_idx) {
                        incoming_targets.insert(syn.dest_id);
                    }
                }
            }

            std::vector<int> targets(incoming_targets.begin(), incoming_targets.end());
            
            comm_map[r].import_offset = current_import_offset;
            comm_map[r].import_count = targets.size();
            comm_map[r].h_recv_buf.resize(targets.size());

            // Map buffer index -> Local Neuron
            for (size_t i = 0; i < targets.size(); ++i) {
                imp_buf_idx.push_back(current_import_offset + i);
                imp_dst.push_back(targets[i] - local_start_idx);
            }
            current_import_offset += targets.size();
        }
        total_import_size = current_import_offset;

        // --- 4. Allocate and Upload to GPU ---
        
        // Local Synapses
        d_local_synapses.allocate(loc_src.size());
        if (!loc_src.empty()) {
            CUDA_CHECK(cudaMemcpy(d_local_synapses.source_idx, loc_src.data(), loc_src.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_synapses.dest_idx, loc_dst.data(), loc_dst.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_synapses.weight, loc_w.data(), loc_w.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        // Export Synapses
        d_export_synapses.allocate(exp_src.size());
        if (!exp_src.empty()) {
            CUDA_CHECK(cudaMemcpy(d_export_synapses.source_idx, exp_src.data(), exp_src.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_export_synapses.export_buf_idx, exp_buf_idx.data(), exp_buf_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_export_synapses.weight, exp_w.data(), exp_w.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        // Import Map
        d_import_map.allocate(imp_buf_idx.size());
        if (!imp_buf_idx.empty()) {
            CUDA_CHECK(cudaMemcpy(d_import_map.import_buf_idx, imp_buf_idx.data(), imp_buf_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_import_map.dest_idx, imp_dst.data(), imp_dst.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        // Monolithic Buffers
        if (total_export_size > 0) CUDA_CHECK(cudaMalloc(&d_export_buffer, total_export_size * sizeof(int)));
        if (total_import_size > 0) CUDA_CHECK(cudaMalloc(&d_import_buffer, total_import_size * sizeof(int)));
    }
};

std::unique_ptr<ISnpSimulator> createCudaMpiSimulator() {
    return std::make_unique<CudaMpiSnpSimulator>();
}