#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <numeric>

// --- CUDA Macros & Constants ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, -1); \
        } \
    } while(0)

constexpr int BLOCK_SIZE = 256;

// --- Device Structures (SoA) ---

// Stores state for neurons owned by this rank
struct LocalNeuronData {
    int* configuration;      // Spike counts
    int* initial_config;     // For reset
    bool* is_open;           // Status
    int* delay_timer;
    int* pending_emission;
    int* spike_production;   // Intermediate production buffer
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&configuration, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&initial_config, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&is_open, n * sizeof(bool)));
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

struct LocalRuleData {
    int* input_threshold;
    int* spikes_consumed;
    int* spikes_produced;
    int* delay;
    int* rule_start_idx;     // Index into arrays above
    int* rule_count;         // How many rules this neuron has
    int total_rules;

    void allocate(int n_neurons, int n_rules) {
        total_rules = n_rules;
        if (n_rules == 0) return;
        CUDA_CHECK(cudaMalloc(&input_threshold, n_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_consumed, n_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_produced, n_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&delay, n_rules * sizeof(int)));
        if (n_neurons > 0) {
            CUDA_CHECK(cudaMalloc(&rule_start_idx, n_neurons * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&rule_count, n_neurons * sizeof(int)));
        }
    }

    void free() {
        if (total_rules == 0) return;
        cudaFree(input_threshold); cudaFree(spikes_consumed);
        cudaFree(spikes_produced); cudaFree(delay);
        cudaFree(rule_start_idx); cudaFree(rule_count);
    }
};

// Synapses where Source and Dest are BOTH on this rank
struct LocalSynapseData {
    int* source_local_idx;
    int* dest_local_idx;
    int* weight;
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&source_local_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dest_local_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weight, n * sizeof(int)));
    }

    void free() {
        if (count == 0) return;
        cudaFree(source_local_idx); cudaFree(dest_local_idx); cudaFree(weight);
    }
};

// Synapses where Source is Local, but Dest is Remote
// We group these by Destination Rank to optimize packing
struct BoundarySynapseData {
    int* source_local_idx;   // Which local neuron fired
    int* buffer_idx;         // Slot index in the outgoing buffer for that rank
    int* weight;
    int count;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&source_local_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&buffer_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weight, n * sizeof(int)));
    }

    void free() {
        if (count == 0) return;
        cudaFree(source_local_idx); cudaFree(buffer_idx); cudaFree(weight);
    }
};

// --- Kernels ---

// 1. Update Delays & Open/Close Status
__global__ void kUpdateStatus(LocalNeuronData neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;

    if (neurons.delay_timer[idx] > 0) {
        neurons.delay_timer[idx]--;
        if (neurons.delay_timer[idx] == 0) {
            neurons.is_open[idx] = true;
        }
    }
}

// 2. Select Rules & Compute Production
__global__ void kSelectRules(LocalNeuronData neurons, LocalRuleData rules) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;
    
    // Default: no production
    neurons.spike_production[idx] = 0;

    if (!neurons.is_open[idx]) return;

    int current_spikes = neurons.configuration[idx];
    int start = rules.rule_start_idx[idx];
    int end = start + rules.rule_count[idx];

    // Deterministic selection (first applicable)
    for (int i = start; i < end; ++i) {
        if (current_spikes >= rules.input_threshold[i]) {
            int consumed = rules.spikes_consumed[i];
            int produced = rules.spikes_produced[i];
            int d = rules.delay[i];

            neurons.configuration[idx] -= consumed;

            if (d > 0) {
                neurons.is_open[idx] = false;
                neurons.delay_timer[idx] = d;
                neurons.pending_emission[idx] = produced;
            } else {
                neurons.spike_production[idx] = produced;
            }
            break; // Apply only one rule
        }
    }
}

// 3. Propagate Pending Emissions (Delayed Spikes becoming active)
// This adds pending spikes to the current production buffer so they are treated uniformly
__global__ void kActivatePending(LocalNeuronData neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;

    // If neuron just opened and had pending spikes
    if (neurons.is_open[idx] && neurons.pending_emission[idx] > 0) {
        neurons.spike_production[idx] += neurons.pending_emission[idx];
        neurons.pending_emission[idx] = 0; 
    }
}

// 4. Propagate Local Synapses (Internal)
__global__ void kPropagateLocal(LocalNeuronData neurons, LocalSynapseData synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.count) return;

    int src = synapses.source_local_idx[idx];
    int produced = neurons.spike_production[src];

    if (produced > 0) {
        int dst = synapses.dest_local_idx[idx];
        int w = synapses.weight[idx];
        
        // Only open neurons receive spikes
        if (neurons.is_open[dst]) {
            atomicAdd(&neurons.configuration[dst], produced * w);
        }
    }
}

// 5. Pack Remote Buffers (Boundary)
// Aggregates spikes for specific remote neurons into a dense buffer
__global__ void kPackRemote(LocalNeuronData neurons, BoundarySynapseData boundary, int* out_buffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundary.count) return;

    int src = boundary.source_local_idx[idx];
    int produced = neurons.spike_production[src];

    if (produced > 0) {
        int buf_idx = boundary.buffer_idx[idx]; // Pre-calculated slot for the destination neuron
        int w = boundary.weight[idx];
        // Aggregate spikes destined for the same remote neuron
        atomicAdd(&out_buffer[buf_idx], produced * w);
    }
}

// 6. Unpack/Integrate Remote Spikes
// Adds received spikes to local neurons
// 'in_buffer' contains aggregated spikes for specific local neurons (mapped by index)
__global__ void kIntegrateRemote(LocalNeuronData neurons, const int* in_buffer, const int* map_buffer_to_local, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int incoming_spikes = in_buffer[idx];
    if (incoming_spikes > 0) {
        int local_neuron_idx = map_buffer_to_local[idx];
        if (neurons.is_open[local_neuron_idx]) {
            atomicAdd(&neurons.configuration[local_neuron_idx], incoming_spikes);
        }
    }
}

// 7. Reset Kernel
__global__ void kReset(LocalNeuronData neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;
    neurons.configuration[idx] = neurons.initial_config[idx];
    neurons.is_open[idx] = true;
    neurons.delay_timer[idx] = 0;
    neurons.pending_emission[idx] = 0;
    neurons.spike_production[idx] = 0;
}


// --- Main Class Implementation ---

class CudaMpiSnpSimulator : public ISnpSimulator {
private:
    int mpi_rank;
    int mpi_size;

    // Partitioning Info
    int global_num_neurons;
    int local_neuron_start; // Inclusive
    int local_neuron_end;   // Exclusive
    int local_neuron_count;

    // CUDA Streams
    cudaStream_t compute_stream;
    cudaStream_t comm_stream;

    // Device Data
    LocalNeuronData d_neurons;
    LocalRuleData d_rules;
    LocalSynapseData d_local_synapses;

    // Communication Data
    // We maintain a separate structure for each neighbor rank we send to
    struct NeighborComm {
        int rank;
        BoundarySynapseData d_boundary_synapses; // Synapses pointing to this rank
        
        // Host buffers (pinned memory for fast transfer)
        int* h_send_buffer; 
        int* h_recv_buffer;
        
        // Device buffers
        int* d_send_buffer;
        int* d_recv_buffer;
        int* d_map_recv_to_local; // Maps recv_buffer[i] -> local_neuron_index

        int send_count; // Number of unique neurons on 'rank' we touch
        int recv_count; // Number of unique neurons on 'my_rank' that 'rank' touches
        
        MPI_Request send_req;
        MPI_Request recv_req;
    };

    std::vector<NeighborComm> neighbors;

    // Performance
    double compute_time = 0.0;
    double comm_time = 0.0;

public:
    CudaMpiSnpSimulator() {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&comm_stream));
    }

    ~CudaMpiSnpSimulator() {
        cleanup();
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(comm_stream);
    }

    bool loadSystem(const SnpSystemConfig& config) override {
        cleanup();
        global_num_neurons = config.neurons.size();

        // 1. Block Partitioning
        int neurons_per_rank = (global_num_neurons + mpi_size - 1) / mpi_size;
        local_neuron_start = mpi_rank * neurons_per_rank;
        local_neuron_end = std::min(local_neuron_start + neurons_per_rank, global_num_neurons);
        local_neuron_count = std::max(0, local_neuron_end - local_neuron_start);

        // 2. Load Local Neurons & Rules
        loadLocalNeurons(config);

        // 3. Analyze Synapses (Classify into Local-Local and Local-Remote)
        // Also builds the communication maps for receiving data
        analyzeSynapses(config);

        return true;
    }

    void step(int steps) override {
        for (int s = 0; s < steps; ++s) {
            auto t_start = MPI_Wtime();

            int threads = BLOCK_SIZE;
            int blocks = (local_neuron_count + threads - 1) / threads;

            // --- Phase 1: Local Compute (Status & Rule Selection) ---
            if (local_neuron_count > 0) {
                kUpdateStatus<<<blocks, threads, 0, compute_stream>>>(d_neurons);
                kSelectRules<<<blocks, threads, 0, compute_stream>>>(d_neurons, d_rules);
                kActivatePending<<<blocks, threads, 0, compute_stream>>>(d_neurons);
            }
            // Sync compute stream needed before packing? 
            // Yes, because Pack reads spike_production written by SelectRules.
            // But we can daisy-chain via events or just issue in order if on same stream.
            // We want Pack on comm_stream to wait for SelectRules on compute_stream.
            cudaEvent_t compute_done;
            cudaEventCreate(&compute_done);
            cudaEventRecord(compute_done, compute_stream);
            cudaStreamWaitEvent(comm_stream, compute_done, 0);

            // --- Phase 2: Communication (Stream 1) ---
            
            // 2a. Pack Outgoing Buffers on GPU
            for (auto& nb : neighbors) {
                if (nb.send_count > 0) {
                    CUDA_CHECK(cudaMemsetAsync(nb.d_send_buffer, 0, nb.send_count * sizeof(int), comm_stream));
                    int syn_blocks = (nb.d_boundary_synapses.count + threads - 1) / threads;
                    kPackRemote<<<syn_blocks, threads, 0, comm_stream>>>(
                        d_neurons, nb.d_boundary_synapses, nb.d_send_buffer
                    );
                    
                    // Copy packed buffer to Pinned Host Memory
                    CUDA_CHECK(cudaMemcpyAsync(nb.h_send_buffer, nb.d_send_buffer, 
                                             nb.send_count * sizeof(int), cudaMemcpyDeviceToHost, comm_stream));
                }
            }
            
            // 2b. Start MPI Recvs (CPU) - Do this early
            for (auto& nb : neighbors) {
                if (nb.recv_count > 0) {
                    MPI_Irecv(nb.h_recv_buffer, nb.recv_count, MPI_INT, nb.rank, 0, MPI_COMM_WORLD, &nb.recv_req);
                }
            }

            // 2c. Wait for D2H copy to finish so we can Send
            cudaStreamSynchronize(comm_stream); 

            // 2d. Start MPI Sends (CPU)
            for (auto& nb : neighbors) {
                if (nb.send_count > 0) {
                    MPI_Isend(nb.h_send_buffer, nb.send_count, MPI_INT, nb.rank, 0, MPI_COMM_WORLD, &nb.send_req);
                }
            }

            // --- Phase 3: Local Propagation (Stream 0 - Overlapped with MPI) ---
            if (d_local_synapses.count > 0) {
                int syn_blocks = (d_local_synapses.count + threads - 1) / threads;
                kPropagateLocal<<<syn_blocks, threads, 0, compute_stream>>>(d_neurons, d_local_synapses);
            }

            // --- Phase 4: Finish Communication & Integrate (Stream 1) ---
            
            // Wait for MPI Recvs
            for (auto& nb : neighbors) {
                if (nb.recv_count > 0) {
                    MPI_Wait(&nb.recv_req, MPI_STATUS_IGNORE);
                    
                    // Copy H2D
                    CUDA_CHECK(cudaMemcpyAsync(nb.d_recv_buffer, nb.h_recv_buffer, 
                                             nb.recv_count * sizeof(int), cudaMemcpyHostToDevice, comm_stream));
                    
                    // Integrate
                    int int_blocks = (nb.recv_count + threads - 1) / threads;
                    kIntegrateRemote<<<int_blocks, threads, 0, comm_stream>>>(
                        d_neurons, nb.d_recv_buffer, nb.d_map_recv_to_local, nb.recv_count
                    );
                }
                if (nb.send_count > 0) {
                    MPI_Wait(&nb.send_req, MPI_STATUS_IGNORE);
                }
            }

            // --- Phase 5: Synchronization ---
            cudaStreamSynchronize(compute_stream);
            cudaStreamSynchronize(comm_stream);
            cudaEventDestroy(compute_done);

            compute_time += (MPI_Wtime() - t_start) * 1000.0;
        }
    }

    std::vector<int> getGlobalState() const override {
        // Gather local state
        std::vector<int> local_state(local_neuron_count);
        if (local_neuron_count > 0) {
            CUDA_CHECK(cudaMemcpy(local_state.data(), d_neurons.configuration, 
                       local_neuron_count * sizeof(int), cudaMemcpyDeviceToHost));
        }

        // Gather counts for Gatherv
        std::vector<int> counts(mpi_size);
        std::vector<int> displs(mpi_size);
        MPI_Allgather(&local_neuron_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        displs[0] = 0;
        for(int i=1; i<mpi_size; i++) displs[i] = displs[i-1] + counts[i-1];

        std::vector<int> global_state(global_num_neurons);
        MPI_Allgatherv(local_state.data(), local_neuron_count, MPI_INT,
                       global_state.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        return global_state;
    }

    void reset() override {
        if (local_neuron_count > 0) {
            int blocks = (local_neuron_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kReset<<<blocks, BLOCK_SIZE, 0, compute_stream>>>(d_neurons);
            cudaStreamSynchronize(compute_stream);
        }
    }

    std::string getPerformanceReport() const override {
        double total_compute, total_comm;
        MPI_Reduce(&compute_time, &total_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            return "MPI+CUDA Simulator: Max Compute Time (ms): " + std::to_string(total_compute);
        }
        return "";
    }

private:
    void cleanup() {
        d_neurons.free();
        d_rules.free();
        d_local_synapses.free();
        for (auto& nb : neighbors) {
            nb.d_boundary_synapses.free();
            if(nb.h_send_buffer) cudaFreeHost(nb.h_send_buffer);
            if(nb.h_recv_buffer) cudaFreeHost(nb.h_recv_buffer);
            if(nb.d_send_buffer) cudaFree(nb.d_send_buffer);
            if(nb.d_recv_buffer) cudaFree(nb.d_recv_buffer);
            if(nb.d_map_recv_to_local) cudaFree(nb.d_map_recv_to_local);
        }
        neighbors.clear();
    }

    void loadLocalNeurons(const SnpSystemConfig& config) {
        d_neurons.allocate(local_neuron_count);

        if (local_neuron_count == 0) return;

        std::vector<int> h_config(local_neuron_count);
        std::vector<char> h_open(local_neuron_count, true); // Is boolean, but use char for cudaMemcpy
        std::vector<int> h_zeros(local_neuron_count, 0);

        // Rules host buffers
        std::vector<int> h_thresh, h_cons, h_prod, h_delay;
        std::vector<int> h_r_start(local_neuron_count), h_r_count(local_neuron_count);
        int total_rules = 0;

        for (int i = 0; i < local_neuron_count; ++i) {
            int global_id = local_neuron_start + i;
            const auto& n = config.neurons[global_id];
            
            h_config[i] = n.initial_spikes;
            h_r_start[i] = total_rules;
            h_r_count[i] = n.rules.size();
            
            for (const auto& r : n.rules) {
                h_thresh.push_back(r.input_threshold);
                h_cons.push_back(r.spikes_consumed);
                h_prod.push_back(r.spikes_produced);
                h_delay.push_back(r.delay);
                total_rules++;
            }
        }

        // Upload Neurons
        CUDA_CHECK(cudaMemcpy(d_neurons.configuration, h_config.data(), local_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.initial_config, h_config.data(), local_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.is_open, h_open.data(), local_neuron_count * sizeof(bool), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_neurons.delay_timer, 0, local_neuron_count * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_neurons.pending_emission, 0, local_neuron_count * sizeof(int)));

        // Upload Rules
        d_rules.allocate(local_neuron_count, total_rules);
        if (total_rules > 0) {
            CUDA_CHECK(cudaMemcpy(d_rules.input_threshold, h_thresh.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_consumed, h_cons.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_produced, h_prod.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.delay, h_delay.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.rule_start_idx, h_r_start.data(), local_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.rule_count, h_r_count.data(), local_neuron_count * sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    void analyzeSynapses(const SnpSystemConfig& config) {
        // 1. Identify owner of each neuron (simple calc due to block partition)
        auto get_owner = [&](int nid) {
            int n_per_rank = (global_num_neurons + mpi_size - 1) / mpi_size;
            return nid / n_per_rank;
        };
        auto get_local_idx = [&](int nid, int rank) {
            int n_per_rank = (global_num_neurons + mpi_size - 1) / mpi_size;
            return nid - (rank * n_per_rank);
        };

        // Temporary storage
        std::vector<int> loc_src, loc_dst, loc_w;
        std::map<int, std::vector<std::tuple<int, int, int>>> remote_sends; // Rank -> {src_local, dest_global, weight}
        std::map<int, std::vector<int>> remote_recvs; // Rank -> {dest_local_on_my_node}

        // Pass 1: Categorize Outgoing Synapses
        for (const auto& syn : config.synapses) {
            int src_owner = get_owner(syn.source_id);
            int dst_owner = get_owner(syn.dest_id);

            // If I own the source
            if (src_owner == mpi_rank) {
                int src_local = get_local_idx(syn.source_id, mpi_rank);
                
                if (dst_owner == mpi_rank) {
                    // Local-Local
                    loc_src.push_back(src_local);
                    loc_dst.push_back(get_local_idx(syn.dest_id, mpi_rank));
                    loc_w.push_back(syn.weight);
                } else {
                    // Local-Remote
                    remote_sends[dst_owner].emplace_back(src_local, syn.dest_id, syn.weight);
                }
            }
            
            // If I own the dest (Recv calculation)
            if (dst_owner == mpi_rank && src_owner != mpi_rank) {
                // Incoming from remote
                // We just need to know which of our local neurons is the target
                // to map the incoming buffer.
                // NOTE: We need a canonical ordering so Sender and Receiver agree on buffer layout.
                // We sort by (Source Global ID, Dest Global ID). 
                // Since we iterate synapses in order, let's assume config.synapses is sorted or we sort.
            }
        }

        // Upload Local-Local Synapses
        d_local_synapses.allocate(loc_src.size());
        if (!loc_src.empty()) {
            CUDA_CHECK(cudaMemcpy(d_local_synapses.source_local_idx, loc_src.data(), loc_src.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_synapses.dest_local_idx, loc_dst.data(), loc_dst.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_synapses.weight, loc_w.data(), loc_w.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        // Process Neighbors (Both Send and Recv)
        // We need to agree on buffer layout with neighbors.
        // Protocol: For Pair (Rank A, Rank B), buffer contains spikes for Unique Dst Neurons on B.
        // Sorted by Dest Neuron ID.
        
        for (int other_rank = 0; other_rank < mpi_size; ++other_rank) {
            if (other_rank == mpi_rank) continue;

            // 1. Calculate Send Map (What I send to Other)
            // Identify unique (DestID) on OtherRank that I touch.
            std::map<int, int> dest_global_to_buffer_idx;
            int buffer_idx_counter = 0;

            // Sort sends to ensure deterministic buffer layout
            auto& sends = remote_sends[other_rank];
            // To match receiver, we must process unique DestIDs in increasing order
            std::map<int, std::vector<std::pair<int, int>>> grouped_by_dest; // DestGlobal -> {SrcLocal, Weight}
            for(auto& t : sends) {
                grouped_by_dest[std::get<1>(t)].push_back({std::get<0>(t), std::get<2>(t)});
            }

            // Flatten for GPU and assign buffer indices
            std::vector<int> h_bound_src, h_bound_buf, h_bound_w;
            for(auto& kv : grouped_by_dest) {
                int dest_global = kv.first;
                dest_global_to_buffer_idx[dest_global] = buffer_idx_counter++;
                for(auto& pair : kv.second) {
                    h_bound_src.push_back(pair.first); // Src Local
                    h_bound_buf.push_back(dest_global_to_buffer_idx[dest_global]);
                    h_bound_w.push_back(pair.second);
                }
            }

            // 2. Calculate Recv Map (What Other sends to Me)
            // I need to know which of my neurons Other is targeting.
            // I iterate synapses where Src=Other, Dest=Me.
            std::map<int, bool> my_neurons_targeted_by_other;
            for(const auto& syn : config.synapses) {
                if (get_owner(syn.source_id) == other_rank && get_owner(syn.dest_id) == mpi_rank) {
                    my_neurons_targeted_by_other[syn.dest_id] = true;
                }
            }
            
            // Map buffer index 0..N to local neuron index
            std::vector<int> h_map_recv;
            for(auto const& [dest_global, _] : my_neurons_targeted_by_other) {
                h_map_recv.push_back(get_local_idx(dest_global, mpi_rank));
            }

            // Only add neighbor if there is traffic
            if (buffer_idx_counter > 0 || !h_map_recv.empty()) {
                NeighborComm nb;
                nb.rank = other_rank;
                nb.send_count = buffer_idx_counter;
                nb.recv_count = h_map_recv.size();
                
                // Allocate Send Structures
                nb.d_boundary_synapses.allocate(h_bound_src.size());
                if (nb.send_count > 0) {
                    CUDA_CHECK(cudaMemcpy(nb.d_boundary_synapses.source_local_idx, h_bound_src.data(), h_bound_src.size()*sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(nb.d_boundary_synapses.buffer_idx, h_bound_buf.data(), h_bound_buf.size()*sizeof(int), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(nb.d_boundary_synapses.weight, h_bound_w.data(), h_bound_w.size()*sizeof(int), cudaMemcpyHostToDevice));
                    
                    CUDA_CHECK(cudaHostAlloc(&nb.h_send_buffer, nb.send_count * sizeof(int), cudaHostAllocDefault));
                    CUDA_CHECK(cudaMalloc(&nb.d_send_buffer, nb.send_count * sizeof(int)));
                } else {
                    nb.h_send_buffer = nullptr; nb.d_send_buffer = nullptr;
                }

                // Allocate Recv Structures
                if (nb.recv_count > 0) {
                    CUDA_CHECK(cudaHostAlloc(&nb.h_recv_buffer, nb.recv_count * sizeof(int), cudaHostAllocDefault));
                    CUDA_CHECK(cudaMalloc(&nb.d_recv_buffer, nb.recv_count * sizeof(int)));
                    CUDA_CHECK(cudaMalloc(&nb.d_map_recv_to_local, nb.recv_count * sizeof(int)));
                    CUDA_CHECK(cudaMemcpy(nb.d_map_recv_to_local, h_map_recv.data(), nb.recv_count * sizeof(int), cudaMemcpyHostToDevice));
                } else {
                    nb.h_recv_buffer = nullptr; nb.d_recv_buffer = nullptr; nb.d_map_recv_to_local = nullptr;
                }

                neighbors.push_back(nb);
            }
        }
    }
};

// Factory Function
std::unique_ptr<ISnpSimulator> createCudaMpiSimulator() {
    return std::make_unique<CudaMpiSnpSimulator>();
}