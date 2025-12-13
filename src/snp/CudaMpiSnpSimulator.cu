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
#include <memory>
#include <cstdio>

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

constexpr int BLOCK_SIZE = 128;

namespace {

// --- Helper: Pinned Memory Wrapper ---
template <typename T>
struct PinnedBuffer {
    T* data = nullptr;
    size_t count = 0;

    void allocate(size_t n) {
        if (data) free();
        count = n;
        if (n > 0) {
            CUDA_CHECK(cudaMallocHost((void**)&data, n * sizeof(T)));
        }
    }

    void free() {
        if (data) {
            CUDA_CHECK(cudaFreeHost(data));
            data = nullptr;
        }
        count = 0;
    }

    ~PinnedBuffer() { free(); }
    T* get() const { return data; }
};

// --- Device Structures (FIXED: Initialized to nullptr) ---

struct DeviceNeuronData {
    int* configuration = nullptr;       
    int* initial_config = nullptr;      
    char* is_open = nullptr;            
    int* delay_timer = nullptr;         
    int* pending_emission = nullptr;    
    int* spike_production = nullptr;    
    int count = 0;

    void allocate(int n) {
        // Always reset count first
        count = n;
        if (n == 0) return; // Returns, but pointers are now safely nullptr
        CUDA_CHECK(cudaMalloc(&configuration, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&initial_config, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&is_open, n * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&delay_timer, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&pending_emission, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spike_production, n * sizeof(int)));
    }

    void free() {
        // Safe to call cudaFree on nullptr
        cudaFree(configuration); cudaFree(initial_config);
        cudaFree(is_open); cudaFree(delay_timer);
        cudaFree(pending_emission); cudaFree(spike_production);
        // Reset pointers to prevent double-free logic issues
        configuration = nullptr; initial_config = nullptr;
        is_open = nullptr; delay_timer = nullptr;
        pending_emission = nullptr; spike_production = nullptr;
        count = 0;
    }
};

struct DeviceRuleData {
    int* input_threshold = nullptr;
    int* spikes_consumed = nullptr;
    int* spikes_produced = nullptr;
    int* delay = nullptr;
    int* rule_start_idx = nullptr;      
    int* rule_count = nullptr;          
    int count = 0;       

    void allocate(int total_rules, int num_neurons) {
        count = total_rules;
        if (total_rules > 0) {
            CUDA_CHECK(cudaMalloc(&input_threshold, total_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&spikes_consumed, total_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&spikes_produced, total_rules * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&delay, total_rules * sizeof(int)));
        }
        if (num_neurons > 0) {
            CUDA_CHECK(cudaMalloc(&rule_start_idx, num_neurons * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&rule_count, num_neurons * sizeof(int)));
        }
    }

    void free() {
        cudaFree(input_threshold);
        cudaFree(spikes_consumed);
        cudaFree(spikes_produced);
        cudaFree(delay);
        cudaFree(rule_start_idx);
        cudaFree(rule_count);
        input_threshold = nullptr;
        rule_start_idx = nullptr;
        count = 0;
    }
};

struct DeviceLocalSynapseData {
    int* source_idx = nullptr; 
    int* dest_idx = nullptr;   
    int* weight = nullptr;
    int count = 0;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&source_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dest_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weight, n * sizeof(int)));
    }

    void free() {
        cudaFree(source_idx); cudaFree(dest_idx); cudaFree(weight);
        source_idx = nullptr;
        count = 0;
    }
};

struct DeviceExportSynapseData {
    int* source_idx = nullptr;      
    int* export_buf_idx = nullptr;  
    int* weight = nullptr;
    int count = 0;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&source_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&export_buf_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weight, n * sizeof(int)));
    }

    void free() {
        cudaFree(source_idx); cudaFree(export_buf_idx); cudaFree(weight);
        source_idx = nullptr;
        count = 0;
    }
};

struct DeviceImportMapData {
    int* import_buf_idx = nullptr; 
    int* dest_idx = nullptr;       
    int count = 0;

    void allocate(int n) {
        count = n;
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&import_buf_idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dest_idx, n * sizeof(int)));
    }

    void free() {
        cudaFree(import_buf_idx); cudaFree(dest_idx);
        import_buf_idx = nullptr;
        count = 0;
    }
};

// --- Kernels (L1 Cache Optimized) ---

__global__ void kUpdateAndSelect(
    DeviceNeuronData neurons, 
    DeviceRuleData rules) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;

    if (neurons.delay_timer[idx] > 0) {
        neurons.delay_timer[idx]--;
        if (neurons.delay_timer[idx] == 0) {
            neurons.is_open[idx] = 1;
        }
    }

    if (!neurons.is_open[idx]) return;

    int current_spikes = neurons.configuration[idx];
    int r_start = rules.rule_start_idx[idx];
    int r_count = rules.rule_count[idx];

    for (int i = 0; i < r_count; ++i) {
        int r_idx = r_start + i;
        // Use __ldg for read-only data (Texture Cache)
        if (current_spikes >= __ldg(&rules.input_threshold[r_idx])) {
            neurons.configuration[idx] -= __ldg(&rules.spikes_consumed[r_idx]);
            int produced = __ldg(&rules.spikes_produced[r_idx]);
            int delay = __ldg(&rules.delay[r_idx]);

            if (delay > 0) {
                neurons.is_open[idx] = 0;
                neurons.delay_timer[idx] = delay;
                neurons.pending_emission[idx] = produced;
            } else {
                neurons.spike_production[idx] = produced;
            }
            break; 
        }
    }
}

__global__ void kPropagateLocal(
    DeviceNeuronData neurons, 
    const DeviceLocalSynapseData synapses) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.count) return;

    int src = __ldg(&synapses.source_idx[idx]);
    int dst = __ldg(&synapses.dest_idx[idx]);
    int w = __ldg(&synapses.weight[idx]);

    int spikes_to_send = neurons.spike_production[src] * w;

    if (neurons.is_open[src]) {
        int pending = neurons.pending_emission[src];
        if (pending > 0) {
            spikes_to_send += pending * w;
        }
    }

    if (spikes_to_send > 0 && neurons.is_open[dst]) {
        atomicAdd(&neurons.configuration[dst], spikes_to_send);
    }
}

__global__ void kPopulateExport(
    DeviceNeuronData neurons, 
    const DeviceExportSynapseData synapses, 
    int* __restrict__ export_buffer) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.count) return;

    int src = __ldg(&synapses.source_idx[idx]);
    int buf_idx = __ldg(&synapses.export_buf_idx[idx]);
    int w = __ldg(&synapses.weight[idx]);

    int spikes_to_send = neurons.spike_production[src] * w;

    if (neurons.is_open[src]) {
        int pending = neurons.pending_emission[src];
        if (pending > 0) spikes_to_send += pending * w;
    }

    if (spikes_to_send > 0) {
        atomicAdd(&export_buffer[buf_idx], spikes_to_send);
    }
}

__global__ void kApplyImports(
    DeviceNeuronData neurons, 
    const DeviceImportMapData imports, 
    const int* __restrict__ import_buffer) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= imports.count) return;

    int buf_idx = __ldg(&imports.import_buf_idx[idx]);
    int dst = __ldg(&imports.dest_idx[idx]);
    
    int spikes = __ldg(&import_buffer[buf_idx]);

    if (spikes > 0 && neurons.is_open[dst]) {
        atomicAdd(&neurons.configuration[dst], spikes);
    }
}

__global__ void kCleanup(DeviceNeuronData neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.count) return;

    neurons.spike_production[idx] = 0;
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

// --- Main Class ---

class CudaMpiSnpSimulator : public ISnpSimulator {
private:
    int mpi_rank, mpi_size;
    
    int global_num_neurons = 0;
    int local_start_idx = 0;
    int local_end_idx = 0;
    int local_num_neurons = 0;

    // Device Data
    DeviceNeuronData d_neurons;
    DeviceRuleData d_rules;
    DeviceLocalSynapseData d_local_synapses;
    DeviceExportSynapseData d_export_synapses;
    DeviceImportMapData d_import_map;

    // Resources
    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_comm = nullptr;
    cudaEvent_t event_compute_done = nullptr;
    cudaEvent_t event_comm_done = nullptr;

    struct RankCommOffsets {
        int export_offset;
        int export_count;
        int import_offset;
        int import_count;
    };
    std::vector<RankCommOffsets> comm_offsets; 

    PinnedBuffer<int> h_pinned_export;
    PinnedBuffer<int> h_pinned_import;
    int* d_export_buffer = nullptr;
    int* d_import_buffer = nullptr;
    int total_export_size = 0;
    int total_import_size = 0;

    double compute_time = 0.0;
    double comm_time = 0.0;
    int steps = 0;

public:
    CudaMpiSnpSimulator() {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_comm, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&event_compute_done));
        CUDA_CHECK(cudaEventCreate(&event_comm_done));
    }

    ~CudaMpiSnpSimulator() {
        d_neurons.free();
        d_rules.free();
        d_local_synapses.free();
        d_export_synapses.free();
        d_import_map.free();
        if (d_export_buffer) cudaFree(d_export_buffer);
        if (d_import_buffer) cudaFree(d_import_buffer);
        
        if (stream_compute) cudaStreamDestroy(stream_compute);
        if (stream_comm) cudaStreamDestroy(stream_comm);
        if (event_compute_done) cudaEventDestroy(event_compute_done);
        if (event_comm_done) cudaEventDestroy(event_comm_done);
    }

    bool loadSystem(const SnpSystemConfig& config) override {
        global_num_neurons = config.neurons.size();

        // 1. Partitioning
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

        // 2. Prepare Data
        d_neurons.allocate(local_num_neurons);
        
        if (local_num_neurons > 0) {
            std::vector<int> h_initial(local_num_neurons);
            std::vector<char> h_open(local_num_neurons, 1);

            for (int i = 0; i < local_num_neurons; ++i) {
                h_initial[i] = config.neurons[local_start_idx + i].initial_spikes;
            }

            CUDA_CHECK(cudaMemcpy(d_neurons.configuration, h_initial.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_neurons.initial_config, h_initial.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_neurons.is_open, h_open.data(), local_num_neurons * sizeof(char), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_neurons.delay_timer, 0, local_num_neurons * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_neurons.pending_emission, 0, local_num_neurons * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_neurons.spike_production, 0, local_num_neurons * sizeof(int)));
        }

        prepareRules(config);
        prepareTopology(config);

        return true;
    }

    void step(int num_steps = 1) override {
        int neuronGrid = std::max(1, (local_num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE);
        int synapseGrid = std::max(1, (d_local_synapses.count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        int exportGrid = std::max(1, (d_export_synapses.count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        int importGrid = std::max(1, (d_import_map.count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        for (int s = 0; s < num_steps; ++s) {
            auto t1 = std::chrono::high_resolution_clock::now();

            // --- Phase 1: Compute (Stream 1) ---
            if (local_num_neurons > 0) {
                if (total_export_size > 0) {
                    CUDA_CHECK(cudaMemsetAsync(d_export_buffer, 0, total_export_size * sizeof(int), stream_compute));
                }

                kUpdateAndSelect<<<neuronGrid, BLOCK_SIZE, 0, stream_compute>>>(d_neurons, d_rules);
                
                if (d_local_synapses.count > 0) {
                    kPropagateLocal<<<synapseGrid, BLOCK_SIZE, 0, stream_compute>>>(d_neurons, d_local_synapses);
                }
                
                if (d_export_synapses.count > 0) {
                    kPopulateExport<<<exportGrid, BLOCK_SIZE, 0, stream_compute>>>(d_neurons, d_export_synapses, d_export_buffer);
                }
            }

            CUDA_CHECK(cudaEventRecord(event_compute_done, stream_compute));

            // --- Phase 2: D2H (Stream 2) ---
            CUDA_CHECK(cudaStreamWaitEvent(stream_comm, event_compute_done, 0));
            
            if (total_export_size > 0) {
                CUDA_CHECK(cudaMemcpyAsync(h_pinned_export.get(), d_export_buffer, 
                           total_export_size * sizeof(int), cudaMemcpyDeviceToHost, stream_comm));
            }

            // --- Phase 3: Cleanup (Stream 1) ---
            if (local_num_neurons > 0) {
                kCleanup<<<neuronGrid, BLOCK_SIZE, 0, stream_compute>>>(d_neurons);
            }

            // --- Phase 4: MPI (Host) ---
            CUDA_CHECK(cudaStreamSynchronize(stream_comm));

            auto t2 = std::chrono::high_resolution_clock::now();
            compute_time += std::chrono::duration<double, std::milli>(t2 - t1).count();

            std::vector<MPI_Request> requests;
            
            for (int r = 0; r < mpi_size; ++r) {
                if (r == mpi_rank) continue;
                int count = comm_offsets[r].import_count;
                if (count > 0) {
                    MPI_Request req;
                    MPI_Irecv(h_pinned_import.get() + comm_offsets[r].import_offset, count, MPI_INT, r, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
            }

            for (int r = 0; r < mpi_size; ++r) {
                if (r == mpi_rank) continue;
                int count = comm_offsets[r].export_count;
                if (count > 0) {
                    MPI_Request req;
                    MPI_Isend(h_pinned_export.get() + comm_offsets[r].export_offset, count, MPI_INT, r, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
            }

            if (!requests.empty()) {
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            }

            auto t3 = std::chrono::high_resolution_clock::now();
            comm_time += std::chrono::duration<double, std::milli>(t3 - t2).count();

            // --- Phase 5: H2D & Apply (Stream 2 -> Stream 1) ---
            if (total_import_size > 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_import_buffer, h_pinned_import.get(), 
                           total_import_size * sizeof(int), cudaMemcpyHostToDevice, stream_comm));
            }
            
            CUDA_CHECK(cudaEventRecord(event_comm_done, stream_comm));
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, event_comm_done, 0));

            if (local_num_neurons > 0 && d_import_map.count > 0) {
                kApplyImports<<<importGrid, BLOCK_SIZE, 0, stream_compute>>>(d_neurons, d_import_map, d_import_buffer);
            }
            
            CUDA_CHECK(cudaStreamSynchronize(stream_compute));

            auto t4 = std::chrono::high_resolution_clock::now();
            compute_time += std::chrono::duration<double, std::milli>(t4 - t3).count();
            
            steps++;
        }
    }

    std::vector<int> getGlobalState() const override {
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::vector<int> local_state(local_num_neurons);
        if (local_num_neurons > 0) {
            CUDA_CHECK(cudaMemcpy(local_state.data(), d_neurons.configuration, local_num_neurons * sizeof(int), cudaMemcpyDeviceToHost));
        }

        std::vector<int> recv_counts(mpi_size);
        int local_n = local_num_neurons;
        MPI_Allgather(&local_n, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> displs(mpi_size);
        displs[0] = 0;
        for (int i = 1; i < mpi_size; ++i) {
            displs[i] = displs[i-1] + recv_counts[i-1];
        }
        int total_neurons = displs.back() + recv_counts.back();

        std::vector<int> global_state(total_neurons);
        MPI_Allgatherv(local_state.data(), local_n, MPI_INT, 
                       global_state.data(), recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

        return global_state;
    }

    void reset() override {
        int gridSize = std::max(1, (local_num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (local_num_neurons > 0) {
            kResetNeurons<<<gridSize, BLOCK_SIZE, 0, stream_compute>>>(d_neurons);
            CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        }
        compute_time = 0;
        comm_time = 0;
        steps = 0;
    }

    std::string getPerformanceReport() const override {
        std::ostringstream ss;
        ss << "=== MPI+CUDA Optimized Rank " << mpi_rank << " Report ===\n";
        ss << "Neurons Owned: " << local_num_neurons << "\n";
        ss << "Compute Time: " << compute_time << " ms\n";
        ss << "Comm Time:    " << comm_time << " ms\n";
        ss << "Total Steps:  " << steps << "\n";
        return ss.str();
    }

private:
    void prepareRules(const SnpSystemConfig& config) {
        std::vector<int> h_thresh, h_cons, h_prod, h_delay;
        std::vector<int> h_start(local_num_neurons, 0); 
        std::vector<int> h_count(local_num_neurons, 0); 

        int rule_cursor = 0;
        for (int i = 0; i < local_num_neurons; ++i) {
            int global_id = local_start_idx + i;
            const auto& neuron = config.neurons[global_id];
            
            h_start[i] = rule_cursor;
            h_count[i] = neuron.rules.size();
            
            for (const auto& r : neuron.rules) {
                h_thresh.push_back(r.input_threshold);
                h_cons.push_back(r.spikes_consumed);
                h_prod.push_back(r.spikes_produced);
                h_delay.push_back(r.delay);
                rule_cursor++;
            }
        }

        d_rules.allocate(rule_cursor, local_num_neurons);

        if (rule_cursor > 0) {
            CUDA_CHECK(cudaMemcpy(d_rules.input_threshold, h_thresh.data(), rule_cursor * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_consumed, h_cons.data(), rule_cursor * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_produced, h_prod.data(), rule_cursor * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.delay, h_delay.data(), rule_cursor * sizeof(int), cudaMemcpyHostToDevice));
        }

        if (local_num_neurons > 0) {
            CUDA_CHECK(cudaMemcpy(d_rules.rule_start_idx, h_start.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.rule_count, h_count.data(), local_num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    void prepareTopology(const SnpSystemConfig& config) {
        std::vector<std::pair<int, int>> rank_ranges(mpi_size);
        int base = global_num_neurons / mpi_size;
        int rem = global_num_neurons % mpi_size;
        for (int r = 0; r < mpi_size; ++r) {
            int n = (r < rem) ? base + 1 : base;
            int start = (r < rem) ? r * n : rem * (base + 1) + (r - rem) * base;
            rank_ranges[r] = {start, start + n};
        }

        std::vector<int> loc_src, loc_dst, loc_w;
        std::map<int, std::set<int>> remote_targets_per_rank; 

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

        comm_offsets.resize(mpi_size);
        
        std::vector<int> exp_src, exp_buf_idx, exp_w;
        int current_export_offset = 0;

        for (int r = 0; r < mpi_size; ++r) {
            if (r == mpi_rank) continue;

            std::vector<int> targets(remote_targets_per_rank[r].begin(), remote_targets_per_rank[r].end());
            
            comm_offsets[r].export_offset = current_export_offset;
            comm_offsets[r].export_count = targets.size();
            
            std::map<int, int> dest_to_chunk_idx;
            for(size_t i=0; i<targets.size(); ++i) {
                dest_to_chunk_idx[targets[i]] = current_export_offset + i;
            }
            current_export_offset += targets.size();

            for (const auto& syn : config.synapses) {
                if (syn.source_id >= local_start_idx && syn.source_id < local_end_idx) { 
                    if (dest_to_chunk_idx.count(syn.dest_id)) {
                        exp_src.push_back(syn.source_id - local_start_idx);
                        exp_buf_idx.push_back(dest_to_chunk_idx[syn.dest_id]);
                        exp_w.push_back(syn.weight);
                    }
                }
            }
        }
        total_export_size = current_export_offset;

        std::vector<int> imp_buf_idx, imp_dst;
        int current_import_offset = 0;

        for (int r = 0; r < mpi_size; ++r) {
            if (r == mpi_rank) continue;

            std::set<int> incoming_targets;
            for (const auto& syn : config.synapses) {
                if (syn.source_id >= rank_ranges[r].first && syn.source_id < rank_ranges[r].second) {
                    if (syn.dest_id >= local_start_idx && syn.dest_id < local_end_idx) {
                        incoming_targets.insert(syn.dest_id);
                    }
                }
            }

            std::vector<int> targets(incoming_targets.begin(), incoming_targets.end());
            
            comm_offsets[r].import_offset = current_import_offset;
            comm_offsets[r].import_count = targets.size();

            for (size_t i = 0; i < targets.size(); ++i) {
                imp_buf_idx.push_back(current_import_offset + i);
                imp_dst.push_back(targets[i] - local_start_idx);
            }
            current_import_offset += targets.size();
        }
        total_import_size = current_import_offset;

        h_pinned_export.allocate(total_export_size);
        h_pinned_import.allocate(total_import_size);

        d_local_synapses.allocate(loc_src.size());
        if (!loc_src.empty()) {
            CUDA_CHECK(cudaMemcpy(d_local_synapses.source_idx, loc_src.data(), loc_src.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_synapses.dest_idx, loc_dst.data(), loc_dst.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_synapses.weight, loc_w.data(), loc_w.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        d_export_synapses.allocate(exp_src.size());
        if (!exp_src.empty()) {
            CUDA_CHECK(cudaMemcpy(d_export_synapses.source_idx, exp_src.data(), exp_src.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_export_synapses.export_buf_idx, exp_buf_idx.data(), exp_buf_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_export_synapses.weight, exp_w.data(), exp_w.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        d_import_map.allocate(imp_buf_idx.size());
        if (!imp_buf_idx.empty()) {
            CUDA_CHECK(cudaMemcpy(d_import_map.import_buf_idx, imp_buf_idx.data(), imp_buf_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_import_map.dest_idx, imp_dst.data(), imp_dst.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        if (total_export_size > 0) CUDA_CHECK(cudaMalloc(&d_export_buffer, total_export_size * sizeof(int)));
        if (total_import_size > 0) CUDA_CHECK(cudaMalloc(&d_import_buffer, total_import_size * sizeof(int)));
    }
};

std::unique_ptr<ISnpSimulator> createCudaMpiSimulator() {
    return std::make_unique<CudaMpiSnpSimulator>();
}