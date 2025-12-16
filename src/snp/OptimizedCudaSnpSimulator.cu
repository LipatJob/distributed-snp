#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include "PerformanceMetrics.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <sstream>
#include <chrono>
#include <iostream>
#include <stdexcept>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while(0)

constexpr int BLOCK_SIZE = 256; // Increased block size for better occupancy

/**
 * @brief Optimized Structure of Arrays (SoA) for Neurons
 * Using int for is_open for alignment
 */
struct DeviceNeuronData {
    int* configuration;       // Spike count
    int* initial_config;      // For reset
    int* is_open;            // 1 = open, 0 = closed (int for alignment)
    int* delay_timer;        // Remaining delay ticks
    int* pending_emission;   // Spikes waiting in delay buffer
    int* current_output;     // Unified output buffer for the current step
    int num_neurons;
    
    void allocate(int n) {
        num_neurons = n;
        CUDA_CHECK(cudaMalloc(&configuration, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&initial_config, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&is_open, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&delay_timer, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&pending_emission, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&current_output, n * sizeof(int)));
    }
    
    void deallocate() {
        if (configuration) cudaFree(configuration);
        if (initial_config) cudaFree(initial_config);
        if (is_open) cudaFree(is_open);
        if (delay_timer) cudaFree(delay_timer);
        if (pending_emission) cudaFree(pending_emission);
        if (current_output) cudaFree(current_output);
    }
};

/**
 * @brief Device Rule Data (Read-Only during execution)
 */
struct DeviceRuleData {
    int* neuron_id;
    int* input_threshold;
    int* spikes_consumed;
    int* spikes_produced;
    int* delay;
    int* rule_start_idx;
    int* rule_count;
    int total_rules;
    
    void allocate(int num_rules, int num_n) {
        total_rules = num_rules;
        CUDA_CHECK(cudaMalloc(&neuron_id, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&input_threshold, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_consumed, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_produced, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&delay, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rule_start_idx, num_n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rule_count, num_n * sizeof(int)));
    }
    
    void deallocate() {
        if (neuron_id) cudaFree(neuron_id);
        if (input_threshold) cudaFree(input_threshold);
        if (spikes_consumed) cudaFree(spikes_consumed);
        if (spikes_produced) cudaFree(spikes_produced);
        if (delay) cudaFree(delay);
        if (rule_start_idx) cudaFree(rule_start_idx);
        if (rule_count) cudaFree(rule_count);
    }
};

/**
 * @brief Device Synapse Data (Read-Only topology)
 */
struct DeviceSynapseData {
    int* source_id;
    int* dest_id;
    int* weight;
    int num_synapses;
    
    void allocate(int n) {
        num_synapses = n;
        CUDA_CHECK(cudaMalloc(&source_id, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dest_id, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weight, n * sizeof(int)));
    }
    
    void deallocate() {
        if (source_id) cudaFree(source_id);
        if (dest_id) cudaFree(dest_id);
        if (weight) cudaFree(weight);
    }
};

/**
 * @brief Fused Neuron Logic Kernel
 * * Handles:
 * 1. Timer updates (decrement delay)
 * 2. Delay expiration (closed -> open, release pending spikes)
 * 3. Rule matching and execution
 * 4. Spike consumption
 * 5. Output scheduling (immediate or delayed)
 * * Writes total spikes to emit this step into `current_output`.
 */
static __global__ void neuronDynamicsKernel(
    DeviceNeuronData neurons,
    const int* __restrict__ rule_start_idx,
    const int* __restrict__ rule_count,
    const int* __restrict__ rule_threshold,
    const int* __restrict__ rule_consumed,
    const int* __restrict__ rule_produced,
    const int* __restrict__ rule_delay
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.num_neurons) return;

    // Local registers for state
    int current_delay = neurons.delay_timer[idx];
    int is_open = neurons.is_open[idx];
    int spikes_to_emit_now = 0;

    // --- Phase 1: Update Delay Status ---
    if (current_delay > 0) {
        current_delay--;
        neurons.delay_timer[idx] = current_delay;
        
        if (current_delay == 0) {
            // Delay expired: Open neuron and release pending buffer
            is_open = 1;
            neurons.is_open[idx] = 1;
            spikes_to_emit_now += neurons.pending_emission[idx];
            neurons.pending_emission[idx] = 0; // Clear buffer
        }
    }

    // --- Phase 2: Apply Rules (Only if open) ---
    if (is_open) {
        int current_spikes = neurons.configuration[idx];
        int r_start = rule_start_idx[idx];
        int r_count = rule_count[idx];
        int r_end = r_start + r_count;

        // Linear scan for first applicable rule
        for (int i = r_start; i < r_end; ++i) {
            if (current_spikes >= rule_threshold[i]) {
                // Apply Rule
                int consumed = rule_consumed[i];
                int produced = rule_produced[i];
                int d = rule_delay[i];

                // Consume spikes
                neurons.configuration[idx] = current_spikes - consumed;

                if (d > 0) {
                    // Close neuron, schedule for later
                    neurons.is_open[idx] = 0;
                    neurons.delay_timer[idx] = d;
                    neurons.pending_emission[idx] = produced;
                } else {
                    // Immediate emission
                    spikes_to_emit_now += produced;
                }
                
                // Determinism: Only one rule fires per step
                break;
            }
        }
    }

    // --- Phase 3: Write Output ---
    // We overwrite current_output every step, effectively clearing it
    neurons.current_output[idx] = spikes_to_emit_now;
}

/**
 * @brief Unified Synapse Propagation Kernel
 * * Reads `current_output` from source neurons and adds to destination.
 * Handles both immediate firings and delayed emissions that just matured.
 */
static __global__ void synapseTransferKernel(
    const int* __restrict__ output_spikes, // Read from neurons.current_output
    int* __restrict__ neuron_config,       // Write to neurons.configuration
    const int* __restrict__ src_ids,
    const int* __restrict__ dest_ids,
    const int* __restrict__ weights,
    int num_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    int src = src_ids[idx];
    int produced_spikes = output_spikes[src];

    // Warp divergence check: most neurons won't fire every step.
    if (produced_spikes > 0) {
        int dest = dest_ids[idx];
        int w = weights[idx];
        
        // Atomic add is necessary because multiple synapses may target the same neuron
        atomicAdd(&neuron_config[dest], produced_spikes * w);
    }
}

static __global__ void resetKernel(DeviceNeuronData neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.num_neurons) return;
    
    neurons.configuration[idx] = neurons.initial_config[idx];
    neurons.is_open[idx] = 1;
    neurons.delay_timer[idx] = 0;
    neurons.pending_emission[idx] = 0;
    neurons.current_output[idx] = 0;
}

class OptimizedCudaSnpSimulator : public ISnpSimulator {
private:
    SnpSystemConfig config;
    int num_neurons = 0;
    int num_synapses = 0;
    int total_rules = 0;
    
    DeviceNeuronData d_neurons;
    DeviceRuleData d_rules;
    DeviceSynapseData d_synapses;
    
    // Pinned memory for fast transfers
    int* h_pinned_state = nullptr;
    
    double total_compute_time_ms = 0.0;
    int steps_executed = 0;
    
    int neuron_grid = 0;
    int synapse_grid = 0;

public:
    OptimizedCudaSnpSimulator() = default;
    
    ~OptimizedCudaSnpSimulator() {
        cleanup();
    }
    
    bool loadSystem(const SnpSystemConfig& sys_config) override {
        try {
            cleanup(); // Ensure clean slate
            config = sys_config;
            num_neurons = config.neurons.size();
            num_synapses = config.synapses.size();
            total_rules = config.getTotalRulesCount();
            
            // Allocate Device Memory
            d_neurons.allocate(num_neurons);
            d_rules.allocate(total_rules, num_neurons);
            d_synapses.allocate(num_synapses);
            
            // Allocate Pinned Memory
            CUDA_CHECK(cudaMallocHost(&h_pinned_state, num_neurons * sizeof(int)));
            
            uploadData();
            
            // Calc grids
            neuron_grid = (num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
            synapse_grid = (num_synapses + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "OptimizedCudaSnpSimulator Error: " << e.what() << std::endl;
            return false;
        }
    }
    
    void step(int steps = 1) override {
        if (num_neurons == 0) return;

        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < steps; ++i) {
            // Kernel 1: Neuron Logic (Update, Fire, Generate Output)
            neuronDynamicsKernel<<<neuron_grid, BLOCK_SIZE>>>(
                d_neurons,
                d_rules.rule_start_idx,
                d_rules.rule_count,
                d_rules.input_threshold,
                d_rules.spikes_consumed,
                d_rules.spikes_produced,
                d_rules.delay
            );
            
            // Kernel 2: Synapse Propagation (Transfer Spikes)
            if (num_synapses > 0) {
                // Ensure Neuron logic is done before propagating
                // (Implicit serialization in stream 0, but good for clarity)
                synapseTransferKernel<<<synapse_grid, BLOCK_SIZE>>>(
                    d_neurons.current_output,
                    d_neurons.configuration,
                    d_synapses.source_id,
                    d_synapses.dest_id,
                    d_synapses.weight,
                    num_synapses
                );
            }
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_compute_time_ms += elapsed.count();
        steps_executed += steps;
    }
    
    std::vector<int> getGlobalState() const override {
        if (num_neurons == 0) return {};
        
        // Fast copy to pinned memory
        CUDA_CHECK(cudaMemcpy(h_pinned_state, d_neurons.configuration, 
                   num_neurons * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Construct vector from pinned memory
        // This is much faster than cudaMemcpy directly to vector.data() if the vector is paged
        return std::vector<int>(h_pinned_state, h_pinned_state + num_neurons);
    }
    
    void reset() override {
        if (num_neurons == 0) return;
        resetKernel<<<neuron_grid, BLOCK_SIZE>>>(d_neurons);
        CUDA_CHECK(cudaDeviceSynchronize());
        steps_executed = 0;
        total_compute_time_ms = 0.0;
    }
    
    PerformanceMetrics getPerformanceMetrics() const override {
        PerformanceMetrics metrics;
        
        // Core metrics
        metrics.steps_executed = steps_executed;
        metrics.total_time_ms = total_compute_time_ms;
        metrics.compute_time_ms = total_compute_time_ms; // For single-GPU, compute = total
        
        // CUDA metrics
        metrics.cuda.kernel_time_ms = total_compute_time_ms;
        // Note: We're not separately tracking memory transfer time in this implementation
        // since transfers only happen on getGlobalState() and loadSystem()
        metrics.cuda.memory_transfer_time_ms = 0.0;
        
        // Algorithm/System metrics
        metrics.algorithm.num_neurons = num_neurons;
        metrics.algorithm.num_synapses = num_synapses;
        metrics.algorithm.total_rules = total_rules;
        
        return metrics;
    }
    
    std::string getPerformanceReport() const override {
        PerformanceMetrics metrics = getPerformanceMetrics();
        return metrics.toReport("CUDA SNP Simulator");
    }

private:
    void cleanup() {
        d_neurons.deallocate();
        d_rules.deallocate();
        d_synapses.deallocate();
        if (h_pinned_state) {
            cudaFreeHost(h_pinned_state);
            h_pinned_state = nullptr;
        }
    }
    
    void uploadData() {
        // --- 1. Neuron Data ---
        std::vector<int> h_config(num_neurons);
        std::vector<int> h_open(num_neurons, 1);
        
        for(int i=0; i<num_neurons; ++i) {
            h_config[i] = config.neurons[i].initial_spikes;
        }
        
        CUDA_CHECK(cudaMemcpy(d_neurons.configuration, h_config.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.initial_config, h_config.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.is_open, h_open.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_neurons.delay_timer, 0, num_neurons * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_neurons.pending_emission, 0, num_neurons * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_neurons.current_output, 0, num_neurons * sizeof(int)));

        // --- 2. Rule Data ---
        std::vector<int> h_threshold, h_consumed, h_produced, h_delay;
        std::vector<int> h_start(num_neurons), h_count(num_neurons);
        
        int current_idx = 0;
        for(int i=0; i<num_neurons; ++i) {
            const auto& rules = config.neurons[i].rules;
            h_start[i] = current_idx;
            h_count[i] = rules.size();
            for(const auto& r : rules) {
                h_threshold.push_back(r.input_threshold);
                h_consumed.push_back(r.spikes_consumed);
                h_produced.push_back(r.spikes_produced);
                h_delay.push_back(r.delay);
                current_idx++;
            }
        }
        
        if (total_rules > 0) {
            CUDA_CHECK(cudaMemcpy(d_rules.input_threshold, h_threshold.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_consumed, h_consumed.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_produced, h_produced.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.delay, h_delay.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_rules.rule_start_idx, h_start.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rules.rule_count, h_count.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice));

        // --- 3. Synapse Data ---
        if (num_synapses > 0) {
            std::vector<int> h_src(num_synapses), h_dest(num_synapses), h_w(num_synapses);
            for(int i=0; i<num_synapses; ++i) {
                h_src[i] = config.synapses[i].source_id;
                h_dest[i] = config.synapses[i].dest_id;
                h_w[i] = config.synapses[i].weight;
            }
            CUDA_CHECK(cudaMemcpy(d_synapses.source_id, h_src.data(), num_synapses * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_synapses.dest_id, h_dest.data(), num_synapses * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_synapses.weight, h_w.data(), num_synapses * sizeof(int), cudaMemcpyHostToDevice));
        }
    }
};

std::unique_ptr<ISnpSimulator> createOptimizedCudaSimulator() {
    return std::make_unique<OptimizedCudaSnpSimulator>();
}