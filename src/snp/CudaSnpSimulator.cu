#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
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

// Optimal block size for most modern GPUs (multiple of warp size 32)
constexpr int BLOCK_SIZE = 256;

/**
 * @brief Structure of Arrays (SoA) for Neurons on Device
 * 
 * Benefits:
 * - Coalesced memory access (threads access consecutive memory)
 * - Better cache utilization
 * - Reduced bank conflicts in shared memory
 */
struct DeviceNeuronData {
    int* configuration;       // Current spike count per neuron
    int* initial_config;      // Initial state for reset
    char* is_open;           // Neuron open/closed status. Is boolean but using char for alignment
    int* delay_timer;        // Remaining delay ticks
    int* pending_emission;   // Spikes to emit when delay expires
    int* spike_production;   // Temporary: spikes produced this step
    int num_neurons;
    
    void allocate(int n) {
        num_neurons = n;
        CUDA_CHECK(cudaMalloc(&configuration, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&initial_config, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&is_open, n * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&delay_timer, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&pending_emission, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spike_production, n * sizeof(int)));
    }
    
    void deallocate() {
        cudaFree(configuration);
        cudaFree(initial_config);
        cudaFree(is_open);
        cudaFree(delay_timer);
        cudaFree(pending_emission);
        cudaFree(spike_production);
    }
};

/**
 * @brief Structure of Arrays (SoA) for Rules on Device
 * 
 * Rules are organized per-neuron for efficient access.
 * Each neuron has a contiguous range of rules.
 */
struct DeviceRuleData {
    int* neuron_id;           // Which neuron this rule belongs to
    int* input_threshold;     // Minimum spikes needed to fire
    int* spikes_consumed;     // Spikes consumed when fired
    int* spikes_produced;     // Spikes produced when fired
    int* delay;              // Delay before emission
    int* rule_start_idx;     // Start index of rules for each neuron
    int* rule_count;         // Number of rules per neuron
    int total_rules;
    int num_neurons;
    
    void allocate(int num_rules, int num_n) {
        total_rules = num_rules;
        num_neurons = num_n;
        CUDA_CHECK(cudaMalloc(&neuron_id, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&input_threshold, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_consumed, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_produced, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&delay, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rule_start_idx, num_n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rule_count, num_n * sizeof(int)));
    }
    
    void deallocate() {
        cudaFree(neuron_id);
        cudaFree(input_threshold);
        cudaFree(spikes_consumed);
        cudaFree(spikes_produced);
        cudaFree(delay);
        cudaFree(rule_start_idx);
        cudaFree(rule_count);
    }
};

/**
 * @brief Structure of Arrays (SoA) for Synapses on Device
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
        cudaFree(source_id);
        cudaFree(dest_id);
        cudaFree(weight);
    }
};

/**
 * @brief CUDA Kernel: Update neuron status based on delay timers
 * 
 * Each thread handles one neuron.
 * - Decrement delay timers
 * - Open neurons when delay reaches 0
 * 
 * Memory access pattern: Each thread accesses its own index (coalesced)
 */
__global__ void updateNeuronStatusKernel(
    DeviceNeuronData neurons
) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= neurons.num_neurons) return;
    
    // Decrement delay timer if active
    if (neurons.delay_timer[neuron_id] > 0) {
        neurons.delay_timer[neuron_id]--;
        
        // Open neuron when delay expires
        if (neurons.delay_timer[neuron_id] == 0) {
            neurons.is_open[neuron_id] = true;
        }
    }
}

/**
 * @brief CUDA Kernel: Select and apply firing rules
 * 
 * Each thread handles one neuron.
 * - Find first applicable rule (deterministic)
 * - Consume spikes
 * - Schedule production (immediate or delayed)
 * 
 * Optimization: Minimize branch divergence by early exit for closed neurons
 */
__global__ void selectAndApplyRulesKernel(
    DeviceNeuronData neurons,
    DeviceRuleData rules
) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= neurons.num_neurons) return;
    
    // Early exit for closed neurons (reduces divergence)
    if (!neurons.is_open[neuron_id]) return;
    
    int current_spikes = neurons.configuration[neuron_id];
    int rule_start = rules.rule_start_idx[neuron_id];
    int rule_end = rule_start + rules.rule_count[neuron_id];
    
    // Find first applicable rule (deterministic by order)
    for (int rule_idx = rule_start; rule_idx < rule_end; ++rule_idx) {
        int threshold = rules.input_threshold[rule_idx];
        
        // Check if rule is applicable
        if (current_spikes >= threshold) {
            int consumed = rules.spikes_consumed[rule_idx];
            int produced = rules.spikes_produced[rule_idx];
            int rule_delay = rules.delay[rule_idx];
            
            // Consume spikes
            neurons.configuration[neuron_id] -= consumed;
            
            // Handle spike production based on delay
            if (rule_delay > 0) {
                // Close neuron and schedule emission
                neurons.is_open[neuron_id] = false;
                neurons.delay_timer[neuron_id] = rule_delay;
                neurons.pending_emission[neuron_id] = produced;
            } else {
                // Immediate production
                neurons.spike_production[neuron_id] = produced;
            }
            
            // Only apply first applicable rule
            break;
        }
    }
}

/**
 * @brief CUDA Kernel: Propagate pending emissions (from delayed rules)
 * 
 * Each thread handles one synapse, reading from neurons that just opened.
 * Uses atomicAdd for concurrent writes to destination neurons.
 * 
 * Only propagates when source neuron is open (delay has expired).
 */
__global__ void propagatePendingEmissionsKernel(
    DeviceNeuronData neurons,
    DeviceSynapseData synapses
) {
    int synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (synapse_id >= synapses.num_synapses) return;
    
    int source = synapses.source_id[synapse_id];
    int dest = synapses.dest_id[synapse_id];
    int weight = synapses.weight[synapse_id];
    
    // Only propagate if source neuron is open (delay expired) AND has pending emission
    if (neurons.is_open[source] && neurons.pending_emission[source] > 0) {
        int spikes_to_send = neurons.pending_emission[source] * weight;
        
        // Only open neurons can receive spikes
        if (neurons.is_open[dest]) {
            atomicAdd(&neurons.configuration[dest], spikes_to_send);
        }
    }
}

/**
 * @brief CUDA Kernel: Clear pending emissions after propagation
 * 
 * Only clears if the neuron is open (meaning the pending emission was just propagated).
 */
__global__ void clearPendingEmissionsKernel(DeviceNeuronData neurons) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= neurons.num_neurons) return;
    
    // Only clear if neuron is open (pending emission was propagated)
    if (neurons.is_open[neuron_id]) {
        neurons.pending_emission[neuron_id] = 0;
    }
}

/**
 * @brief CUDA Kernel: Propagate immediate spike production through synapses
 * 
 * Each thread handles one synapse.
 * Uses atomicAdd for safe concurrent writes to destination neurons.
 */
__global__ void propagateImmediateSpikesKernel(
    DeviceNeuronData neurons,
    DeviceSynapseData synapses
) {
    int synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (synapse_id >= synapses.num_synapses) return;
    
    int source = synapses.source_id[synapse_id];
    int dest = synapses.dest_id[synapse_id];
    int weight = synapses.weight[synapse_id];
    
    int spikes = neurons.spike_production[source];
    if (spikes > 0) {
        int spikes_to_send = spikes * weight;
        
        // Only open neurons can receive spikes
        if (neurons.is_open[dest]) {
            atomicAdd(&neurons.configuration[dest], spikes_to_send);
        }
    }
}

/**
 * @brief CUDA Kernel: Clear spike production buffer
 */
__global__ void clearSpikeProductionKernel(DeviceNeuronData neurons) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= neurons.num_neurons) return;
    
    neurons.spike_production[neuron_id] = 0;
}

/**
 * @brief CUDA Kernel: Reset neurons to initial state
 */
__global__ void resetNeuronsKernel(DeviceNeuronData neurons) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_id >= neurons.num_neurons) return;
    
    neurons.configuration[neuron_id] = neurons.initial_config[neuron_id];
    neurons.is_open[neuron_id] = true;
    neurons.delay_timer[neuron_id] = 0;
    neurons.pending_emission[neuron_id] = 0;
    neurons.spike_production[neuron_id] = 0;
}

/**
 * @brief High-Performance CUDA Implementation of SN P System Simulator
 * 
 * Key Optimizations:
 * 1. Structure of Arrays (SoA) for coalesced memory access
 * 2. Optimized kernel launch configurations for GPU occupancy
 * 3. Minimized branch divergence via early exits
 * 4. Atomic operations for safe concurrent spike propagation
 * 5. Separate kernels for different phases to reduce complexity
 */
class CudaSnpSimulator : public ISnpSimulator {
private:
    // Host-side configuration
    SnpSystemConfig config;
    int num_neurons = 0;
    int num_synapses = 0;
    int total_rules = 0;
    
    // Device-side data structures
    DeviceNeuronData d_neurons;
    DeviceRuleData d_rules;
    DeviceSynapseData d_synapses;
    
    // Performance tracking
    double total_compute_time_ms = 0.0;
    double total_kernel_time_ms = 0.0;
    double total_memory_time_ms = 0.0;
    int steps_executed = 0;
    
    // Kernel launch configurations
    int neuron_grid_size = 0;
    int synapse_grid_size = 0;
    
public:
    CudaSnpSimulator() = default;
    
    ~CudaSnpSimulator() {
        cleanup();
    }
    
    bool loadSystem(const SnpSystemConfig& sys_config) override {
        try {
            config = sys_config;
            num_neurons = config.neurons.size();
            num_synapses = config.synapses.size();
            total_rules = config.getTotalRulesCount();
            
            // Allocate device memory
            d_neurons.allocate(num_neurons);
            d_rules.allocate(total_rules, num_neurons);
            d_synapses.allocate(num_synapses);
            
            // Prepare and upload neuron data
            uploadNeuronData();
            
            // Prepare and upload rule data
            uploadRuleData();
            
            // Upload synapse data
            uploadSynapseData();
            
            // Calculate optimal kernel launch configurations
            neuron_grid_size = (num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
            synapse_grid_size = (num_synapses + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading system: " << e.what() << std::endl;
            cleanup();
            return false;
        }
    }
    
    void step(int steps = 1) override {
        for (int step_num = 0; step_num < steps; ++step_num) {
            auto start = std::chrono::high_resolution_clock::now();
            
            executeOneStep();
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_compute_time_ms += elapsed.count();
            steps_executed++;
        }
    }
    
    std::vector<int> getLocalState() const override {
        std::vector<int> state(num_neurons);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        CUDA_CHECK(cudaMemcpy(state.data(), d_neurons.configuration, 
                   num_neurons * sizeof(int), cudaMemcpyDeviceToHost));
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        const_cast<CudaSnpSimulator*>(this)->total_memory_time_ms += elapsed.count();
        
        return state;
    }
    
    void reset() override {
        if (num_neurons == 0) return;
        
        // Reset using CUDA kernel for efficiency
        resetNeuronsKernel<<<neuron_grid_size, BLOCK_SIZE>>>(d_neurons);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        total_compute_time_ms = 0.0;
        total_kernel_time_ms = 0.0;
        total_memory_time_ms = 0.0;
        steps_executed = 0;
    }
    
    std::string getPerformanceReport() const override {
        std::ostringstream report;
        report << "=== CUDA SNP Simulator Performance Report ===\n";
        report << "Total Steps: " << steps_executed << "\n";
        report << "Total Compute Time: " << total_compute_time_ms << " ms\n";
        report << "  - Kernel Time: " << total_kernel_time_ms << " ms\n";
        report << "  - Memory Transfer Time: " << total_memory_time_ms << " ms\n";
        if (steps_executed > 0) {
            report << "Average Time per Step: " 
                   << (total_compute_time_ms / steps_executed) << " ms\n";
        }
        report << "System Size: " << num_neurons << " neurons, " 
               << num_synapses << " synapses, " << total_rules << " rules\n";
        report << "Kernel Config: " << neuron_grid_size << " blocks x " 
               << BLOCK_SIZE << " threads (neurons)\n";
        report << "               " << synapse_grid_size << " blocks x " 
               << BLOCK_SIZE << " threads (synapses)\n";
        return report.str();
    }
    
private:
    void cleanup() {
        if (num_neurons > 0) d_neurons.deallocate();
        if (total_rules > 0) d_rules.deallocate();
        if (num_synapses > 0) d_synapses.deallocate();
        num_neurons = 0;
        num_synapses = 0;
        total_rules = 0;
    }
    
    void uploadNeuronData() {
        // Prepare host arrays
        std::vector<int> h_configuration(num_neurons);
        std::vector<char> h_is_open(num_neurons, true);
        std::vector<int> h_delay_timer(num_neurons, 0);
        std::vector<int> h_pending_emission(num_neurons, 0);
        std::vector<int> h_spike_production(num_neurons, 0);
        
        for (int i = 0; i < num_neurons; ++i) {
            h_configuration[i] = config.neurons[i].initial_spikes;
        }
        
        // Upload to device
        CUDA_CHECK(cudaMemcpy(d_neurons.configuration, h_configuration.data(), 
                   num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.initial_config, h_configuration.data(), 
                   num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.is_open, h_is_open.data(), 
                   num_neurons * sizeof(bool), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.delay_timer, h_delay_timer.data(), 
                   num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.pending_emission, h_pending_emission.data(), 
                   num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neurons.spike_production, h_spike_production.data(), 
                   num_neurons * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    void uploadRuleData() {
        // Flatten rules into SoA format
        std::vector<int> h_neuron_id;
        std::vector<int> h_threshold;
        std::vector<int> h_consumed;
        std::vector<int> h_produced;
        std::vector<int> h_delay;
        std::vector<int> h_rule_start_idx(num_neurons);
        std::vector<int> h_rule_count(num_neurons);
        
        int rule_idx = 0;
        for (int neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
            const auto& neuron = config.neurons[neuron_id];
            h_rule_start_idx[neuron_id] = rule_idx;
            h_rule_count[neuron_id] = neuron.rules.size();
            
            for (const auto& rule : neuron.rules) {
                h_neuron_id.push_back(neuron_id);
                h_threshold.push_back(rule.input_threshold);
                h_consumed.push_back(rule.spikes_consumed);
                h_produced.push_back(rule.spikes_produced);
                h_delay.push_back(rule.delay);
                rule_idx++;
            }
        }
        
        // Upload to device
        if (total_rules > 0) {
            CUDA_CHECK(cudaMemcpy(d_rules.neuron_id, h_neuron_id.data(), 
                       total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.input_threshold, h_threshold.data(), 
                       total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_consumed, h_consumed.data(), 
                       total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.spikes_produced, h_produced.data(), 
                       total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rules.delay, h_delay.data(), 
                       total_rules * sizeof(int), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_rules.rule_start_idx, h_rule_start_idx.data(), 
                   num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rules.rule_count, h_rule_count.data(), 
                   num_neurons * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    void uploadSynapseData() {
        if (num_synapses == 0) return;
        
        std::vector<int> h_source_id(num_synapses);
        std::vector<int> h_dest_id(num_synapses);
        std::vector<int> h_weight(num_synapses);
        
        for (int i = 0; i < num_synapses; ++i) {
            h_source_id[i] = config.synapses[i].source_id;
            h_dest_id[i] = config.synapses[i].dest_id;
            h_weight[i] = config.synapses[i].weight;
        }
        
        CUDA_CHECK(cudaMemcpy(d_synapses.source_id, h_source_id.data(), 
                   num_synapses * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_synapses.dest_id, h_dest_id.data(), 
                   num_synapses * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_synapses.weight, h_weight.data(), 
                   num_synapses * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    void executeOneStep() {
        auto kernel_start = std::chrono::high_resolution_clock::now();
        
        // Phase 1: Update neuron status (delay timers, open/closed)
        updateNeuronStatusKernel<<<neuron_grid_size, BLOCK_SIZE>>>(d_neurons);
        CUDA_CHECK(cudaGetLastError());
        
        // Phase 2: Select and apply firing rules
        selectAndApplyRulesKernel<<<neuron_grid_size, BLOCK_SIZE>>>(d_neurons, d_rules);
        CUDA_CHECK(cudaGetLastError());
        
        // Phase 3: Propagate pending emissions (from delayed rules)
        if (num_synapses > 0) {
            propagatePendingEmissionsKernel<<<synapse_grid_size, BLOCK_SIZE>>>(
                d_neurons, d_synapses);
            CUDA_CHECK(cudaGetLastError());
            
            clearPendingEmissionsKernel<<<neuron_grid_size, BLOCK_SIZE>>>(d_neurons);
            CUDA_CHECK(cudaGetLastError());
        }
        
        // Phase 4: Propagate immediate spike production
        if (num_synapses > 0) {
            propagateImmediateSpikesKernel<<<synapse_grid_size, BLOCK_SIZE>>>(
                d_neurons, d_synapses);
            CUDA_CHECK(cudaGetLastError());
        }
        
        // Phase 5: Clear spike production buffer
        clearSpikeProductionKernel<<<neuron_grid_size, BLOCK_SIZE>>>(d_neurons);
        CUDA_CHECK(cudaGetLastError());
        
        // Wait for all kernels to complete
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto kernel_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> kernel_elapsed = kernel_end - kernel_start;
        total_kernel_time_ms += kernel_elapsed.count();
    }
};

// Factory function implementation
std::unique_ptr<ISnpSimulator> createCudaSimulator() {
    return std::make_unique<CudaSnpSimulator>();
}
