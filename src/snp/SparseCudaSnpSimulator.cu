#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <sstream>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <algorithm>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while(0)

// Optimal block size for modern GPUs (multiple of warp size 32)
constexpr int BLOCK_SIZE = 64;

/**
 * @brief Sparse CUDA Implementation of SN P System Simulator
 * 
 * This implementation uses sparse matrix representation as described in:
 * "Sparse Spiking Neural-like Membrane Systems on Graphics Processing Units"
 * (Hernández-Tello et al., 2024)
 * 
 * Key optimizations:
 * - Compressed Synapse Matrix (Sy_π) instead of full transition matrix
 * - ELL-like format for efficient GPU memory access
 * - Rule Vector (RV_π) with CSR-like indexing
 * - Neuron-Rule Map Vector (N_π) for fast rule lookup
 * 
 * Sparse representation benefits:
 * - Reduced memory footprint (especially for sparse graphs)
 * - Better cache utilization
 * - Coalesced memory access patterns
 */

/**
 * @brief Device structure for Rule Vector (RV_π)
 * 
 * Stores rule information in CSR-like format for efficient access.
 * Rules are organized per-neuron for locality.
 */
struct DeviceRuleVector {
    int* Ei;              // Regular expression type (0=min, 1=exact)
    int* En;              // Regular expression multiplicity
    int* c;               // Spikes consumed
    int* p;               // Spikes produced
    int* d;               // Delay
    int* nid;             // Neuron ID containing this rule
    int total_rules;
    
    void allocate(int num_rules) {
        total_rules = num_rules;
        CUDA_CHECK(cudaMalloc(&Ei, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&En, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&c, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&p, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d, num_rules * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&nid, num_rules * sizeof(int)));
    }
    
    void deallocate() {
        cudaFree(Ei);
        cudaFree(En);
        cudaFree(c);
        cudaFree(p);
        cudaFree(d);
        cudaFree(nid);
    }
};

/**
 * @brief Device structure for Neuron-Rule Map Vector (N_π)
 * 
 * Maps each neuron to its rule set using start indices.
 * N_π[i] = start index of rules for neuron i
 * N_π[i+1] - N_π[i] = number of rules for neuron i
 */
struct DeviceNeuronRuleMap {
    int* rule_start_idx;  // Start index for each neuron's rules
    int num_neurons;
    
    void allocate(int n) {
        num_neurons = n;
        // Allocate n+1 to store end index
        CUDA_CHECK(cudaMalloc(&rule_start_idx, (n + 1) * sizeof(int)));
    }
    
    void deallocate() {
        cudaFree(rule_start_idx);
    }
};

/**
 * @brief Device structure for Compressed Synapse Matrix (Sy_π)
 * 
 * Stores synapse connectivity and weights in ELL-like format:
 * - Rows = max out-degree of any neuron
 * - Columns = number of neurons
 * - Values = target neuron IDs (or -1 for padding) and weights
 * 
 * This eliminates redundant storage of spike production values.
 */
struct DeviceSynapseMatrix {
    int* target_neurons;  // Flattened 2D array: [max_out_degree][num_neurons]
    int* weights;         // Flattened 2D array: [max_out_degree][num_neurons]
    int max_out_degree;   // Maximum number of outgoing synapses
    int num_neurons;
    
    void allocate(int max_deg, int n) {
        max_out_degree = max_deg;
        num_neurons = n;
        CUDA_CHECK(cudaMalloc(&target_neurons, max_deg * n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weights, max_deg * n * sizeof(int)));
    }
    
    void deallocate() {
        cudaFree(target_neurons);
        cudaFree(weights);
    }
    
    __device__ int getTarget(int row, int col) const {
        return target_neurons[row * num_neurons + col];
    }
    
    __device__ int getWeight(int row, int col) const {
        return weights[row * num_neurons + col];
    }
};

/**
 * @brief Device structure for Configuration and State Vectors
 */
struct DeviceNeuronState {
    int* C;               // Configuration vector (spike counts)
    int* C_initial;       // Initial configuration for reset
    int* D;               // Delays vector (0=open, >0=closed)
    int* S;               // Spiking vector (selected rule per neuron, -1=none)
    int* pending_spikes;  // Spikes to emit when delay expires
    int num_neurons;
    
    void allocate(int n) {
        num_neurons = n;
        CUDA_CHECK(cudaMalloc(&C, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&C_initial, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&D, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&S, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&pending_spikes, n * sizeof(int)));
    }
    
    void deallocate() {
        cudaFree(C);
        cudaFree(C_initial);
        cudaFree(D);
        cudaFree(S);
        cudaFree(pending_spikes);
    }
};

/**
 * @brief CUDA Kernel: Calculate Spiking Vector (SV_CALC)
 * 
 * Algorithm from paper (Algorithm 2):
 * - Each thread handles one neuron
 * - Select first applicable rule based on regular expression
 * - Store selected rule index in spiking vector
 * 
 * Regular expression check (3 types supported):
 * 1. e* (Ei=0, En=0): Always applicable
 * 2. e+ (Ei=0, En=1): Applicable if n >= 1
 * 3. e^n (Ei=1, En=n): Applicable if n == En
 */
__global__ void calcSpikingVectorKernel(
    DeviceNeuronState state,
    DeviceRuleVector rules,
    DeviceNeuronRuleMap rule_map
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (nid >= state.num_neurons) return;
    
    // Initialize spiking vector entry to -1 (no rule selected)
    state.S[nid] = -1;
    
    // Only open neurons can fire rules
    if (state.D[nid] != 0) return;
    
    int current_spikes = state.C[nid];
    int rule_start = rule_map.rule_start_idx[nid];
    int rule_end = rule_map.rule_start_idx[nid + 1];
    
    // Find first applicable rule (deterministic order)
    for (int r = rule_start; r < rule_end; ++r) {
        int Ei = rules.Ei[r];  // Regex type
        int En = rules.En[r];  // Regex multiplicity
        
        // Check if regular expression matches current spike count
        bool regtype_1_2 = (Ei == 0) && (current_spikes >= En);
        bool regtype_3 = (Ei == 1) && (current_spikes == En);
        
        if (regtype_1_2 || regtype_3) {
            // Rule is applicable, select it
            state.S[nid] = r;
            break;  // Only select first applicable rule
        }
    }
}

/**
 * @brief CUDA Kernel: Consume spikes from firing rules
 */
__global__ void consumeSpikesKernel(
    DeviceNeuronState state,
    DeviceRuleVector rules
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (nid >= state.num_neurons) return;
    
    // Check if neuron has an active rule
    if (state.S[nid] < 0) return;
    
    int rid = state.S[nid];
    int c = rules.c[rid];
    
    // Consume spikes
    atomicSub(&state.C[nid], c);
}

/**
 * @brief CUDA Kernel: Propagate spikes (immediate or after delay)
 */
__global__ void propagateSpikesKernel(
    DeviceNeuronState state,
    DeviceRuleVector rules,
    DeviceSynapseMatrix synapses
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (nid >= state.num_neurons) return;
    
    int spikes_to_send = 0;
    
    // Check if neuron just opened (delay expired) and has pending spikes
    if (state.D[nid] == 0 && state.pending_spikes[nid] > 0) {
        spikes_to_send = state.pending_spikes[nid];
        state.pending_spikes[nid] = 0;
    }
    // Or if neuron fired a rule with no delay
    else if (state.S[nid] >= 0) {
        int rid = state.S[nid];
        int delay = rules.d[rid];
        int p = rules.p[rid];
        
        if (delay == 0) {
            // Immediate spike production
            spikes_to_send = p;
        } else {
            // Schedule spikes for later emission
            state.pending_spikes[nid] = p;
        }
    }
    
    // Propagate spikes through synapses if any to send
    if (spikes_to_send > 0) {
        for (int i = 0; i < synapses.max_out_degree; ++i) {
            int target_nid = synapses.getTarget(i, nid);
            
            // Check for padding (null synapse)
            if (target_nid < 0) break;
            
            int weight = synapses.getWeight(i, nid);
            
            // Only deliver spikes to open neurons
            if (state.D[target_nid] == 0) {
                atomicAdd(&state.C[target_nid], spikes_to_send * weight);
            }
        }
    }
}

/**
 * @brief CUDA Kernel: Update Delays Vector (UPDATE_DELAYS)
 * 
 * Algorithm from paper:
 * - Set delay for neurons with active delayed rules (BEFORE decrement)
 * - Decrement delay counters for closed neurons
 * - Each thread handles one neuron
 */
__global__ void updateDelaysKernel(
    DeviceNeuronState state,
    DeviceRuleVector rules
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (nid >= state.num_neurons) return;
    
    // If a rule with delay was selected, set the delay counter
    if (state.S[nid] >= 0) {
        int rid = state.S[nid];
        int delay = rules.d[rid];
        
        if (delay > 0) {
            state.D[nid] = delay;
        }
    }
    // Otherwise, decrement delay counter for closed neurons
    else if (state.D[nid] > 0) {
        state.D[nid]--;
    }
}

/**
 * @brief CUDA Kernel: Reset spiking vector
 */
__global__ void resetSpikingVectorKernel(
    DeviceNeuronState state
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (nid >= state.num_neurons) return;
    
    state.S[nid] = -1;
}

/**
 * @brief CUDA Kernel: Check if computation can continue
 * 
 * Returns true if any rule is applicable or any neuron is closed (delayed)
 */
__global__ void checkContinueKernel(
    DeviceNeuronState state,
    bool* can_continue
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (nid >= state.num_neurons) return;
    
    // Check if neuron has active rule or is delayed
    if (state.S[nid] >= 0 || state.D[nid] > 0) {
        *can_continue = true;
    }
}

/**
 * @brief Sparse CUDA SNP Simulator Implementation
 * 
 * Uses compressed sparse representation for efficient GPU execution.
 */
class SparseCudaSnpSimulator : public ISnpSimulator {
private:
    // Host-side configuration
    SnpSystemConfig config;
    int num_neurons;
    int num_rules;
    int max_out_degree;
    
    // Device memory structures
    DeviceNeuronState d_state;
    DeviceRuleVector d_rules;
    DeviceNeuronRuleMap d_rule_map;
    DeviceSynapseMatrix d_synapses;
    
    // Performance tracking
    double total_compute_time_ms = 0.0;
    double total_comm_time_ms = 0.0;
    int steps_executed = 0;
    
    // Host-side buffer for state retrieval
    std::vector<int> host_config;
    
public:
    SparseCudaSnpSimulator() = default;
    
    ~SparseCudaSnpSimulator() {
        // Clean up device memory
        if (num_neurons > 0) {
            d_state.deallocate();
            d_rules.deallocate();
            d_rule_map.deallocate();
            d_synapses.deallocate();
        }
    }
    
    bool loadSystem(const SnpSystemConfig& sys_config) override {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            config = sys_config;
            num_neurons = static_cast<int>(config.neurons.size());
            num_rules = static_cast<int>(config.getTotalRulesCount());
            
            // Compute max out-degree for synapse matrix
            max_out_degree = computeMaxOutDegree();
            
            // Allocate device memory
            d_state.allocate(num_neurons);
            d_rules.allocate(num_rules);
            d_rule_map.allocate(num_neurons);
            d_synapses.allocate(max_out_degree, num_neurons);
            
            // Initialize data structures
            initializeRuleVector();
            initializeNeuronRuleMap();
            initializeSynapseMatrix();
            initializeState();
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_comm_time_ms += elapsed.count();
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading system: " << e.what() << std::endl;
            return false;
        }
    }
    
    void step(int steps = 1) override {
        for (int i = 0; i < steps; ++i) {
            executeOneStep();
        }
    }
    
    std::vector<int> getGlobalState() const override {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<int> result(num_neurons);
        CUDA_CHECK(cudaMemcpy(result.data(), d_state.C, 
                              num_neurons * sizeof(int), 
                              cudaMemcpyDeviceToHost));
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        const_cast<SparseCudaSnpSimulator*>(this)->total_comm_time_ms += elapsed.count();
        
        return result;
    }
    
    void reset() override {
        // Copy initial configuration back
        CUDA_CHECK(cudaMemcpy(d_state.C, d_state.C_initial,
                              num_neurons * sizeof(int),
                              cudaMemcpyDeviceToDevice));
        
        // Reset delays, spiking vector, and pending spikes
        CUDA_CHECK(cudaMemset(d_state.D, 0, num_neurons * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_state.S, -1, num_neurons * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_state.pending_spikes, 0, num_neurons * sizeof(int)));
        
        total_compute_time_ms = 0.0;
        total_comm_time_ms = 0.0;
        steps_executed = 0;
    }
    
    std::string getPerformanceReport() const override {
        std::ostringstream report;
        report << "=== Sparse CUDA Simulator Performance Report ===\n";
        report << "Total Steps: " << steps_executed << "\n";
        report << "Total Compute Time: " << total_compute_time_ms << " ms\n";
        report << "Total Communication Time: " << total_comm_time_ms << " ms\n";
        if (steps_executed > 0) {
            report << "Average Compute per Step: " 
                   << (total_compute_time_ms / steps_executed) << " ms\n";
            report << "Average Communication per Step: " 
                   << (total_comm_time_ms / steps_executed) << " ms\n";
        }
        report << "Compression Stats:\n";
        report << "  Neurons: " << num_neurons << "\n";
        report << "  Rules: " << num_rules << "\n";
        report << "  Max Out-Degree: " << max_out_degree << "\n";
        report << "  Synapse Matrix Size: " 
               << (max_out_degree * num_neurons * sizeof(int)) / 1024.0 << " KB\n";
        return report.str();
    }
    
private:
    /**
     * @brief Execute one simulation step
     * 
     * Follows Algorithm 1 from the paper:
     * 1. Update delays (decrement counters)
     * 2. Calculate spiking vector (SV_CALC)
     * 3. Consume spikes from selected rules
     * 4. Update delays (set new delays for firing rules)
     * 5. Propagate spikes (immediate or scheduled)
     */
    void executeOneStep() {
        auto start = std::chrono::high_resolution_clock::now();
        
        int num_blocks = (num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Step 1: Calculate spiking vector (which rules can fire)
        calcSpikingVectorKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_state, d_rules, d_rule_map
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Step 2: Consume spikes from selected rules
        consumeSpikesKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_state, d_rules
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Step 3: Update delays (set delays for rules that just fired)
        updateDelaysKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_state, d_rules
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Step 4: Propagate spikes (immediate or emit pending after delay)
        propagateSpikesKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_state, d_rules, d_synapses
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Synchronize to ensure step completion
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_compute_time_ms += elapsed.count();
        steps_executed++;
    }
    
    /**
     * @brief Compute maximum out-degree of the synapse graph
     */
    int computeMaxOutDegree() {
        std::vector<int> out_degree(num_neurons, 0);
        
        for (const auto& synapse : config.synapses) {
            out_degree[synapse.source_id]++;
        }
        
        int max_deg = 0;
        for (int deg : out_degree) {
            max_deg = std::max(max_deg, deg);
        }
        
        return max_deg;
    }
    
    /**
     * @brief Initialize Rule Vector (RV_π) on device
     * 
     * Builds CSR-like structure with rule information.
     */
    void initializeRuleVector() {
        std::vector<int> Ei(num_rules);
        std::vector<int> En(num_rules);
        std::vector<int> c(num_rules);
        std::vector<int> p(num_rules);
        std::vector<int> d(num_rules);
        std::vector<int> nid(num_rules);
        
        int rule_idx = 0;
        for (int neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
            const auto& neuron = config.neurons[neuron_id];
            
            for (const auto& rule : neuron.rules) {
                // All rules use threshold as minimum (>=)
                // Type 1 (e*): threshold=0 → Ei=0, En=0 (always fires)
                // Type 2 (a^n+): threshold=n → Ei=0, En=n (fires if spikes >= n)
                // We use Ei=0 for all rules (minimum match, not exact)
                Ei[rule_idx] = 0;
                En[rule_idx] = rule.input_threshold;
                
                c[rule_idx] = rule.spikes_consumed;
                p[rule_idx] = rule.spikes_produced;
                d[rule_idx] = rule.delay;
                nid[rule_idx] = neuron_id;
                
                rule_idx++;
            }
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_rules.Ei, Ei.data(), 
                              num_rules * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rules.En, En.data(), 
                              num_rules * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rules.c, c.data(), 
                              num_rules * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rules.p, p.data(), 
                              num_rules * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rules.d, d.data(), 
                              num_rules * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rules.nid, nid.data(), 
                              num_rules * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    /**
     * @brief Initialize Neuron-Rule Map Vector (N_π) on device
     * 
     * N_π[i] = starting index of rules for neuron i
     */
    void initializeNeuronRuleMap() {
        std::vector<int> rule_start_idx(num_neurons + 1);
        
        rule_start_idx[0] = 0;
        for (int i = 0; i < num_neurons; ++i) {
            rule_start_idx[i + 1] = rule_start_idx[i] + 
                                    static_cast<int>(config.neurons[i].rules.size());
        }
        
        CUDA_CHECK(cudaMemcpy(d_rule_map.rule_start_idx, rule_start_idx.data(),
                              (num_neurons + 1) * sizeof(int), 
                              cudaMemcpyHostToDevice));
    }
    
    /**
     * @brief Initialize Compressed Synapse Matrix (Sy_π) on device
     * 
     * ELL-like format:
     * - Rows = max out-degree
     * - Columns = neurons
     * - Values = target neuron IDs (-1 for padding) and weights
     */
    void initializeSynapseMatrix() {
        std::vector<int> synapse_matrix(max_out_degree * num_neurons, -1);
        std::vector<int> weight_matrix(max_out_degree * num_neurons, 1);
        
        // Track how many synapses added per neuron
        std::vector<int> synapse_count(num_neurons, 0);
        
        for (const auto& synapse : config.synapses) {
            int src = synapse.source_id;
            int dst = synapse.dest_id;
            int weight = synapse.weight;
            int row = synapse_count[src];
            
            if (row < max_out_degree) {
                synapse_matrix[row * num_neurons + src] = dst;
                weight_matrix[row * num_neurons + src] = weight;
                synapse_count[src]++;
            }
        }
        
        CUDA_CHECK(cudaMemcpy(d_synapses.target_neurons, synapse_matrix.data(),
                              max_out_degree * num_neurons * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_synapses.weights, weight_matrix.data(),
                              max_out_degree * num_neurons * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    
    /**
     * @brief Initialize neuron state on device
     */
    void initializeState() {
        std::vector<int> initial_config(num_neurons);
        std::vector<int> delays(num_neurons, 0);
        std::vector<int> spiking(num_neurons, -1);
        std::vector<int> pending(num_neurons, 0);
        
        for (int i = 0; i < num_neurons; ++i) {
            initial_config[i] = config.neurons[i].initial_spikes;
        }
        
        // Copy initial configuration
        CUDA_CHECK(cudaMemcpy(d_state.C, initial_config.data(),
                              num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_state.C_initial, initial_config.data(),
                              num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        
        // Initialize delays, spiking vector, and pending spikes
        CUDA_CHECK(cudaMemcpy(d_state.D, delays.data(),
                              num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_state.S, spiking.data(),
                              num_neurons * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_state.pending_spikes, pending.data(),
                              num_neurons * sizeof(int), cudaMemcpyHostToDevice));
    }
};

// Factory function implementation
std::unique_ptr<ISnpSimulator> createSparseCudaSimulator() {
    return std::make_unique<SparseCudaSnpSimulator>();
}
