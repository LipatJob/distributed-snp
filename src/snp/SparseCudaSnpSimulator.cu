#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <sstream>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <stdexcept>

// --- Constants & Macros ---

#define BLOCK_SIZE 128
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while(0)

// --- Device Structures ---

/**
 * @brief Compressed Rule Vector (RV_Pi)
 * Represents rule properties optimized for direct access by rule index.
 */
struct DeviceRuleVector {
    int* input_threshold;
    int* spikes_consumed;
    int* spikes_produced; // 'p' value in paper
    int* delay; 
    int total_rules;

    void allocate(int n) {
        total_rules = n;
        CUDA_CHECK(cudaMalloc(&input_threshold, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_consumed, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spikes_produced, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&delay, n * sizeof(int)));
    }

    void deallocate() {
        cudaFree(input_threshold);
        cudaFree(spikes_consumed);
        cudaFree(spikes_produced);
        cudaFree(delay);
    }
};

/**
 * @brief Neuron-Rule Map (N_Pi)
 * Maps neuron ID to its range of rules [start, end).
 */
struct DeviceNeuronRuleMap {
    int* rule_start_idx;
    int* rule_count;

    void allocate(int n_neurons) {
        CUDA_CHECK(cudaMalloc(&rule_start_idx, n_neurons * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rule_count, n_neurons * sizeof(int)));
    }

    void deallocate() {
        cudaFree(rule_start_idx);
        cudaFree(rule_count);
    }
};

/**
 * @brief Compressed Synapse Matrix (Sy_Pi)
 * Concept from Section 5.3.3:
 * A matrix of size (Z x q), where Z is the max output degree and q is number of neurons.
 * Sy_Pi[row, col] stores the destination neuron ID for the synapse.
 * -1 (or -2 in paper) represents null/padding.
 * Stored in Column-Major order to allow threads (neurons) to iterate their synapses efficiently.
 */
struct DeviceSynapseMatrix {
    int* matrix; // Flattened Z * q (destination IDs)
    int* weights; // Flattened Z * q (synapse weights)
    int max_out_degree; // Z
    int num_neurons;    // q

    void allocate(int neurons, int z) {
        num_neurons = neurons;
        max_out_degree = z;
        // Allocate flattened arrays
        CUDA_CHECK(cudaMalloc(&matrix, neurons * z * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&weights, neurons * z * sizeof(int)));
    }

    void deallocate() {
        cudaFree(matrix);
        cudaFree(weights);
    }
};

/**
 * @brief Simulation State Vectors
 */
struct DeviceState {
    int* config_vector;     // C_k: spikes per neuron
    int* delay_vector;      // D_k: remaining delay
    int* spiking_vector;    // S_k: Index of active rule per neuron (-1 if none)
    int* pending_emission;  // Spikes waiting to be emitted after delay
    int* initial_config;    // C_0: For reset

    void allocate(int n) {
        CUDA_CHECK(cudaMalloc(&config_vector, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&delay_vector, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&spiking_vector, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&pending_emission, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&initial_config, n * sizeof(int)));
    }

    void deallocate() {
        cudaFree(config_vector);
        cudaFree(delay_vector);
        cudaFree(spiking_vector);
        cudaFree(pending_emission);
        cudaFree(initial_config);
    }
};

// --- Kernels ---

/**
 * @brief Kernel: Calculate Spiking Vector (SV_CALC)
 * Corresponds to Algorithm 2 in paper.
 * * Each thread (neuron) checks its rules.
 * If a rule is applicable, its index is stored in S_k[nid].
 * Deterministic: First applicable rule wins.
 */
__global__ void k_calc_spiking_vector(
    int num_neurons,
    const int* __restrict__ config,      // C_k
    const int* __restrict__ delay,       // D_k
    int* __restrict__ spiking_vector,    // S_k (output)
    const int* __restrict__ rule_start,  // N_Pi
    const int* __restrict__ rule_count,  // N_Pi
    const int* __restrict__ r_threshold  // RV.En (simplification for standard rules)
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= num_neurons) return;

    // Reset spiking vector for this step
    spiking_vector[nid] = -1;

    // Only open neurons can fire
    if (delay[nid] == 0) {
        int current_spikes = config[nid];
        int start = rule_start[nid];
        int count = rule_count[nid];

        // Iterate rules belonging to this neuron
        for (int i = 0; i < count; ++i) {
            int rid = start + i;
            // Simple check: spikes >= threshold
            // (Assuming standard rules; regular expressions can be added here)
            if (current_spikes >= r_threshold[rid]) {
                spiking_vector[nid] = rid;
                break; // Deterministic choice: lowest index fires
            }
        }
    }
}

/**
 * @brief Kernel: Transition Step (STEP) - Compressed Format
 * Corresponds to Algorithm 5 (Compressed format) in paper.
 * * 1. Consumes spikes.
 * 2. Sets delays.
 * 3. Iterates Synapse Matrix (Sy_Pi) to distribute produced spikes.
 */
__global__ void k_step_compressed(
    int num_neurons,
    int max_out_degree,
    int* config,                     // C_k (Read/Write)
    int* delay_vector,               // D_k (Write)
    int* pending_emission,           // Pending spikes (Write)
    const int* __restrict__ spiking_vector, // S_k (Read)
    const int* __restrict__ synapse_matrix, // Sy_Pi (Read)
    const int* __restrict__ synapse_weights, // Synapse weights (Read)
    const int* __restrict__ r_consumed,     // RV.C
    const int* __restrict__ r_produced,     // RV.P
    const int* __restrict__ r_delay         // RV.Delay
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= num_neurons) return;

    // Check if this neuron is firing (S_k has valid rule index)
    // AND neuron is open (delay == 0)
    int active_rule_idx = spiking_vector[nid];

    if (active_rule_idx >= 0 && delay_vector[nid] == 0) {
        int consumed = r_consumed[active_rule_idx];
        int produced = r_produced[active_rule_idx];
        int new_delay = r_delay[active_rule_idx];

        // 1. Consume spikes
        atomicSub(&config[nid], consumed);

        // 2. Handle delay and spike production
        if (new_delay > 0) {
            // Neuron closes for delay period, store pending emission
            delay_vector[nid] = new_delay;
            pending_emission[nid] = produced;
        } else {
            // No delay: send spikes immediately
            for (int i = 0; i < max_out_degree; ++i) {
                int dest_nid = synapse_matrix[nid * max_out_degree + i];
                if (dest_nid >= 0) {
                    int weight = synapse_weights[nid * max_out_degree + i];
                    if (delay_vector[dest_nid] == 0) {
                        atomicAdd(&config[dest_nid], produced * weight);
                    }
                } else {
                    break;
                }
            }
        }
    }
}

/**
 * @brief Kernel: Update Delays and Emit Pending Spikes
 * Decrements delay counters for closed neurons.
 * When delay reaches 0, sends pending emissions.
 */
__global__ void k_update_delays(
    int num_neurons,
    int max_out_degree,
    int* delay_vector,
    int* pending_emission,
    int* config,
    const int* __restrict__ synapse_matrix,
    const int* __restrict__ synapse_weights
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= num_neurons) return;

    if (delay_vector[nid] > 0) {
        delay_vector[nid]--;
        
        // When delay reaches 0, emit pending spikes
        if (delay_vector[nid] == 0 && pending_emission[nid] > 0) {
            int produced = pending_emission[nid];
            pending_emission[nid] = 0; // Clear pending
            
            // Send to all connected neurons
            for (int i = 0; i < max_out_degree; ++i) {
                int dest_nid = synapse_matrix[nid * max_out_degree + i];
                if (dest_nid >= 0) {
                    int weight = synapse_weights[nid * max_out_degree + i];
                    if (delay_vector[dest_nid] == 0) {
                        atomicAdd(&config[dest_nid], produced * weight);
                    }
                } else {
                    break;
                }
            }
        }
    }
}

/**
 * @brief Kernel: Reset
 */
__global__ void k_reset_state(
    int num_neurons,
    int* config,
    const int* initial_config,
    int* delay_vector,
    int* spiking_vector,
    int* pending_emission
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= num_neurons) return;

    config[nid] = initial_config[nid];
    delay_vector[nid] = 0;
    spiking_vector[nid] = -1;
    pending_emission[nid] = 0;
}

// --- Simulator Class ---

class SparseCudaSnpSimulator : public ISnpSimulator {
private:
    // Host Config
    int num_neurons = 0;
    int total_rules = 0;
    int max_out_degree = 0; // Z

    // Device Memory Wrappers
    DeviceState d_state;
    DeviceRuleVector d_rv;
    DeviceNeuronRuleMap d_map;
    DeviceSynapseMatrix d_sy;

    // Performance Metrics
    double total_compute_time = 0.0;
    int steps_executed = 0;
    
    // Launch Config
    int grid_size = 0;

public:
    SparseCudaSnpSimulator() = default;

    ~SparseCudaSnpSimulator() {
        if (num_neurons > 0) {
            d_state.deallocate();
            d_rv.deallocate();
            d_map.deallocate();
            d_sy.deallocate();
        }
    }

    bool loadSystem(const SnpSystemConfig& config) override {
        try {
            num_neurons = config.neurons.size();
            total_rules = config.getTotalRulesCount();
            
            // 1. Analyze Graph for Max Out-Degree (Z)
            std::vector<int> out_degrees(num_neurons, 0);
            for (const auto& syn : config.synapses) {
                if (syn.source_id < num_neurons) {
                    out_degrees[syn.source_id]++;
                }
            }
            max_out_degree = 0;
            for (int d : out_degrees) {
                if (d > max_out_degree) max_out_degree = d;
            }

            // 2. Allocate Device Memory
            d_state.allocate(num_neurons);
            d_rv.allocate(total_rules);
            d_map.allocate(num_neurons);
            d_sy.allocate(num_neurons, max_out_degree);

            // 3. Prepare Host Data for Rule Vector & Map
            std::vector<int> h_threshold;
            std::vector<int> h_consumed;
            std::vector<int> h_produced;
            std::vector<int> h_delay;
            
            std::vector<int> h_rule_start(num_neurons);
            std::vector<int> h_rule_count(num_neurons);

            int current_rule_idx = 0;
            for (int i = 0; i < num_neurons; ++i) {
                h_rule_start[i] = current_rule_idx;
                h_rule_count[i] = config.neurons[i].rules.size();

                for (const auto& r : config.neurons[i].rules) {
                    h_threshold.push_back(r.input_threshold);
                    h_consumed.push_back(r.spikes_consumed);
                    h_produced.push_back(r.spikes_produced);
                    h_delay.push_back(r.delay);
                    current_rule_idx++;
                }
            }

            // 4. Prepare Host Data for Synapse Matrix (Sy_Pi)
            // Flattened: [neuron 0 synapses...][neuron 1 synapses...]
            // Size: num_neurons * max_out_degree
            // Init with -1 (padding)
            std::vector<int> h_synapse_matrix(num_neurons * max_out_degree, -1);
            std::vector<int> h_synapse_weights(num_neurons * max_out_degree, 1);
            
            // Temporary counter for filling rows
            std::vector<int> current_col_fill(num_neurons, 0);

            for (const auto& syn : config.synapses) {
                int src = syn.source_id;
                int dst = syn.dest_id;
                int weight = syn.weight;
                int row_idx = current_col_fill[src];
                
                // Store at: matrix[src * max_out_degree + row_idx]
                h_synapse_matrix[src * max_out_degree + row_idx] = dst;
                h_synapse_weights[src * max_out_degree + row_idx] = weight;
                current_col_fill[src]++;
            }

            // 5. Prepare Initial State
            std::vector<int> h_config(num_neurons);
            for(int i=0; i<num_neurons; ++i) {
                h_config[i] = config.neurons[i].initial_spikes;
            }

            // 6. Upload to Device
            // Rules
            CUDA_CHECK(cudaMemcpy(d_rv.input_threshold, h_threshold.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rv.spikes_consumed, h_consumed.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rv.spikes_produced, h_produced.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rv.delay, h_delay.data(), total_rules * sizeof(int), cudaMemcpyHostToDevice));
            
            // Map
            CUDA_CHECK(cudaMemcpy(d_map.rule_start_idx, h_rule_start.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_map.rule_count, h_rule_count.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice));

            // Synapse Matrix
            CUDA_CHECK(cudaMemcpy(d_sy.matrix, h_synapse_matrix.data(), h_synapse_matrix.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_sy.weights, h_synapse_weights.data(), h_synapse_weights.size() * sizeof(int), cudaMemcpyHostToDevice));

            // State
            CUDA_CHECK(cudaMemcpy(d_state.config_vector, h_config.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_state.initial_config, h_config.data(), num_neurons * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_state.delay_vector, 0, num_neurons * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_state.spiking_vector, -1, num_neurons * sizeof(int)));
            CUDA_CHECK(cudaMemset(d_state.pending_emission, 0, num_neurons * sizeof(int)));

            // 7. Config Launch
            grid_size = (num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "SparseCudaSnpSimulator Initialization Failed: " << e.what() << std::endl;
            return false;
        }
    }

    void step(int steps = 1) override {
        for(int k=0; k<steps; ++k) {
            auto start = std::chrono::high_resolution_clock::now();

            // 1. Update Delays and Emit Pending Spikes (from previous step)
            // This must happen FIRST so neurons can open and pending spikes can be sent
            k_update_delays<<<grid_size, BLOCK_SIZE>>>(
                num_neurons,
                max_out_degree,
                d_state.delay_vector,
                d_state.pending_emission,
                d_state.config_vector,
                d_sy.matrix,
                d_sy.weights
            );
            CUDA_CHECK(cudaGetLastError());

            // 2. Calculate Spiking Vector (Determine active rules)
            // Only open neurons (delay==0) can fire
            k_calc_spiking_vector<<<grid_size, BLOCK_SIZE>>>(
                num_neurons,
                d_state.config_vector,
                d_state.delay_vector,
                d_state.spiking_vector,
                d_map.rule_start_idx,
                d_map.rule_count,
                d_rv.input_threshold
            );
            CUDA_CHECK(cudaGetLastError());

            // 3. Perform Transition (Consume, Produce via Sy_Pi, Set Delays)
            k_step_compressed<<<grid_size, BLOCK_SIZE>>>(
                num_neurons,
                max_out_degree,
                d_state.config_vector,
                d_state.delay_vector,
                d_state.pending_emission,
                d_state.spiking_vector,
                d_sy.matrix,
                d_sy.weights,
                d_rv.spikes_consumed,
                d_rv.spikes_produced,
                d_rv.delay
            );
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaDeviceSynchronize());

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            total_compute_time += elapsed.count();
            steps_executed++;
        }
    }

    std::vector<int> getGlobalState() const override {
        std::vector<int> result(num_neurons);
        CUDA_CHECK(cudaMemcpy(result.data(), d_state.config_vector, num_neurons * sizeof(int), cudaMemcpyDeviceToHost));
        return result;
    }

    void reset() override {
        k_reset_state<<<grid_size, BLOCK_SIZE>>>(
            num_neurons,
            d_state.config_vector,
            d_state.initial_config,
            d_state.delay_vector,
            d_state.spiking_vector,
            d_state.pending_emission
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        steps_executed = 0;
        total_compute_time = 0.0;
    }

    std::string getPerformanceReport() const override {
        std::ostringstream ss;
        ss << "=== Sparse CUDA SNP Simulator (Compressed/Optimized) ===\n";
        ss << "Neurons: " << num_neurons << ", Max Out-Degree (Z): " << max_out_degree << "\n";
        ss << "Total Rules: " << total_rules << "\n";
        ss << "Steps Executed: " << steps_executed << "\n";
        ss << "Total Compute Time: " << total_compute_time << " ms\n";
        if (steps_executed > 0)
            ss << "Avg Time/Step: " << (total_compute_time / steps_executed) << " ms\n";
        ss << "Matrix Strategy: Compressed Synapse Matrix (Sy_Pi)\n";
        return ss.str();
    }
};

// --- Factory Implementation ---

std::unique_ptr<ISnpSimulator> createSparseCudaSimulator() {
    return std::make_unique<SparseCudaSnpSimulator>();
}