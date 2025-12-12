/**
 * @file CudaMpiSnpSimulator.cu
 * @brief Distributed CUDA+MPI Implementation of Spiking Neural P System Simulator
 * 
 * This implementation distributes neurons across MPI ranks and uses CUDA for
 * parallel computation on each node's GPU. It minimizes:
 * - Process <-> Process communication (MPI)
 * - Processor <-> GPU transfers
 * - GPU <-> Global Memory accesses
 * 
 * Distribution Strategy:
 * - Neurons are partitioned by ID across MPI ranks
 * - Each rank manages a subset of neurons on its local GPU
 * - Synapses are replicated where needed (source/dest mapping)
 * - Configuration vectors are synchronized via MPI_Allgatherv
 * 
 * Algorithm: Implements C(k+1) = C(k) + St(k+1) ⊙ (Sp(k) + STv(k))
 */

#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <stdexcept>

// ============================================================================
// CUDA Kernels for SNP Operations
// ============================================================================

/**
 * @brief Kernel to select firing rules for each local neuron
 * 
 * For each neuron, finds the applicable rule with highest priority.
 * Deterministic: Higher priority value wins; ties broken by rule order.
 * 
 * @param config Current spike count per neuron
 * @param status Neuron open/closed state (1=open, 0=closed)
 * @param rule_neuron_id Which neuron each rule belongs to
 * @param rule_threshold Spike threshold for rule applicability
 * @param rule_priority Rule priority for selection
 * @param selected_rule Output: selected rule index per neuron (-1 if none)
 * @param n_neurons Number of local neurons
 * @param n_rules Total number of rules
 */
__global__ void kSelectRules(
    const int* config,
    const int* status,
    const int* rule_neuron_id,
    const int* rule_threshold,
    const int* rule_priority,
    int* selected_rule,
    int n_neurons,
    int n_rules)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= n_neurons) return;
    
    // Only open neurons can fire
    if (status[neuron_idx] == 0) {
        selected_rule[neuron_idx] = -1;
        return;
    }
    
    int current_spikes = config[neuron_idx];
    int best_rule = -1;
    int best_priority = -1;
    
    // Linear search for applicable rule with highest priority
    for (int rule_idx = 0; rule_idx < n_rules; rule_idx++) {
        if (rule_neuron_id[rule_idx] == neuron_idx) {
            int threshold = rule_threshold[rule_idx];
            int priority = rule_priority[rule_idx];
            
            // Check applicability and priority
            if (current_spikes >= threshold) {
                if (best_rule == -1 || priority > best_priority) {
                    best_rule = rule_idx;
                    best_priority = priority;
                }
            }
        }
    }
    
    selected_rule[neuron_idx] = best_rule;
}

/**
 * @brief Kernel to apply selected rules: consume spikes and schedule production
 * 
 * @param config Current configuration (spike counts)
 * @param status Neuron status (open/closed)
 * @param delay_timer Remaining delay per neuron
 * @param pending_emission Pending spikes to emit when delay expires
 * @param selected_rule Selected rule per neuron
 * @param rule_consumed Spikes consumed by each rule
 * @param rule_produced Spikes produced by each rule
 * @param rule_delay Delay associated with each rule
 * @param spike_production Output: immediate spike production per neuron
 * @param n_neurons Number of local neurons
 */
__global__ void kApplyRules(
    int* config,
    int* status,
    int* delay_timer,
    int* pending_emission,
    const int* selected_rule,
    const int* rule_consumed,
    const int* rule_produced,
    const int* rule_delay,
    int* spike_production,
    int n_neurons)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= n_neurons) return;
    
    int rule_idx = selected_rule[neuron_idx];
    if (rule_idx < 0) {
        spike_production[neuron_idx] = 0;
        return;
    }
    
    // Consume spikes
    config[neuron_idx] -= rule_consumed[rule_idx];
    
    int delay = rule_delay[rule_idx];
    int produced = rule_produced[rule_idx];
    
    if (delay > 0) {
        // Close neuron and schedule emission
        status[neuron_idx] = 0;
        delay_timer[neuron_idx] = delay;
        pending_emission[neuron_idx] = produced;
        spike_production[neuron_idx] = 0;
    } else {
        // Immediate emission
        spike_production[neuron_idx] = produced;
    }
}

/**
 * @brief Kernel to update neuron status based on delay timers
 * 
 * Decrements delay timers and opens neurons when delay expires.
 */
__global__ void kUpdateStatus(
    int* status,
    int* delay_timer,
    int* pending_emission,
    int* delayed_production,
    int n_neurons)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= n_neurons) return;
    
    if (delay_timer[neuron_idx] > 0) {
        delay_timer[neuron_idx]--;
        
        if (delay_timer[neuron_idx] == 0) {
            // Neuron opens
            status[neuron_idx] = 1;
            // Transfer pending spikes to production
            delayed_production[neuron_idx] = pending_emission[neuron_idx];
            pending_emission[neuron_idx] = 0;
        } else {
            delayed_production[neuron_idx] = 0;
        }
    } else {
        delayed_production[neuron_idx] = 0;
    }
}

/**
 * @brief Kernel to propagate spikes through synapses
 * 
 * For each synapse, adds weighted spikes from source to destination.
 * Only updates neurons if they are open (status = 1).
 * 
 * @param config Configuration vector (modified in-place)
 * @param status Neuron status (for filtering)
 * @param spike_production Spikes produced by each neuron this step
 * @param synapse_src Source neuron ID for each synapse
 * @param synapse_dest Destination neuron ID for each synapse
 * @param synapse_weight Weight for each synapse
 * @param global_to_local Maps global neuron ID to local index
 * @param n_synapses Number of synapses
 * @param n_neurons_global Total neurons in system
 */
__global__ void kPropagateSpikes(
    int* config,
    const int* status,
    const int* spike_production,
    const int* synapse_src,
    const int* synapse_dest,
    const int* synapse_weight,
    const int* global_to_local,
    int n_synapses,
    int n_neurons_global)
{
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= n_synapses) return;
    
    int src_global = synapse_src[synapse_idx];
    int dest_global = synapse_dest[synapse_idx];
    int weight = synapse_weight[synapse_idx];
    
    // Check if destination is local to this rank
    if (dest_global < 0 || dest_global >= n_neurons_global) return;
    int dest_local = global_to_local[dest_global];
    if (dest_local < 0) return; // Not local to this rank
    
    // Only deliver to open neurons
    if (status[dest_local] == 0) return;
    
    // Atomic add to handle multiple synapses to same destination
    int spikes = spike_production[src_global] * weight;
    if (spikes > 0) {
        atomicAdd(&config[dest_local], spikes);
    }
}

// ============================================================================
// CudaMpiSnpSimulator Class Implementation
// ============================================================================

class CudaMpiSnpSimulator : public ISnpSimulator {
private:
    // MPI Configuration
    int rank;
    int size;
    
    // System Configuration (Host)
    SnpSystemConfig config;
    int n_neurons_global;
    int n_neurons_local;
    int local_start_id;  // First neuron ID owned by this rank
    int local_end_id;    // Last neuron ID owned by this rank (exclusive)
    
    // Host State Vectors
    std::vector<int> h_config_local;       // Local neuron spike counts
    std::vector<int> h_config_global;      // Global config (after gather)
    std::vector<int> h_initial_config;     // For reset
    std::vector<int> h_status;             // Local neuron status
    std::vector<int> h_delay_timer;        // Local delay timers
    std::vector<int> h_pending_emission;   // Local pending emissions
    
    // Device State Vectors
    int* d_config_local;
    int* d_config_global;
    int* d_status;
    int* d_delay_timer;
    int* d_pending_emission;
    int* d_spike_production;
    int* d_delayed_production;
    int* d_selected_rule;
    
    // Device Rule Data
    int n_rules_local;
    int* d_rule_neuron_id;
    int* d_rule_threshold;
    int* d_rule_priority;
    int* d_rule_consumed;
    int* d_rule_produced;
    int* d_rule_delay;
    
    // Device Synapse Data
    int n_synapses;
    int* d_synapse_src;
    int* d_synapse_dest;
    int* d_synapse_weight;
    int* d_global_to_local;
    
    // MPI Communication Buffers
    std::vector<int> recvcounts;
    std::vector<int> displs;
    
    // Performance Tracking
    double compute_time_ms = 0.0;
    double comm_time_ms = 0.0;
    int steps_executed = 0;
    
public:
    CudaMpiSnpSimulator() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Set GPU device based on rank
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        if (num_devices > 0) {
            cudaSetDevice(rank % num_devices);
        }
        
        // Initialize device pointers to null
        d_config_local = nullptr;
        d_config_global = nullptr;
        d_status = nullptr;
        d_delay_timer = nullptr;
        d_pending_emission = nullptr;
        d_spike_production = nullptr;
        d_delayed_production = nullptr;
        d_selected_rule = nullptr;
        d_rule_neuron_id = nullptr;
        d_rule_threshold = nullptr;
        d_rule_priority = nullptr;
        d_rule_consumed = nullptr;
        d_rule_produced = nullptr;
        d_rule_delay = nullptr;
        d_synapse_src = nullptr;
        d_synapse_dest = nullptr;
        d_synapse_weight = nullptr;
        d_global_to_local = nullptr;
    }
    
    ~CudaMpiSnpSimulator() {
        cleanup();
    }
    
    bool loadSystem(const SnpSystemConfig& sys_config) override {
        try {
            config = sys_config;
            n_neurons_global = config.neurons.size();
            
            // Distribute neurons across ranks (1D decomposition)
            computeLocalNeuronRange();
            
            // Initialize host state vectors
            initializeHostState();
            
            // Allocate and initialize device memory
            allocateDeviceMemory();
            
            // Upload rule and synapse data to GPU
            uploadRuleData();
            uploadSynapseData();
            
            // Upload initial state to GPU
            uploadStateToDevice();
            
            return true;
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "Error loading system: " << e.what() << std::endl;
            }
            return false;
        }
    }
    
    void step(int steps = 1) override {
        for (int step_num = 0; step_num < steps; step_num++) {
            auto start_step = std::chrono::high_resolution_clock::now();
            
            // Phase 0: Reset temporary arrays
            cudaMemset(d_spike_production, 0, n_neurons_global * sizeof(int));
            cudaMemset(d_selected_rule, -1, n_neurons_local * sizeof(int));
            // Note: d_delayed_production is set by updateNeuronStatus, no need to zero
            
            // Phase 1: Update neuron status (GPU)
            auto start_compute = std::chrono::high_resolution_clock::now();
            updateNeuronStatus();
            
            // Phase 2: Select firing rules (GPU)
            selectFiringRules();
            
            // Phase 3: Apply rules (GPU)
            applySelectedRules();
            auto end_compute = std::chrono::high_resolution_clock::now();
            compute_time_ms += std::chrono::duration<double, std::milli>(
                end_compute - start_compute).count();
            
            // Phase 4: Synchronize global configuration (MPI)
            auto start_comm = std::chrono::high_resolution_clock::now();
            synchronizeGlobalConfiguration();
            auto end_comm = std::chrono::high_resolution_clock::now();
            comm_time_ms += std::chrono::duration<double, std::milli>(
                end_comm - start_comm).count();
            
            // Phase 5: Propagate spikes through synapses (GPU)
            start_compute = std::chrono::high_resolution_clock::now();
            propagateSpikes();
            cudaDeviceSynchronize();
            end_compute = std::chrono::high_resolution_clock::now();
            compute_time_ms += std::chrono::duration<double, std::milli>(
                end_compute - start_compute).count();
            
            steps_executed++;
        }
    }
    
    std::vector<int> getLocalState() const override {
        // Download current local configuration from GPU
        std::vector<int> local_state(n_neurons_local);
        cudaMemcpy(local_state.data(), d_config_local,
                   n_neurons_local * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Gather global state from all ranks
        std::vector<int> global_state(n_neurons_global);
        MPI_Allgatherv(
            local_state.data(), n_neurons_local, MPI_INT,
            global_state.data(), 
            const_cast<int*>(recvcounts.data()), 
            const_cast<int*>(displs.data()), 
            MPI_INT, MPI_COMM_WORLD);
        
        return global_state;
    }
    
    void reset() override {
        // Reset to initial configuration
        h_config_local = h_initial_config;
        h_status.assign(n_neurons_local, 1);
        h_delay_timer.assign(n_neurons_local, 0);
        h_pending_emission.assign(n_neurons_local, 0);
        
        // Upload reset state to device
        uploadStateToDevice();
        
        // Reset performance counters
        compute_time_ms = 0.0;
        comm_time_ms = 0.0;
        steps_executed = 0;
    }
    
    std::string getPerformanceReport() const override {
        std::ostringstream report;
        report << "=== CUDA+MPI SNP Simulator Performance Report ===\n";
        report << "MPI Rank: " << rank << " / " << size << "\n";
        report << "Local Neurons: " << n_neurons_local 
               << " (IDs " << local_start_id << "-" << (local_end_id-1) << ")\n";
        report << "Total Steps: " << steps_executed << "\n";
        report << "Compute Time: " << compute_time_ms << " ms\n";
        report << "Communication Time: " << comm_time_ms << " ms\n";
        if (steps_executed > 0) {
            report << "Avg Compute/Step: " 
                   << (compute_time_ms / steps_executed) << " ms\n";
            report << "Avg Comm/Step: " 
                   << (comm_time_ms / steps_executed) << " ms\n";
            double total_time = compute_time_ms + comm_time_ms;
            report << "Compute %: " 
                   << (100.0 * compute_time_ms / total_time) << "%\n";
            report << "Comm %: " 
                   << (100.0 * comm_time_ms / total_time) << "%\n";
        }
        return report.str();
    }
    
private:
    /**
     * @brief Compute which neurons belong to this rank
     * 
     * Uses 1D decomposition: distributes neurons evenly across ranks.
     * Remainder neurons assigned to first ranks.
     */
    void computeLocalNeuronRange() {
        int base_neurons = n_neurons_global / size;
        int remainder = n_neurons_global % size;
        
        // Ranks 0 to remainder-1 get base+1 neurons
        // Ranks remainder to size-1 get base neurons
        if (rank < remainder) {
            n_neurons_local = base_neurons + 1;
            local_start_id = rank * n_neurons_local;
        } else {
            n_neurons_local = base_neurons;
            local_start_id = remainder * (base_neurons + 1) +
                           (rank - remainder) * base_neurons;
        }
        local_end_id = local_start_id + n_neurons_local;
        
        // Prepare MPI communication parameters
        recvcounts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; r++) {
            int n_local_r = (r < remainder) ? (base_neurons + 1) : base_neurons;
            recvcounts[r] = n_local_r;
            if (r == 0) {
                displs[r] = 0;
            } else {
                displs[r] = displs[r-1] + recvcounts[r-1];
            }
        }
    }
    
    /**
     * @brief Initialize host-side state vectors
     */
    void initializeHostState() {
        h_config_local.resize(n_neurons_local);
        h_config_global.resize(n_neurons_global);
        h_status.resize(n_neurons_local);
        h_delay_timer.resize(n_neurons_local);
        h_pending_emission.resize(n_neurons_local);
        
        // Initialize local configuration from global config
        for (int i = 0; i < n_neurons_local; i++) {
            int global_id = local_start_id + i;
            h_config_local[i] = config.neurons[global_id].initial_spikes;
            h_status[i] = 1; // All neurons start open
            h_delay_timer[i] = 0;
            h_pending_emission[i] = 0;
        }
        
        h_initial_config = h_config_local;
    }
    
    /**
     * @brief Allocate GPU memory for state and computation
     */
    void allocateDeviceMemory() {
        // State vectors
        cudaMalloc(&d_config_local, n_neurons_local * sizeof(int));
        cudaMalloc(&d_config_global, n_neurons_global * sizeof(int));
        cudaMalloc(&d_status, n_neurons_local * sizeof(int));
        cudaMalloc(&d_delay_timer, n_neurons_local * sizeof(int));
        cudaMalloc(&d_pending_emission, n_neurons_local * sizeof(int));
        cudaMalloc(&d_spike_production, n_neurons_global * sizeof(int));
        cudaMalloc(&d_delayed_production, n_neurons_local * sizeof(int));
        cudaMalloc(&d_selected_rule, n_neurons_local * sizeof(int));
        
        // Initialize temporary arrays to zero/invalid values
        cudaMemset(d_spike_production, 0, n_neurons_global * sizeof(int));
        cudaMemset(d_selected_rule, -1, n_neurons_local * sizeof(int));
        // Note: d_delayed_production is always set by updateNeuronStatus
    }
    
    /**
     * @brief Upload initial state from host to device
     */
    void uploadStateToDevice() {
        cudaMemcpy(d_config_local, h_config_local.data(),
                   n_neurons_local * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_status, h_status.data(),
                   n_neurons_local * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_delay_timer, h_delay_timer.data(),
                   n_neurons_local * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pending_emission, h_pending_emission.data(),
                   n_neurons_local * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    /**
     * @brief Upload rule data to GPU
     * 
     * Flattens all rules into arrays indexed by rule ID.
     * Only includes rules for local neurons.
     */
    void uploadRuleData() {
        std::vector<int> h_rule_neuron_id;
        std::vector<int> h_rule_threshold;
        std::vector<int> h_rule_priority;
        std::vector<int> h_rule_consumed;
        std::vector<int> h_rule_produced;
        std::vector<int> h_rule_delay;
        
        // Flatten rules from local neurons
        for (int local_idx = 0; local_idx < n_neurons_local; local_idx++) {
            int global_id = local_start_id + local_idx;
            const auto& neuron = config.neurons[global_id];
            
            for (const auto& rule : neuron.rules) {
                h_rule_neuron_id.push_back(local_idx);
                h_rule_threshold.push_back(rule.input_threshold);
                h_rule_priority.push_back(rule.priority);
                h_rule_consumed.push_back(rule.spikes_consumed);
                h_rule_produced.push_back(rule.spikes_produced);
                h_rule_delay.push_back(rule.delay);
            }
        }
        
        n_rules_local = h_rule_neuron_id.size();
        
        if (n_rules_local > 0) {
            cudaMalloc(&d_rule_neuron_id, n_rules_local * sizeof(int));
            cudaMalloc(&d_rule_threshold, n_rules_local * sizeof(int));
            cudaMalloc(&d_rule_priority, n_rules_local * sizeof(int));
            cudaMalloc(&d_rule_consumed, n_rules_local * sizeof(int));
            cudaMalloc(&d_rule_produced, n_rules_local * sizeof(int));
            cudaMalloc(&d_rule_delay, n_rules_local * sizeof(int));
            
            cudaMemcpy(d_rule_neuron_id, h_rule_neuron_id.data(),
                       n_rules_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rule_threshold, h_rule_threshold.data(),
                       n_rules_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rule_priority, h_rule_priority.data(),
                       n_rules_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rule_consumed, h_rule_consumed.data(),
                       n_rules_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rule_produced, h_rule_produced.data(),
                       n_rules_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rule_delay, h_rule_delay.data(),
                       n_rules_local * sizeof(int), cudaMemcpyHostToDevice);
        }
    }
    
    /**
     * @brief Upload synapse data to GPU
     * 
     * Creates a global-to-local mapping for efficient synapse processing.
     */
    void uploadSynapseData() {
        n_synapses = config.synapses.size();
        
        // Create global-to-local neuron ID mapping
        std::vector<int> h_global_to_local(n_neurons_global, -1);
        for (int local_idx = 0; local_idx < n_neurons_local; local_idx++) {
            int global_id = local_start_id + local_idx;
            h_global_to_local[global_id] = local_idx;
        }
        
        cudaMalloc(&d_global_to_local, n_neurons_global * sizeof(int));
        cudaMemcpy(d_global_to_local, h_global_to_local.data(),
                   n_neurons_global * sizeof(int), cudaMemcpyHostToDevice);
        
        if (n_synapses > 0) {
            std::vector<int> h_synapse_src(n_synapses);
            std::vector<int> h_synapse_dest(n_synapses);
            std::vector<int> h_synapse_weight(n_synapses);
            
            for (int i = 0; i < n_synapses; i++) {
                h_synapse_src[i] = config.synapses[i].source_id;
                h_synapse_dest[i] = config.synapses[i].dest_id;
                h_synapse_weight[i] = config.synapses[i].weight;
            }
            
            cudaMalloc(&d_synapse_src, n_synapses * sizeof(int));
            cudaMalloc(&d_synapse_dest, n_synapses * sizeof(int));
            cudaMalloc(&d_synapse_weight, n_synapses * sizeof(int));
            
            cudaMemcpy(d_synapse_src, h_synapse_src.data(),
                       n_synapses * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_synapse_dest, h_synapse_dest.data(),
                       n_synapses * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_synapse_weight, h_synapse_weight.data(),
                       n_synapses * sizeof(int), cudaMemcpyHostToDevice);
        }
    }
    
    /**
     * @brief Update neuron status based on delay timers (GPU)
     */
    void updateNeuronStatus() {
        if (n_neurons_local == 0) return;
        
        int threads = 256;
        int blocks = (n_neurons_local + threads - 1) / threads;
        
        kUpdateStatus<<<blocks, threads>>>(
            d_status,
            d_delay_timer,
            d_pending_emission,
            d_delayed_production,
            n_neurons_local
        );
    }
    
    /**
     * @brief Select firing rules for each local neuron (GPU)
     */
    void selectFiringRules() {
        if (n_neurons_local == 0 || n_rules_local == 0) return;
        
        int threads = 256;
        int blocks = (n_neurons_local + threads - 1) / threads;
        
        kSelectRules<<<blocks, threads>>>(
            d_config_local,
            d_status,
            d_rule_neuron_id,
            d_rule_threshold,
            d_rule_priority,
            d_selected_rule,
            n_neurons_local,
            n_rules_local
        );
    }
    
    /**
     * @brief Apply selected rules to neurons (GPU)
     */
    void applySelectedRules() {
        if (n_neurons_local == 0) return;
        
        int threads = 256;
        int blocks = (n_neurons_local + threads - 1) / threads;
        
        kApplyRules<<<blocks, threads>>>(
            d_config_local,
            d_status,
            d_delay_timer,
            d_pending_emission,
            d_selected_rule,
            d_rule_consumed,
            d_rule_produced,
            d_rule_delay,
            d_spike_production + local_start_id,
            n_neurons_local
        );
    }
    
    /**
     * @brief Synchronize global configuration across all ranks (MPI)
     * 
     * Uses MPI_Allgatherv to collect local configurations into global view.
     * This is necessary because spike propagation needs to know all neuron states.
     */
    void synchronizeGlobalConfiguration() {
        // Ensure all GPU operations are complete before downloading
        cudaDeviceSynchronize();
        
        // Download local config from GPU
        cudaMemcpy(h_config_local.data(), d_config_local,
                   n_neurons_local * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Gather all local configs into global config
        MPI_Allgatherv(
            h_config_local.data(), n_neurons_local, MPI_INT,
            h_config_global.data(), recvcounts.data(), displs.data(), MPI_INT,
            MPI_COMM_WORLD
        );
        
        // Upload global config to GPU
        cudaMemcpy(d_config_global, h_config_global.data(),
                   n_neurons_global * sizeof(int), cudaMemcpyHostToDevice);
        
        // Gather spike production (needed for synapse propagation)
        std::vector<int> h_spike_prod_local(n_neurons_local);
        cudaMemcpy(h_spike_prod_local.data(), d_spike_production + local_start_id,
                   n_neurons_local * sizeof(int), cudaMemcpyDeviceToHost);
        
        std::vector<int> h_spike_prod_global(n_neurons_global);
        MPI_Allgatherv(
            h_spike_prod_local.data(), n_neurons_local, MPI_INT,
            h_spike_prod_global.data(), recvcounts.data(), displs.data(), MPI_INT,
            MPI_COMM_WORLD
        );
        
        // Gather delayed production from all ranks
        std::vector<int> h_delayed_prod_local(n_neurons_local);
        cudaMemcpy(h_delayed_prod_local.data(), d_delayed_production,
                   n_neurons_local * sizeof(int), cudaMemcpyDeviceToHost);
        
        std::vector<int> h_delayed_prod_global(n_neurons_global);
        MPI_Allgatherv(
            h_delayed_prod_local.data(), n_neurons_local, MPI_INT,
            h_delayed_prod_global.data(), recvcounts.data(), displs.data(), MPI_INT,
            MPI_COMM_WORLD
        );
        
        // All ranks now have the complete delayed production, add it to spike production
        for (int i = 0; i < n_neurons_global; i++) {
            h_spike_prod_global[i] += h_delayed_prod_global[i];
        }
        
        // Upload global spike production to GPU
        cudaMemcpy(d_spike_production, h_spike_prod_global.data(),
                   n_neurons_global * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    /**
     * @brief Propagate spikes through synapses (GPU)
     */
    void propagateSpikes() {
        if (n_synapses == 0) return;
        
        int threads = 256;
        int blocks = (n_synapses + threads - 1) / threads;
        
        kPropagateSpikes<<<blocks, threads>>>(
            d_config_local,
            d_status,
            d_spike_production,
            d_synapse_src,
            d_synapse_dest,
            d_synapse_weight,
            d_global_to_local,
            n_synapses,
            n_neurons_global
        );
    }
    
    /**
     * @brief Free all GPU memory
     */
    void cleanup() {
        cudaFree(d_config_local);
        cudaFree(d_config_global);
        cudaFree(d_status);
        cudaFree(d_delay_timer);
        cudaFree(d_pending_emission);
        cudaFree(d_spike_production);
        cudaFree(d_delayed_production);
        cudaFree(d_selected_rule);
        cudaFree(d_rule_neuron_id);
        cudaFree(d_rule_threshold);
        cudaFree(d_rule_priority);
        cudaFree(d_rule_consumed);
        cudaFree(d_rule_produced);
        cudaFree(d_rule_delay);
        cudaFree(d_synapse_src);
        cudaFree(d_synapse_dest);
        cudaFree(d_synapse_weight);
        cudaFree(d_global_to_local);
    }
};

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<ISnpSimulator> createCudaMpiSimulator() {
    return std::make_unique<CudaMpiSnpSimulator>();
}
