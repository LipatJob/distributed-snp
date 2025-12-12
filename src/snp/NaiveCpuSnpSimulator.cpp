#include "ISnpSimulator.hpp"
#include "SnpSystemConfig.hpp"
#include <vector>
#include <memory>
#include <sstream>
#include <chrono>

/**
 * @brief Naive CPU Implementation of SN P System Simulator
 * 
 * This implementation prioritizes simplicity and readability over performance.
 * It follows the mathematical model described in the reference document:
 * - Configuration Vector C(k): spike counts per neuron
 * - Status Vector St(k): open/closed state per neuron
 * - Firing rules with delays
 * 
 * Design Philosophy:
 * - Simple linear search for rule selection
 * - No optimization tricks
 * - Clear variable names matching the mathematical notation
 * - Step-by-step execution matching the algorithm description
 */
class NaiveCpuSnpSimulator : public ISnpSimulator {
private:
    // System Configuration (immutable after load)
    SnpSystemConfig config;
    
    // Current State Vectors
    std::vector<int> configuration;      // C(k): Current spike count per neuron
    std::vector<int> initial_config;     // C(0): For reset
    std::vector<bool> neuron_is_open;    // St(k): true = open, false = closed
    std::vector<int> delay_timer;        // Remaining delay ticks per neuron
    std::vector<int> pending_emission;   // Spikes scheduled for emission when delay expires
    
    // Performance Tracking
    double total_compute_time_ms = 0.0;
    int steps_executed = 0;
    
    /**
     * @brief Structure to track a rule's location and properties
     */
    struct RuleReference {
        int neuron_id;
        int rule_index;
        const SnpRule* rule;
        
        RuleReference(int nid, int ridx, const SnpRule* r)
            : neuron_id(nid), rule_index(ridx), rule(r) {}
    };
    
public:
    NaiveCpuSnpSimulator() = default;
    
    bool loadSystem(const SnpSystemConfig& sys_config) override {
        config = sys_config;
        
        // Initialize configuration vector C(0)
        configuration.clear();
        configuration.resize(config.neurons.size(), 0);
        
        for (size_t i = 0; i < config.neurons.size(); ++i) {
            configuration[i] = config.neurons[i].initial_spikes;
        }
        
        // Save initial state for reset
        initial_config = configuration;
        
        // Initialize status vector (all neurons start open)
        neuron_is_open.clear();
        neuron_is_open.resize(config.neurons.size(), true);
        
        // Initialize delay timers (all at 0)
        delay_timer.clear();
        delay_timer.resize(config.neurons.size(), 0);
        
        // Initialize pending emissions (all at 0)
        pending_emission.clear();
        pending_emission.resize(config.neurons.size(), 0);
        
        return true;
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
        return configuration;
    }
    
    void reset() override {
        configuration = initial_config;
        neuron_is_open.assign(neuron_is_open.size(), true);
        delay_timer.assign(delay_timer.size(), 0);
        pending_emission.assign(pending_emission.size(), 0);
        total_compute_time_ms = 0.0;
        steps_executed = 0;
    }
    
    std::string getPerformanceReport() const override {
        std::ostringstream report;
        report << "=== Naive CPU Simulator Performance Report ===\n";
        report << "Total Steps: " << steps_executed << "\n";
        report << "Total Compute Time: " << total_compute_time_ms << " ms\n";
        if (steps_executed > 0) {
            report << "Average Time per Step: " 
                   << (total_compute_time_ms / steps_executed) << " ms\n";
        }
        report << "Note: This is a naive implementation without optimization.\n";
        return report.str();
    }
    
private:
    /**
     * @brief Execute one complete simulation step
     * 
     * Algorithm (from reference document):
     * 1. Update neuron status based on delay timers
     * 2. For each open neuron, select applicable rule (deterministic by order)
     * 3. Apply selected rules (consume spikes, schedule production)
     * 4. Propagate spikes through synapses to open neurons
     */
    void executeOneStep() {
        // Phase 1: Update neuron open/closed status based on delays
        updateNeuronStatus();
        
        // Phase 2: Select rules for each open neuron (deterministic by order)
        std::vector<RuleReference> selected_rules;
        selectFiringRules(selected_rules);
        
        // Phase 3: Apply rules - consume spikes and compute production
        std::vector<int> spike_production(config.neurons.size(), 0);
        applyRules(selected_rules, spike_production);
        
        // Phase 4: Propagate spikes through synapses
        propagateSpikes(spike_production);
    }
    
    /**
     * @brief Update neuron status: decrement delay timers and open neurons when ready
     * 
     * When a delay expires (timer reaches 0), the neuron opens and emits its pending spikes.
     */
    void updateNeuronStatus() {
        for (size_t neuron_id = 0; neuron_id < config.neurons.size(); neuron_id++) {
            if (delay_timer[neuron_id] > 0) {
                delay_timer[neuron_id]--;
                
                // Neuron opens when delay reaches 0
                if (delay_timer[neuron_id] == 0) {
                    neuron_is_open[neuron_id] = true;
                    // Note: Pending spikes will be propagated in the propagateSpikes phase
                }
            }
        }
    }
    
    /**
     * @brief Select one rule per neuron based on applicability and order
     * 
     * Deterministic Rule Selection (from reference):
     * - Only open neurons can fire rules
     * - A rule is applicable if: current_spikes >= threshold
     * - If multiple rules applicable, the first rule
     */
    void selectFiringRules(std::vector<RuleReference>& selected_rules) {
        selected_rules.clear();
        
        for (size_t neuron_id = 0; neuron_id < config.neurons.size(); ++neuron_id) {
            // Skip closed neurons
            if (!neuron_is_open[neuron_id]) {
                continue;
            }
            
            const auto& neuron = config.neurons[neuron_id];
            int current_spikes = configuration[neuron_id];
            
            // Find applicable rules for this neuron
            for (size_t rule_idx = 0; rule_idx < neuron.rules.size(); ++rule_idx) {
                const auto& rule = neuron.rules[rule_idx];
                
                // Check if rule is applicable
                if (current_spikes >= rule.input_threshold) {
                    // For simplicity, select the first applicable rule
                    selected_rules.emplace_back(neuron_id, rule_idx, &rule);
                    break; // Only one rule per neuron
                }
            }
        }
    }
    
    /**
     * @brief Apply selected rules: consume spikes, set delays, compute production
     * 
     * For each firing rule:
     * - Consume spikes from source neuron
     * - If delay > 0: close neuron, set timer, and schedule spike emission
     * - If delay = 0: add spikes to immediate production
     */
    void applyRules(const std::vector<RuleReference>& selected_rules,
                    std::vector<int>& spike_production) {
        
        for (const auto& rule_ref : selected_rules) {
            int neuron_id = rule_ref.neuron_id;
            const SnpRule* rule = rule_ref.rule;
            
            // Consume spikes from neuron
            configuration[neuron_id] -= rule->spikes_consumed;
            
            // Handle spike production based on delay
            if (rule->delay > 0) {
                // Rule has delay: close neuron and schedule emission
                neuron_is_open[neuron_id] = false;
                delay_timer[neuron_id] = rule->delay;
                
                // Schedule spikes to be emitted when delay expires
                pending_emission[neuron_id] = rule->spikes_produced;
            } else {
                // No delay: emit spikes immediately
                spike_production[neuron_id] += rule->spikes_produced;
            }
        }
    }
    
    /**
     * @brief Propagate spikes through synapses to target neurons
     * 
     * This handles both:
     * 1. Immediate spike production (from rules with delay=0)
     * 2. Delayed spike emission (when neuron's delay expires)
     * 
     * For each synapse (source -> dest):
     * - If dest neuron is open: add (weight * source_production) spikes
     * - If dest neuron is closed: spikes are lost (per reference document)
     */
    void propagateSpikes(const std::vector<int>& spike_production) {
        // First, emit any pending spikes from neurons whose delays just expired
        for (size_t neuron_id = 0; neuron_id < config.neurons.size(); ++neuron_id) {
            // If neuron just opened (delay was > 0 last step and is now 0) and has pending emission
            if (neuron_is_open[neuron_id] && pending_emission[neuron_id] > 0) {
                // Propagate the pending spikes through synapses
                for (const auto& synapse : config.synapses) {
                    if (synapse.source_id == static_cast<int>(neuron_id)) {
                        int dest = synapse.dest_id;
                        int weight = synapse.weight;
                        int spikes_to_send = pending_emission[neuron_id] * weight;
                        
                        // Only open neurons can receive spikes
                        if (neuron_is_open[dest]) {
                            configuration[dest] += spikes_to_send;
                        }
                    }
                }
                
                // Clear pending emission after propagation
                pending_emission[neuron_id] = 0;
            }
        }
        
        // Then, propagate immediate spike production (from rules with delay=0)
        for (const auto& synapse : config.synapses) {
            int source = synapse.source_id;
            int dest = synapse.dest_id;
            int weight = synapse.weight;
            
            int spikes_to_send = spike_production[source] * weight;
            
            // Only open neurons can receive spikes
            if (spikes_to_send > 0 && neuron_is_open[dest]) {
                configuration[dest] += spikes_to_send;
            }
            // If dest is closed, spikes are lost
        }
    }
};

// Factory function implementation
std::unique_ptr<ISnpSimulator> createNaiveCpuSimulator() {
    return std::make_unique<NaiveCpuSnpSimulator>();
}
