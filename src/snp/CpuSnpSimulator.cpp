/*
 * CpuSnpSimulator.cpp
 * * CPU-Only Spiking Neural P System Simulator
 * * Focus: Readability, Reference Implementation, Verification
 */

#include "SnpSimulator.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>

// -------------------------------------------------------------------------
// CPU SIMULATOR CLASS
// -------------------------------------------------------------------------

class CpuSnpSimulator : public ISnpSimulator {
private:
    // System Definition
    int numNeurons;
    int numRules;

    // Neuron State (Configuration C^k)
    // Stored densely: C[i] = number of spikes in neuron i 
    std::vector<SpikeCount> neuronSpikes;
    
    // Rule State
    // ruleDelays[i]: The static delay defined for rule i
    // ruleCurrentDelays[i]: The countdown timer for rule i (0 = ready)
    std::vector<int> ruleDelays;
    std::vector<int> ruleCurrentDelays;

    // Rule Logic
    // Maps rule_idx -> neuron_idx (Which neuron owns this rule?)
    std::vector<int> ruleOwners;
    
    // Matrix M_Pi Representation (Parsed for Readability)
    // We split the matrix conceptually into Consumption (negative) and Production (positive)
    // for clearer logic, though they come from the same input arrays.
    std::vector<int> consumption; // consumption[rule_idx] = spikes to consume
    
    // Production List: Maps Rule -> [TargetNeuron, SpikeAmount]
    struct Synapse {
        int targetNeuron;
        int spikes;
    };
    std::vector<std::vector<Synapse>> productionRules;

    // Simulation State
    uint64_t currentTick = 0;

public:
    CpuSnpSimulator() = default;
    ~CpuSnpSimulator() = default;

    void Initialize(
        int globalNumNeurons, 
        int globalNumRules,
        const std::vector<int>& ruleOwnersMap,
        const std::vector<float>& matrixValues,
        const std::vector<int>& matrixColIndices,
        const std::vector<int>& matrixRowPtrs,
        const std::vector<SpikeCount>& initialSpikes
    ) override {
        numNeurons = globalNumNeurons;
        numRules = globalNumRules;
        neuronSpikes = initialSpikes;
        ruleOwners = ruleOwnersMap;

        // Initialize delays to 0 (assuming no delay info passed in interface for now)
        ruleDelays.assign(numRules, 0); 
        ruleCurrentDelays.assign(numRules, 0);
        consumption.assign(numRules, 0);
        productionRules.resize(numRules);

        // Parse CSR Matrix M_Pi
        // We iterate through every rule (row) and parse its connections (cols)
        for (int r = 0; r < numRules; ++r) {
            int start = matrixRowPtrs[r];
            int end = matrixRowPtrs[r+1];

            for (int i = start; i < end; ++i) {
                int targetNeuron = matrixColIndices[i];
                int val = static_cast<int>(matrixValues[i]);

                if (val < 0) {
                    // Negative value in M_Pi means consumption [cite: 102]
                    // The rule consumes 'val' spikes from its owner
                    consumption[r] = -val; 
                } else if (val > 0) {
                    // Positive value in M_Pi means production [cite: 103]
                    // The rule produces 'val' spikes sent to 'targetNeuron'
                    productionRules[r].push_back({targetNeuron, val});
                }
            }
        }
    }

    void Step(int steps) override {
        // Temporary buffer to store the net change for each neuron for the current step.
        // We cannot modify 'neuronSpikes' directly while iterating because SN P systems 
        // are synchronized: all neurons read state at t, and write state at t+1. [cite: 76]
        std::vector<SpikeCount> delta(numNeurons);

        // Vector to track which rules fired this step
        std::vector<bool> ruleFired(numRules);

        for (int s = 0; s < steps; ++s) {
            // Reset per-step buffers
            std::fill(delta.begin(), delta.end(), 0);
            std::fill(ruleFired.begin(), ruleFired.end(), false);

            // -----------------------------------------------------------------
            // PHASE 1: SELECTION & FIRING
            // Iterate over all rules to see which ones are applicable.
            // -----------------------------------------------------------------
            for (int r = 0; r < numRules; ++r) {
                int owner = ruleOwners[r];
                
                // Check Delay Status
                if (ruleCurrentDelays[r] > 0) {
                    // Rule is busy/delayed. It cannot fire.
                    // Decrement logic happens at end of tick.
                    continue; 
                }

                // Check Applicability (Spikes >= Consumption)
                // Note: Real SN P systems use regex E. Here we check simple threshold. [cite: 71]
                int spikesNeeded = consumption[r];
                if (neuronSpikes[owner] >= spikesNeeded) {
                    // Rule applies!
                    ruleFired[r] = true;
                    
                    // Mark consumption (remove spikes from owner)
                    // Note: If multiple rules define consumption for the same neuron, 
                    // this naive logic sums them. Deterministic systems usually have 
                    // disjoint conditions or priorities.
                    delta[owner] -= spikesNeeded;

                    // If we had delay logic, we would set the timer here:
                    // ruleCurrentDelays[r] = ruleDelays[r]; // [cite: 79]
                }
            }

            // -----------------------------------------------------------------
            // PHASE 2: EXECUTION & PRODUCTION
            // Process the firing rules to send spikes to neighbors.
            // -----------------------------------------------------------------
            for (int r = 0; r < numRules; ++r) {
                if (ruleFired[r]) {
                    // Send spikes to all target neurons connected by synapses
                    for (const auto& synapse : productionRules[r]) {
                        delta[synapse.targetNeuron] += synapse.spikes;
                    }
                }
            }
            
            // -----------------------------------------------------------------
            // PHASE 3: STATE UPDATE
            // Apply the computed deltas to the system state. [cite: 127]
            // -----------------------------------------------------------------
            for (int n = 0; n < numNeurons; ++n) {
                neuronSpikes[n] += delta[n];
                
                // Sanity check: Spikes cannot be negative
                if (neuronSpikes[n] < 0) {
                    std::cerr << "Warning: Neuron " << n << " has negative spikes!" << std::endl;
                    neuronSpikes[n] = 0;
                }
            }
            
            // Update Delays (Decrement countdowns)
            for (int r = 0; r < numRules; ++r) {
                 if (ruleCurrentDelays[r] > 0) {
                     ruleCurrentDelays[r]--;
                 }
            }

            currentTick++;
        }
    }

    SimulationState GetState() const override {
        SimulationState state;
        state.neuronSpikes = neuronSpikes;
        state.neuronDelays = ruleCurrentDelays; // Simplified mapping
        state.currentTick = currentTick;
        return state;
    }

    void Reset() override {
        currentTick = 0;
        std::fill(neuronSpikes.begin(), neuronSpikes.end(), 0);
        std::fill(ruleCurrentDelays.begin(), ruleCurrentDelays.end(), 0);
        // Note: In a real app, you'd store the initial configuration and restore it here.
    }
};

std::unique_ptr<ISnpSimulator> makeCpuSimulator() {
    return std::make_unique<CpuSnpSimulator>();
}