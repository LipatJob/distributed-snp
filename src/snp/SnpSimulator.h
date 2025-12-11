#ifndef SNP_SIMULATOR_H
#define SNP_SIMULATOR_H

#include <vector>
#include <memory>
#include "MatrixOps.h"

// --- Helper Structures for SN P Logic ---

struct SnpRule {
    int id;
    int neuronIndex;       // The neuron this rule belongs to
    double spikesConsumed; // c
    double spikesProduced; // p
    int delay;             // d
    double threshold;      // Minimum spikes needed to fire
    bool isForgetting;     // True for forgetting rules (λ)

    bool canFire(double currentSpikes) const {
        return currentSpikes >= threshold;
    }
};

struct Neuron {
    int id;
    double spikes;
    int remainingDelay; // 0 means open
};

// --- The Simulator Class ---

class SnpSimulator {
private:
    std::unique_ptr<IMatrixOps<double>> matrixOps;
    
    // System dimensions
    size_t numNeurons; // m
    size_t numRules;   // n

    // Host Data (CPU side logic)
    std::vector<Neuron> neurons;
    std::vector<SnpRule> rules;
    
    // Matrix Representation
    std::vector<double> host_M_Pi; 

    // Vectors for current step
    std::vector<double> host_C;    // Configuration (1 x m)
    std::vector<double> host_Iv;   // Indicator Vector (1 x n)
    std::vector<double> host_St;   // Status Vector (1 x m)
    
    // Temporary buffers
    std::vector<double> buffer_Delta;
    std::vector<double> buffer_MaskedDelta;
    std::vector<double> buffer_NextC;

public:
    SnpSimulator(BackendType backend, size_t m, size_t n);

    // Load a rule into the system and update M_Pi matrix
    void addRule(int ruleIdx, int sourceNeuron, int destNeuron, 
                 double consumed, double produced, double threshold, 
                 int delay, bool isForgetting = false);

    void setInitialSpikes(int neuronIdx, double count);
    
    // Core simulation step
    void step(int tick);

    // Getters for testing
    const std::vector<double>& getConfiguration() const { return host_C; }
    const std::vector<double>& getStatusVector() const { return host_St; }
    const std::vector<double>& getIndicatorVector() const { return host_Iv; }
    const std::vector<Neuron>& getNeurons() const { return neurons; }
    double getSpikeCount(int neuronIdx) const { return neurons[neuronIdx].spikes; }
    bool isNeuronOpen(int neuronIdx) const { return neurons[neuronIdx].remainingDelay == 0; }
};

#endif // SNP_SIMULATOR_H
