Create a parallel and distributed implementation of the spiking neural p system using CUDA and MPI. I've attached the necessary papers for reference. Assume that the SNP system is deterministic for now. Each node has a T4 GPU. Assume that the input size is large. The system must distribute the work among different nodes to maximize performance. It must also minimize the communication between Processes, Processor <-> GPU, and GPU <-> Global Memory


Create an SnpSimulator class that implements the following interface:

```
#pragma once
#include <vector>
#include <cstdint>

// Forward declaration for device pointers if needed in other contexts
// using distinct types helps type safety
using SpikeCount = int32_t;

struct SimulationState {
    std::vector<SpikeCount> neuronSpikes;
    std::vector<int> neuronDelays;
    uint64_t currentTick;
};

class ISnpSimulator {
public:
    virtual ~ISnpSimulator() = default;

    /**
     * Initialize the simulator with the system definition.
     * @param adjMatrix The sparse transition matrix M_Pi (CSR format typically)
     * @param initialSpikes Initial configuration vector C^0
     * @param rules Definition of rules per neuron (regex/thresholds)
     */
    virtual void Initialize(
        int numNeurons, 
        int numRules,
        const std::vector<int>& ruleOwners, // Map rule_idx -> neuron_idx
        const std::vector<float>& matrixValues,
        const std::vector<int>& matrixColIndices,
        const std::vector<int>& matrixRowPtrs,
        const std::vector<SpikeCount>& initialSpikes
    ) = 0;

    /**
     * Advances the simulation by a specific number of time steps.
     * This encapsulates the heavy compute loop to minimize host interaction.
     */
    virtual void Step(int steps = 1) = 0;

    /**
     * Retrieves the current state from the GPU to the Host.
     * Useful for verification (unit tests) or checkpoints.
     */
    virtual SimulationState GetState() const = 0;
    
    /**
     * Resets the simulation to t=0 without reallocating GPU memory.
     */
    virtual void Reset() = 0;
};
```