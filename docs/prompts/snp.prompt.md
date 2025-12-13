Create a parallel and distributed implementation of the spiking neural p system using CUDA and MPI. See ./snp_explanation.tex to learn more about what Spiking Neural P Systems are

Each node has a T4 GPU. Assume that the input size is large. The system must distribute the work among different nodes to maximize performance. It must also minimize the communication Process <-> Process, Processor <-> GPU, and GPU <-> Global Memory.

Create an CudaMpiSnpSimulator class that implements the following interface. Make sure to keep the code simple and readable. Do not change anything but the implementation of the CudaMpiSnpSimulator class. You can add helper functions and classes as needed, but do not change the interface.

ISnpSimulator.hpp
```
#pragma once
#include "SnpSystemConfig.hpp"
#include <vector>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <memory>

// Forward declaration for system configuration structures
struct SnpSystemConfig; 

/**
 * @brief Interface for a Spiking Neural P System Simulator.
 * * Designed for ease of use and testability. Implementation details regarding
 * CUDA/MPI are hidden from the consumer.
 */
class ISnpSimulator {
public:
    virtual ~ISnpSimulator() = default;

    /**
     * @brief Loads the SN P System configuration and initializes device memory.
     * * This method handles the distribution of the system matrix (partitioning)
     * based on the MPI rank.
     * * @param config The structural definition of the SN P System (rules, synapses).
     * @return true if initialization was successful.
     */
    virtual bool loadSystem(const SnpSystemConfig& config) = 0;

    /**
     * @brief Advances the simulation by a specified number of time steps.
     * * Implements the matrix equation: C(k+1) = C(k) + Sp(k) * M
     * Handles GPU kernel launches and MPI synchronization automatically.
     * * @param steps Number of ticks to simulate.
     */
    virtual void step(int steps = 1) = 0;

    /**
     * @brief Retrieves the current spike count of local neurons.
     * * @return std::vector<int> Local partition of the configuration vector C.
     */
    virtual std::vector<int> getLocalState() const = 0;

    /**
     * @brief Resets the system to the initial configuration (C0).
     */
    virtual void reset() = 0;

    /**
     * @brief Returns performance metrics (compute time vs communication time).
     * useful for identifying bottlenecks.
     */
    virtual std::string getPerformanceReport() const = 0;
};
```

SnpSystemConfig.hpp

```
#pragma once
#include <vector>
#include <algorithm>

// --- Data Definitions ---

struct SnpRule {
    int input_threshold = 0;
    int spikes_consumed = 0;
    int spikes_produced = 0;
    int delay = 0;
    int priority = 0;

    // RESTORED: Default constructor required for struct usage in vectors/other structs
    SnpRule() = default;

    // Convenience Constructor for the Builder/Tests
    SnpRule(int threshold, int consumed, int produced, int d = 0, int p = 0)
        : input_threshold(threshold), spikes_consumed(consumed), 
          spikes_produced(produced), delay(d), priority(p) {}
};

struct SnpNeuron {
    int id;
    int initial_spikes;
    std::vector<SnpRule> rules;
    bool is_output;

    // Default constructor
    SnpNeuron() : id(-1), initial_spikes(0), is_output(false) {}

    // Convenience constructor
    SnpNeuron(int id, int spikes = 0, bool output = false)
        : id(id), initial_spikes(spikes), is_output(output) {}
};

struct SnpSynapse {
    int source_id;
    int dest_id;
    int weight;

    // Default constructor
    SnpSynapse() : source_id(-1), dest_id(-1), weight(1) {}

    // Convenience constructor
    SnpSynapse(int src, int dst, int w = 1)
        : source_id(src), dest_id(dst), weight(w) {}
};

struct SnpSystemConfig {
    std::vector<SnpNeuron> neurons;
    std::vector<SnpSynapse> synapses;

    size_t getTotalRulesCount() const {
        size_t count = 0;
        for (const auto& n : neurons) count += n.rules.size();
        return count;
    }
};
```