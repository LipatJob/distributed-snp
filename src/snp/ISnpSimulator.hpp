#pragma once
#include "SnpSystemConfig.hpp"
#include <vector>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

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
     * @brief Retrieves the current spike count of all neurons in the system.
     * * @return std::vector<int> Current global configuration vector C(k).
     */
    virtual std::vector<int> getGlobalState() const = 0;

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

// Factory functions for dependency injection / ease of testing
std::unique_ptr<ISnpSimulator> createNaiveCpuSimulator();
std::unique_ptr<ISnpSimulator> createCudaSimulator();
std::unique_ptr<ISnpSimulator> createCudaMpiSimulator();
