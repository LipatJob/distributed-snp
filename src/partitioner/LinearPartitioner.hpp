#pragma once

#include "IPartitioner.hpp"
#include <vector>
#include <cmath>

/**
 * @brief Simple linear (block) partitioning strategy.
 * 
 * Assigns neurons to partitions sequentially based on their ID.
 * Rank 0 gets indices [0, N/P), Rank 1 gets [N/P, 2N/P), etc.
 * This is the default "naive" distribution which assumes spatial locality
 * in the original index order or simply balances neuron counts.
 */
class LinearPartitioner : public IPartitioner {
public:
    std::vector<int> partition(const SnpSystemConfig& config, int num_partitions) override {
        int num_neurons = config.neurons.size();
        std::vector<int> assignments(num_neurons);
        
        if (num_partitions <= 0) return assignments;

        // Calculate chunk size (ceiling division to ensure coverage)
        // Actually, standard block distribution usually does N/P.
        // Let's do simple block distribution.
        int chunk_size = (num_neurons + num_partitions - 1) / num_partitions;

        for (int i = 0; i < num_neurons; ++i) {
            int partition_id = i / chunk_size;
            if (partition_id >= num_partitions) {
                partition_id = num_partitions - 1;
            }
            assignments[i] = partition_id;
        }

        return assignments;
    }

    PartitionerType getType() const override {
        return PartitionerType::LINEAR;
    }
};
