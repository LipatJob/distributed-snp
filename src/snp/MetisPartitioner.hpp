#pragma once

#include "IPartitioner.hpp"
#include "SnpSystemConfig.hpp"
#include <vector>

/**
 * @brief Partitioner implementation using the METIS library.
 * 
 * Uses METIS (k-way graph partitioning) to minimize edge cuts (communication volume)
 * while balancing the number of vertices (computational load) across partitions.
 */
class MetisPartitioner : public IPartitioner {
public:
    MetisPartitioner() = default;
    ~MetisPartitioner() override = default;

    std::vector<int> partition(const SnpSystemConfig& config, int num_partitions) override;

    PartitionerType getType() const override {
        return PartitionerType::METIS;
    }
};
