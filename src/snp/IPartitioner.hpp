#pragma once

#include "SnpSystemConfig.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

enum class PartitionerType {
    LINEAR,
    LOUVAIN,
    RED_BLUE_BFS,
    METIS,
    CUSTOM
};

/**
 * @brief Interface for partitioning strategies in distributed SNP simulations.
 * 
 * Different algorithms can be used to assign neurons to MPI ranks (partitions)
 * to optimize for different metrics (e.g., minimizing communication volume,
 * balancing computational load, or optimizing for specific network topologies).
 */
class IPartitioner {
public:
    virtual ~IPartitioner() = default;

    /**
     * @brief Computes the partition assignment for each neuron.
     * 
     * @param config The SNP system configuration containing the graph structure.
     * @param num_partitions The number of partitions to create (typically MPI world size).
     * @return std::vector<int> A vector where index is neuron ID and value is partition ID.
     */
    virtual std::vector<int> partition(const SnpSystemConfig& config, int num_partitions) = 0;
    
    /**
     * @brief Returns the type of the partitioning strategy.
     */
    virtual PartitionerType getType() const = 0;

    /**
     * @brief Helper to get string representation of partitioner type.
     *
     * @param PartitionerType The type of partitioner to use.
     */
    static std::string getPartitionerName(PartitionerType type) {
        switch (type) {
            case PartitionerType::LINEAR: return "Linear (Block) Partitioning";
            case PartitionerType::LOUVAIN: return "Louvain Community Detection";
            case PartitionerType::RED_BLUE_BFS: return "Red-Blue Pebbling (BFS)";
            case PartitionerType::METIS: return "METIS Graph Partitioning";
            case PartitionerType::CUSTOM: return "Custom";
            default: return "Unknown";
        }
    }

    /**
     * @brief Parse a partitioner type from a string name.
     * 
     * Accepts case-insensitive names: "linear", "louvain", "red-blue", "redblue", "custom"
     * 
     * @param name The string name of the partitioner
     * @param defaultType The default type to return if parsing fails
     * @return PartitionerType The parsed partitioner type or default
     */
    static PartitionerType parsePartitionerType(const std::string& name, 
                                                  PartitionerType defaultType = PartitionerType::LINEAR) {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        
        if (lower == "linear" || lower == "block") {
            return PartitionerType::LINEAR;
        } else if (lower == "louvain" || lower == "community") {
            return PartitionerType::LOUVAIN;
        } else if (lower == "red-blue" || lower == "redblue" || lower == "rb" || lower == "bfs") {
            return PartitionerType::RED_BLUE_BFS;
        } else if (lower == "custom") {
            return PartitionerType::CUSTOM;
        }
        
        return defaultType;
    }
};
