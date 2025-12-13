#pragma once

#include "SnpSystemConfig.hpp"
#include <vector>
#include <map>

class LouvainPartitioner {
public:
    /**
     * @brief Partitions the SNP system into k partitions using the Louvain algorithm.
     * 
     * @param config The SNP system configuration.
     * @param num_partitions The desired number of partitions (e.g., MPI size).
     * @return std::vector<int> A vector of size num_neurons, where index is neuron ID and value is partition ID.
     */
    static std::vector<int> partition(const SnpSystemConfig& config, int num_partitions);

private:
    struct Graph {
        int num_nodes;
        std::vector<std::vector<std::pair<int, double>>> adj; // Adjacency list (neighbor, weight)
        std::vector<double> node_weights; // Sum of weights incident to node
        double total_weight; // Sum of all edge weights (m)
    };

    static Graph buildGraph(const SnpSystemConfig& config);
    static std::vector<int> runLouvain(const Graph& graph);
    static std::vector<int> balancePartitions(const std::vector<int>& communities, int num_nodes, int num_partitions);
};
