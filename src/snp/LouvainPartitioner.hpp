#pragma once

#include "IPartitioner.hpp"
#include "SnpSystemConfig.hpp"
#include <vector>
#include <map>

class LouvainPartitioner : public IPartitioner
{
public:
  /**
   * @brief Partitions the SNP system into k partitions using the Louvain algorithm.
   */
  std::vector<int> partition(const SnpSystemConfig &config, int num_partitions) override;

  PartitionerType getType() const override
  {
    return PartitionerType::LOUVAIN;
  }

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
