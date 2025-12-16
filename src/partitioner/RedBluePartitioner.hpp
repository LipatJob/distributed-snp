#pragma once

#include "IPartitioner.hpp"
#include <queue>
#include <set>
#include <algorithm>
#include <cmath>

/**
 * @brief Partitioning strategy inspired by the Red-Blue Pebbling Game.
 * 
 * In the Red-Blue Pebbling Game, "Red" pebbles represent data in fast (local) memory,
 * and "Blue" pebbles represent data in slow (remote) memory. The goal is to minimize
 * Blue pebble operations (I/O or communication).
 * 
 * This partitioner uses a BFS-based expansion (wavefront) to form partitions.
 * This tends to create "clumped" partitions with minimal surface area (boundary),
 * thereby minimizing the number of synapses crossing partition boundaries (communication).
 */
class RedBluePartitioner : public IPartitioner {
public:
    std::vector<int> partition(const SnpSystemConfig& config, int num_partitions) override {
        int num_neurons = config.neurons.size();
        std::vector<int> assignments(num_neurons, -1);
        
        // Build adjacency list for faster traversal
        std::vector<std::vector<int>> adj(num_neurons);
        for (const auto& synapse : config.synapses) {
            // Undirected graph for partitioning purposes usually works better 
            // to keep connected components together
            adj[synapse.source_id].push_back(synapse.dest_id);
            adj[synapse.dest_id].push_back(synapse.source_id);
        }

        int target_size = (num_neurons + num_partitions - 1) / num_partitions;
        std::vector<bool> visited(num_neurons, false);
        
        for (int p = 0; p < num_partitions; ++p) {
            int current_partition_size = 0;
            std::queue<int> q;

            // Find a seed node for this partition
            // Heuristic: Pick the first unvisited node (or could be random)
            int seed = -1;
            for (int i = 0; i < num_neurons; ++i) {
                if (!visited[i]) {
                    seed = i;
                    break;
                }
            }

            if (seed == -1) break; // All nodes assigned

            q.push(seed);
            visited[seed] = true;
            assignments[seed] = p;
            current_partition_size++;

            // Expand "Red" region (local partition)
            while (!q.empty() && current_partition_size < target_size) {
                int u = q.front();
                q.pop();

                for (int v : adj[u]) {
                    if (!visited[v]) {
                        visited[v] = true;
                        assignments[v] = p;
                        current_partition_size++;
                        q.push(v);
                        
                        if (current_partition_size >= target_size) break;
                    }
                }
            }
        }

        // Handle any disconnected components or leftovers
        for (int i = 0; i < num_neurons; ++i) {
            if (assignments[i] == -1) {
                assignments[i] = num_partitions - 1; // Dump into last partition
            }
        }

        return assignments;
    }

    PartitionerType getType() const override {
        return PartitionerType::RED_BLUE_BFS;
    }
};
