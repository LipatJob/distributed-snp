#include "MetisPartitioner.hpp"
#include <metis.h>
#include <iostream>
#include <numeric>
#include <map>

std::vector<int> MetisPartitioner::partition(const SnpSystemConfig& config, int num_partitions) {
    int n = config.neurons.size();
    if (n == 0) return {};
    if (num_partitions <= 1) return std::vector<int>(n, 0);

    idx_t nvtxs = n;
    idx_t ncon = 1; // Number of balancing constraints
    
    // Build Adjacency List (Undirected)
    // Use map to consolidate multi-edges between same pair
    std::vector<std::map<int, int>> adj_map(n);
    
    for (const auto& syn : config.synapses) {
        if (syn.source_id >= n || syn.dest_id >= n) continue;
        if (syn.source_id == syn.dest_id) continue; // Self-loops usually ignored in partitioning

        // Forward
        adj_map[syn.source_id][syn.dest_id] += syn.weight;
        // Backward
        adj_map[syn.dest_id][syn.source_id] += syn.weight;
    }

    // Convert to CSR
    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    std::vector<idx_t> adjwgt; // Edge weights
    std::vector<idx_t> vwgt(n); // Vertex weights

    xadj.reserve(n + 1);
    xadj.push_back(0);

    for (int i = 0; i < n; ++i) {
        // Calculate vertex weight based on computational load (rules)
        // Base weight 1 + number of rules
        vwgt[i] = 1 + config.neurons[i].rules.size();

        for (auto const& [neighbor, weight] : adj_map[i]) {
            adjncy.push_back(neighbor);
            adjwgt.push_back(weight);
        }
        xadj.push_back(adjncy.size());
    }

    idx_t nparts = num_partitions;
    std::vector<idx_t> part(n);
    idx_t objval;

    // Options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // Minimize edge cut
    options[METIS_OPTION_SEED] = 42; // Deterministic

    // If graph has no edges, METIS might fail or return trivial partition.
    // If adjncy is empty, we can just do linear partitioning.
    if (adjncy.empty()) {
         std::vector<int> fallback(n);
         int chunk = (n + num_partitions - 1) / num_partitions;
         for(int i=0; i<n; ++i) fallback[i] = std::min(i / chunk, num_partitions - 1);
         return fallback;
    }

    int status = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                     vwgt.data(), NULL, adjwgt.data(), &nparts, NULL,
                                     NULL, options, &objval, part.data());

    if (status != METIS_OK) {
        std::cerr << "METIS partitioning failed with error code: " << status << std::endl;
        // Fallback to linear
        std::vector<int> fallback(n);
        int chunk = (n + num_partitions - 1) / num_partitions;
        for(int i=0; i<n; ++i) fallback[i] = std::min(i / chunk, num_partitions - 1);
        return fallback;
    }

    // Convert idx_t back to int
    std::vector<int> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = static_cast<int>(part[i]);
    }

    return result;
}
