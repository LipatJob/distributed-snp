#include "LouvainPartitioner.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iostream>

std::vector<int> LouvainPartitioner::partition(const SnpSystemConfig& config, int num_partitions) {
    if (config.neurons.empty()) return {};
    if (num_partitions <= 1) return std::vector<int>(config.neurons.size(), 0);

    // 1. Build Graph (Undirected, Weighted)
    Graph graph = buildGraph(config);

    // 2. Run Louvain (Modularity Optimization)
    // For simplicity, we run one pass of modularity optimization (Phase 1).
    // A full implementation would recursively aggregate nodes, but Phase 1 
    // often provides sufficient granularity for load balancing.
    std::vector<int> communities = runLouvain(graph);

    // 3. Balance Partitions (Bin Packing)
    return balancePartitions(communities, graph.num_nodes, num_partitions);
}

LouvainPartitioner::Graph LouvainPartitioner::buildGraph(const SnpSystemConfig& config) {
    Graph g;
    g.num_nodes = config.neurons.size();
    g.adj.resize(g.num_nodes);
    g.node_weights.assign(g.num_nodes, 0.0);
    g.total_weight = 0.0;

    for (const auto& syn : config.synapses) {
        if (syn.source_id >= g.num_nodes || syn.dest_id >= g.num_nodes) continue;
        
        // Treat as undirected: add weight to both directions
        // If multiple synapses exist, weights sum up
        double w = static_cast<double>(syn.weight);
        
        // Add forward
        g.adj[syn.source_id].push_back({syn.dest_id, w});
        // Add backward
        g.adj[syn.dest_id].push_back({syn.source_id, w});
        
        g.node_weights[syn.source_id] += w;
        g.node_weights[syn.dest_id] += w;
        g.total_weight += w; // Counted once per edge (undirected logic usually counts 2m, but here we sum incident)
    }
    
    // Consolidate duplicates in adjacency list
    for (int i = 0; i < g.num_nodes; ++i) {
        std::sort(g.adj[i].begin(), g.adj[i].end());
        std::vector<std::pair<int, double>> unique_adj;
        if (!g.adj[i].empty()) {
            unique_adj.push_back(g.adj[i][0]);
            for (size_t k = 1; k < g.adj[i].size(); ++k) {
                if (g.adj[i][k].first == unique_adj.back().first) {
                    unique_adj.back().second += g.adj[i][k].second;
                } else {
                    unique_adj.push_back(g.adj[i][k]);
                }
            }
        }
        g.adj[i] = unique_adj;
    }

    // Adjust total weight to be 2m (sum of all node weights)
    g.total_weight = 0;
    for(double w : g.node_weights) g.total_weight += w;

    return g;
}

std::vector<int> LouvainPartitioner::runLouvain(const Graph& graph) {
    int n = graph.num_nodes;
    std::vector<int> community(n);
    std::iota(community.begin(), community.end(), 0); // Each node in its own community

    std::vector<double> k_i = graph.node_weights;
    std::vector<double> sigma_tot(n); // Sum of weights in each community
    for(int i=0; i<n; ++i) sigma_tot[i] = k_i[i];

    double m2 = graph.total_weight; // 2m
    if (m2 == 0) return std::vector<int>(n, 0); // Disconnected or empty

    bool improvement = true;
    int max_iter = 20; // Safety break
    int iter = 0;

    // Randomize order of node traversal
    std::vector<int> nodes(n);
    std::iota(nodes.begin(), nodes.end(), 0);
    std::mt19937 rng(42); // Deterministic seed

    while (improvement && iter < max_iter) {
        improvement = false;
        std::shuffle(nodes.begin(), nodes.end(), rng);

        for (int i : nodes) {
            int old_comm = community[i];
            double k_i_in_old = 0;

            // Find neighbors and their communities
            // Also calculate k_i_in for current community
            std::map<int, double> neighbor_communities; // comm -> weight sum
            
            for (auto& edge : graph.adj[i]) {
                int neighbor = edge.first;
                double w = edge.second;
                int neighbor_comm = community[neighbor];
                neighbor_communities[neighbor_comm] += w;
                
                if (neighbor_comm == old_comm) {
                    // Note: self-loops are handled if they exist in adj
                    k_i_in_old += w; 
                }
            }

            // Remove i from old community
            sigma_tot[old_comm] -= k_i[i];
            
            // Find best community
            int best_comm = old_comm;
            double best_gain = 0.0;
            
            // Gain formula: Delta Q = [k_i_in / m2] - [Sigma_tot * k_i / (m2^2)] * 2 (approx)
            // We maximize: k_i_in - (Sigma_tot * k_i) / m2
            // (Factor of 2m cancels out or is constant)
            
            // Current cost (removed)
            // We compare gain of moving to NEW vs staying (which is 0 gain relative to current state)
            // Actually, standard way: calculate modularity increase.
            
            for (auto const& [comm, k_i_in] : neighbor_communities) {
                if (comm == old_comm) continue;
                
                double delta = k_i_in - (sigma_tot[comm] * k_i[i]) / m2;
                if (delta > best_gain) {
                    best_gain = delta;
                    best_comm = comm;
                }
            }
            
            // Check if staying is better (or rather, if moving back to old is better than best new)
            // But we already removed it. So we treat "best_comm" as the target.
            // If best_gain > 0, we move.
            // Wait, if we removed it, we need to compare placing it in old_comm vs others.
            // Gain of placing in comm C: k_i_in_C - (Sigma_tot_C * k_i) / m2
            
            double gain_old = k_i_in_old - (sigma_tot[old_comm] * k_i[i]) / m2;
            
            // If we found a neighbor community with better gain than putting it back in old
            if (best_gain > gain_old) {
                community[i] = best_comm;
                sigma_tot[best_comm] += k_i[i];
                improvement = true;
            } else {
                community[i] = old_comm;
                sigma_tot[old_comm] += k_i[i];
            }
        }
        iter++;
    }
    
    // Renumber communities to 0..K-1
    std::map<int, int> renumber;
    int next_id = 0;
    for(int& c : community) {
        if (renumber.find(c) == renumber.end()) {
            renumber[c] = next_id++;
        }
        c = renumber[c];
    }

    return community;
}

std::vector<int> LouvainPartitioner::balancePartitions(const std::vector<int>& communities, int num_nodes, int num_partitions) {
    // 1. Count community sizes
    std::map<int, int> comm_sizes;
    for (int c : communities) comm_sizes[c]++;

    // 2. Sort communities by size (descending)
    std::vector<std::pair<int, int>> sorted_comms;
    for (auto const& [comm, size] : comm_sizes) {
        sorted_comms.push_back({comm, size});
    }
    std::sort(sorted_comms.begin(), sorted_comms.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // 3. Bin Packing (Greedy LPT)
    std::vector<int> bin_weights(num_partitions, 0);
    std::map<int, int> comm_to_partition;

    for (auto const& [comm, size] : sorted_comms) {
        // Find smallest bin
        int best_bin = 0;
        int min_weight = bin_weights[0];
        for (int i = 1; i < num_partitions; ++i) {
            if (bin_weights[i] < min_weight) {
                min_weight = bin_weights[i];
                best_bin = i;
            }
        }
        
        // Assign
        comm_to_partition[comm] = best_bin;
        bin_weights[best_bin] += size;
    }

    // 4. Build result
    std::vector<int> partition(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        partition[i] = comm_to_partition[communities[i]];
    }

    return partition;
}
