#pragma once

#include "SnpSystemConfig.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>

class SnpSystemPermuter {
public:
    struct PermutationResult {
        SnpSystemConfig config;
        std::vector<int> old_to_new;
        std::vector<int> new_to_old;
        std::vector<int> partition_offsets;
        std::vector<int> partition_counts;
    };

    static PermutationResult permute(const SnpSystemConfig& original, const std::vector<int>& partition, int num_partitions) {
        PermutationResult result;
        int n = original.neurons.size();
        result.old_to_new.resize(n);
        result.new_to_old.resize(n);
        result.partition_counts.assign(num_partitions, 0);
        result.partition_offsets.resize(num_partitions);

        // 1. Count partition sizes
        for (int p : partition) {
            if (p >= 0 && p < num_partitions) {
                result.partition_counts[p]++;
            }
        }

        // 2. Calculate offsets
        int current_offset = 0;
        for (int i = 0; i < num_partitions; ++i) {
            result.partition_offsets[i] = current_offset;
            current_offset += result.partition_counts[i];
        }

        // 3. Build mappings
        std::vector<int> current_pos = result.partition_offsets;
        for (int i = 0; i < n; ++i) {
            int p = partition[i];
            int new_idx = current_pos[p]++;
            result.old_to_new[i] = new_idx;
            result.new_to_old[new_idx] = i;
        }

        // 4. Build new config
        result.config.neurons.resize(n);
        
        // Copy neurons (updating IDs)
        for (int i = 0; i < n; ++i) {
            int new_id = result.old_to_new[i];
            result.config.neurons[new_id] = original.neurons[i];
            result.config.neurons[new_id].id = new_id; // Update internal ID
        }

        // Copy synapses (updating source/dest IDs)
        result.config.synapses.reserve(original.synapses.size());
        for (const auto& syn : original.synapses) {
            if (syn.source_id < n && syn.dest_id < n) {
                result.config.synapses.emplace_back(
                    result.old_to_new[syn.source_id],
                    result.old_to_new[syn.dest_id],
                    syn.weight
                );
            }
        }

        return result;
    }
};
