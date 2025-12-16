#pragma once
#include <cstdint>
#include <vector>

namespace bigdata {

constexpr uint64_t PARTITION_MAGIC = 0x534E504441544131; // "SNPDATA1"

struct PartitionHeader {
    uint64_t magic;
    int32_t rank_id;
    int32_t num_ranks;
    uint64_t num_local_neurons;
    uint64_t num_local_synapses;
    uint64_t num_export_groups;
    uint64_t num_import_groups;
};

// On disk, Neuron is:
// int32_t id
// int32_t initial_spikes
// int32_t num_rules
// RuleData[num_rules]

struct RuleData {
    int32_t input_threshold;
    int32_t spikes_consumed;
    int32_t spikes_produced;
    int32_t delay;
};

struct LocalSynapseData {
    int32_t source_local_idx;
    int32_t dest_local_idx;
    int32_t weight;
};

struct ExportGroupHeader {
    int32_t target_rank;
    uint64_t num_synapses;
};

struct ExportSynapseData {
    int32_t source_local_idx;
    int32_t weight;
};

struct ImportGroupHeader {
    int32_t source_rank;
    uint64_t num_synapses;
};

struct ImportSynapseData {
    int32_t export_index; 
    int32_t dest_local_idx;
};

} // namespace bigdata
