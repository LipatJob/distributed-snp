#pragma once
#include <vector>
#include <algorithm>

// --- Data Definitions ---

struct SnpRule {
    int input_threshold = 0;
    int spikes_consumed = 0;
    int spikes_produced = 0;
    int delay = 0;

    // RESTORED: Default constructor required for struct usage in vectors/other structs
    SnpRule() = default;

    // Convenience Constructor for the Builder/Tests
    SnpRule(int threshold, int consumed, int produced, int d = 0)
        : input_threshold(threshold), spikes_consumed(consumed), 
          spikes_produced(produced), delay(d) {}
};

struct SnpNeuron {
    int id;
    int initial_spikes;
    std::vector<SnpRule> rules;
    bool is_output;

    // Default constructor
    SnpNeuron() : id(-1), initial_spikes(0), is_output(false) {}

    // Convenience constructor
    SnpNeuron(int id, int spikes = 0, bool output = false)
        : id(id), initial_spikes(spikes), is_output(output) {}
};

struct SnpSynapse {
    int source_id;
    int dest_id;
    int weight;

    // Default constructor
    SnpSynapse() : source_id(-1), dest_id(-1), weight(1) {}

    // Convenience constructor
    SnpSynapse(int src, int dst, int w = 1)
        : source_id(src), dest_id(dst), weight(w) {}
};

struct SnpSystemConfig {
    std::vector<SnpNeuron> neurons;
    std::vector<SnpSynapse> synapses;

    size_t getTotalRulesCount() const {
        size_t count = 0;
        for (const auto& n : neurons) count += n.rules.size();
        return count;
    }
};

// --- Fluent Builder for Readable Tests ---

class SnpSystemBuilder {
    SnpSystemConfig config;

public:
    SnpSystemBuilder& addNeuron(int id, int initial_spikes = 0) {
        if (config.neurons.size() <= static_cast<size_t>(id)) {
            // Fill gaps with default-constructed neurons
            config.neurons.resize(id + 1); 
        }
        config.neurons[id] = SnpNeuron(id, initial_spikes);
        return *this;
    }

    SnpSystemBuilder& addRule(int neuron_id, int threshold, int consumed, int produced, int delay = 0) {
        if (config.neurons.size() <= static_cast<size_t>(neuron_id)) {
            addNeuron(neuron_id);
        }
        config.neurons[neuron_id].rules.emplace_back(threshold, consumed, produced, delay);
        return *this;
    }

    SnpSystemBuilder& addSynapse(int source, int dest, int weight = 1) {
        config.synapses.emplace_back(source, dest, weight);
        return *this;
    }

    SnpSystemBuilder& chainNeurons(const std::vector<int>& ids) {
        for (size_t i = 0; i < ids.size() - 1; ++i) {
            addSynapse(ids[i], ids[i+1]);
        }
        return *this;
    }

    SnpSystemConfig build() const {
        return config;
    }
};