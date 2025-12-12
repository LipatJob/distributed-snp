#include "ISort.hpp"
#include "../snp/ISnpSimulator.hpp"
#include "../snp/SnpSystemConfig.hpp"
#include <algorithm>
#include <iostream>

class SnpSort : public ISort {
private:
    std::unique_ptr<ISnpSimulator> simulator;

public:
    explicit SnpSort(std::unique_ptr<ISnpSimulator> sim) 
        : simulator(std::move(sim)) {}

    void sort(int* data, size_t size) override {
        if (size == 0) return;
        
        std::vector<int> inputNumbers(data, data + size);
        std::vector<int> result = runSort(inputNumbers);
        
        // Copy results back to original array
        for (size_t i = 0; i < size; ++i) {
            data[i] = result[i];
        }
    }

private:
    std::vector<int> runSort(const std::vector<int>& inputNumbers) {
        int N = inputNumbers.size();
        if (N == 0) return {};

        // Determine max value to calculate simulation ticks needed
        int maxVal = 0;
        for(int n : inputNumbers) {
            if(n > maxVal) maxVal = n;
        }
        
        // Layout: [ Inputs (0..N-1) | Sorters (N..2N-1) | Outputs (2N..3N-1) ]
        int startInput = 0;
        int startSorter = N;
        int startOutput = 2 * N;
        int totalNeurons = 3 * N;

        // --- A. Build System Configuration using Builder ---
        SnpSystemBuilder builder;
        
        // Create all neurons
        for (int i = 0; i < totalNeurons; ++i) {
            builder.addNeuron(i, 0);
        }

        // --- B. Build Topology ---
        // Inputs -> All Sorters
        for(int i = 0; i < N; ++i) {
            for(int s = 0; s < N; ++s) {
                builder.addSynapse(startInput + i, startSorter + s);
            }
        }

        // Sorters -> Outputs
        // Sorter s (detects N-s spikes) connects to Outputs [s...N-1]
        for(int s = 0; s < N; ++s) {
            for(int o = s; o < N; ++o) {
                builder.addSynapse(startSorter + s, startOutput + o);
            }
        }

        // --- C. Define Rules ---
        // Input Streams (Neurons 0 to N-1)
        for(int i = 0; i < N; ++i) {
            int neuronId = startInput + i;
            int spikes = inputNumbers[i];
            
            // Reset neuron with initial spikes
            builder.addNeuron(neuronId, spikes);
            
            // Stream rule: consume 1, produce 1, no delay, priority 1
            builder.addRule(neuronId, 1, 1, 1, 0, 1);
        }

        // Sorters (Neurons N to 2N-1)
        for(int s = 0; s < N; ++s) {
            int sorterID = startSorter + s;
            int target = N - s;

            // 1. Forget High (Higher priority - processed first)
            for(int k = N; k > target; --k) {
                // threshold=k, consume=k, produce=0, delay=0, priority=k (greedy)
                builder.addRule(sorterID, k, k, 0, 0, k);
            }
            
            // 2. Fire Exact (at target count)
            // threshold=target, consume=target, produce=1, delay=0, priority=target
            builder.addRule(sorterID, target, target, 1, 0, target);

            // 3. Forget Low (Lower priority)
            for(int k = target - 1; k >= 1; --k) {
                // threshold=k, consume=k, produce=0, delay=0, priority=k
                builder.addRule(sorterID, k, k, 0, 0, k);
            }
        }

        // Mark output neurons
        SnpSystemConfig config = builder.build();
        for(int o = 0; o < N; ++o) {
            config.neurons[startOutput + o].is_output = true;
        }

        // --- D. Load System and Execute ---
        if (!simulator->loadSystem(config)) {
            std::cerr << "Failed to load SNP system for sorting" << std::endl;
            return std::vector<int>(N, 0);
        }

        // Execute simulation
        int ticks = maxVal + 3; // Buffer for signal propagation
        simulator->step(ticks);

        // --- E. Collect Results ---
        std::vector<int> localState = simulator->getLocalState();
        std::vector<int> result;
        
        // Extract output neuron values
        for(int o = 0; o < N; ++o) {
            int outputIdx = startOutput + o;
            if (outputIdx < static_cast<int>(localState.size())) {
                result.push_back(localState[outputIdx]);
            } else {
                result.push_back(0);
            }
        }
        
        return result;
    }
};

// Factory function to create SnpSort with CPU simulator
std::unique_ptr<ISort> createSnpSort() {
    return std::make_unique<SnpSort>(createNaiveCpuSimulator());
}

// Factory function to create SnpSort with CUDA/MPI simulator
std::unique_ptr<ISort> createSnpSortCudaMpi() {
    return std::make_unique<SnpSort>(createCudaMpiSimulator());
}

