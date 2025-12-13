#include "ISort.hpp"
#include "../snp/ISnpSimulator.hpp"
#include "../snp/SnpSystemConfig.hpp"
#include <iostream>

class SnpSort : public ISort {
private:
    std::unique_ptr<ISnpSimulator> simulator;
    SnpSystemConfig config;
    std::vector<int> inputNumbers;
    int offset;
    int maxVal;
    size_t N;

public:
    explicit SnpSort(std::unique_ptr<ISnpSimulator> sim) 
        : simulator(std::move(sim)), offset(0), maxVal(0), N(0) {}

    void load(const int* data, size_t size) override {
        if (size == 0) {
            N = 0;
            return;
        }
        
        N = size;
        inputNumbers.assign(data, data + size);
        
        // Find min and max values
        int minVal = inputNumbers[0];
        maxVal = inputNumbers[0];
        for(int n : inputNumbers) {
            if(n < minVal) minVal = n;
            if(n > maxVal) maxVal = n;
        }
        
        // Offset to make all values non-negative (SNP systems can't have negative spikes)
        offset = (minVal < 0) ? -minVal : 0;
        std::vector<int> transformedNumbers;
        transformedNumbers.reserve(N);
        for(int n : inputNumbers) {
            transformedNumbers.push_back(n + offset);
        }
        
        // Update maxVal with offset
        maxVal += offset;
        
        // Build SNP system configuration
        config = buildSortingSystem(transformedNumbers);
        
        // Load system into simulator
        if (!simulator->loadSystem(config)) {
            std::cerr << "Failed to load SNP system for sorting" << std::endl;
        }
    }

    std::vector<int> execute() override {
        if (N == 0) return {};
        
        // Execute simulation
        int ticks = maxVal + 3; // Buffer for signal propagation
        simulator->step(ticks);

        // Collect results
        std::vector<int> localState = simulator->getGlobalState();
        std::vector<int> result;
        
        int startOutput = 2 * N;
        
        // Extract output neuron values and transform back to original range
        for(size_t o = 0; o < N; ++o) {
            int outputIdx = startOutput + o;
            if (outputIdx < static_cast<int>(localState.size())) {
                result.push_back(localState[outputIdx] - offset);
            } else {
                result.push_back(-offset);
            }
        }
        
        return result;
    }

    void sort(int* data, size_t size) override {
        load(data, size);
        std::vector<int> result = execute();
        
        // Copy results back to original array
        for (size_t i = 0; i < size && i < result.size(); ++i) {
            data[i] = result[i];
        }
    }

    std::string getPerformanceReport() const override {
        if (!simulator) {
            return "No simulator available";
        }
        return simulator->getPerformanceReport();
    }

private:
    SnpSystemConfig buildSortingSystem(const std::vector<int>& transformedNumbers) {
        int N = transformedNumbers.size();
        
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
            int spikes = transformedNumbers[i];  // Use transformed (non-negative) values
            
            // Reset neuron with initial spikes
            builder.addNeuron(neuronId, spikes);
            
            // Stream rule: consume 1, produce 1, no delay
            builder.addRule(neuronId, 1, 1, 1, 0);
        }

        // Sorters (Neurons N to 2N-1)
        for(int s = 0; s < N; ++s) {
            int sorterID = startSorter + s;
            int target = N - s;

            // 1. Forget High (Higher priority - processed first)
            for(int k = N; k > target; --k) {
                // threshold=k, consume=k, produce=0, delay=0
                builder.addRule(sorterID, k, k, 0, 0);
            }
            
            // 2. Fire Exact (at target count)
            // threshold=target, consume=target, produce=1, delay=0, priority=target
            builder.addRule(sorterID, target, target, 1, 0);

            // 3. Forget Low (Lower priority)
            for(int k = target - 1; k >= 1; --k) {
                // threshold=k, consume=k, produce=0, delay=0
                builder.addRule(sorterID, k, k, 0, 0);
            }
        }

        // Mark output neurons
        SnpSystemConfig config = builder.build();
        for(int o = 0; o < N; ++o) {
            config.neurons[startOutput + o].is_output = true;
        }
        
        return config;
    }
};

// Factory function to create SnpSort with CPU simulator
std::unique_ptr<ISort> createNaiveCpuSnpSort() {
    return std::make_unique<SnpSort>(createNaiveCpuSimulator());
}

// Factory function to create SnpSort with CUDA simulator
std::unique_ptr<ISort> createCudaSnpSort() {
    return std::make_unique<SnpSort>(createCudaSimulator());
}

// Factory function to create SnpSort with Sparse CUDA simulator
std::unique_ptr<ISort> createSparseCudaSnpSort() {
    return std::make_unique<SnpSort>(createSparseCudaSimulator());
}

// Factory function to create SnpSort with Naive CUDA/MPI simulator
std::unique_ptr<ISort> createNaiveCudaMpiSnpSort() {
    return std::make_unique<SnpSort>(createNaiveCudaMpiSimulator());
}

// Factory function to create SnpSort with CUDA/MPI simulator
std::unique_ptr<ISort> createCudaMpiSnpSort() {
    return std::make_unique<SnpSort>(createCudaMpiSimulator());
}