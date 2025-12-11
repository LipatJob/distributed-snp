#include "SnpSimulator.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <mpi.h>

SnpSimulator::SnpSimulator(BackendType backend, size_t m, size_t n) 
    : numNeurons(m), numRules(n) {
    
    // 1. Initialize Matrix Ops Backend
    matrixOps = createMatrixOps<double>(backend);
    std::cout << "Simulator initialized using backend: " << matrixOps->getBackendName() << std::endl;

    // 2. Resize Host Vectors
    host_M_Pi.resize(numRules * numNeurons, 0.0);
    host_C.resize(numNeurons, 0.0);
    host_Iv.resize(numRules, 0.0);
    host_St.resize(numNeurons, 1.0); // Default open

    buffer_Delta.resize(numNeurons);
    buffer_MaskedDelta.resize(numNeurons);
    buffer_NextC.resize(numNeurons);
    
    // Initialize neurons
    for(size_t i=0; i<m; i++) neurons.push_back({(int)i, 0.0, 0});
}

void SnpSimulator::addRule(int ruleIdx, int sourceNeuron, int destNeuron, 
                           double consumed, double produced, double threshold, 
                           int delay, bool isForgetting) {
    rules.push_back({ruleIdx, sourceNeuron, consumed, produced, delay, threshold, isForgetting});

    // Update Spiking Transition Matrix (M_Pi)
    // Row = ruleIdx, Col = Neuron
    
    // 1. Consumption (Negative at source)
    host_M_Pi[ruleIdx * numNeurons + sourceNeuron] -= consumed;
    
    // 2. Production (Positive at destination)
    if (!isForgetting && destNeuron >= 0 && destNeuron < (int)numNeurons) {
        host_M_Pi[ruleIdx * numNeurons + destNeuron] += produced;
    }
}

void SnpSimulator::setInitialSpikes(int neuronIdx, double count) {
    neurons[neuronIdx].spikes = count;
    host_C[neuronIdx] = count;
}

void SnpSimulator::step(int tick) {
        std::cout << "\n--- Tick " << tick << " ---" << std::endl;

        // Step 1: CPU Logic - Determine Firing Rules (Iv) and Neuron Status (St)
        std::fill(host_Iv.begin(), host_Iv.end(), 0.0);
        std::fill(host_St.begin(), host_St.end(), 0.0);

        // Update Neuron Status Vector (St)
        for (size_t i = 0; i < numNeurons; ++i) {
            if (neurons[i].remainingDelay > 0) {
                neurons[i].remainingDelay--; // Decrement delay
                host_St[i] = 0.0;            // Neuron is Closed
            } else {
                host_St[i] = 1.0;            // Neuron is Open
            }
        }

        // Determine Indicator Vector (Iv) - Which rules fire?
        // Note: This is where non-determinism would be handled. 
        // We assume a deterministic "first applicable rule" strategy here.
        std::vector<bool> neuronFired(numNeurons, false);

        for (const auto& rule : rules) {
            // Check if neuron is open, has enough spikes, and hasn't fired yet in this tick
            if (host_St[rule.neuronIndex] == 1.0 && 
                !neuronFired[rule.neuronIndex] && 
                rule.canFire(neurons[rule.neuronIndex].spikes)) {
                
                host_Iv[rule.id] = 1.0; // Mark rule as firing
                neuronFired[rule.neuronIndex] = true;
                
                // If rule has delay, set it for the neuron (effective next tick)
                if (rule.delay > 0) {
                    neurons[rule.neuronIndex].remainingDelay = rule.delay;
                }
            }
        }

        // --- GPU/MPI Execution Block ---
        // We now map the equation: C_next = C + St * (Iv * M_Pi)

        // 1. Matrix Multiplication: Delta = Iv * M_Pi
        // Iv is (1 x n), M_Pi is (n x m) -> Delta is (1 x m)
        // This calculates the net spike change based on fired rules
        matrixOps->multiply(host_Iv.data(), host_M_Pi.data(), buffer_Delta.data(), 
                            1, numNeurons, numRules);

        // 2. Hadamard Product: MaskedDelta = St (element-wise) Delta
        // This ensures closed neurons do not receive/emit spikes 
        // (per WebSnapse definition of Status Vector application)
        matrixOps->hadamard(host_St.data(), buffer_Delta.data(), buffer_MaskedDelta.data(), 
                            1, numNeurons);

        // 3. Matrix Addition: C_next = C + MaskedDelta
        matrixOps->add(host_C.data(), buffer_MaskedDelta.data(), buffer_NextC.data(), 
                       1, numNeurons);


        // Step 4: CPU Sync - Update Host State
        host_C = buffer_NextC;
        for(size_t i=0; i<numNeurons; i++) {
            neurons[i].spikes = host_C[i];
            // Clamp negative spikes to 0 (sanity check)
            if(neurons[i].spikes < 0) neurons[i].spikes = 0; 
            
            std::cout << "Neuron " << i << ": " << neurons[i].spikes << " spikes" 
                      << (host_St[i] == 0.0 ? " [CLOSED]" : " [OPEN]") << std::endl;
        }
}

#ifdef BUILD_SNP_DEMO_MAIN
// Demo main function (only compiled when building standalone demo)
int main(int argc, char** argv) {
    // Initialize MPI (Required because MpiCudaMatrixOps uses it)
    MPI_Init(&argc, &argv);

    {
        // Example: 3 Neurons, 2 Rules
        // Backend choices: BackendType::CPU, BackendType::CUDA, BackendType::MPI_CUDA
        SnpSimulator sim(BackendType::CUDA, 3, 2);

        // Setup System
        // Neuron 0 has 2 spikes
        sim.setInitialSpikes(0, 2.0);

        // Rule 0: Neuron 0 consumes 2, produces 1 at Neuron 1 (Threshold 2, Delay 0)
        // Rule index, Source, Dest, Consumed, Produced, Threshold, Delay
        sim.addRule(0, 0, 1, 2.0, 1.0, 2.0, 0);

        // Rule 1: Neuron 1 consumes 1, produces 1 at Neuron 2 (Threshold 1, Delay 0)
        sim.addRule(1, 1, 2, 1.0, 1.0, 1.0, 0);

        // Run Simulation
        for (int i = 0; i < 5; ++i) {
            sim.step(i);
        }
    }

    MPI_Finalize();
    return 0;
}
#endif