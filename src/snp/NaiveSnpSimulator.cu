/*
 * NaiveSnpSimulator.cu
 * * Naive Distributed Spiking Neural P System Simulator
 * * Pattern: Replicated State / Global Reduction
 */

#include "SnpSimulator.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// -------------------------------------------------------------------------
// NAIVE CUDA KERNELS
// -------------------------------------------------------------------------

// 1. Compute Deltas
// Checks local rules and adds their effects to a global delta vector.
// Note: This writes to a vector of size N (Global), but only reads local state.
__global__ void ComputeLocalDeltasKernel(
    int numLocalRules,
    const int* __restrict__ ruleOwners,        // Neuron index owning the rule
    const SpikeCount* __restrict__ fullState,  // Replicated global state
    const int* __restrict__ consumption,       // Spikes to consume
    const int* __restrict__ rowPtr,            // CSR Row Pointers for production
    const int* __restrict__ colInd,            // CSR Column Indices (Targets)
    const int* __restrict__ values,            // CSR Values (Weights)
    SpikeCount* __restrict__ globalDelta       // Output: The changes caused by this rank
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numLocalRules) return;

    int neuronIdx = ruleOwners[tid];
    int spikesAvailable = fullState[neuronIdx];
    int spikesNeeded = consumption[tid];

    // Naive Firing Condition: Spikes >= Consumption
    // (Ignores delay logic for maximum simplicity as per naive requirement)
    if (spikesAvailable >= spikesNeeded) {
        
        // 1. Record Consumption (Negative Delta)
        // Atomic add is needed because multiple rules might fire for the same neuron
        atomicAdd(&globalDelta[neuronIdx], -spikesNeeded);

        // 2. Record Production (Positive Deltas)
        int start = rowPtr[tid];
        int end = rowPtr[tid + 1];

        for (int i = start; i < end; ++i) {
            int targetNeuron = colInd[i];
            int weight = values[i];
            
            // Atomic add to target in the delta vector
            // This handles both local and remote targets uniformly
            atomicAdd(&globalDelta[targetNeuron], weight);
        }
    }
}

// 2. Apply Updates
// Simple vector addition: State = State + Delta
__global__ void ApplyGlobalUpdatesKernel(
    int totalNeurons,
    SpikeCount* __restrict__ fullState,
    const SpikeCount* __restrict__ globalDelta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalNeurons) return;

    fullState[tid] += globalDelta[tid];
    
    // Clamp to 0 to prevent underflow if logic is loose
    if (fullState[tid] < 0) fullState[tid] = 0;
}

// -------------------------------------------------------------------------
// SIMULATOR CLASS
// -------------------------------------------------------------------------

class NaiveSnpSimulator : public ISnpSimulator {
private:
    int rank, size;
    int totalNeurons;
    int myRuleStart, myRuleEnd, numLocalRules;

    // Device Memory
    SpikeCount* d_fullState = nullptr;    // Size: N (Replicated)
    SpikeCount* d_globalDelta = nullptr;  // Size: N (Replicated/Reduced)
    
    // Local Rules Definition (CSR)
    int* d_ruleOwners = nullptr;
    int* d_consumption = nullptr;
    int* d_rowPtr = nullptr;
    int* d_colInd = nullptr;
    int* d_values = nullptr;

    // Host Buffers for MPI
    SpikeCount* h_localDelta = nullptr;
    SpikeCount* h_globalDelta = nullptr;

    uint64_t currentTick = 0;

public:
    NaiveSnpSimulator() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    ~NaiveSnpSimulator() {
        if (d_fullState) cudaFree(d_fullState);
        if (d_globalDelta) cudaFree(d_globalDelta);
        if (d_ruleOwners) cudaFree(d_ruleOwners);
        if (d_consumption) cudaFree(d_consumption);
        if (d_rowPtr) cudaFree(d_rowPtr);
        if (d_colInd) cudaFree(d_colInd);
        if (d_values) cudaFree(d_values);
        if (h_localDelta) delete[] h_localDelta;
        if (h_globalDelta) delete[] h_globalDelta;
    }

    void Initialize(
        int globalNumNeurons, 
        int globalNumRules,
        const std::vector<int>& ruleOwners,
        const std::vector<float>& matrixValues,
        const std::vector<int>& matrixColIndices,
        const std::vector<int>& matrixRowPtrs,
        const std::vector<SpikeCount>& initialSpikes
    ) override {
        totalNeurons = globalNumNeurons;

        // 1. Load Balanced Partitioning of RULES (not neurons)
        // Since rules drive computation, we split rules evenly.
        int rulesPerRank = (globalNumRules + size - 1) / size;
        myRuleStart = rank * rulesPerRank;
        myRuleEnd = std::min(myRuleStart + rulesPerRank, globalNumRules);
        numLocalRules = (myRuleEnd > myRuleStart) ? (myRuleEnd - myRuleStart) : 0;

        // 2. Extract Data for Local Rules
        std::vector<int> l_owners;
        std::vector<int> l_consumption;
        std::vector<int> l_rowPtr;
        std::vector<int> l_colInd;
        std::vector<int> l_vals;
        
        l_rowPtr.push_back(0); // CSR start

        for (int r = myRuleStart; r < myRuleEnd; ++r) {
            l_owners.push_back(ruleOwners[r]);
            
            // Extract CSR row for this rule
            int start = matrixRowPtrs[r];
            int end = matrixRowPtrs[r+1];
            
            int consumption = 0;
            
            for (int nz = start; nz < end; ++nz) {
                int col = matrixColIndices[nz];
                int val = (int)matrixValues[nz];

                if (val < 0) {
                    consumption = -val; // Store consumption separately
                } else {
                    l_colInd.push_back(col); // Direct Global Index
                    l_vals.push_back(val);
                }
            }
            l_consumption.push_back(consumption);
            l_rowPtr.push_back(l_colInd.size());
        }

        // 3. Allocate Replicated State on GPU
        cudaMalloc(&d_fullState, totalNeurons * sizeof(SpikeCount));
        cudaMalloc(&d_globalDelta, totalNeurons * sizeof(SpikeCount));
        cudaMemcpy(d_fullState, initialSpikes.data(), totalNeurons * sizeof(SpikeCount), cudaMemcpyHostToDevice);

        // 4. Allocate Local Rules on GPU
        if (numLocalRules > 0) {
            cudaMalloc(&d_ruleOwners, numLocalRules * sizeof(int));
            cudaMalloc(&d_consumption, numLocalRules * sizeof(int));
            cudaMalloc(&d_rowPtr, l_rowPtr.size() * sizeof(int));
            cudaMalloc(&d_colInd, l_colInd.size() * sizeof(int));
            cudaMalloc(&d_values, l_vals.size() * sizeof(int));

            cudaMemcpy(d_ruleOwners, l_owners.data(), numLocalRules * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_consumption, l_consumption.data(), numLocalRules * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_rowPtr, l_rowPtr.data(), l_rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_colInd, l_colInd.data(), l_colInd.size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_values, l_vals.data(), l_vals.size() * sizeof(int), cudaMemcpyHostToDevice);
        }

        // 5. Host buffers for reduction
        h_localDelta = new SpikeCount[totalNeurons];
        h_globalDelta = new SpikeCount[totalNeurons];
    }

    void Step(int steps) override {
        int blockSize = 256;
        int gridRules = (numLocalRules + blockSize - 1) / blockSize;
        int gridState = (totalNeurons + blockSize - 1) / blockSize;

        for (int s = 0; s < steps; ++s) {
            // A. Reset Delta Vector (Zero out)
            cudaMemset(d_globalDelta, 0, totalNeurons * sizeof(SpikeCount));

            // B. Compute Local Contributions
            if (numLocalRules > 0) {
                ComputeLocalDeltasKernel<<<gridRules, blockSize>>>(
                    numLocalRules, 
                    d_ruleOwners, 
                    d_fullState, // Reads full global state
                    d_consumption, 
                    d_rowPtr, 
                    d_colInd, 
                    d_values, 
                    d_globalDelta // Accumulates into local delta buffer
                );
            }
            cudaDeviceSynchronize();

            // C. Global Synchronization (The Naive Part)
            // 1. Copy GPU Delta -> CPU
            cudaMemcpy(h_localDelta, d_globalDelta, totalNeurons * sizeof(SpikeCount), cudaMemcpyDeviceToHost);

            // 2. MPI AllReduce: Sum everyone's deltas -> Everyone gets the Total Delta
            // This scales poorly (O(N) communication), but logic is trivial.
            MPI_Allreduce(
                h_localDelta, 
                h_globalDelta, 
                totalNeurons, 
                MPI_INT32_T, // Assuming SpikeCount is int32
                MPI_SUM, 
                MPI_COMM_WORLD
            );

            // 3. Copy Total Delta -> GPU
            cudaMemcpy(d_globalDelta, h_globalDelta, totalNeurons * sizeof(SpikeCount), cudaMemcpyHostToDevice);

            // D. Apply Global Update
            ApplyGlobalUpdatesKernel<<<gridState, blockSize>>>(
                totalNeurons, d_fullState, d_globalDelta
            );
            
            currentTick++;
        }
    }

    SimulationState GetState() const override {
        // Since state is replicated, every node has the answer.
        // We just return Rank 0's copy.
        SimulationState state;
        state.currentTick = currentTick;

        if (rank == 0) {
            state.neuronSpikes.resize(totalNeurons);
            cudaMemcpy(state.neuronSpikes.data(), d_fullState, totalNeurons * sizeof(SpikeCount), cudaMemcpyDeviceToHost);
        }
        
        return state;
    }

    void Reset() override {
        // Simplified reset logic
        currentTick = 0;
        cudaMemset(d_fullState, 0, totalNeurons * sizeof(SpikeCount));
        // Note: Real reset would reload initialSpikes
    }
};