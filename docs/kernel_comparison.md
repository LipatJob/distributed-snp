# CUDA Kernel Comparison

This document provides a comparison of the CUDA kernels used across different simulator implementations in the Distributed SNP project, highlighting semantic equivalents and unique implementation details.

## Kernel Comparison Matrix

| Functionality | Sparse CUDA | Optimized CUDA (Single) | Naive MPI+CUDA | Optimized MPI+CUDA |
| :--- | :--- | :--- | :--- | :--- |
| **Reset State** | `resetNeuronsKernel` | `resetNeuronsKernel` | `resetNeuronsKernel` | `resetNeuronsKernel` |
| **Evaluation**<br>*(Update Delays, Check Rules, Fire)* | `computeSpikingVectorKernel`<br>*(Calculates active rules)* | `updateNeuronDynamicsKernel`<br>*(Fused: Delay + Rules + Output Gen)* | `updateNeuronDynamicsKernel`<br>*(Fused: Delay + Rules + Local Output)* | `updateNeuronDynamicsKernel`<br>*(Fused: Delay + Rules + Prod Set)* |
| **Propagation**<br>*(Move flow from neuron to neuron)* | `consumeSpikesAndProduceKernel`<br>*(Immediate firing)*<br><br>`updateDelaysAndEmitKernel`<br>*(Delayed firing)* | `propagateSpikesKernel`<br>*(Unified propagation)* | `propagateSpikesKernel`<br>*(Reads from global gather buffer)* | `propagateLocalSpikesKernel`<br>*(intra-node only)* |
| **MPI Communication**<br>*(Buffers for network)* | *N/A* | *N/A* | *N/A*<br>*(Handled by host gather)* | `populateExportBufferKernel`<br>*(Pack outgoing)*<br><br>`applyImportedSpikesKernel`<br>*(Unpack incoming)* |
| **Cleanup** | *Implicit* | *Implicit* | *Implicit* | `clearSpikeProductionKernel` |

## Key Differences & Implementation Notes

### 1. Sparse Simulator (`SparseCudaSnpSimulator.cu`)
*   **Approach**: Uses a multi-pass approach optimized for sparse matrices.
*   **Flow**:
    1.  `computeSpikingVectorKernel`: Determines which rules *will* fire.
    2.  `consumeSpikesAndProduceKernel`: Updates neuron state and matrix state.
    3.  `updateDelaysAndEmitKernel`: Handles the temporal aspect of delayed spikes.

### 2. Optimized Simulators (Single & MPI)
*   **Approach**: Focuses on minimizing global memory accesses and kernel launch overhead.
*   **Fusion**: The "Neuron Dynamics" phase fuses delay updates, rule checking, and initial output generation into a single kernel (`updateNeuronDynamicsKernel`).

### 3. MPI Implementations
*   **Naive (`NaiveCudaMpiSnpSimulator.cu`)**:
    *   **Communication**: "All-to-All" style logic on the host. It gathers *all* firing events into a global buffer.
    *   **Propagation**: The kernel reads this massive global buffer. Every thread (neuron) checks if any of its sources fired. This is simpler to implement but bandwidth-heavy.
*   **Optimized (`OptimizedCudaMpiSnpSimulator.cu`)**:
    *   **Communication**: "Point-to-Point" style logic (graph partition aware). Segregates local vs. remote traffic.
    *   **Propagation**:
        *   `propagateLocalSpikesKernel`: Fast path for intra-GPU synapses.
        *   `populateExportBufferKernel`: Only packs spikes destined for other nodes.
        *   `applyImportedSpikesKernel`: Unpacks only relevant incoming spikes.
