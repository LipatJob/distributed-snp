# CudaMpiSnpSimulator: Scalable Distributed Architecture

This document explains the architecture of the optimized `CudaMpiSnpSimulator`. Unlike the "Naive" implementation which replicates all synapses, this version uses a **Graph Partitioning** approach with explicit **Edge-Cut Communication**. This design reduces memory usage and communication volume, making it suitable for large-scale systems where the synapse graph cannot fit on a single GPU.

## 1. Architectural Overview

The simulator partitions the Spiking Neural P System across multiple MPI ranks. Each rank is responsible for:
1.  **State Management:** Owning a subset of neurons (Partition $P_i$).
2.  **Local Computation:** Executing rules and updating state for $P_i$.
3.  **Intra-Node Routing:** Handling synapses where both Source and Destination are local ($u, v \in P_i$).
4.  **Inter-Node Routing:** Handling "Export" synapses ($u \in P_i, v \notin P_i$) and "Import" synapses ($u \notin P_i, v \in P_i$).


## 2. Data Structures (Structure of Arrays)

To maximize GPU memory bandwidth, data is organized into specialized Structure of Arrays (SoA) layouts.

### DeviceNeuronData (State)
Stores the dynamic state of local neurons.
* `configuration`: Current spike counts ($C_k$).
* `delay_timer` & `pending_emission`: Logic for handling rule delays.
* `spike_production`: Temporary buffer for spikes produced in the current tick.

### Synapse Classification
Synapses are split into three categories to optimize memory access patterns:

1.  **`DeviceLocalSynapseData`**: Standard adjacency list for local-to-local connections.
    * Source $\to$ Dest (both indices are local).
2.  **`DeviceExportSynapseData`**: Local-to-Remote connections.
    * Source (Local Index) $\to$ Export Buffer Index.
    * *Optimization:* Synaptic weights are applied here (Sender-side), reducing the data needed in the buffer.
3.  **`DeviceImportMapData`**: Remote-to-Local connections.
    * Import Buffer Index $\to$ Dest (Local Index).

## 3. The Simulation Cycle (`step`)

Each time step is executed in three distinct phases: **Compute**, **Communicate**, and **Apply**.

### Phase 1: Local Compute (GPU)
This phase performs all logic that does not require data from other nodes.

1.  **`kUpdateStatus`**: Decrements delay timers. If a timer hits 0, the neuron opens.
2.  **`kSelectAndFire`**: Scans rules for open neurons. If a rule fires, it updates the neuron's spike count and sets the `spike_production` or `pending_emission` state.
3.  **`kPropagateLocal`**: Iterates `DeviceLocalSynapseData`. It adds spikes directly from source to destination neurons using atomic adds.
4.  **`kPopulateExport`**: Iterates `DeviceExportSynapseData`.
    * It calculates the *weighted* spike contribution: $Spikes \times Weight$.
    * It accumulates this value into a contiguous **Export Buffer** on the GPU.

### Phase 2: Communication (Hybrid MPI+CUDA)
This phase exchanges the "Edge Cut" data.

1.  **Device-to-Host**: The monolithic Export Buffer is copied from GPU to CPU.
2.  **Packing**: The host code segments the monolithic buffer into specific send buffers for each target MPI rank.
3.  **MPI Exchange**:
    * Uses non-blocking `MPI_Isend` and `MPI_Irecv`.
    * Ranks exchange only the active spike data destined for their specific boundary neurons.
4.  **Unpacking**: Received buffers are aggregated into a monolithic Import Buffer.
5.  **Host-to-Device**: The Import Buffer is copied from CPU to GPU.


### Phase 3: Apply Imports (GPU)
This phase integrates the external data into the local state.

1.  **`kApplyImports`**: Iterates `DeviceImportMapData`.
    * Reads weighted spike counts from the Import Buffer.
    * Atomically adds them to the corresponding local destination neurons.
2.  **`kCleanup`**: Resets `spike_production` and processed `pending_emission` states to prepare for the next step.

## 4. Topology Analysis (`prepareTopology`)

A critical part of this implementation is the setup phase, where the communication graph is built. This runs once during initialization.

1.  **Range Identification**: The system calculates which neuron IDs belong to which MPI rank.
2.  **Synapse sorting**:
    * Synapses entirely within the local range are added to `d_local_synapses`.
    * Synapses crossing boundaries are identified.
3.  **Handshake / Mapping**:
    * **Export Map**: The rank determines the unique set of *remote* neurons it feeds. It assigns each a slot in the Export Buffer.
    * **Import Map**: The rank determines which *local* neurons are fed by remote ranks. It assigns each a slot in the Import Buffer.
    * **Consistency**: The code relies on a deterministic sorting of synapse targets (by Global ID) to ensure that Rank A knows exactly which slot in the buffer corresponds to which neuron on Rank B, without sending explicit metadata every step.

## 5. Key Optimizations

* **Sender-Side Weighting**: Synaptic weights are applied before communication. If multiple synapses connect to the same remote neuron with different weights, they are summed into a single integer transmission, reducing bandwidth.
* **Buffer Coalescing**: Small messages are aggregated into monolithic buffers to minimize `cudaMemcpy` overhead and MPI latency.
* **Atomic Operations**: Used for spike aggregation (`atomicAdd`) to handle cases where multiple source neurons feed a single destination concurrently.