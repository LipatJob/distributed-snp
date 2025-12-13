# CudaSnpSimulator: Single-GPU Implementation Guide

The `CudaSnpSimulator` is a high-performance, single-node implementation of Spiking Neural P Systems designed for NVIDIA GPUs. It serves as the foundational baseline for parallelizing the system before introducing MPI distribution.

## 1. Memory Architecture: Structure of Arrays (SoA)

The most critical design choice in this simulator is the use of **Structure of Arrays (SoA)** rather than Array of Structures (AoS).

### Why SoA?
In a standard C++ object approach (AoS), a `Neuron` object might hold its spike count, status, and rules together in memory. However, on a GPU, threads (cuda cores) execute in groups called warps.
* **Coalesced Access:** When thread $i$ reads `configuration[i]` and thread $i+1$ reads `configuration[i+1]`, the GPU can fetch a single continuous chunk of memory.
* **Cache Efficiency:** SoA ensures that data required for a specific kernel (e.g., just the `delay_timers` for the status update kernel) is packed tightly, maximizing cache hit rates. 
### Data Structures
The simulator defines three main device structures:

1.  **`DeviceNeuronData`**:
    * `configuration`: The current spike count vector $C(k)$.
    * `delay_timer`: Tracks how many ticks remain before a neuron opens.
    * `is_open`: A boolean mask (stored as `char` for alignment) acting as the status vector $St(k)$.
    * `pending_emission`: Buffer for spikes waiting for a delay to expire.
    * `spike_production`: Buffer for spikes produced *immediately* in the current step.

2.  **`DeviceRuleData`**:
    * Rules are flattened. Instead of `vector<Rule>`, we use arrays like `input_threshold[]`, `spikes_consumed[]`, etc.
    * **CSR-like Indexing:** `rule_start_idx` and `rule_count` allow each neuron to find its specific block of rules within the flattened arrays.

3.  **`DeviceSynapseData`**:
    * Simple arrays of `source_id`, `dest_id`, and `weight`.

## 2. The Simulation Pipeline (`executeOneStep`)

The simulation advances in discrete time steps. Unlike the CPU version which processes neurons sequentially, the CUDA version launches independent kernels for different phases of the algorithm.

### Phase 1: Status Update
**Kernel:** `updateNeuronStatusKernel`
* **Parallelism:** One thread per neuron.
* **Logic:** Decrements `delay_timer`. If the timer reaches 0, the neuron flag `is_open` is set to true.

### Phase 2: Rule Selection & Firing
**Kernel:** `selectAndApplyRulesKernel`
* **Parallelism:** One thread per neuron.
* **Divergence Optimization:** Threads assigned to closed neurons exit immediately (`if (!is_open) return`), freeing up warp resources.
* **Logic:**
    1.  Reads current spike count.
    2.  Iterates through the neuron's rules (linear search).
    3.  **Deterministic Choice:** Applies the first rule where `spikes >= threshold`.
    4.  **Branching:**
        * If `rule.delay > 0`: Closes neuron, sets `delay_timer`, and writes to `pending_emission`.
        * If `rule.delay == 0`: Writes directly to `spike_production`.

### Phase 3: Delayed Spike Propagation
**Kernel:** `propagatePendingEmissionsKernel`
* **Parallelism:** One thread per **synapse**.
* **Logic:** This handles spikes from neurons that *just* opened (delay finished).
    * Check: `if (neurons.is_open[source] && neurons.pending_emission[source] > 0)`
    * Action: Add `pending_emission * weight` to destination.
* **Safety:** Uses `atomicAdd` because multiple synapses may target the same destination neuron simultaneously. 
### Phase 4: Immediate Spike Propagation
**Kernel:** `propagateImmediateSpikesKernel`
* **Parallelism:** One thread per **synapse**.
* **Logic:** Handles rules that fired this step with no delay.
    * Check: `if (neurons.spike_production[source] > 0)`
    * Action: Add `spike_production * weight` to destination.

### Phase 5: Cleanup
**Kernels:** `clearPendingEmissionsKernel`, `clearSpikeProductionKernel`
* Resets the temporary buffers to zero to prepare for the next time step.

## 3. Kernel Configuration & Performance

* **Block Size:** The system uses a fixed `BLOCK_SIZE = 64` (a multiple of the warp size, 32) to ensure efficient thread scheduling.
* **Grid Size:** Calculated dynamically based on the number of neurons or synapses: `(N + BLOCK_SIZE - 1) / BLOCK_SIZE`.
* **Synchronization:** `cudaDeviceSynchronize()` is called at the end of `executeOneStep` to ensure all GPU tasks finish before the host records the step duration.