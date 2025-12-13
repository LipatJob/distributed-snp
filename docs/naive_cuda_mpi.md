# NaiveCudaMpiSnpSimulator: Architecture & Implementation Guide

This document details the architectural decisions and implementation logic for the `NaiveCudaMpiSnpSimulator`. This class provides a hybrid **MPI + CUDA** implementation of Spiking Neural P Systems, designed to run on distributed clusters with NVIDIA T4 GPUs.

## 1. Design Philosophy

The implementation follows the **"Distributed State, Replicated Synapses"** pattern. This approach prioritizes implementation simplicity and correctness over minimal memory footprint.

* **Distributed Neurons:** The state vector $C^{(k)}$ (configuration) is partitioned across MPI ranks. Each rank owns a disjoint subset of neurons.
* **Replicated Synapses:** The connectivity graph (synapses) is fully replicated on every GPU. This avoids complex graph partitioning and ghost-cell logic, simplifying the "spike propagation" phase to a linear scan.
* **Global Synchronization:** The system synchronizes the "production vector" (spikes emitted) at every time step, ensuring all nodes have a coherent view of system activity.

## 2. Partitioning Strategy

Given $N$ total neurons and $P$ MPI ranks:

1.  **Neuron Ownership:** Neurons are partitioned using a 1D block distribution.
    * Rank $r$ owns neurons in range $[Start_r, End_r)$.
    * Each rank allocates device memory **only** for its local neurons (`LocalNeuronData`).
2.  **Rule Ownership:** Rules associated with local neurons are stored locally (`LocalRuleData`).
3.  **Synapse Storage:** Every rank stores the complete list of $M$ synapses (`GlobalSynapseData`).

## 3. Data Structures

To optimize for GPU throughput, we use **Structure of Arrays (SoA)** layouts.

### Device: `LocalNeuronData`
Stores the state for neurons owned by the local rank.
* `current_spikes`: $C^{(k)}$ for local neurons.
* `delay_timer`: Tracks the delay $d$ for firing rules.
* `is_open`: Boolean status mask $St^{(k)}$.
* `pending_emission`: Spikes scheduled for future release.

### Device: `GlobalSynapseData`
A read-only, fully replicated list of all connections in the system.
* `source_global_id`: Global index of the source neuron.
* `dest_global_id`: Global index of the destination neuron.
* `weight`: Synaptic weight.

## 4. Execution Loop (The Step Function)

The simulation proceeds in discrete time steps $k$. Each step consists of three distinct phases:

### Phase 1: Local Compute (CUDA)
**Kernel:** `kLocalComputeAndProduce`
**Goal:** Determine which local neurons fire and how many spikes they produce.

1.  **Update Delays:** Decrement `delay_timer` for closed neurons. If a timer reaches 0, the neuron opens ($St_i = 1$).
2.  **Rule Selection:** For every open neuron, the kernel scans applicable rules.
    * **Determinism:** As per the reference guide, we strictly ignore non-determinism and select the first applicable rule (lowest ID).
3.  **Consumption & Production:**
    * Spikes are consumed from `current_spikes`.
    * If `rule.delay > 0`, the neuron closes, and production is stored in `pending_emission`.
    * If `rule.delay == 0`, spikes are added immediately to the output buffer `local_production`.

### Phase 2: Global Synchronization (MPI)
**Operation:** `MPI_Allgatherv`
**Goal:** Create a unified view of system activity.

1.  Each rank copies its `local_production` vector (size $N_{local}$) from Device to Host.
2.  `MPI_Allgatherv` combines these chunks into a `global_production` vector (size $N_{total}$).
3.  The full `global_production` vector is copied back to Device memory on all ranks.

This vector corresponds to the term $(Iv^{(k)} \cdot M_{\Pi})$ in the transition equation, representing the net spikes generated before distribution.

### Phase 3: Global Distribution (CUDA)
**Kernel:** `kDistributeGlobalSpikes`
**Goal:** Propagate spikes from *any* neuron in the system to *local* neurons.

1.  Threads iterate over the **Global Synapse List**.
2.  For each synapse $(src, dest)$:
    * Check `global_production[src]`. If 0, skip.
    * Check if `dest` belongs to the **current rank**.
    * If both are true, add `spikes * weight` to `local_neurons[dest]`.
3.  **Masking:** Spikes are only added if `is_open[dest]` is true, satisfying the system requirement that closed neurons ignore inputs.

## 5. Edge Case Handling

* **Zero Neurons/Rules:** All memory allocations and kernel launches are guarded by size checks (`if (count > 0)`). This allows ranks to operate safely even if they are assigned an empty partition (e.g., if $Ranks > Neurons$).
* **Small Inputs:** The 1D partitioning handles remainders ( $N \% P$ ) by assigning one extra neuron to the first $R$ ranks, ensuring load balancing.
* **Atomic Operations:** Phase 3 uses `atomicAdd` to handle multiple synapses feeding into a single local neuron simultaneously.

## 6. Performance Considerations

* **Latency vs. Bandwidth:** This "Naive" implementation is bandwidth-heavy due to `MPI_Allgatherv`. It scales well for dense spiking activity but may be inefficient for sparse systems compared to a point-to-point communication model.
* **Memory:** Replicating synapses costs $O(M)$ memory per GPU. For massive graphs exceeding GPU memory, a graph-partitioned approach would be required.