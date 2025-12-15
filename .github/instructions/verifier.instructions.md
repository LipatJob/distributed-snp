# Verification Instructions: Sparse CUDA SNP Simulator Comparison

## Objective
Compare the implementation of `distributed-snp/src/snp/SparseCudaSnpSimulator.cu` with the reference implementation `sparse_snp/src/snp_static_optimized.cu`.

## Files to Analyze
1.  **Target Implementation**: `distributed-snp/src/snp/SparseCudaSnpSimulator.cu`
2.  **Reference Implementation**: `sparse_snp/src/snp_static_optimized.cu`

## Comparison Criteria

### 1. Data Structures & Memory Layout
*   **Synapse Storage**: Compare `DeviceSynapseMatrix` in `SparseCudaSnpSimulator.cu` with the `trans_matrix` in `snp_static_optimized.cu`.
    *   **Layout Analysis**: Determine if the matrices are stored in Row-Major (Neuron-Major) or Column-Major (Interleaved) order.
    *   **Coalescing**: Analyze which layout enables coalesced memory access when threads (neurons) iterate through their outgoing synapses.
*   **Rule Storage**: Compare how rules are accessed. `SparseCudaSnpSimulator.cu` uses `DeviceRuleVector` and `DeviceNeuronRuleMap`. Compare this with the rule access pattern in `snp_static_optimized.cu`.

### 2. Kernel Implementation
*   **Spiking Vector Calculation**: Compare `k_calc_spiking_vector` (Target) with `kalc_spiking_vector_for_optimized` (Reference).
    *   Both use 1 thread per neuron. Compare how they iterate rules and select the active rule.
*   **System Transition**: Compare `k_step_compressed` (Target) with `kalc_transition_optimized` (Reference).
    *   **Loop Structure**: Compare the inner loop over `max_out_degree` (or `z`).
    *   **Memory Access**: Specifically check the index calculation for reading the destination neuron ID.
        *   Target: `synapse_matrix[nid * max_out_degree + i]`
        *   Reference: `trans_matrix[j * n + nid]` (Verify this pattern).
    *   **Atomics**: Compare the usage of `atomicAdd` for spike distribution.

### 3. Performance & Optimization
*   **Memory Coalescing**: This is the critical comparison point.
    *   Explain why one implementation achieves coalesced access while the other might not.
*   **Divergence**: Both implementations use 1 thread per neuron. Discuss if the memory layout impacts warp divergence or cache efficiency.

### 4. Correctness & Features
*   **Delay Handling**: Compare how delays are handled.
    *   Target: Uses `pending_emission` buffer.
    *   Reference: Checks `delays_vector` inside the transition loop.
*   **Edge Cases**: Check handling of padding (e.g., `-1` entries in the matrix).

## Output Format
Provide a detailed comparison report with the following sections:
1.  **Executive Summary**: High-level findings, focusing on the memory layout difference.
2.  **Structural Comparison**: Detailed analysis of `DeviceSynapseMatrix` vs `trans_matrix`.
3.  **Algorithmic Comparison**: Kernel logic differences.
4.  **Performance Analysis**: Theoretical impact of the memory layout on bandwidth.
5.  **Recommendations**: Specific code improvements for `SparseCudaSnpSimulator.cu` to match or exceed the reference optimization.
