Create a parallel implementation of the spiking neural p system using CUDA. See snp_explanation.tex to learn more about what Spiking Neural P Systems are. Make sure to apply techniques that optimize memory access patterns and minimize latency. Make sure to use the proper representation/data structures of spikes, rules, and etc to maximize parallelism on the GPU and minimize latency.

Here are some techniques to consider when implementing the CudaSnpSimulator class. You do not have to use all of them, but consider which ones are most appropriate for your implementation:
- Use Structure of Arrays (SoA) instead of Array of Structures (AoS) for storing neuron states and rules to improve memory coalescing.
- Implement tiling strategies to load data into shared memory for faster access during simulation steps.
- Minimize divergent branches in CUDA kernels by grouping neurons with similar behavior together.
- Bank Conflict Avoidance: When using shared memory, ensure that accesses are designed to avoid bank conflicts, which can lead to serialization of memory accesses.
- Optimize kernel launch configurations (block size, grid size) based on the specific GPU architecture to maximize occupancy.
- Cache Optimization: Leverage the GPU's cache hierarchy effectively by accessing memory in a way that maximizes cache hits.

Create the CudaSnpSimulator class that implements the interface in ISnpSimulator.hpp and SnpSystemConfig.hpp. Make sure to keep the implementation efficient but simple and readable at the same time. Do not change anything but the implementation of the CudaSnpSimulator class. You can add helper functions, classes, and create new files as needed, but do not change the interface.