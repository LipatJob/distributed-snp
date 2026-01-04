I'm planning to create visualizations that compare the results of nsys reports of different SNP implementations. The purpose is to understand the performance characteristic of the implementation better. These visualizations will be included on our paper

Here's folder structure of the reports: nsys/<timestamp>/<implementation>/<report>

Here are the implementations available:
  - optimized-cuda
  - sparse-cuda
  - naive-cuda-mpi:linear
  - naive-cuda-mpi:louvain
  - naive-cuda-mpi:red-blue
  - optimized-cuda-mpi:linear
  - optimized-cuda-mpi:louvain
  - optimized-cuda-mpi:red-blue

I want you to list down a list of visualizations that we must have in the analysis of our implementations of our paper.

Here's what I am thinking so far. Feel free to change and add more as you see fit.

For CUDA implementations,
- Show how much time is spent on CPU vs CUDA kernel
- Show how much time is spent on CUDA API
- Show how much time is spent per kernel
- Show how much data is transferred from host to device

For CUDA+MPI implementations,
- include visualizations for CUDA implementations
- Show how much time is spend on CPU vs CUDA vs MPI
- Show how much data is sent through MPI
- Compare the partitioning strategies for MPI implementations (linear, louvain, red-blue)