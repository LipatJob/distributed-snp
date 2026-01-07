I want you to revamp this notebook for visualizing various profiling/benchmarks. I've attached some sample data that is uses. They are going to be attached to in our paper. Consider also that our paper uses a two column format. Simplify the code and make the notebooks more consistent. Make the visualizations simple, easy to read and pleasant to look at. Create a new notebook for this.

Here are our implementations:
- cpu
- optimized-cuda
- sparse-cuda
- naive-cuda-mpi:linear
- naive-cuda-mpi:louvain
- naive-cuda-mpi:red-blue
- optimized-cuda-mpi:linear
- optimized-cuda-mpi:louvain
- optimized-cuda-mpi:red-blue

There needs to be 4 timestamp inputs:
- timestamp of scaling benchmark
- timestamp of distribution benchmark
- timestamp of nsys profiling
- timestamp of ncu profiling

Here's what I want to show. Feel free to add, modify, and remove as necessary
- execution time as input size increases (scaling benchmark)
- how stable the different implementations are depending on various input distributions (distribution benchmark)
- break down of MPI vs CPU vs CUDA time (nsys profiling)
- Global memory vs L1 Cache vs L2 cache usage (ncu profiling)
- Comparison of partitioning strategies

Other notes:
- the visualization of the scaling benchmark is doing to be displayed in full paper width.
- I've also attached a latex file that provides an overview of the project