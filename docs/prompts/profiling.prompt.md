I want you to profilet eh different SNP Implementations. It want to profile using NVIDIA Nsight Systems CLI. It should profile CPU, CUDA, and MPI while the SNP model is running. It should profile these SnpSimulators when sorting a medium sized array like in the benchmark. It should be easy to add new implementations to profile. Since these some of these are distributed and needs MPI, make sure to copy the binaries to all the nodes first. Also update the Makefile to make it easy to run the profiling.

Make sure to read the following files for context on how to do this:
- benchmark/SortBenchmark.cpp
- scripts/run_distributed_benchmark.sh
- src/snp/NaiveCudaMpiSnpSimulator.cu
- src/snp/CudaMpiSnpSimulator.cu
- src/snp/SparseCudaSnpSimulator.cu
- src/snp/NaiveCpuSnpSimulator.cpp
- src/sort/SnpSort.cpp
- src/sort/ISort.hpp
- Makefile