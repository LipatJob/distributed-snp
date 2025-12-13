I want you to profile CudaMpiSnpSimulator.cu and NaiveCudaMpiSnpSimulator.cu. It want to profile using NVIDIA Nsight Systems. It should profile these SnpSimulators when sorting a medium sized array like in the benchmark. It should be easy to add new implementations to profile. I want a make command that will build, distribute, and ready the program for profiling. Then I want a script that will run the program which I could use in the Nsight Systems UI.

Make sure to read the following files for context on how to do this:
- benchmark/SortBenchmark.cpp
- src/snp/NaiveCudaMpiSnpSimulator.cu
- src/snp/CudaMpiSnpSimulator.cu
- src/sort/SnpSort.cpp
- src/sort/ISort.hpp
- Makefile

