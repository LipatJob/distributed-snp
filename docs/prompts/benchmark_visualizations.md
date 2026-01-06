I now want you to generate visualizations using python notebooks for the two benchmarks. The main benchmark driver is in SortBenchmark.cpp and this is ran through run_benchmark.sh and run_all.sh. These visualizations will be included on our paper to compare the different SNP implementations.

There are two benchmarks here (which are defined in run_all.sh):
- varied distributions (random, reverse sorted, etc) -> this compared whether the performance the implementation is table across different distributions (i.e. grouping should be by implementation)
- scaling from tiny to large inputs sizes

I want to input the timestamp the two timestamps associated to the benchmarks.

I've attached python notebook for reference of another visualization. I've also provided sample data.