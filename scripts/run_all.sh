# !/bin/bash

echo "Building and testing all implementations..."
make build test

echo "Running benchmarks for all implementations..."
make benchmark

echo "Setting up for profiling..."
sudo sysctl kernel.perf_event_paranoid=1
ssh shared@10.0.0.2 "sudo sysctl kernel.perf_event_paranoid=1"

echo "Profiling all implementations..."
make profile ARGS="-i cuda -p nsys"
sleep 2
make profile ARGS="-i sparse-cuda -p nsys"
sleep 2
make profile ARGS="-i cuda-mpi -p nsys"
sleep 2
make profile ARGS="-i naive-cuda-mpi -p nsys"
sleep 2

echo "Collecting NVIDIA Compute Utility profiles..."
make profile ARGS="-i cuda -p ncu -s 3"
sleep 2
make profile ARGS="-i sparse-cuda -s 3"
sleep 2