#!/bin/bash
# run_distributed_tests.sh - Run tests across distributed hosts

set -e

# Configuration
BUILD_DIR="/home/shared/distributed-snp-new/build"
HOSTS="localhost,10.0.0.2"
NUM_PROCS=2

echo "=============================================="
echo "Running distributed tests"
echo "Hosts: $HOSTS"
echo "Processes: $NUM_PROCS"
echo "=============================================="

# Set library path for both local and remote execution
export LD_LIBRARY_PATH=${BUILD_DIR}/lib:${BUILD_DIR}/_deps/googletest-build/lib:$LD_LIBRARY_PATH

# # Run matrix operations tests with MPI
# echo ""
# echo "Running Matrix Operations Tests..."
# echo "----------------------------------------------"
# mpirun -np ${NUM_PROCS} \
#        --host ${HOSTS} \
#        --allow-run-as-root \
#        --mca btl_tcp_if_include ens5 \
# 	   --mca oob_tcp_if_include ens5 \
#        -x LD_LIBRARY_PATH \
#        ${BUILD_DIR}/test_matrix_ops

# Run SNP simulator tests with MPI
echo ""
echo "Running SNP Simulator Tests..."
echo "----------------------------------------------"
mpirun -np ${NUM_PROCS} \
       --host ${HOSTS} \
       --allow-run-as-root \
       --mca btl_tcp_if_include ens5 \
       --mca oob_tcp_if_include ens5 \
       -x LD_LIBRARY_PATH \
       ${BUILD_DIR}/test_snp_simulator

# Run sorting tests with MPI
echo ""
echo "Running Sorting Tests..."
echo "----------------------------------------------"
mpirun -np ${NUM_PROCS} \
       --host ${HOSTS} \
       --allow-run-as-root \
       --mca btl_tcp_if_include ens5 \
       --mca oob_tcp_if_include ens5 \
       -x LD_LIBRARY_PATH \
       ${BUILD_DIR}/test_sort

echo ""
echo "=============================================="
echo "All distributed tests complete!"
echo "=============================================="
