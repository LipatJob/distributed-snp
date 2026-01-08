#!/bin/bash
# Run distributed tests across MPI nodes

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BUILD_DIR="/home/shared/tmp/distributed-snp-new"
HOSTS="localhost,10.0.0.3,10.0.1.3"
NUM_PROCS=3

echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SNP Distributed Tests${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
echo -e "  Hosts: ${HOSTS}"
echo -e "  Processes: ${NUM_PROCS}"
echo ""

# Set library path
export LD_LIBRARY_PATH=${BUILD_DIR}/lib:${BUILD_DIR}/_deps/googletest-build/lib:$LD_LIBRARY_PATH

MPI_CMD="mpirun -np ${NUM_PROCS} --host ${HOSTS} --allow-run-as-root \
  --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5 -x LD_LIBRARY_PATH"

# Run SNP simulator tests
echo -e "${BLUE}▶${NC} Running SNP simulator tests"
$MPI_CMD ${BUILD_DIR}/bin/test_snp_simulator
echo -e "${GREEN}✓${NC} SNP simulator tests passed"
echo ""

# Run sorting tests
echo -e "${BLUE}▶${NC} Running sort tests"
$MPI_CMD ${BUILD_DIR}/bin/test_sort
echo -e "${GREEN}✓${NC} Sort tests passed"

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All tests passed!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
