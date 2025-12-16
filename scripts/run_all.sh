#!/bin/bash
# Complete SNP System Test, Benchmark, and Profile Pipeline
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_step() { echo -e "${BLUE}▶${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_info() { echo -e "${YELLOW}→${NC} $1"; }

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SNP System - Complete Test & Benchmark Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Build and Test
log_step "Building and testing implementations"
make build test
log_success "Build and tests complete"
echo ""

# Benchmark
log_step "Running benchmarks"
make benchmark
log_success "Benchmarks complete"
echo ""

# Setup profiling permissions
log_step "Configuring profiling permissions"
sudo sysctl -q kernel.perf_event_paranoid=1
ssh shared@10.0.0.2 "sudo sysctl -q kernel.perf_event_paranoid=1" 2>/dev/null || log_info "Remote node unavailable, skipping"
ssh shared@10.0.1.2 "sudo sysctl -q kernel.perf_event_paranoid=1" 2>/dev/null || log_info "Remote node unavailable, skipping"
log_success "Profiling setup complete"
echo ""

# Profile implementations
log_step "Profiling implementations with Nsight Systems"

implementations=(
    "cpu"
    "optimized-cuda"
    "sparse-cuda"
    "naive-cuda-mpi:linear"
    "naive-cuda-mpi:louvain"
    "naive-cuda-mpi:rb"
    "optimized-cuda-mpi:linear"
    "optimized-cuda-mpi:louvain"
    "optimized-cuda-mpi:red-blue"
)
make profile ARGS="-p nsys -i $(IFS=,; echo "${implementations[*]}")" && sleep 1
log_success "Nsight Systems profiling complete"
echo ""

# NCU profiling
log_step "Collecting kernel metrics with Nsight Compute"
implementations=(
    "optimized-cuda"
    "sparse-cuda"
)
make profile ARGS="-s 3 -p ncu -i $(IFS=,; echo "${implementations[*]}")" && sleep 1
log_success "Nsight Compute profiling complete"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Pipeline Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
