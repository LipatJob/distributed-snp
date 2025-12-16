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
log_success "Profiling setup complete"
echo ""

# Profile implementations
log_step "Profiling implementations with Nsight Systems"
for impl in "sparse-cuda" "optimized-cuda"; do
    log_info "Profiling $impl"
    make profile ARGS="-i $impl -p nsys" && sleep 1
done

for part in "linear" "louvain" "red-blue"; do
    log_info "Profiling naive-cuda-mpi ($part)"
    make profile ARGS="-i naive-cuda-mpi -p nsys -pt $part" && sleep 1
    log_info "Profiling optimized-cuda-mpi ($part)"
    make profile ARGS="-i optimized-cuda-mpi -p nsys -pt $part" && sleep 1
done
log_success "Nsight Systems profiling complete"
echo ""

# NCU profiling
log_step "Collecting kernel metrics with Nsight Compute"
for impl in "sparse-cuda" "optimized-cuda"; do
    log_info "Profiling $impl kernels"
    make profile ARGS="-i $impl -p ncu -s 3" && sleep 1
done
log_success "Nsight Compute profiling complete"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Pipeline Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
