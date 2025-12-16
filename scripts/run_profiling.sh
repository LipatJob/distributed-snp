#!/bin/bash

# SNP System Profiling Script with NVIDIA Nsight Systems and Nsight Compute
# Profiles different SNP implementations (CPU, CUDA, MPI) using nsys and ncu CLI
# Handles distributed deployments by copying binaries to all nodes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HOSTFILE="${PROJECT_ROOT}/hostfile.txt"
OUTPUT_DIR="${PROJECT_ROOT}/profiling/results"


BUILD_DIR="/home/shared/tmp/distributed-snp-new"
PROFILE_EXEC="${BUILD_DIR}/bin/snp_profile"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Default options
NUM_PROCS=2
IMPLEMENTATION="all"
USE_HOSTFILE=true
NSYS_OPTS=""
NCU_OPTS=""
PROFILER="nsys"  # Options: nsys, ncu, both
OUTPUT_PREFIX="snp_profile"
STEPS=""  # Empty means run to completion (max steps)
PARTITIONER=""  # Empty means use default (linear); Options: linear, louvain, red-blue
ARRAY_SIZE="2048"  # Default array size

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Logging functions
log_step() { echo -e "${BLUE}▶${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_warn() { echo -e "${YELLOW}→${NC} $1"; }

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  SNP System Profiling${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
    echo ""
}

print_usage() {
    cat << EOF
Usage: $0 [options]

Options:
  -i, --implementation NAME  Implementation (default: all)
  -p, --profiler TOOL       nsys, ncu, or both (default: nsys)
  -s, --steps N             Number of steps (default: max)
  --partitioner TYPE        linear, louvain, or red-blue (default: linear)
  --array-size N            Array size (default: 2048)
  -h, --help                Show help

Implementations: cpu, optimized-cuda, sparse-cuda, naive-cuda-mpi, optimized-cuda-mpi

Examples:
  $0 -i optimized-cuda -p ncu
  $0 -i optimized-cuda-mpi --partitioner louvain -s 100
EOF
}

check_dependencies() {
    if [[ "$PROFILER" == "nsys" ]] || [[ "$PROFILER" == "both" ]]; then
        if ! command -v nsys &> /dev/null; then
            log_error "NVIDIA Nsight Systems (nsys) not found"
            exit 1
        fi
    fi
    
    if [[ "$PROFILER" == "ncu" ]] || [[ "$PROFILER" == "both" ]]; then
        if ! command -v ncu &> /dev/null; then
            log_error "NVIDIA Nsight Compute (ncu) not found"
            exit 1
        fi
    fi
    
    if [ ! -f "$PROFILE_EXEC" ]; then
        log_error "Profile executable not found at $PROFILE_EXEC"
        exit 1
    fi
    
    log_success "Dependencies verified"
}

run_profiling_nsys() {
    local impl="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="${OUTPUT_DIR}/${OUTPUT_PREFIX}_nsys_${impl}_${timestamp}_%h_rank%q{OMPI_COMM_WORLD_RANK}"
    
    log_step "Profiling with Nsight Systems: ${impl}"
    
    local nsys_cmd="/usr/local/cuda/bin/nsys profile --capture-range=cudaProfilerApi \
--trace=cuda,mpi,nvtx,osrt --output=$output_file --force-overwrite=true --stats=true"
    
    [ -n "$NSYS_OPTS" ] && nsys_cmd="$nsys_cmd $NSYS_OPTS"

    local mpi_cmd="mpirun -np 2 --host localhost,10.0.0.2 --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5"
    
    local args="$impl ${STEPS:-0} ${PARTITIONER:-linear} $ARRAY_SIZE"
    local full_cmd="$mpi_cmd $nsys_cmd $PROFILE_EXEC $args"
    
    eval $full_cmd 2>&1 | grep -v "^Collecting" || true
    
    log_success "Profile saved: ${output_file}.nsys-rep"
    echo ""
}

run_profiling_ncu() {
    local impl="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="${OUTPUT_DIR}/${OUTPUT_PREFIX}_ncu_${impl}_${timestamp}"
    
    if [[ "$impl" != *"cuda"* ]]; then
        log_warn "Skipping NCU for non-GPU implementation: $impl"
        return
    fi
    
    log_step "Profiling with Nsight Compute: ${impl}"
    
    local ncu_cmd="/usr/local/cuda/bin/ncu --set full --export $output_file --force-overwrite --call-stack"
    [ -n "$NCU_OPTS" ] && ncu_cmd="/usr/local/cuda/bin/ncu $NCU_OPTS --export $output_file --force-overwrite"

    if [[ "$impl" == *"mpi"* ]]; then
        output_file="${output_file}_%h_rank%q{OMPI_COMM_WORLD_RANK}"
        local mpi_cmd="mpirun -np 2 --host localhost,10.0.0.2 --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5"
        local args="$impl ${STEPS:-0} ${PARTITIONER:-linear} $ARRAY_SIZE"
        eval "$mpi_cmd $ncu_cmd $PROFILE_EXEC $args" 2>&1 | grep -v "^==" || true
    else
        local args="$impl ${STEPS:-0} linear $ARRAY_SIZE"
        eval "$ncu_cmd $PROFILE_EXEC $args" 2>&1 | grep -v "^==" || true
    fi
    
    log_success "Profile saved: ${output_file}.ncu-rep"
    
    # Extract metrics
    local relevant_metrics=(
        "gpu__time_duration.sum" "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.pct"
        "l1tex__t_bytes_lookup_hit.sum" "l1tex__t_bytes_lookup_miss.sum"
        "dram__bytes_read.sum" "dram__bytes_write.sum"
    )
    ncu --import $output_file.ncu-rep --metrics $(IFS=, ; echo "${relevant_metrics[*]}") \
        --page raw --csv > "${output_file}_metrics.csv" 2>/dev/null
    log_success "Metrics exported: ${output_file}_metrics.csv"
    echo ""
}

run_profiling() {
    local impl="$1"
    
    case $PROFILER in
        nsys)
            run_profiling_nsys "$impl"
            ;;
        ncu)
            run_profiling_ncu "$impl"
            ;;
        both)
            run_profiling_nsys "$impl"
            run_profiling_ncu "$impl"
            ;;
        *)
            echo -e "${RED}Error: Unknown profiler '$PROFILER'${NC}"
            exit 1
            ;;
    esac
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--implementation)
            IMPLEMENTATION="$2"
            shift 2
            ;;
        -n|--num-procs)
            NUM_PROCS="$2"
            shift 2
            ;;
        --hostfile)
            HOSTFILE="$2"
            shift 2
            ;;
        --no-hostfile)
            USE_HOSTFILE=false
            shift
            ;;
        -p|--profiler)
            PROFILER="$2"
            shift 2
            ;;
        --nsys-opts)
            NSYS_OPTS="$2"
            shift 2
            ;;
        --ncu-opts)
            NCU_OPTS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        -s|--steps)
            STEPS="$2"
            shift 2
            ;;
        -pt|--partitioner)
            PARTITIONER="$2"
            shift 2
            ;;
        -as|--array-size)
            ARRAY_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Execution
# ============================================================================

print_header

echo "  Implementation: $IMPLEMENTATION"
echo "  Profiler: $PROFILER"
[ -n "$STEPS" ] && echo "  Steps: $STEPS" || echo "  Steps: max"
[ -n "$PARTITIONER" ] && echo "  Partitioner: $PARTITIONER"
echo "  Array Size: $ARRAY_SIZE"
echo ""

check_dependencies

# Run profiling
case $IMPLEMENTATION in
    cpu|optimized-cuda|sparse-cuda|naive-cuda-mpi|optimized-cuda-mpi)
        run_profiling "$IMPLEMENTATION"
        ;;
    *)
        log_error "Unknown implementation: $IMPLEMENTATION"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Profiling Complete${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "  Results: $OUTPUT_DIR"
[[ "$PROFILER" == "nsys" ]] || [[ "$PROFILER" == "both" ]] && \
    echo -e "  View: nsys-ui $OUTPUT_DIR/*_nsys_*.nsys-rep"
[[ "$PROFILER" == "ncu" ]] || [[ "$PROFILER" == "both" ]] && \
    echo -e "  View: ncu-ui $OUTPUT_DIR/*_ncu_*.ncu-rep"
echo ""
