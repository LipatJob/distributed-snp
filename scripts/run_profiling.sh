#!/bin/bash

# SNP System Profiling Script - Simplified Version
# Profiles SNP implementations using nsys and ncu

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/profiling/results"
BUILD_DIR="/home/shared/tmp/distributed-snp-new"
PROFILE_EXEC="${BUILD_DIR}/bin/snp_profile"

mkdir -p "$OUTPUT_DIR"

HOSTS=(
    "localhost"
    "10.0.0.2"
    "10.0.1.2"
)

# Defaults
IMPLEMENTATIONS=("optimized-cuda-mpi")
PARTITIONERS=("linear")
PROFILER="nsys"
STEPS=""
ARRAY_SIZE="2048"
OUTPUT_PREFIX="snp_profile"
NSYS_OPTS=""
NCU_OPTS=""
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

relevant_ncu_metrics=(
    "gpu__time_duration.sum"
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.pct"
    "sm__sass_branch_targets.avg"
    "sm__sass_branch_targets_threads_divergent.sum"
    "l1tex__t_bytes_lookup_hit.sum"
    "l1tex__t_bytes_lookup_miss.sum"
    "lts__t_sectors_lookup_hit.sum"
    "lts__t_sectors_lookup_miss.sum"
    "lts__t_requests_lookup_hit.sum"
    "lts__t_requests_lookup_miss.sum"
    "dram__bytes_read.sum"
    "dram__bytes_write.sum"
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"
    "smsp__warp_issue_stalled_wait_per_warp_active.pct"
    "smsp__warp_issue_stalled_membar_per_warp_active.pct"
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"
    "l1tex__t_bytes_lookup_hit.sum" "l1tex__t_bytes_lookup_miss.sum"
    "dram__bytes_read.sum" "dram__bytes_write.sum"
)

log_step() { echo -e "${BLUE}▶${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_warn() { echo -e "${YELLOW}→${NC} $1"; }

print_usage() {
    cat << EOF
Usage: $0 [options]

Options:
  -i, --implementation LIST  Comma-separated implementations with optional partitioner
                            Format: impl[:partitioner]
                            Implementations: cpu, optimized-cuda, sparse-cuda, naive-cuda-mpi, optimized-cuda-mpi
                            Partitioners: linear, louvain, rb (default: linear for MPI)
  -p, --profiler TOOL       nsys, ncu, or both (default: nsys)
  -s, --steps N             Number of steps (default: max)
  -as, --array-size N       Array size (default: 2048)
  -h, --help                Show help

Examples:
  $0 -i optimized-cuda -p ncu
  $0 -i optimized-cuda-mpi:rb
  $0 -i optimized-cuda-mpi:linear,naive-cuda-mpi:rb,sparse-cuda -p both
  $0 -i optimized-cuda-mpi:louvain -s 100
EOF
}

check_dependencies() {
    [[ "$PROFILER" =~ ^(nsys|both)$ ]] && ! command -v nsys &> /dev/null && log_error "nsys not found" && exit 1
    [[ "$PROFILER" =~ ^(ncu|both)$ ]] && ! command -v ncu &> /dev/null && log_error "ncu not found" && exit 1
    [ ! -f "$PROFILE_EXEC" ] && log_error "Profile executable not found: $PROFILE_EXEC" && exit 1
    log_success "Dependencies verified"
}

get_output_dir() {
    local tool="$1"
    local impl="$2"
    local partitioner="$3"
    
    # Only include partitioner in folder name for MPI implementations
    if [[ "$impl" == *"mpi"* ]]; then
        echo "${OUTPUT_DIR}/${tool}/${BATCH_TIMESTAMP}/${impl}-${partitioner}"
    else
        echo "${OUTPUT_DIR}/${tool}/${BATCH_TIMESTAMP}/${impl}"
    fi
}

get_base_filename() {
    local impl="$1"
    local tool="$2"
    local partitioner="$3"
    local outdir=$(get_output_dir "$tool" "$impl" "$partitioner")
    mkdir -p "$outdir"
    echo "${outdir}/${OUTPUT_PREFIX}"
}

extract_nsys_stats() {
    local base_pattern="$1"
    local is_mpi="$2"

    if [ "$is_mpi" = true ]; then
        log_step "Collecting nsys files from worker nodes..."
        local master="${HOSTS[0]}"
        local workers=("${HOSTS[@]:1}")
        for node in "${workers[@]}"; do
            local worker_file_paths=$(ssh shared@"$node" "ls ${base_pattern}"*nsys-rep 2>/dev/null || echo "")
            for worker_file in $worker_file_paths; do
                scp shared@"$node":"$worker_file" "${base_pattern}$(basename "$worker_file")"
            done
            
        done
    fi
        
    
    log_step "Extracting nsys metrics..."
    
    # Find actual generated files (nsys expands %h_rank%q patterns)
    local rep_files=$(ls ${base_pattern}*.nsys-rep 2>/dev/null || echo "")
    
    if [ -z "$rep_files" ]; then
        log_warn "No nsys-rep files found"
        return 1
    fi
    
    for rep_file in $rep_files; do
        local base="${rep_file%.nsys-rep}"
        local relevant_metrics=(
            "cuda_api_sum"
            "cuda_gpu_kern_sum"
            "cuda_gpu_mem_size_sum"
            "cuda_gpu_mem_time_sum"
            "cuda_gpu_sum"
            "cuda_kern_exec_sum"
            "cuda_api_gpu_sum"
            "mpi_event_sum"
            "mpi_msg_size_sum"
        )

        nsys stats --report $(IFS=, ; echo "${relevant_metrics[*]}") \
            --format csv --quiet --output "${base}_stats" "$rep_file" 2>/dev/null || true
    done
    
    log_success "Metrics exported"
}

extract_ncu_stats() {
    local base_pattern="$1"
    
    log_step "Extracting ncu metrics..."
    
    local rep_files=$(ls ${base_pattern}*.ncu-rep 2>/dev/null || echo "")
    for rep_file in $rep_files; do
        local base="${rep_file%.ncu-rep}"
        ncu --import "$rep_file" --metrics $(IFS=, ; echo "${relevant_ncu_metrics[*]}") --page raw --csv > "${base}_metrics.csv"
        echo -e "${GREEN}✓ Metrics saved to: ${base}_metrics.csv${NC}"
    done
}

run_mpi_profiler() {
    local tool="$1"
    local impl="$2"
    local partitioner="$3"
    local output_dir=$(get_output_dir "$tool" "$impl" "$partitioner")
    local base=$(get_base_filename "$impl" "$tool" "$partitioner")
    
    # Determine if this is an MPI implementation
    local is_mpi=false
    [[ "$impl" == *"mpi"* ]] && is_mpi=true
    
    # Set output filename pattern based on whether it's MPI
    local output="$base"
    if [ "$is_mpi" = true ]; then
        output="${base}_%h_rank%q{OMPI_COMM_WORLD_RANK}"
    fi
    
    # Build profiler command
    local profiler_cmd=""
    if [ "$tool" == "nsys" ]; then
        profiler_cmd="/usr/local/cuda/bin/nsys profile --capture-range=cudaProfilerApi --cpuctxsw=system-wide \
            --trace=cuda,mpi,nvtx,osrt --output=$output \
            --force-overwrite=true $NSYS_OPTS"
    else
        local ncu_opts="${NCU_OPTS:---section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --call-stack --launch-count 20} --metrics $(IFS=, ; echo "${relevant_ncu_metrics[*]}")"
        profiler_cmd="/usr/local/cuda/bin/ncu $ncu_opts --export $output --force-overwrite --lockstep-kernel-launch"
    fi
    
    local app_cmd
    if [ "$is_mpi" = true ]; then
        app_cmd="$PROFILE_EXEC $impl ${STEPS:-0} $partitioner $ARRAY_SIZE"
    else
        # Non-MPI implementations don't take a partitioner argument
        app_cmd="$PROFILE_EXEC $impl ${STEPS:-0} $ARRAY_SIZE"
    fi
    
    # Execute with or without MPI
    if [ "$is_mpi" = true ]; then
        for node in "${HOSTS[@]}"; do
            ssh shared@"$node" "mkdir -p $output_dir"  2>&1 || true
        done
        # Suppress MPI error messages when process exits during cleanup
        local mpi_cmd="mpirun -np ${#HOSTS[@]} --host $(IFS=, ; echo "${HOSTS[*]}") \
            --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5"
        eval "$mpi_cmd $profiler_cmd $app_cmd"
    else
        eval "$profiler_cmd $app_cmd" 2>&1 | grep -v "^Collecting\|^==" || true
    fi
    
    log_success "Profile saved: ${base}*"
    
    case "$tool" in
        "nsys") extract_nsys_stats "$base" "$is_mpi" ;;
        "ncu")  extract_ncu_stats "$base" ;;
    esac
}

run_profiling_for_impl() {
    local impl="$1"
    local partitioner="$2"
    
    echo ""
    log_step "Profiling: $impl $( [ -n "$partitioner" ] && echo "with $partitioner partitioner" )"
    
    case $PROFILER in
        nsys)
            run_mpi_profiler "nsys" "$impl" "$partitioner"
            ;;
        ncu)
            if [[ "$impl" != *"cuda"* ]]; then
                log_warn "Skipping NCU for non-CUDA implementation: $impl"
                return
            fi
            run_mpi_profiler "ncu" "$impl" "$partitioner"
            ;;
        both)
            run_mpi_profiler "nsys" "$impl" "$partitioner"
            if [[ "$impl" == *"cuda"* ]]; then
                echo ""
                run_mpi_profiler "ncu" "$impl" "$partitioner"
            fi
            ;;
    esac
}

run_profiling_batch() {
    echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  SNP System Profiling${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
    echo "  Implementations:"
    for i in "${!IMPLEMENTATIONS[@]}"; do
        echo "    - ${IMPLEMENTATIONS[$i]} $( [ -n "${PARTITIONERS[$i]}" ] && echo "with ${PARTITIONERS[$i]} partitioner" )"
    done
    echo "  Profiler: $PROFILER"
    echo "  Steps: ${STEPS:-max}"
    echo "  Array Size: $ARRAY_SIZE"
    echo "  Batch ID: $BATCH_TIMESTAMP"
    echo ""
    
    check_dependencies
    
    for i in "${!IMPLEMENTATIONS[@]}"; do
        run_profiling_for_impl "${IMPLEMENTATIONS[$i]}" "${PARTITIONERS[$i]}"
    done
    
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Profiling Complete${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
    echo -e "  Results: ${OUTPUT_DIR}/${PROFILER}/${BATCH_TIMESTAMP}/"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--implementation) 
            IFS=',' read -ra IMPL_LIST <<< "$2"
            IMPLEMENTATIONS=()
            PARTITIONERS=()
            for item in "${IMPL_LIST[@]}"; do
                if [[ "$item" == *":"* ]]; then
                    # Split impl:partitioner
                    IMPLEMENTATIONS+=("${item%%:*}")
                    PARTITIONERS+=("${item##*:}")
                else
                    # No partitioner specified, use default (ignored for non-MPI)
                    IMPLEMENTATIONS+=("$item")
                    PARTITIONERS+=("")
                fi
            done
            shift 2 
            ;;
        -p|--profiler) PROFILER="$2"; shift 2 ;;
        -s|--steps) STEPS="$2"; shift 2 ;;
        -as|--array-size) ARRAY_SIZE="$2"; shift 2 ;;
        --nsys-opts) NSYS_OPTS="$2"; shift 2 ;;
        --ncu-opts) NCU_OPTS="$2"; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) log_error "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

# Run
run_profiling_batch
