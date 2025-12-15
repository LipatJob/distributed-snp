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

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   SNP System Profiling with NVIDIA Nsight Tools       ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_usage() {
    cat << EOF
Usage: $0 [options]

Profile SNP implementations using NVIDIA Nsight Systems and/or Nsight Compute.

Options:
  -i, --implementation NAME  Implementation to profile (default: all)
                            Options: cpu, cuda, sparse-cuda, naive-cuda-mpi,
                                     cuda-mpi, mpi, all
  -n, --num-procs N         Number of MPI processes (default: 2)
  --hostfile FILE           Path to MPI hostfile (default: ./hostfile.txt)
  --no-hostfile             Don't use hostfile, run all on localhost
  -p, --profiler TOOL       Profiler to use (default: nsys)
                            Options: nsys (Nsight Systems), ncu (Nsight Compute), both
  --nsys-opts "OPTIONS"     Additional nsys options (default: none)
  --ncu-opts "OPTIONS"      Additional ncu options (default: none)
  -o, --output PREFIX       Output file prefix (default: snp_profile)
  -s, --steps N             Number of simulation steps to run (default: max/all)
  -h, --help                Show this help message

Examples:
  $0                                     # Profile all with Nsight Systems
  $0 -i cuda -p ncu                      # Profile CUDA with Nsight Compute
  $0 -i cuda -p both                     # Profile CUDA with both tools
  $0 -i mpi -n 4 -p nsys                 # Profile MPI with Nsight Systems
  $0 -i sparse-cuda -p ncu --ncu-opts "--set full"  # Detailed kernel profiling
  $0 -i cuda -s 100                      # Profile CUDA for 100 steps only

Available Implementations:
  cpu           - NaiveCpuSnp (single process)
  cuda          - CudaSnp (single process, requires GPU)
  sparse-cuda   - SparseCudaSnp (single process, requires GPU)
  naive-cuda-mpi- NaiveCudaMpiSnp (distributed, requires GPU)
  cuda-mpi      - CudaMpiSnp (distributed, requires GPU)
  mpi           - All MPI implementations
  all           - All implementations (default)

Profiler Tools:
  nsys  - Nsight Systems: System-wide performance analysis (CPU, GPU, MPI)
          Generates .nsys-rep files viewable with nsys-ui
  ncu   - Nsight Compute: Detailed GPU kernel profiling
          Generates .ncu-rep files viewable with ncu-ui
  both  - Run both profilers sequentially

Notes:
  - For distributed implementations, binaries are automatically copied to all nodes
  - Nsight Systems output (.nsys-rep) for system-wide analysis
  - Nsight Compute output (.ncu-rep) for detailed kernel metrics
  - Use 'nsys-ui' or 'ncu-ui' to view results graphically
  - Requires NVIDIA Nsight Systems and/or Nsight Compute installed
EOF
}

check_dependencies() {
    # Check for nsys if needed
    if [[ "$PROFILER" == "nsys" ]] || [[ "$PROFILER" == "both" ]]; then
        if ! command -v nsys &> /dev/null; then
            echo -e "${RED}Error: NVIDIA Nsight Systems (nsys) not found!${NC}"
            echo "Please install Nsight Systems from:"
            echo "  https://developer.nvidia.com/nsight-systems"
            echo ""
            echo "Or install via package manager:"
            echo "  Ubuntu/Debian: apt install nsight-systems-cli"
            exit 1
        fi
    fi
    
    # Check for ncu if needed
    if [[ "$PROFILER" == "ncu" ]] || [[ "$PROFILER" == "both" ]]; then
        if ! command -v ncu &> /dev/null; then
            echo -e "${RED}Error: NVIDIA Nsight Compute (ncu) not found!${NC}"
            echo "Please install Nsight Compute from:"
            echo "  https://developer.nvidia.com/nsight-compute"
            echo ""
            echo "Or install via package manager:"
            echo "  Ubuntu/Debian: apt install nsight-compute-cli"
            exit 1
        fi
    fi
    
    # Check if profile executable exists
    if [ ! -f "$PROFILE_EXEC" ]; then
        echo -e "${RED}Error: Profile executable not found at $PROFILE_EXEC${NC}"
        echo "Please build the project first with 'make build'"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Dependencies verified${NC}"
}

copy_binaries_to_nodes() {
    if [ "$USE_HOSTFILE" = false ]; then
        echo -e "${YELLOW}Running locally only, skipping distribution${NC}"
        return
    fi
    
    if [ ! -f "$HOSTFILE" ]; then
        echo -e "${YELLOW}Warning: Hostfile not found at $HOSTFILE${NC}"
        echo "Creating default hostfile for localhost..."
        echo "localhost slots=2" > "$HOSTFILE"
        return
    fi
    
    echo -e "${BLUE}Copying binaries to remote nodes...${NC}"
    
    # Parse hostfile to get unique nodes (excluding localhost)
    NODES=$(grep -v "^#" "$HOSTFILE" | grep -v "^localhost" | awk '{print $1}' | sort -u || true)
    
    if [ -z "$NODES" ]; then
        echo -e "${YELLOW}No remote nodes found in hostfile${NC}"
        return
    fi
    
    for node in $NODES; do
        echo -e "  ${BLUE}→ $node${NC}"
        
        # Create remote directory if needed
        ssh "$node" "mkdir -p ~/distributed-snp-new/build" 2>/dev/null || {
            echo -e "${RED}    ✗ Failed to connect to $node${NC}"
            continue
        }
        
        # Copy the profile executable
        scp -q "$PROFILE_EXEC" "$node:~/distributed-snp-new/build/" || {
            echo -e "${RED}    ✗ Failed to copy to $node${NC}"
            continue
        }
        
        echo -e "${GREEN}    ✓ Complete${NC}"
    done
    
    echo -e "${GREEN}Binary distribution complete${NC}"
}

run_profiling_nsys() {
    local impl="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="${OUTPUT_DIR}/${OUTPUT_PREFIX}_nsys_${impl}_${timestamp}_%h_rank%q{OMPI_COMM_WORLD_RANK}"
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Profiling with Nsight Systems: ${impl}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    
    # Build nsys command
    # We use nsys profile to capture CPU, CUDA, and MPI activity
    local nsys_trace="cuda,mpi,nvtx,osrt"
    local nsys_cmd="/usr/local/cuda/bin/nsys profile --capture-range=cudaProfilerApi --trace=$nsys_trace --output=$output_file --force-overwrite=true --stats=true"
    
    # Add custom options if provided
    if [ -n "$NSYS_OPTS" ]; then
        nsys_cmd="$nsys_cmd $NSYS_OPTS"
    fi

    # Build MPI command
    mpi_cmd="mpirun -np 2 --host localhost,10.0.0.2 --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5"
    
    # Construct full command with optional steps parameter
    if [ -n "$STEPS" ]; then
        local full_cmd="$mpi_cmd $nsys_cmd $PROFILE_EXEC $impl $STEPS"
    else
        local full_cmd="$mpi_cmd $nsys_cmd $PROFILE_EXEC $impl"
    fi
    
    echo -e "${YELLOW}Command:${NC} $full_cmd"
    echo ""
    
    # Execute
    eval $full_cmd
    
    echo ""
    echo -e "${GREEN}✓ Nsight Systems profile saved to: ${output_file}.nsys-rep${NC}"
    echo -e "${YELLOW}  View with: nsys-ui ${output_file}.nsys-rep${NC}"
    echo ""
}

run_profiling_ncu() {
    local impl="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="${OUTPUT_DIR}/${OUTPUT_PREFIX}_ncu_${impl}_${timestamp}"
    mkdir -p "$OUTPUT_DIR"
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Profiling with Nsight Compute: ${impl}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    
    # Check if this is a GPU implementation
    if [[ "$impl" != *"cuda"* ]]; then
        echo -e "${YELLOW}Warning: Nsight Compute is for GPU kernel profiling.${NC}"
        echo -e "${YELLOW}Implementation '$impl' may not have GPU kernels to profile.${NC}"
        echo -e "${YELLOW}Skipping Nsight Compute profiling for this implementation.${NC}"
        return
    fi
    
    # Build ncu command
    # Default: profile all kernels with detailed metrics
    local ncu_cmd="/usr/local/cuda/bin/ncu --set full --export $output_file --force-overwrite --call-stack"
    
    # Add custom options if provided
    if [ -n "$NCU_OPTS" ]; then
        ncu_cmd="/usr/local/cuda/bin/ncu $NCU_OPTS --export $output_file --force-overwrite"
    fi

    # For MPI implementations, we need to use mpirun
    if [[ "$impl" == *"mpi"* ]]; then
        output_file="${output_file}_%h_rank%q{OMPI_COMM_WORLD_RANK}"
        mpi_cmd="mpirun -np 2 --host localhost,10.0.0.2 --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5"
        if [ -n "$STEPS" ]; then
            local full_cmd="$mpi_cmd $ncu_cmd $PROFILE_EXEC $impl $STEPS"
        else
            local full_cmd="$mpi_cmd $ncu_cmd $PROFILE_EXEC $impl"
        fi
    else
        if [ -n "$STEPS" ]; then
            local full_cmd="$ncu_cmd $PROFILE_EXEC $impl $STEPS"
        else
            local full_cmd="$ncu_cmd $PROFILE_EXEC $impl"
        fi
    fi
    
    echo -e "${YELLOW}Command:${NC} $full_cmd"
    echo ""
    
    # Execute
    eval $full_cmd
    
    echo ""
    echo -e "${GREEN}✓ Nsight Compute profile saved to: ${output_file}.ncu-rep${NC}"
    echo -e "${YELLOW}  View with: ncu-ui ${output_file}.ncu-rep${NC}"
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
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Execution
# ============================================================================

print_header

echo "Configuration:"
echo "  Implementation: $IMPLEMENTATION"
echo "  Profiler: $PROFILER"
echo "  MPI Processes: $NUM_PROCS"
if [ -n "$STEPS" ]; then
    echo "  Steps: $STEPS"
else
    echo "  Steps: max (run to completion)"
fi
echo "  Output Directory: $OUTPUT_DIR"
echo "  Hostfile: $HOSTFILE"
echo ""

check_dependencies

# Copy binaries for distributed implementations
if [[ "$IMPLEMENTATION" == *"mpi"* ]] || [ "$IMPLEMENTATION" = "all" ]; then
    copy_binaries_to_nodes
fi

# Run profiling based on selected implementation
case $IMPLEMENTATION in
    cpu)
        run_profiling "cpu"
        ;;
    cuda)
        run_profiling "cuda"
        ;;
    sparse-cuda)
        run_profiling "sparse-cuda"
        ;;
    naive-cuda-mpi)
        run_profiling "naive-cuda-mpi"
        ;;
    cuda-mpi)
        run_profiling "cuda-mpi"
        ;;
    *)
        echo -e "${RED}Error: Unknown implementation '$IMPLEMENTATION'${NC}"
        print_usage
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       Profiling Complete!                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"

if [[ "$PROFILER" == "nsys" ]] || [[ "$PROFILER" == "both" ]]; then
    echo "  Nsight Systems:"
    echo "    - View profiles: nsys-ui $OUTPUT_DIR/*_nsys_*.nsys-rep"
    echo "    - Generate reports: nsys stats $OUTPUT_DIR/*_nsys_*.nsys-rep"
    echo "    - Compare implementations side-by-side in nsys-ui"
fi

if [[ "$PROFILER" == "ncu" ]] || [[ "$PROFILER" == "both" ]]; then
    echo "  Nsight Compute:"
    echo "    - View kernel profiles: ncu-ui $OUTPUT_DIR/*_ncu_*.ncu-rep"
    echo "    - Generate reports: ncu --import $OUTPUT_DIR/*_ncu_*.ncu-rep"
    echo "    - Compare kernels across runs"
fi

echo ""
