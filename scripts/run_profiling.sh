#!/bin/bash

# SNP System Profiling Script with NVIDIA Nsight Systems
# Profiles different SNP implementations (CPU, CUDA, MPI) using nsys CLI
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
OUTPUT_PREFIX="snp_profile"

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
    echo -e "${BLUE}║       SNP System Profiling with NVIDIA Nsight         ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_usage() {
    cat << EOF
Usage: $0 [options]

Profile SNP implementations using NVIDIA Nsight Systems CLI.

Options:
  -i, --implementation NAME  Implementation to profile (default: all)
                            Options: cpu, cuda, sparse-cuda, naive-cuda-mpi,
                                     cuda-mpi, mpi, all
  -n, --num-procs N         Number of MPI processes (default: 2)
  --hostfile FILE           Path to MPI hostfile (default: ./hostfile.txt)
  --no-hostfile             Don't use hostfile, run all on localhost
  --nsys-opts "OPTIONS"     Additional nsys options (default: none)
  -o, --output PREFIX       Output file prefix (default: snp_profile)
  -h, --help                Show this help message

Examples:
  $0                                     # Profile all implementations
  $0 -i cuda                             # Profile only CUDA implementation
  $0 -i mpi -n 4                         # Profile MPI implementations with 4 processes
  $0 -i all --nsys-opts "--trace=cuda,mpi,nvtx"  # Custom nsys tracing

Available Implementations:
  cpu           - NaiveCpuSnp (single process)
  cuda          - CudaSnp (single process, requires GPU)
  sparse-cuda   - SparseCudaSnp (single process, requires GPU)
  naive-cuda-mpi- NaiveCudaMpiSnp (distributed, requires GPU)
  cuda-mpi      - CudaMpiSnp (distributed, requires GPU)
  mpi           - All MPI implementations
  all           - All implementations (default)

Notes:
  - For distributed implementations, binaries are automatically copied to all nodes
  - Nsight Systems output (.nsys-rep files) are saved to profiling/results/
  - Use 'nsys-ui' to view the generated .nsys-rep files graphically
  - Requires NVIDIA Nsight Systems to be installed (nsys command)
EOF
}

check_dependencies() {
    # Check for nsys
    if ! command -v nsys &> /dev/null; then
        echo -e "${RED}Error: NVIDIA Nsight Systems (nsys) not found!${NC}"
        echo "Please install Nsight Systems from:"
        echo "  https://developer.nvidia.com/nsight-systems"
        echo ""
        echo "Or install via package manager:"
        echo "  Ubuntu/Debian: apt install nsight-systems-cli"
        exit 1
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

run_profiling() {
    local impl="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${impl}_${timestamp}_%h_rank%q{OMPI_COMM_WORLD_RANK}"
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Profiling: ${impl}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    
    # Build nsys command
    # We use nsys profile to capture CPU, CUDA, and MPI activity
    local nsys_trace="cuda,mpi,nvtx,osrt"
    local nsys_cmd="/usr/local/cuda/bin/nsys profile --trace=$nsys_trace --output=$output_file --force-overwrite=true --stats=true"

    # Build MPI command
    mpi_cmd="mpirun -np 2 --host localhost,10.0.0.2 --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5"
    
    # Construct full command
    local full_cmd="$mpi_cmd $nsys_cmd $PROFILE_EXEC $impl"
    
    echo -e "${YELLOW}Command:${NC} $full_cmd"
    echo ""
    
    # Execute
    eval $full_cmd
    
    echo ""
    echo -e "${GREEN}✓ Profile saved to: ${output_file}.nsys-rep${NC}"
    echo -e "${YELLOW}  View with: nsys-ui ${output_file}.nsys-rep${NC}"
    echo ""
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
        --nsys-opts)
            NSYS_OPTS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PREFIX="$2"
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
echo "  MPI Processes: $NUM_PROCS"
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
echo "  1. View profiles graphically: nsys-ui $OUTPUT_DIR/*.nsys-rep"
echo "  2. Generate reports: nsys stats $OUTPUT_DIR/*.nsys-rep"
echo "  3. Compare implementations side-by-side in nsys-ui"
echo ""
