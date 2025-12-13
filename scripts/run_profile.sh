#!/bin/bash

# ============================================================================
# run_profile.sh - Runner script for NVIDIA Nsight Systems profiling
# ============================================================================
#
# This script is designed to run the snp_profile executable with MPI
# in a format suitable for NVIDIA Nsight Systems UI.
#
# Usage:
#   ./run_profile.sh [options]
#
# Options:
#   -s, --simulator <name>   Simulator to profile (default: CudaMpiSnp)
#   -n, --size <size>        Array size (default: 1000)
#   -i, --iterations <iter>  Number of iterations (default: 3)
#   -np, --np <num>          Number of MPI processes (default: 2)
#   -h, --hostfile <file>    Hostfile for MPI (default: hostfile.txt)
#   --help                   Show this help message
#
# Examples:
#   # Profile CudaMpiSnp with default settings
#   ./run_profile.sh
#
#   # Profile NaiveCudaMpiSnp with larger array
#   ./run_profile.sh -s NaiveCudaMpiSnp -n 5000
#
#   # Profile with 4 MPI processes and 5 iterations
#   ./run_profile.sh -np 4 -i 5
#
# For NVIDIA Nsight Systems:
#   1. Open Nsight Systems and create a new profile session
#   2. Set the command to: /path/to/run_profile.sh [options]
#   3. Configure MPI settings in Nsight Systems UI
#   4. Run the profile

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
SIMULATOR="CudaMpiSnp"
ARRAY_SIZE=100
ITERATIONS=3
NUM_PROCS=2
HOSTFILE="${PROJECT_ROOT}/hostfile.txt"
PROFILER_BIN="${PROJECT_ROOT}/build/snp_profile"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Functions
# ============================================================================

print_usage() {
    cat << EOF
Usage: $0 [options]

Options:
  -s, --simulator <name>   Simulator to profile (default: $SIMULATOR)
                           Available: CudaMpiSnp, NaiveCudaMpiSnp
  -n, --size <size>        Array size (default: $ARRAY_SIZE)
  -i, --iterations <iter>  Number of iterations (default: $ITERATIONS)
  -np, --np <num>          Number of MPI processes (default: $NUM_PROCS)
  -h, --hostfile <file>    Hostfile for MPI (default: $HOSTFILE)
  --help                   Show this help message

Examples:
  $0
  $0 -s NaiveCudaMpiSnp -n 5000
  $0 -np 4 -i 5

For NVIDIA Nsight Systems:
  1. Create new profile session
  2. Set command: $SCRIPT_DIR/run_profile.sh [options]
  3. Configure and run
EOF
}

check_profiler() {
    if [ ! -f "$PROFILER_BIN" ]; then
        echo -e "${YELLOW}ERROR: Profiler not found at $PROFILER_BIN${NC}"
        echo "Please build the profiler first:"
        echo "  cd $PROJECT_ROOT"
        echo "  make profile-build"
        exit 1
    fi
}

check_hostfile() {
    if [ ! -f "$HOSTFILE" ]; then
        echo -e "${YELLOW}WARNING: Hostfile not found at $HOSTFILE${NC}"
        echo "Generating hostfile..."
        mkdir -p "$(dirname "$HOSTFILE")"
        # Generate a simple hostfile with localhost
        echo "localhost slots=1" > "$HOSTFILE"
        echo -e "${GREEN}Generated: $HOSTFILE${NC}"
    fi
}

print_info() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}SNP Profiler Configuration${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Simulator:        $SIMULATOR"
    echo "Array Size:       $ARRAY_SIZE"
    echo "Iterations:       $ITERATIONS"
    echo "MPI Processes:    $NUM_PROCS"
    echo "Hostfile:         $HOSTFILE"
    echo "Profiler Binary:  $PROFILER_BIN"
    echo -e "${GREEN}========================================${NC}"
}

# ============================================================================
# Parse arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--simulator)
            SIMULATOR="$2"
            shift 2
            ;;
        -n|--size)
            ARRAY_SIZE="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -np|--np)
            NUM_PROCS="$2"
            shift 2
            ;;
        -h|--hostfile)
            HOSTFILE="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Pre-flight checks
# ============================================================================

check_profiler
check_hostfile
print_info

# ============================================================================
# Run profiler with MPI
# ============================================================================

echo ""
echo -e "${BLUE}Starting MPI profiler session...${NC}"
echo ""

# Build MPI command
MPI_CMD="mpirun \
    --allow-run-as-root \
    --mca btl_tcp_if_include ens5 \
    --mca oob_tcp_if_include ens5 \
    -np 2 \
    --host localhost,10.0.0.2"

# For distributed systems, you may want to add:
# --mca btl_tcp_if_include <interface> \
# --mca oob_tcp_if_include <interface> \

# Add bind-to-socket for better performance (optional, adjust as needed)
MPI_CMD="$MPI_CMD --bind-to socket"

# Build profiler arguments
PROFILE_ARGS="-s $SIMULATOR -n $ARRAY_SIZE -i $ITERATIONS"

# Construct full command
FULL_CMD="$MPI_CMD $PROFILER_BIN $PROFILE_ARGS"

echo -e "${BLUE}Command:${NC}"
echo "$FULL_CMD"
echo ""

# Execute
exec $FULL_CMD
