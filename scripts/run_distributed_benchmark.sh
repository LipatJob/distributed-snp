#!/bin/bash

# Run SNP Sort Benchmarks across distributed nodes
# Usage: ./scripts/run_distributed_benchmark.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
BENCHMARK_EXEC="${BUILD_DIR}/sort_benchmark"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark_results"
HOSTFILE="${PROJECT_ROOT}/hostfile.txt"
HOSTS="localhost,10.0.0.2"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Default options
NUM_PROCS=2
OUTPUT_FORMAT="console"
FILTER=""
BENCHMARK_ARGS=""
USE_HOSTFILE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-procs)
            NUM_PROCS="$2"
            shift 2
            ;;
        -f|--filter)
            FILTER="$2"
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
        --json)
            OUTPUT_FORMAT="json"
            shift
            ;;
        --csv)
            OUTPUT_FORMAT="csv"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -n, --num-procs N      Number of MPI processes (default: 2)"
            echo "  -f, --filter REGEX     Run only benchmarks matching REGEX"
            echo "  --hostfile FILE        Path to MPI hostfile (default: ./hostfile.txt)"
            echo "  --no-hostfile          Don't use hostfile, run all on localhost"
            echo "  --json                 Output results in JSON format"
            echo "  --csv                  Output results in CSV format"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Run with 2 processes using hostfile"
            echo "  $0 -n 4 -f 'CudaMpi.*500'                  # Run 500-element benchmarks with 4 processes"
            echo "  $0 --no-hostfile -n 2                       # Run locally without hostfile"
            echo "  $0 --json --filter 'RANDOM'                 # Output RANDOM benchmarks in JSON"
            exit 0
            ;;
        *)
            BENCHMARK_ARGS="$BENCHMARK_ARGS $1"
            shift
            ;;
    esac
done

# Check if benchmark executable exists
if [ ! -f "$BENCHMARK_EXEC" ]; then
    echo "Error: Benchmark executable not found at $BENCHMARK_EXEC"
    echo "Please build the project first with 'make' or 'cmake --build build'"
    exit 1
fi

# Check hostfile if using it
if [ "$USE_HOSTFILE" = true ] && [ ! -f "$HOSTFILE" ]; then
    echo "Warning: Hostfile not found at $HOSTFILE"
    echo "Creating default hostfile for localhost..."
    echo "localhost slots=2" > "$HOSTFILE"
fi

# Generate timestamp for output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build benchmark command
BENCHMARK_CMD="$BENCHMARK_EXEC"

if [ -n "$FILTER" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_filter=$FILTER"
fi

case $OUTPUT_FORMAT in
    json)
        OUTPUT_FILE="${OUTPUT_DIR}/benchmark_distributed_${TIMESTAMP}.json"
        BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_format=json --benchmark_out=$OUTPUT_FILE"
        ;;
    csv)
        OUTPUT_FILE="${OUTPUT_DIR}/benchmark_distributed_${TIMESTAMP}.csv"
        BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_format=csv --benchmark_out=$OUTPUT_FILE"
        ;;
esac

BENCHMARK_CMD="$BENCHMARK_CMD $BENCHMARK_ARGS"

echo "========================================="
echo "Running Distributed SNP Sort Benchmarks"
echo "========================================="
echo "MPI Processes:  $NUM_PROCS"
if [ "$USE_HOSTFILE" = true ]; then
    echo "Hostfile:       $HOSTFILE"
    echo "Hosts:"
    cat "$HOSTFILE" | grep -v '^#' | grep -v '^$' | sed 's/^/  /'
fi
echo "Build Dir:      $BUILD_DIR"
echo "Benchmark Exec: $BENCHMARK_EXEC"
if [ -n "$FILTER" ]; then
    echo "Filter:         $FILTER"
fi
if [ "$OUTPUT_FORMAT" != "console" ]; then
    echo "Output File:    $OUTPUT_FILE"
fi
echo "========================================="
echo ""

# Build MPI command
MPI_CMD="mpirun -np $NUM_PROCS --host ${HOSTS} --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5 --allow-run-as-root --oversubscribe"

if [ "$USE_HOSTFILE" = true ]; then
    # Extract hosts from hostfile
    HOSTS=$(grep -v '^#' "$HOSTFILE" | grep -v '^$' | awk '{print $1}' | paste -sd, -)
    if [ -n "$HOSTS" ]; then
        MPI_CMD="$MPI_CMD --host $HOSTS"
    fi
fi

# Run benchmark with MPI
$MPI_CMD $BENCHMARK_CMD

echo ""
echo "========================================="
echo "Distributed Benchmark Complete"
if [ "$OUTPUT_FORMAT" != "console" ]; then
    echo "Results saved to: $OUTPUT_FILE"
fi
echo "========================================="
