#!/bin/bash

# Run SNP Sort Benchmarks on single node
# Usage: ./scripts/run_benchmark.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
BENCHMARK_EXEC="${BUILD_DIR}/sort_benchmark"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark_results"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Default options
NUM_PROCS=1
OUTPUT_FORMAT="console"
FILTER=""
BENCHMARK_ARGS=""

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
            echo "  -n, --num-procs N    Number of MPI processes (default: 1)"
            echo "  -f, --filter REGEX   Run only benchmarks matching REGEX"
            echo "  --json               Output results in JSON format"
            echo "  --csv                Output results in CSV format"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all benchmarks with 1 process"
            echo "  $0 -n 2 -f 'CudaMpi.*RANDOM'         # Run CUDA/MPI RANDOM benchmarks with 2 processes"
            echo "  $0 --json                             # Output in JSON format"
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

# Generate timestamp for output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build benchmark command
BENCHMARK_CMD="$BENCHMARK_EXEC"

if [ -n "$FILTER" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_filter=$FILTER"
fi

case $OUTPUT_FORMAT in
    json)
        OUTPUT_FILE="${OUTPUT_DIR}/benchmark_${TIMESTAMP}.json"
        BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_format=json --benchmark_out=$OUTPUT_FILE"
        ;;
    csv)
        OUTPUT_FILE="${OUTPUT_DIR}/benchmark_${TIMESTAMP}.csv"
        BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_format=csv --benchmark_out=$OUTPUT_FILE"
        ;;
esac

BENCHMARK_CMD="$BENCHMARK_CMD $BENCHMARK_ARGS"

echo "========================================="
echo "Running SNP Sort Benchmarks"
echo "========================================="
echo "MPI Processes:  $NUM_PROCS"
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

# Run benchmark with MPI
if [ "$NUM_PROCS" -eq 1 ]; then
    # Single process - run directly
    $BENCHMARK_CMD
else
    # Multiple processes - use MPI
    mpirun -np $NUM_PROCS --allow-run-as-root --oversubscribe $BENCHMARK_CMD
fi

echo ""
echo "========================================="
echo "Benchmark Complete"
if [ "$OUTPUT_FORMAT" != "console" ]; then
    echo "Results saved to: $OUTPUT_FILE"
fi
echo "========================================="
