#!/bin/bash

# Run SNP Sort Benchmarks across distributed nodes
# Usage: ./scripts/run_distributed_benchmark.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark/results"
HOSTFILE="${PROJECT_ROOT}/hostfile.txt"
mkdir -p "$OUTPUT_DIR"

BUILD_DIR="/home/shared/tmp/distributed-snp-new"
BENCHMARK_EXEC="${BUILD_DIR}/bin/sort_benchmark"

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
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -n, --num-procs N      Number of MPI processes (default: 2)"
            echo "  -f, --filter REGEX     Run only benchmarks matching REGEX"
            echo "  --hostfile FILE        Path to MPI hostfile (default: ./hostfile.txt)"
            echo "  --no-hostfile          Don't use hostfile, run all on localhost"
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

# Create timestamped directory structure
RESULT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# Build benchmark command
BENCHMARK_CMD="$BENCHMARK_EXEC"

if [ -n "$FILTER" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_filter=$FILTER"
fi

# Save to temporary location first (for compatibility)
DATA_OUTPUT_FILE="${RESULT_DIR}/data.json"
BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_out_format=json --benchmark_out=$DATA_OUTPUT_FILE"
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
echo "Result Dir:     $RESULT_DIR"
echo "========================================="
echo ""

# Build MPI command
MPI_CMD="mpirun -np $NUM_PROCS --host ${HOSTS} --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5 --allow-run-as-root --oversubscribe"

# Run benchmark with MPI
$MPI_CMD $BENCHMARK_CMD

echo ""
echo "========================================="
echo "Distributed Benchmark Complete"
echo "========================================="

# Generate report (only on rank 0)
if [ -f "$DATA_OUTPUT_FILE" ]; then
    echo ""
    echo "Generating visualizations and report..."
    
    # Check if Python 3 is available
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        echo "Warning: Python not found. Skipping report generation."
        echo "Raw results saved to: $DATA_OUTPUT_FILE"
        exit 0
    fi
    
    # Run report generation
    REPORT_SCRIPT="${PROJECT_ROOT}/benchmark/generate_report.py"
    if [ -f "$REPORT_SCRIPT" ]; then
        $PYTHON_CMD "$REPORT_SCRIPT" "$DATA_OUTPUT_FILE"
        
        echo ""
        echo "========================================="
        echo "Report Generated Successfully"
        echo "========================================="
        echo "Location: $RESULT_DIR"
        echo "  - data.json: Raw benchmark data"
        echo "  - table.txt: Tabular comparison"
        echo "  - viz/: Visualization plots"
        echo "========================================="
    else
        echo "Warning: Report script not found at $REPORT_SCRIPT"
        echo "Raw results saved to: $DATA_OUTPUT_FILE"
    fi
else
    echo "Warning: Benchmark output file not found"
fi
