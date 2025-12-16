#!/bin/bash
# Run SNP sort benchmarks across distributed nodes
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark/results"
HOSTFILE="${PROJECT_ROOT}/hostfile.txt"
BUILD_DIR="/home/shared/tmp/distributed-snp-new"
BENCHMARK_EXEC="${BUILD_DIR}/bin/sort_benchmark"
HOSTS="localhost,10.0.0.2,10.0.1.2"

mkdir -p "$OUTPUT_DIR"

# Default options
NUM_PROCS=3
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
            cat << EOF
Usage: $0 [options]

Options:
  -n, --num-procs N   Number of MPI processes (default: 2)
  -f, --filter REGEX  Run only benchmarks matching REGEX
  -h, --help          Show this help

Examples:
  $0                          # Run all benchmarks
  $0 -n 4 -f 'CudaMpi.*500'   # Filter 500-element benchmarks
EOF
            exit 0
            ;;
        *)
            BENCHMARK_ARGS="$BENCHMARK_ARGS $1"
            shift
            ;;
    esac
done

# Setup output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$RESULT_DIR"
DATA_OUTPUT_FILE="${RESULT_DIR}/data.json"

# Build command
BENCHMARK_CMD="$BENCHMARK_EXEC --benchmark_out_format=json --benchmark_out=$DATA_OUTPUT_FILE"
[ -n "$FILTER" ] && BENCHMARK_CMD="$BENCHMARK_CMD --benchmark_filter=$FILTER"
BENCHMARK_CMD="$BENCHMARK_CMD $BENCHMARK_ARGS"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SNP Sort Benchmarks${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "  MPI Processes: ${NUM_PROCS}"
[ -n "$FILTER" ] && echo -e "  Filter: ${FILTER}"
echo -e "  Results: ${RESULT_DIR}"
echo ""

# Run benchmark
MPI_CMD="mpirun -np $NUM_PROCS --host ${HOSTS} --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5"

echo -e "${BLUE}▶${NC} Running benchmarks..."
$MPI_CMD $BENCHMARK_CMD
echo -e "${GREEN}✓${NC} Benchmarks complete"
echo ""

# Generate report
if [ -f "$DATA_OUTPUT_FILE" ]; then
    PYTHON_CMD=$(command -v python3 || command -v python)
    REPORT_SCRIPT="${PROJECT_ROOT}/benchmark/generate_report.py"
    
    if [ -n "$PYTHON_CMD" ] && [ -f "$REPORT_SCRIPT" ]; then
        echo -e "${BLUE}▶${NC} Generating report and visualizations"
        $PYTHON_CMD "$REPORT_SCRIPT" "$DATA_OUTPUT_FILE" 2>/dev/null
        echo -e "${GREEN}✓${NC} Report generated"
        echo ""
        echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  Results: ${RESULT_DIR}${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
        echo -e "  • data.json - Raw benchmark data"
        echo -e "  • table.txt - Tabular comparison"
        echo -e "  • viz/ - Performance plots"
    else
        echo -e "${YELLOW}→${NC} Results saved to: $DATA_OUTPUT_FILE"
    fi
fi
