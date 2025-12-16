#!/bin/bash
BENCHMARK_DIR="$(dirname "$0")/../benchmark"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RESULTS_DIR=$BENCHMARK_DIR/results/$TIMESTAMP
mkdir -p $RESULTS_DIR

echo "Running benchmarks at $TIMESTAMP"
reframe -C $BENCHMARK_DIR/reframe_config.py -c $BENCHMARK_DIR/snp_benchmark.py -r --report-file $RESULTS_DIR/report.json
echo "Report is saved at $RESULTS_DIR/report.json"
echo ""

echo "Generating summary..."
echo "Summary is saved at $RESULTS_DIR/summary.txt"

echo "Generating visualizations..."
VIZ_OUTPUT_DIR=$RESULTS_DIR/figures
mkdir -p $VIZ_OUTPUT_DIR
python3 $BENCHMARK_DIR/generate_visualizations.py --report $RESULTS_DIR/report.json --output $VIZ_OUTPUT_DIR

echo "Figures are saved at $VIZ_OUTPUT_DIR"