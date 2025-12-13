#!/bin/bash

# Run benchmarks and automatically generate visualizations
# Usage: ./scripts/benchmark_and_visualize.sh [optional_benchmark_args]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"

echo "========================================"
echo "SNP Sort Benchmark & Visualization"
echo "========================================"
echo ""

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_JSON="$RESULTS_DIR/benchmark_${TIMESTAMP}.json"

echo "Step 1: Running benchmarks..."
echo "----------------------------------------"

# Change to project directory
cd "$PROJECT_ROOT"

# Run the distributed benchmark and save JSON output
if make benchmark-distributed BENCHMARK_ARGS="--benchmark_format=json --benchmark_out=$OUTPUT_JSON $*"; then
    echo ""
    echo "✓ Benchmarks completed successfully"
    echo "  Results saved to: $OUTPUT_JSON"
else
    echo ""
    echo "✗ Benchmark execution failed"
    exit 1
fi

echo ""
echo "Step 2: Generating visualizations..."
echo "----------------------------------------"

# Check if matplotlib is available
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "⚠ Warning: matplotlib not found. Installing..."
    pip install matplotlib numpy
fi

# Run visualization script
if python3 "$SCRIPT_DIR/visualize_benchmarks.py" "$OUTPUT_JSON" -o "$RESULTS_DIR/viz_${TIMESTAMP}"; then
    echo ""
    echo "✓ Visualizations generated successfully"
    echo "  Output directory: $RESULTS_DIR/viz_${TIMESTAMP}"
else
    echo ""
    echo "⚠ Visualization generation failed, but benchmark data is saved"
fi

echo ""
echo "Step 3: Comparing with previous results (if available)..."
echo "----------------------------------------"

# Find all JSON files in results directory
JSON_FILES=("$RESULTS_DIR"/*.json)

if [ ${#JSON_FILES[@]} -gt 1 ]; then
    echo "Found ${#JSON_FILES[@]} benchmark result files"
    
    # Run comparison script
    if python3 "$SCRIPT_DIR/compare_benchmarks.py" "${JSON_FILES[@]}" > "$RESULTS_DIR/comparison_${TIMESTAMP}.txt"; then
        echo "✓ Comparison report saved to: $RESULTS_DIR/comparison_${TIMESTAMP}.txt"
        
        # Also print to console
        echo ""
        echo "Latest Comparison Results:"
        echo "----------------------------------------"
        cat "$RESULTS_DIR/comparison_${TIMESTAMP}.txt"
    fi
else
    echo "Only one benchmark result available. Run again to compare."
fi

echo ""
echo "========================================"
echo "Complete!"
echo "========================================"
echo ""
echo "Files generated:"
echo "  - Benchmark data:    $OUTPUT_JSON"
echo "  - Visualizations:    $RESULTS_DIR/viz_${TIMESTAMP}/"
echo "  - Summary report:    $RESULTS_DIR/viz_${TIMESTAMP}/benchmark_summary.txt"
echo ""
echo "To view the visualizations:"
echo "  ls -lh $RESULTS_DIR/viz_${TIMESTAMP}/"
echo ""
