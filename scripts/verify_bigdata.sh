#!/bin/bash
set -e

# Arguments
NODES_LIST=$1
OUTDIR=$2
STEPS=$3
REMOTE_DIR=$4
BUILD_DIR=$5
NEURONS=${6:-1000} # Default to small size for verification

echo "=== Big Data Verification ==="
echo "Nodes: $NODES_LIST"
echo "Neurons: $NEURONS"
echo "Steps: $STEPS"

# 1. Run Distributed (2 Nodes)
echo "--- Running Distributed Simulation (2 Nodes) ---"
make bigdata-generate NEURONS=$NEURONS HOSTS="$NODES_LIST" OUTDIR="$OUTDIR/dist"
make bigdata-run HOSTS="$NODES_LIST" OUTDIR="$OUTDIR/dist" STEPS=$STEPS

# Move distributed result immediately to avoid overwrite
mkdir -p $OUTDIR/dist
if [ -f "bigdata/results/run_2nodes.json" ]; then
    mv "bigdata/results/run_2nodes.json" "$OUTDIR/dist/result.json"
else
    echo "Error: Distributed result file not found."
    exit 1
fi

# 2. Run Local Reference (2 Ranks on Localhost)
echo "--- Running Local Reference (2 Ranks on Localhost) ---"
# Generate locally for 2 ranks
mkdir -p $OUTDIR/local
$BUILD_DIR/bin/bigdata_generator \
    --neurons $NEURONS \
    --ranks 2 \
    --intra 10 \
    --inter 3 \
    --outdir $OUTDIR/local \
    --seed 123 \
    --mem-limit 0 \
    --rank -1 # Generate all ranks

# Run locally
mpirun -np 2 --host localhost:2 \
    $BUILD_DIR/bin/bigdata_run $OUTDIR/local/descriptor.json $STEPS > $OUTDIR/local/output.txt

# 3. Compare Results
echo "--- Comparing Results ---"

# Move local result
if [ -f "bigdata/results/run_2nodes.json" ]; then
    mv "bigdata/results/run_2nodes.json" "$OUTDIR/local/result.json"
else
    echo "Error: Local result file not found."
    exit 1
fi

DIST_CHECKSUM=$(grep "state_checksum" $OUTDIR/dist/result.json | awk -F': ' '{print $2}' | tr -d ',')
LOCAL_CHECKSUM=$(grep "state_checksum" $OUTDIR/local/result.json | awk -F': ' '{print $2}' | tr -d ',')

echo "Distributed Checksum: $DIST_CHECKSUM"
echo "Local Checksum:       $LOCAL_CHECKSUM"

if [ "$DIST_CHECKSUM" == "$LOCAL_CHECKSUM" ]; then
    echo "SUCCESS: Checksums match! Distributed simulation is consistent with local simulation."
    exit 0
else
    echo "FAILURE: Checksums do not match!"
    exit 1
fi
