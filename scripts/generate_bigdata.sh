#!/bin/bash
set -e

# Arguments
NODES_LIST=$1
OUTDIR=$2
REMOTE_DIR=$3
BUILD_DIR=$4
NEURONS=${5:-10000}
MAX_MEM_GB=${6:-0}

# Convert space-separated list to array
IFS=' ' read -r -a NODES <<< "$NODES_LIST"

echo "Nodes: ${NODES[@]}"
echo "Output Dir: $OUTDIR"
echo "Remote Dir: $REMOTE_DIR"
echo "Neurons: $NEURONS"
echo "Max Mem (GB): $MAX_MEM_GB"

# Generate Data Locally
echo "Generating data locally on each node..."

# Ensure remote directories exist and run generator
for i in "${!NODES[@]}"; do
    NODE=${NODES[$i]}
    echo "  Generating on node $NODE (Rank $i)..."
    
    if [ "$NODE" == "localhost" ] || [ "$NODE" == "127.0.0.1" ]; then
        # Local node
        mkdir -p $OUTDIR
        $BUILD_DIR/bin/bigdata_generator \
            --neurons $NEURONS \
            --ranks ${#NODES[@]} \
            --intra 10 \
            --inter 1 \
            --outdir $OUTDIR \
            --seed 123 \
            --max-mem $MAX_MEM_GB \
            --rank $i
    else
        # Remote node
        ssh $NODE "mkdir -p $OUTDIR"
        ssh $NODE "$REMOTE_DIR/bin/bigdata_generator \
            --neurons $NEURONS \
            --ranks ${#NODES[@]} \
            --intra 10 \
            --inter 1 \
            --outdir $OUTDIR \
            --seed 123 \
            --max-mem $MAX_MEM_GB \
            --rank $i"
    fi
done
