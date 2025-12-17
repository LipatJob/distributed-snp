#!/bin/bash
set -e

# Arguments
NODES_LIST=$1
OUTDIR=$2
STEPS=$3
REMOTE_DIR=$4
IMPL=${5:-optimized}
PARTITIONER=${6:-linear}

# Convert space-separated list to array
IFS=' ' read -r -a NODES <<< "$NODES_LIST"

echo "Nodes: ${NODES[@]}"
echo "Output Dir: $OUTDIR"
echo "Remote Dir: $REMOTE_DIR"
echo "Implementation: $IMPL"
echo "Partitioner: $PARTITIONER"

# 2. Run MPI
echo "Running MPI Simulation..."
MPI_HOSTS=$(echo $NODES_LIST | tr ' ' ',')

mpirun -np ${#NODES[@]} --host $MPI_HOSTS -wd $REMOTE_DIR \
    --mca btl_tcp_if_include ens5 --mca oob_tcp_if_include ens5 \
    $REMOTE_DIR/bin/bigdata_run $OUTDIR/descriptor.json $STEPS "$IMPL" "$PARTITIONER"

# 3. Collect Results
RANK0_NODE=${NODES[0]}
RESULT_FILE="bigdata/results/run_${IMPL}_${#NODES[@]}nodes.json"
LOCAL_RESULT_DIR="bigdata/results"

mkdir -p $LOCAL_RESULT_DIR

echo "Retrieving results from Rank 0 ($RANK0_NODE)..."

if [ "$RANK0_NODE" == "localhost" ] || [ "$RANK0_NODE" == "127.0.0.1" ]; then
    if [ -f "$REMOTE_DIR/$RESULT_FILE" ]; then
        cp "$REMOTE_DIR/$RESULT_FILE" "$LOCAL_RESULT_DIR/"
        echo "Results saved to $LOCAL_RESULT_DIR/$(basename $RESULT_FILE)"
    else
        echo "Warning: Result file not found at $REMOTE_DIR/$RESULT_FILE"
    fi
else
    scp "$RANK0_NODE:$REMOTE_DIR/$RESULT_FILE" "$LOCAL_RESULT_DIR/" && \
    echo "Results saved to $LOCAL_RESULT_DIR/$(basename $RESULT_FILE)" || \
    echo "Warning: Failed to retrieve results file from $RANK0_NODE"
fi
