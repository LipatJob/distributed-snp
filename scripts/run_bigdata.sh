#!/bin/bash
set -e

# Arguments
NODES_LIST=$1
OUTDIR=$2
STEPS=$3
REMOTE_DIR=$4
BUILD_DIR=$5

# Convert space-separated list to array
IFS=' ' read -r -a NODES <<< "$NODES_LIST"

echo "Nodes: ${NODES[@]}"
echo "Output Dir: $OUTDIR"
echo "Remote Dir: $REMOTE_DIR"

# 1. Distribute Data
echo "Distributing data..."

# Ensure remote directories exist and copy descriptor
for i in "${!NODES[@]}"; do
    NODE=${NODES[$i]}
    echo "  Preparing node $NODE (Rank $i)..."
    
    if [ "$NODE" == "localhost" ] || [ "$NODE" == "127.0.0.1" ]; then
        # Local node
        mkdir -p $REMOTE_DIR/$OUTDIR
        cp $OUTDIR/descriptor.json $REMOTE_DIR/$OUTDIR/
        cp $OUTDIR/partition_$i.dat $REMOTE_DIR/$OUTDIR/
    else
        # Remote node
        ssh $NODE "mkdir -p $REMOTE_DIR/$OUTDIR"
        scp $OUTDIR/descriptor.json $NODE:$REMOTE_DIR/$OUTDIR/
        scp $OUTDIR/partition_$i.dat $NODE:$REMOTE_DIR/$OUTDIR/
    fi
done

# 2. Run MPI
echo "Running MPI Simulation..."
MPI_HOSTS=$(echo $NODES_LIST | tr ' ' ',')

mpirun -np ${#NODES[@]} --host $MPI_HOSTS \
    $REMOTE_DIR/bin/bigdata_run $REMOTE_DIR/$OUTDIR/descriptor.json $STEPS

# 3. Collect Results
# The result is printed to stdout by Rank 0 (localhost), so we should see it.
# But we also saved it to a file in bigdata/results/run_Xnodes.json on Rank 0.
# Since Rank 0 is localhost (usually), it should be in $REMOTE_DIR/bigdata/results/...
# We might want to copy it back to the workspace.

if [ -f "$REMOTE_DIR/bigdata/results/run_${#NODES[@]}nodes.json" ]; then
    mkdir -p bigdata/results
    cp $REMOTE_DIR/bigdata/results/run_${#NODES[@]}nodes.json bigdata/results/
    echo "Results saved to bigdata/results/run_${#NODES[@]}nodes.json"
fi
