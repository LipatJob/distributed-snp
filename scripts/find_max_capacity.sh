#!/bin/bash
# scripts/find_max_capacity.sh
#
# This script incrementally increases the dataset size to find the maximum
# capacity of the system before failure (either OOM crash or safety check).
#
# Usage: ./scripts/find_max_capacity.sh [START_NEURONS] [MULTIPLIER] [HOSTS] [MEM_LIMIT_GB]

START_NEURONS=${1:-1000000}
MULTIPLIER=${2:-2}
HOSTS=${3:-"localhost,10.0.0.2"}
MEM_LIMIT_GB=${4:-0}

CURRENT_NEURONS=$START_NEURONS
LAST_SUCCESS=0

echo "================================================================"
echo "Capacity Stress Test"
echo "================================================================"
echo "  Start Neurons: $START_NEURONS"
echo "  Multiplier:    $MULTIPLIER"
echo "  Hosts:         $HOSTS"
echo "  Mem Limit:     $MEM_LIMIT_GB GB"
echo "================================================================"

while true; do
    echo ""
    echo ">>> Testing size: $CURRENT_NEURONS neurons"
    
    # 1. Generate
    echo "[Step 1] Generating dataset..."
    make bigdata-generate NEURONS=$CURRENT_NEURONS HOSTS="$HOSTS" MEM_LIMIT_GB=$MEM_LIMIT_GB
    GEN_EXIT=$?

    if [ $GEN_EXIT -ne 0 ]; then
        echo "❌ Generation failed at $CURRENT_NEURONS neurons (Exit Code: $GEN_EXIT)."
        break
    fi

    # 2. Run
    echo "[Step 2] Running simulation..."
    make bigdata-run HOSTS="$HOSTS"
    RUN_EXIT=$?

    if [ $RUN_EXIT -ne 0 ]; then
        echo "❌ Simulation failed at $CURRENT_NEURONS neurons (Exit Code: $RUN_EXIT)."
        break
    fi

    echo "✅ Success at $CURRENT_NEURONS neurons."
    LAST_SUCCESS=$CURRENT_NEURONS
    
    # Calculate next size
    # Use python for integer arithmetic with float multiplier support if needed
    CURRENT_NEURONS=$(python3 -c "print(int($CURRENT_NEURONS * $MULTIPLIER))")
done

echo ""
echo "================================================================"
echo "Test Complete"
if [ "$LAST_SUCCESS" -eq "0" ]; then
    echo "❌ Failed at the starting size ($START_NEURONS)."
else
    echo "✅ Maximum successful size: $LAST_SUCCESS neurons"
    echo "❌ Failure occurred at:     $CURRENT_NEURONS neurons"
fi
echo "================================================================"
