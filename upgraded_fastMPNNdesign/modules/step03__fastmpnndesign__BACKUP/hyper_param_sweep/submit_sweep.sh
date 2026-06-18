#!/bin/bash
# Submit hyperparameter sweep jobs
#
# Usage:
#   ./submit_sweep.sh [num_parallel]
#
# Examples:
#   ./submit_sweep.sh      # Run all jobs sequentially
#   ./submit_sweep.sh 4    # Run 4 jobs in parallel
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMDS_FILE="${SCRIPT_DIR}/cmds/all_sweep_commands.txt"
LOGS_DIR="${SCRIPT_DIR}/logs"

NUM_PARALLEL="${1:-1}"

# Generate commands if not exists
if [ ! -f "$CMDS_FILE" ]; then
    echo "Generating sweep commands..."
    python "${SCRIPT_DIR}/generate_sweep.py"
fi

# Count jobs
NUM_JOBS=$(wc -l < "$CMDS_FILE")
echo "Submitting $NUM_JOBS jobs with $NUM_PARALLEL parallel..."

# Create logs directory
mkdir -p "$LOGS_DIR"

# Run jobs
JOB_NUM=0
while IFS= read -r CMD; do
    JOB_NUM=$((JOB_NUM + 1))
    LOG_OUT="${LOGS_DIR}/sweep_${JOB_NUM}.stdout"
    LOG_ERR="${LOGS_DIR}/sweep_${JOB_NUM}.stderr"

    echo "[$JOB_NUM/$NUM_JOBS] Starting..."

    if [ "$NUM_PARALLEL" -eq 1 ]; then
        # Sequential execution
        eval "$CMD" > "$LOG_OUT" 2> "$LOG_ERR"
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            echo "[$JOB_NUM/$NUM_JOBS] Completed successfully"
        else
            echo "[$JOB_NUM/$NUM_JOBS] Failed with exit code $STATUS"
        fi
    else
        # Parallel execution using background jobs
        eval "$CMD" > "$LOG_OUT" 2> "$LOG_ERR" &

        # Limit parallel jobs
        while [ $(jobs -r | wc -l) -ge "$NUM_PARALLEL" ]; do
            sleep 5
        done
    fi
done < "$CMDS_FILE"

# Wait for all background jobs
if [ "$NUM_PARALLEL" -gt 1 ]; then
    echo "Waiting for remaining jobs to complete..."
    wait
fi

echo ""
echo "All jobs completed!"
echo "Logs: $LOGS_DIR"
echo ""
echo "Run analysis:"
echo "  python ${SCRIPT_DIR}/analyze_sweep.py"
