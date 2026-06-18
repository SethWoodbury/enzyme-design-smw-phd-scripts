#!/bin/bash

# =============================================================================
# Monitor Step03 Hyperparameter Sweep Jobs
# =============================================================================
#
# This script monitors the status of SLURM array jobs for the hyperparameter
# sweep and parses log files to show success/failure counts.
#
# USAGE:
#   ./monitor_jobs.sh [OPTIONS]
#
# OPTIONS:
#   --job-id JOB_ID      Monitor specific SLURM job ID
#   --watch              Continuously monitor (refresh every 10 seconds)
#   --failed             Show only failed jobs with error messages
#   --detailed           Show detailed status for each task
#   --help               Show this help message
#
# EXAMPLES:
#   # Show current status
#   ./monitor_jobs.sh
#
#   # Monitor specific job
#   ./monitor_jobs.sh --job-id 12345
#
#   # Continuously monitor
#   ./monitor_jobs.sh --watch
#
#   # Show failed jobs with errors
#   ./monitor_jobs.sh --failed
#
# =============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# Default parameters
JOB_ID=""
WATCH_MODE=false
SHOW_FAILED=false
DETAILED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --job-id)
            JOB_ID="$2"
            shift 2
            ;;
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --failed)
            SHOW_FAILED=true
            shift
            ;;
        --detailed)
            DETAILED=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to show job status
show_status() {
    clear
    echo "============================================================================="
    echo "Step03 Hyperparameter Sweep - Job Status"
    echo "============================================================================="
    echo "Time: $(date)"
    echo ""

    # Show SLURM queue status
    if [ -n "$JOB_ID" ]; then
        echo "SLURM Job: $JOB_ID"
        echo "-----------------------------------------------------------------------------"
        if squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; then
            squeue -j "$JOB_ID"
        else
            echo "Job $JOB_ID not found in queue (may be completed)"
        fi
    else
        echo "Your SLURM Jobs:"
        echo "-----------------------------------------------------------------------------"
        squeue -u "$USER" | grep -E "step03_sweep|JOBID" || echo "No active jobs"
    fi

    echo ""
    echo "============================================================================="
    echo "Log File Analysis"
    echo "============================================================================="

    # Count log files
    local total_logs=$(find "$LOGS_DIR" -name "sweep_*.out" 2>/dev/null | wc -l)
    local total_err_logs=$(find "$LOGS_DIR" -name "sweep_*.err" 2>/dev/null | wc -l)

    if [ "$total_logs" -eq 0 ]; then
        echo "No log files found yet."
        echo ""
        return
    fi

    # Analyze completion status
    local completed=0
    local failed=0
    local running=0

    # Check each output log for completion
    for log in "$LOGS_DIR"/sweep_*.out; do
        if [ -f "$log" ]; then
            if grep -q "Task .* Completed" "$log" 2>/dev/null; then
                if grep -q "Exit status:  0" "$log" 2>/dev/null; then
                    ((completed++))
                else
                    ((failed++))
                fi
            else
                # Check if file has content but no completion marker
                if [ -s "$log" ]; then
                    ((running++))
                fi
            fi
        fi
    done

    echo "Total tasks:          $total_logs"
    echo "Completed (success):  $completed"
    echo "Failed:               $failed"
    echo "Running:              $running"
    echo ""

    # Check for output files
    if [ -d "$OUTPUT_DIR" ]; then
        local num_outputs=$(find "$OUTPUT_DIR" -name "fastmpnn_design_results.json" 2>/dev/null | wc -l)
        echo "Output JSON files:    $num_outputs"
        echo ""
    fi

    # Show failed jobs if requested
    if [ "$SHOW_FAILED" = true ] && [ "$failed" -gt 0 ]; then
        echo "============================================================================="
        echo "Failed Jobs"
        echo "============================================================================="

        for log in "$LOGS_DIR"/sweep_*.out; do
            if [ -f "$log" ]; then
                if grep -q "Exit status:  [^0]" "$log" 2>/dev/null; then
                    local task_id=$(basename "$log" | grep -oP 'sweep_\d+_\K\d+')
                    local exit_code=$(grep "Exit status:" "$log" | tail -1 | grep -oP '\d+$')
                    local err_log="${log%.out}.err"

                    echo ""
                    echo "Task $task_id (exit code: $exit_code)"
                    echo "-----------------------------------------------------------------------------"
                    echo "Log: $log"

                    # Show last 10 lines of error log if it exists
                    if [ -f "$err_log" ] && [ -s "$err_log" ]; then
                        echo "Error (last 10 lines):"
                        tail -10 "$err_log" | sed 's/^/  /'
                    fi

                    # Show last 20 lines of output log
                    echo "Output (last 20 lines):"
                    tail -20 "$log" | sed 's/^/  /'
                    echo "-----------------------------------------------------------------------------"
                fi
            fi
        done
        echo ""
    fi

    # Show detailed status if requested
    if [ "$DETAILED" = true ]; then
        echo "============================================================================="
        echo "Detailed Task Status"
        echo "============================================================================="
        printf "%-8s %-12s %-10s %s\n" "Task" "Status" "Exit" "Job Name"
        echo "-----------------------------------------------------------------------------"

        for log in "$LOGS_DIR"/sweep_*.out; do
            if [ -f "$log" ]; then
                local task_id=$(basename "$log" | grep -oP 'sweep_\d+_\K\d+')
                local status="UNKNOWN"
                local exit_code="-"
                local job_name="-"

                # Extract job name from log
                job_name=$(grep -oP 'outputs/\K[^/]+' "$log" | head -1 || echo "-")

                if grep -q "Task .* Completed" "$log" 2>/dev/null; then
                    exit_code=$(grep "Exit status:" "$log" | tail -1 | grep -oP '\d+$')
                    if [ "$exit_code" = "0" ]; then
                        status="COMPLETE"
                    else
                        status="FAILED"
                    fi
                elif [ -s "$log" ]; then
                    status="RUNNING"
                else
                    status="PENDING"
                fi

                printf "%-8s %-12s %-10s %s\n" "$task_id" "$status" "$exit_code" "${job_name:0:50}"
            fi
        done | sort -n
        echo ""
    fi

    echo "============================================================================="
    echo "Commands"
    echo "============================================================================="
    echo "View specific log:    tail -f logs/sweep_JOBID_TASKID.out"
    echo "Cancel jobs:          scancel JOB_ID"
    echo "Collect results:      ./collect_results.sh"
    echo "Analyze results:      python analyze_sweep.py"
    echo "============================================================================="
    echo ""
}

# Main execution
if [ "$WATCH_MODE" = true ]; then
    echo "Starting watch mode (Ctrl+C to exit)..."
    while true; do
        show_status
        sleep 10
    done
else
    show_status
fi
