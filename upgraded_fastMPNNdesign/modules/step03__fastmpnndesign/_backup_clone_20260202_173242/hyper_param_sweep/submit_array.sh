#!/bin/bash

# =============================================================================
# Submit Step03 Hyperparameter Sweep as SLURM Array Job
# =============================================================================
#
# This script submits the hyperparameter sweep as a SLURM array job.
#
# USAGE:
#   ./submit_array.sh [OPTIONS]
#
# OPTIONS:
#   --max-parallel N     Maximum number of array tasks to run in parallel
#                        (default: 20)
#   --time HH:MM:SS      Time limit per job (default: 4:00:00)
#   --partition PART     SLURM partition to use (default: batch)
#   --mem MEM            Memory per job, e.g., 16G, 32G (default: 16G)
#   --cpus N             CPUs per task (default: 4)
#   --commands FILE      Commands file to use (default: cmds/all_sweep_commands.txt)
#   --dry-run            Show what would be submitted without actually submitting
#   --help               Show this help message
#
# EXAMPLES:
#   # Submit with default settings (20 jobs in parallel)
#   ./submit_array.sh
#
#   # Submit with 10 jobs in parallel
#   ./submit_array.sh --max-parallel 10
#
#   # Submit with custom time limit and partition
#   ./submit_array.sh --time 8:00:00 --partition long
#
#   # Dry run to see what would be submitted
#   ./submit_array.sh --dry-run
#
# =============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMDS_FILE=""  # Will be set from --commands or default
TEMPLATE="${SCRIPT_DIR}/sbatch_template.sh"
LOGS_DIR="${SCRIPT_DIR}/logs"

# Default parameters
MAX_PARALLEL=20
TIME_LIMIT="4:00:00"
PARTITION="batch"
MEMORY="16G"
CPUS=4
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --commands)
            CMDS_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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

# Set default commands file if not specified
if [ -z "$CMDS_FILE" ]; then
    CMDS_FILE="${SCRIPT_DIR}/cmds/all_sweep_commands.txt"
fi

# Handle relative paths
if [[ ! "$CMDS_FILE" = /* ]]; then
    CMDS_FILE="${SCRIPT_DIR}/${CMDS_FILE}"
fi

# Generate commands if they don't exist
if [ ! -f "$CMDS_FILE" ]; then
    echo "Commands file not found. Generating sweep commands..."
    python "${SCRIPT_DIR}/generate_sweep.py"
    echo ""
fi

# Verify commands file exists
if [ ! -f "$CMDS_FILE" ]; then
    echo "ERROR: Failed to generate commands file: $CMDS_FILE"
    exit 1
fi

# Count number of jobs
NUM_JOBS=$(wc -l < "$CMDS_FILE")

if [ "$NUM_JOBS" -eq 0 ]; then
    echo "ERROR: No commands found in $CMDS_FILE"
    exit 1
fi

# Create logs directory
mkdir -p "$LOGS_DIR"

# Create submission script from template
SUBMIT_SCRIPT="${SCRIPT_DIR}/.sbatch_submit.sh"
# Use | as delimiter for sed since paths contain /
sed -e "s/NUM_JOBS/${NUM_JOBS}/" \
    -e "s/MAX_PARALLEL/${MAX_PARALLEL}/" \
    -e "s|SWEEP_DIR_PLACEHOLDER|${SCRIPT_DIR}|" \
    -e "s|CMDS_FILE_PLACEHOLDER|${CMDS_FILE}|" \
    -e "s|LOGS_DIR_PLACEHOLDER|${LOGS_DIR}|" \
    -e "s/#SBATCH --time=.*/#SBATCH --time=${TIME_LIMIT}/" \
    -e "s/#SBATCH --partition=.*/#SBATCH --partition=${PARTITION}/" \
    -e "s/#SBATCH --mem=.*/#SBATCH --mem=${MEMORY}/" \
    -e "s/#SBATCH --cpus-per-task=.*/#SBATCH --cpus-per-task=${CPUS}/" \
    "$TEMPLATE" > "$SUBMIT_SCRIPT"

# Print summary
echo "============================================================================="
echo "SLURM Array Job Submission Summary"
echo "============================================================================="
echo "Total jobs:           $NUM_JOBS"
echo "Max parallel:         $MAX_PARALLEL"
echo "Time limit:           $TIME_LIMIT"
echo "Partition:            $PARTITION"
echo "Memory per job:       $MEMORY"
echo "CPUs per job:         $CPUS"
echo "Commands file:        $CMDS_FILE"
echo "Logs directory:       $LOGS_DIR"
echo "Submit script:        $SUBMIT_SCRIPT"
echo "============================================================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "DRY RUN - Would submit with the following sbatch script:"
    echo "-----------------------------------------------------------------------------"
    head -20 "$SUBMIT_SCRIPT"
    echo "... (see $SUBMIT_SCRIPT for full script)"
    echo "-----------------------------------------------------------------------------"
    echo ""
    echo "To actually submit, remove the --dry-run flag"
    exit 0
fi

# Submit to SLURM
echo ""
echo "Submitting to SLURM..."
SUBMIT_OUTPUT=$(sbatch "$SUBMIT_SCRIPT")
echo "$SUBMIT_OUTPUT"

# Extract job ID
JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -oP 'Submitted batch job \K\d+')

if [ -n "$JOB_ID" ]; then
    echo ""
    echo "============================================================================="
    echo "Job submitted successfully!"
    echo "============================================================================="
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor jobs with:"
    echo "  ./monitor_jobs.sh"
    echo "  squeue -u \$USER"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View logs in real-time:"
    echo "  tail -f logs/sweep_${JOB_ID}_*.out"
    echo ""
    echo "Cancel jobs:"
    echo "  scancel $JOB_ID"
    echo "============================================================================="
else
    echo ""
    echo "ERROR: Failed to extract job ID from sbatch output"
    exit 1
fi
