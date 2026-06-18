#!/bin/bash

# =============================================================================
# Collect Step03 Hyperparameter Sweep Results
# =============================================================================
#
# This script collects results from completed hyperparameter sweep jobs and
# creates summary reports.
#
# USAGE:
#   ./collect_results.sh [OPTIONS]
#
# OPTIONS:
#   --output FILE        Output CSV file (default: sweep_results.csv)
#   --format FORMAT      Output format: csv, json, or both (default: both)
#   --min-complete N     Minimum number of completed jobs required (default: 1)
#   --help               Show this help message
#
# EXAMPLES:
#   # Collect results to default files
#   ./collect_results.sh
#
#   # Collect to custom CSV file
#   ./collect_results.sh --output my_results.csv
#
#   # Export as JSON only
#   ./collect_results.sh --format json
#
# OUTPUT:
#   Creates summary files with key metrics from all completed jobs:
#   - CSV: Spreadsheet-compatible format
#   - JSON: Machine-readable format
#   - Summary table printed to console
#
# =============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
LOGS_DIR="${SCRIPT_DIR}/logs"

# Default parameters
OUTPUT_CSV="sweep_results.csv"
OUTPUT_JSON="sweep_results.json"
FORMAT="both"
MIN_COMPLETE=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_CSV="$2"
            OUTPUT_JSON="${OUTPUT_CSV%.csv}.json"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --min-complete)
            MIN_COMPLETE="$2"
            shift 2
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

# Validate format
if [[ ! "$FORMAT" =~ ^(csv|json|both)$ ]]; then
    echo "ERROR: Invalid format: $FORMAT (must be csv, json, or both)"
    exit 1
fi

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory not found: $OUTPUT_DIR"
    echo "No jobs have been run yet."
    exit 1
fi

echo "============================================================================="
echo "Collecting Step03 Hyperparameter Sweep Results"
echo "============================================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Count jobs and results
total_jobs=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
json_files=$(find "$OUTPUT_DIR" -name "fastmpnn_design_results.json" 2>/dev/null | wc -l)

echo "Total job directories: $total_jobs"
echo "Completed jobs (with JSON): $json_files"
echo ""

if [ "$json_files" -lt "$MIN_COMPLETE" ]; then
    echo "ERROR: Not enough completed jobs ($json_files < $MIN_COMPLETE)"
    echo "Wait for more jobs to complete or reduce --min-complete"
    exit 1
fi

# Use the analyze_sweep.py script to generate results
echo "Analyzing results..."
echo "-----------------------------------------------------------------------------"

if [ "$FORMAT" = "csv" ] || [ "$FORMAT" = "both" ]; then
    python "${SCRIPT_DIR}/analyze_sweep.py" --output_dir "$OUTPUT_DIR" --export_csv "$OUTPUT_CSV"
    if [ -f "$OUTPUT_CSV" ]; then
        echo ""
        echo "CSV results written to: $OUTPUT_CSV"
    fi
fi

if [ "$FORMAT" = "json" ]; then
    # Generate JSON output using Python
    python3 << 'EOF'
import json
import sys
from pathlib import Path

# Import from analyze_sweep.py
sys.path.insert(0, "${SCRIPT_DIR}")
from analyze_sweep import load_results

output_dir = "${OUTPUT_DIR}"
output_file = "${OUTPUT_JSON}"

results = load_results(output_dir)

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nJSON results written to: {output_file}")
EOF
fi

if [ "$FORMAT" = "both" ]; then
    # Generate JSON in addition to CSV
    python3 << 'EOF'
import json
import sys
from pathlib import Path

sys.path.insert(0, "${SCRIPT_DIR}")
from analyze_sweep import load_results

output_dir = "${OUTPUT_DIR}"
output_file = "${OUTPUT_JSON}"

results = load_results(output_dir)

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"JSON results written to: {output_file}")
EOF
fi

echo ""
echo "============================================================================="
echo "Results Collection Complete"
echo "============================================================================="
echo ""

# Show summary statistics
if [ -f "$OUTPUT_CSV" ]; then
    echo "Summary Statistics:"
    echo "-----------------------------------------------------------------------------"

    # Count completed vs failed
    completed=$(tail -n +2 "$OUTPUT_CSV" | wc -l)
    echo "Jobs analyzed: $completed"

    # Show best result
    echo ""
    echo "Best result (by bond deviation):"
    # Skip header, sort by bond_dev column (6th column), show top result
    tail -n +2 "$OUTPUT_CSV" | sort -t',' -k6 -n | head -1 | \
        awk -F',' '{printf "  Job: %s\n  Bond deviation: %s\n  Mutations: %s\n", $1, $6, $4}'

    echo ""
    echo "View full results:"
    echo "  cat $OUTPUT_CSV"
    echo "  column -t -s',' $OUTPUT_CSV | less -S"
    echo ""
fi

# Check for failed jobs
if [ -d "$LOGS_DIR" ]; then
    failed_count=$(grep -l "Exit status:  [^0]" "$LOGS_DIR"/sweep_*.out 2>/dev/null | wc -l || echo 0)
    if [ "$failed_count" -gt 0 ]; then
        echo "WARNING: $failed_count jobs failed"
        echo "Use './monitor_jobs.sh --failed' to see error messages"
        echo ""
    fi
fi

echo "Next steps:"
echo "  - Review results: open $OUTPUT_CSV in Excel or a text editor"
echo "  - Analyze in detail: python analyze_sweep.py"
echo "  - Check failed jobs: ./monitor_jobs.sh --failed"
echo "============================================================================="
