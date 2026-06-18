#!/bin/bash
# Test script for step03 FastMPNN design
#
# Usage:
#   ./run_test.sh [protocol]
#
# Examples:
#   ./run_test.sh          # Run with 'fast' protocol (default for testing)
#   ./run_test.sh default  # Run with 'default' protocol
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"

# Default protocol for testing (faster)
PROTOCOL="${1:-fast}"

# Paths
STEP02_JSON="${SCRIPT_DIR}/step02_outputs/input_pdb_aligned_relaxed_metrics.json"
PARAMS="${SCRIPT_DIR}/params/XDW.params"
OUTPUT_DIR="${SCRIPT_DIR}/output_dir"

# Verify inputs exist
if [ ! -f "$STEP02_JSON" ]; then
    echo "ERROR: Step02 JSON not found: $STEP02_JSON"
    echo "Make sure step02 test has been run first."
    exit 1
fi

if [ ! -f "$PARAMS" ]; then
    echo "ERROR: Params file not found: $PARAMS"
    exit 1
fi

# Clean output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "Step03 FastMPNN Design Test"
echo "=================================="
echo "Protocol: $PROTOCOL"
echo "Step02 JSON: $STEP02_JSON"
echo "Params: $PARAMS"
echo "Output: $OUTPUT_DIR"
echo ""

# Run the design (as module to handle relative imports)
# Must run from project root for relative imports to work
PROJECT_ROOT="$(dirname "$(dirname "$MODULE_DIR")")"
cd "$PROJECT_ROOT"
python -m modules.step03__fastmpnndesign.fastmpnn_design \
    --step02_json "$STEP02_JSON" \
    --params "$PARAMS" \
    --output_dir "$OUTPUT_DIR" \
    --protocol "$PROTOCOL" \
    --num_final_designs 2 \
    --debug

echo ""
echo "=================================="
echo "Test completed!"
echo "=================================="
echo "Check output in: $OUTPUT_DIR"
