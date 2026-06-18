#!/bin/bash
# Integration tests for step03 FastMPNN design module
#
# Tests various protocol configurations to ensure all features work correctly.
# These are lightweight tests designed to run quickly while exercising key features.
#
# Usage:
#   ./run_integration_tests.sh          # Run all tests
#   ./run_integration_tests.sh test1    # Run specific test
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"

# Test inputs
STEP02_JSON="${SCRIPT_DIR}/step02_outputs/input_pdb_aligned_relaxed_metrics.json"
PARAMS="${SCRIPT_DIR}/params/XDW.params"
STEP01_PDB="${SCRIPT_DIR}/step01_outputs/input_pdb_aligned.pdb"
OUTPUT_BASE="${SCRIPT_DIR}/output_dir"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verify inputs exist
if [ ! -f "$STEP02_JSON" ]; then
    echo -e "${RED}ERROR: Step02 JSON not found: $STEP02_JSON${NC}"
    exit 1
fi

if [ ! -f "$PARAMS" ]; then
    echo -e "${RED}ERROR: Params file not found: $PARAMS${NC}"
    exit 1
fi

run_test() {
    local test_name="$1"
    local test_dir="${OUTPUT_BASE}/${test_name}"
    shift

    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}TEST: ${test_name}${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # Clean and create output directory
    rm -rf "$test_dir"
    mkdir -p "$test_dir"

    # Run the test
    cd "$PROJECT_ROOT"
    if python -m modules.step03__fastmpnndesign.fastmpnn_design \
        --step02_json "$STEP02_JSON" \
        --params "$PARAMS" \
        --output_dir "$test_dir" \
        --step01_pdb "$STEP01_PDB" \
        --max_runtime 600 \
        "$@" ; then

        # Check output exists
        if [ -f "${test_dir}/fastmpnn_design_results.json" ]; then
            echo -e "${GREEN}PASSED: ${test_name}${NC}"
            echo "  Output: ${test_dir}/fastmpnn_design_results.json"
            return 0
        else
            echo -e "${RED}FAILED: ${test_name} - No results JSON produced${NC}"
            return 1
        fi
    else
        echo -e "${RED}FAILED: ${test_name} - Script exited with error${NC}"
        return 1
    fi
}

# ============================================================================
# TEST DEFINITIONS
# ============================================================================

test_design_only() {
    # Test 1: Design-only preset (fastest, no relaxation)
    run_test "test01_design_only" \
        --preset design_only \
        --num_final_designs 2 \
        --mpnn_num_designs 2
}

test_geometry_only() {
    # Test 2: Geometry-only preset (cart relax only, no design)
    run_test "test02_geometry_only" \
        --preset geometry_only \
        --num_final_designs 1
}

test_fast_preset() {
    # Test 3: Fast preset
    run_test "test03_fast_preset" \
        --preset fast \
        --num_final_designs 2
}

test_skip_initial_cart() {
    # Test 4: Skip initial cart_relax (since step02 already did it)
    run_test "test04_skip_initial_cart" \
        --preset balanced \
        --skip_initial_cart_relax \
        --num_final_designs 2 \
        --mpnn_num_designs 2
}

test_custom_protocol_string() {
    # Test 5: Custom protocol string
    run_test "test05_custom_protocol" \
        --protocol "mpnn:T0.15:N2 -> torsional_relax:R1S2" \
        --num_final_designs 2
}

test_protocol_file_json() {
    # Test 6: Protocol from JSON file
    cat > "${OUTPUT_BASE}/test_protocol.json" << 'EOF'
{
  "steps": [
    {"type": "mpnn", "temperature": 0.1, "num_designs": 2},
    {"type": "torsional_relax", "repeats": 1, "stages": 2}
  ]
}
EOF
    run_test "test06_protocol_file_json" \
        --protocol_file "${OUTPUT_BASE}/test_protocol.json" \
        --num_final_designs 2
}

test_protocol_file_txt() {
    # Test 7: Protocol from text file
    cat > "${OUTPUT_BASE}/test_protocol.txt" << 'EOF'
# Light MPNN then relax
mpnn:T0.1:N2
torsional_relax:R1S2
EOF
    run_test "test07_protocol_file_txt" \
        --protocol_file "${OUTPUT_BASE}/test_protocol.txt" \
        --num_final_designs 2
}

test_mpnn_ending_protocol() {
    # Test 8: Protocol ending with MPNN (tests restoration)
    run_test "test08_mpnn_ending" \
        --protocol "mpnn:T0.1:N3" \
        --num_final_designs 2
}

test_design_secondary_sphere() {
    # Test 9: Design secondary sphere
    run_test "test09_secondary_sphere" \
        --preset design_only \
        --design_secondary_sphere \
        --num_final_designs 2 \
        --mpnn_num_designs 2
}

test_no_conserve_interactions() {
    # Test 10: Disable interaction conservation
    run_test "test10_no_conserve" \
        --preset design_only \
        --no_conserve_interactions \
        --num_final_designs 2 \
        --mpnn_num_designs 2
}

# ============================================================================
# MAIN
# ============================================================================

echo "=================================================="
echo "Step03 FastMPNN Design Integration Tests"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Step02 JSON: $STEP02_JSON"
echo "Params: $PARAMS"
echo "Output base: $OUTPUT_BASE"

# Create output base
mkdir -p "$OUTPUT_BASE"

# Track results
PASSED=0
FAILED=0
TESTS_RUN=()

# Run specific test or all tests
if [ -n "$1" ]; then
    # Run specific test
    case "$1" in
        test1|test01|design_only)
            test_design_only && ((PASSED++)) || ((FAILED++))
            ;;
        test2|test02|geometry_only)
            test_geometry_only && ((PASSED++)) || ((FAILED++))
            ;;
        test3|test03|fast)
            test_fast_preset && ((PASSED++)) || ((FAILED++))
            ;;
        test4|test04|skip_cart)
            test_skip_initial_cart && ((PASSED++)) || ((FAILED++))
            ;;
        test5|test05|custom_protocol)
            test_custom_protocol_string && ((PASSED++)) || ((FAILED++))
            ;;
        test6|test06|json_protocol)
            test_protocol_file_json && ((PASSED++)) || ((FAILED++))
            ;;
        test7|test07|txt_protocol)
            test_protocol_file_txt && ((PASSED++)) || ((FAILED++))
            ;;
        test8|test08|mpnn_ending)
            test_mpnn_ending_protocol && ((PASSED++)) || ((FAILED++))
            ;;
        test9|test09|secondary)
            test_design_secondary_sphere && ((PASSED++)) || ((FAILED++))
            ;;
        test10|no_conserve)
            test_no_conserve_interactions && ((PASSED++)) || ((FAILED++))
            ;;
        quick)
            # Quick smoke test - just design_only
            test_design_only && ((PASSED++)) || ((FAILED++))
            ;;
        *)
            echo "Unknown test: $1"
            echo "Available: test1-test10, quick"
            exit 1
            ;;
    esac
else
    # Run all tests
    echo ""
    echo "Running all integration tests..."

    test_design_only && ((PASSED++)) || ((FAILED++))
    test_geometry_only && ((PASSED++)) || ((FAILED++))
    test_fast_preset && ((PASSED++)) || ((FAILED++))
    test_skip_initial_cart && ((PASSED++)) || ((FAILED++))
    test_custom_protocol_string && ((PASSED++)) || ((FAILED++))
    test_protocol_file_json && ((PASSED++)) || ((FAILED++))
    test_protocol_file_txt && ((PASSED++)) || ((FAILED++))
    test_mpnn_ending_protocol && ((PASSED++)) || ((FAILED++))
    test_design_secondary_sphere && ((PASSED++)) || ((FAILED++))
    test_no_conserve_interactions && ((PASSED++)) || ((FAILED++))
fi

# Summary
echo ""
echo "=================================================="
echo "INTEGRATION TEST SUMMARY"
echo "=================================================="
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"
echo "Output: $OUTPUT_BASE"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
exit 0
