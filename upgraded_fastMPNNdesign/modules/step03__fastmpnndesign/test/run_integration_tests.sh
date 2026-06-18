#!/bin/bash
# Integration tests for step03 FastMPNN design module
#
# Modernized tests for JSON-based protocols only.
# These are lightweight runs that exercise:
# - mpnn_multi pooling
# - clustering + keep_cluster_best
# - scale/reset score term overrides
# - repack/minimize
# - final_diversify path
#
# Usage:
#   ./run_integration_tests.sh          # Run all tests
#   ./run_integration_tests.sh quick    # Minimal smoke (validate + minimal protocol)
#   ./run_integration_tests.sh validate # Protocol validation only
#

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Test inputs
PARAMS="${SCRIPT_DIR}/params/XDW.params"
STEP01_PDB="${SCRIPT_DIR}/step01_outputs/input_pdb_aligned.pdb"
STEP01_JSON="${SCRIPT_DIR}/step01_outputs/input_pdb_recommended_atom_cst.json"
OUTPUT_BASE="${SCRIPT_DIR}/output_dir"
PROTO_DIR="${OUTPUT_BASE}/protocols"
STEP02_JSON="${OUTPUT_BASE}/step02_stub.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

fail() {
  echo -e "${RED}ERROR: $1${NC}"
  exit 1
}

# Verify inputs exist
[ -f "$PARAMS" ] || fail "Params file not found: $PARAMS"
[ -f "$STEP01_PDB" ] || fail "Step01 PDB not found: $STEP01_PDB"
[ -f "$STEP01_JSON" ] || fail "Step01 JSON not found: $STEP01_JSON"

prepare_step02_stub() {
  mkdir -p "$OUTPUT_BASE"
  local step01_pdb_abs
  local step01_json_abs
  step01_pdb_abs="$(python -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$STEP01_PDB")"
  step01_json_abs="$(python -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$STEP01_JSON")"

  cat > "$STEP02_JSON" << EOF
{
  "metadata": {
    "output_pdb": "${step01_pdb_abs}",
    "input_pdb": "${step01_pdb_abs}",
    "ref_pdb": "${step01_pdb_abs}",
    "step01_pdb": "${step01_pdb_abs}",
    "step01_json": "${step01_json_abs}"
  }
}
EOF
}

prepare_protocol_fixtures() {
  mkdir -p "$PROTO_DIR"

  # Minimal protocol that hits mpnn_multi, clustering, scale reset, repack, minimize.
  cat > "${PROTO_DIR}/minimal.json" << 'EOF'
{
  "steps": [
    {"type": "scale", "terms": {"fa_rep": 0.2}},
    {"type": "mpnn_multi", "strategies": [
      {"type": "design_core_shell", "temperature": 0.1, "num_designs": 2, "batch_size": 1},
      {"type": "design_core_shell", "temperature": 0.2, "num_designs": 2, "batch_size": 1, "use_sc_context": false}
    ]},
    {"type": "cluster", "method": "sequence", "n_clusters": 2},
    {"type": "keep_cluster_best", "n": 1, "metric": "smart"},
    {"type": "repack", "repack_scope": "core_shell", "scorefunction": "beta_jan25"},
    {"type": "scale", "terms": {"fa_rep": "reset"}},
    {"type": "minimize", "tolerance": 0.01, "scorefunction": "beta_jan25", "minimize_scope": "core_shell"}
  ]
}
EOF

  # Final diversify path with small target count.
  cat > "${PROTO_DIR}/final_diversify.json" << 'EOF'
{
  "steps": [
    {"type": "mpnn_multi", "strategies": [
      {"type": "design_core_shell", "temperature": 0.1, "num_designs": 1, "batch_size": 1}
    ]},
    {"type": "final_diversify",
     "temperatures": [0.1],
     "target_count": 2,
     "designs_per_temp": 1,
     "max_iterations": 1,
     "design_scope": "shell_only",
     "batch_size": 1,
     "overshoot_threshold": 0.5,
     "fallback_include_flex": false
    }
  ]
}
EOF
}

prepare_step02_stub
prepare_protocol_fixtures

[ -f "$STEP02_JSON" ] || fail "Step02 JSON stub not created: $STEP02_JSON"

run_test() {
  local test_name="$1"
  local protocol_path="$2"
  local expected_min="${3:-1}"
  shift 3

  local test_dir="${OUTPUT_BASE}/${test_name}"
  echo ""
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}TEST: ${test_name}${NC}"
  echo -e "${YELLOW}========================================${NC}"

  rm -rf "$test_dir"
  mkdir -p "$test_dir"

  cd "$PROJECT_ROOT" || exit 1
  if python -m modules.step03__fastmpnndesign.fastmpnn_design \
      --step02_json "$STEP02_JSON" \
      --params "$PARAMS" \
      --output_dir "$test_dir" \
      --step01_pdb "$STEP01_PDB" \
      --protocol_file "$protocol_path" \
      --max_runtime 300 \
      --no-mpnn-server \
      --mpnn_no_gpu \
      --num_final_designs 2 \
      "$@" ; then

      if [ ! -f "${test_dir}/fastmpnn_design_results.json" ]; then
          echo -e "${RED}FAILED: ${test_name} - No results JSON produced${NC}"
          return 1
      fi

      python - << PY
import json, sys
from collections import Counter
from pathlib import Path
path = r"${test_dir}/fastmpnn_design_results.json"
data = json.load(open(path))
count = len(data.get("output_designs", []))
if count < ${expected_min}:
    print(f"FAILED: expected >= ${expected_min} designs, got {count}")
    sys.exit(1)

# Check for basename collisions in final outputs
stems = [Path(d.get("pdb_path","")).stem for d in data.get("output_designs", []) if d.get("pdb_path")]
dupes = {k:v for k,v in Counter(stems).items() if v > 1}
if dupes:
    print(f"FAILED: basename collisions in output_designs: {list(dupes.items())[:5]}")
    sys.exit(1)

# If final_diversify exists, ensure its recorded PDBs are unique
finals = [m for m in data.get("metrics_history", []) if m.get("step_type") == "final_diversify"]
if finals:
    f = finals[-1]
    stems = [Path(s.get(\"pdb\",\"\" )).stem for s in f.get(\"structures\", []) if s.get(\"pdb\")]
    dupes = {k:v for k,v in Counter(stems).items() if v > 1}
    if dupes:
        print(f\"FAILED: basename collisions in final_diversify structures: {list(dupes.items())[:5]}\")
        sys.exit(1)

print(f"OK: {count} designs")
PY
      if [ $? -ne 0 ]; then
          echo -e "${RED}FAILED: ${test_name} - Output count check failed${NC}"
          return 1
      fi

      echo -e "${GREEN}PASSED: ${test_name}${NC}"
      echo "  Output: ${test_dir}/fastmpnn_design_results.json"
      return 0
  else
      echo -e "${RED}FAILED: ${test_name} - Script exited with error${NC}"
      return 1
  fi
}

validate_protocols() {
  echo ""
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}TEST: validate_protocols${NC}"
  echo -e "${YELLOW}========================================${NC}"

  python - << PY
import sys
from pathlib import Path
from modules.step03__fastmpnndesign.protocol_parser import ProtocolFileParser

paths = [
    Path("modules/step03__fastmpnndesign/protocols/default.json"),
    Path(r"${PROTO_DIR}/minimal.json"),
    Path(r"${PROTO_DIR}/final_diversify.json"),
]

for p in paths:
    result = ProtocolFileParser.validate_file(str(p))
    if not result.get("valid"):
        print(f"INVALID: {p} -> {result.get('errors')}")
        sys.exit(1)
print("OK: protocol validation")
PY
  if [ $? -ne 0 ]; then
      echo -e "${RED}FAILED: validate_protocols${NC}"
      return 1
  fi
  echo -e "${GREEN}PASSED: validate_protocols${NC}"
  return 0
}

naming_collision_dry() {
  echo ""
  echo -e "${YELLOW}========================================${NC}"
  echo -e "${YELLOW}TEST: naming_collision_dry${NC}"
  echo -e "${YELLOW}========================================${NC}"

  python -m modules.step03__fastmpnndesign.test.test_naming_collisions
  if [ $? -ne 0 ]; then
      echo -e "${RED}FAILED: naming_collision_dry${NC}"
      return 1
  fi
  echo -e "${GREEN}PASSED: naming_collision_dry${NC}"
  return 0
}

run_and_count() {
  if "$@"; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
  fi
}

echo "=================================================="
echo "Step03 FastMPNN Design Integration Tests"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Step02 JSON: $STEP02_JSON"
echo "Params: $PARAMS"
echo "Output base: $OUTPUT_BASE"

mkdir -p "$OUTPUT_BASE"

PASSED=0
FAILED=0

if [ -n "$1" ]; then
  case "$1" in
    quick)
      run_and_count validate_protocols
      run_and_count naming_collision_dry
      run_and_count run_test "test_minimal_protocol" "${PROTO_DIR}/minimal.json" 1
      ;;
    validate)
      run_and_count validate_protocols
      run_and_count naming_collision_dry
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_integration_tests.sh [quick|validate]"
      exit 1
      ;;
  esac
else
  echo ""
  echo "Running all integration tests..."
  run_and_count validate_protocols
  run_and_count naming_collision_dry
  run_and_count run_test "test_minimal_protocol" "${PROTO_DIR}/minimal.json" 1
  run_and_count run_test "test_final_diversify" "${PROTO_DIR}/final_diversify.json" 1
fi

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
