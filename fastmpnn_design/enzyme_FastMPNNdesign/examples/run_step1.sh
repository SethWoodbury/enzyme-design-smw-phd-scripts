#!/bin/bash
# Example script for running Step 1: Coordinate Transformation
#
# Just run this script directly - no pip install needed!
# Copy this directory anywhere and it will work.

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration - edit these paths for your run
INPUT_PDB="/net/scratch/woodbuse/organophosphatase/round2/af3_out/filtered_i1/alignment/PTE_wKCX_set6_lig_XDW_ORI_04_C9_i_4_model_0_cfg_T__cfgsc_1_50__step_1_50__gam0_0_60__gamMIN_0_10__jit_1_50_eV1_T0_15__8_1_hAF3_idx_3_model_aligned.pdb"
REF_PDB="/net/scratch/woodbuse/organophosphatase/round2/af3_out/filtered_i1/ref_pdbs/PTE_wKCX_set6_lig_XDW_ORI_04_C9_i_4_model_0_cfg_T__cfgsc_1_50__step_1_50__gam0_0_60__gamMIN_0_10__jit_1_50_eV1_T0_15__8_1.pdb"

# Output directory
OUTPUT_DIR="./step1_output"
mkdir -p "$OUTPUT_DIR"

# Catres subset indices (1-indexed REMARK 666 line positions)
# These residues get their geometry constrained to match the reference.
# Residues NOT in this list become "conserved_motif" (sequence fixed, geometry free)
CATRES_SUBSET="1,2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19"

echo "========================================"
echo "Step 1: Coordinate Transformation"
echo "========================================"
echo "Input PDB: $INPUT_PDB"
echo "Reference PDB: $REF_PDB"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run Step 1 using the standalone runner
python "$PIPELINE_DIR/run_step1.py" \
    --input_pdb "$INPUT_PDB" \
    --ref_pdb "$REF_PDB" \
    --catres_subset "$CATRES_SUBSET" \
    --output_pdb "$OUTPUT_DIR/transformed.pdb" \
    --output_json "$OUTPUT_DIR/residue_registry.json" \
    --verbose

echo ""
echo "Done! Output files:"
echo "  - $OUTPUT_DIR/transformed.pdb"
echo "  - $OUTPUT_DIR/residue_registry.json"
