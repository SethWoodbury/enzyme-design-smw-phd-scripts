#!/usr/bin/env python3
"""Comprehensive test suite for step03 FastMPNN design module.

Run from project root:
    cd /home/woodbuse/special_scripts/upgraded_fastMPNNdesign
    python -m modules.step03__fastmpnndesign.test.test_all
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Ensure we can import from the right place
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)


def test_interaction_analyzer():
    """Test interaction analyzer module."""
    from modules.step03__fastmpnndesign.interaction_analyzer import (
        InteractionType,
        InteractionConfig,
    )

    print("Testing InteractionType enum...")
    assert InteractionType.HBOND.value == "hbond"
    assert InteractionType.from_string("spodium") == InteractionType.METAL
    assert InteractionType.from_string("h_bond") == InteractionType.HBOND
    assert InteractionType.from_string("salt_bridge") == InteractionType.CHARGED
    assert len(InteractionType.all_types()) == 7
    print("  InteractionType: PASSED")

    print("Testing InteractionConfig...")
    config = InteractionConfig()
    assert config.include_ligand_interactions == False
    assert config.include_catres_interactions == True
    assert InteractionType.HBOND in config.interaction_types
    print("  InteractionConfig defaults: PASSED")

    config2 = InteractionConfig(
        interaction_types=[InteractionType.HBOND, InteractionType.METAL],
        include_ligand_interactions=True
    )
    assert len(config2.interaction_types) == 2
    assert config2.include_ligand_interactions == True
    print("  InteractionConfig custom: PASSED")

    assert config.get_bias_for_type(InteractionType.HBOND) == 2.0
    assert config.get_bias_for_type(InteractionType.METAL) == 2.5
    assert config.is_strong_interaction(InteractionType.HBOND) == True
    assert config.is_strong_interaction(InteractionType.HYDROPHOBIC) == False
    print("  Bias/weight methods: PASSED")


def test_protocol_parser():
    """Test protocol parser module."""
    from modules.step03__fastmpnndesign.protocol_parser import (
        ProtocolParser,
        ProtocolFileParser,
        MPNNStep,
        CartRelaxStep,
        TorsionalRelaxStep,
    )

    parser = ProtocolParser()

    # Test protocol string parsing
    print("Testing protocol string parsing...")
    steps = parser.parse('mpnn:T0.2:N4 -> cart_relax:R2S3 -> mpnn:T0.1:N8 -> torsional_relax:R1S3')
    assert len(steps) == 4
    assert isinstance(steps[0], MPNNStep)
    assert steps[0].temperature == 0.2
    assert steps[0].num_designs == 4
    assert isinstance(steps[1], CartRelaxStep)
    assert steps[1].repeats == 2
    assert steps[1].stages == 3
    print("  Protocol string: PASSED")

    # Test JSON protocol file
    print("Testing JSON protocol file...")
    json_content = '''
{
  "steps": [
    {"type": "mpnn", "temperature": 0.15, "num_designs": 4},
    {"type": "torsional_relax", "repeats": 2}
  ]
}
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(json_content)
        json_file = f.name

    file_parser = ProtocolFileParser()
    json_steps = file_parser.load_from_file(json_file)
    assert len(json_steps) == 2
    assert isinstance(json_steps[0], MPNNStep)
    assert json_steps[0].temperature == 0.15
    os.unlink(json_file)
    print("  JSON file: PASSED")

    # Test text protocol file
    print("Testing text protocol file...")
    txt_content = '''# Test text protocol
mpnn:T0.25:N6
cart_relax:R3S4
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(txt_content)
        txt_file = f.name

    txt_steps = file_parser.load_from_file(txt_file)
    assert len(txt_steps) == 2
    assert isinstance(txt_steps[0], MPNNStep)
    assert txt_steps[0].temperature == 0.25
    os.unlink(txt_file)
    print("  Text file: PASSED")

def test_pdb_restoration():
    """Test PDB restoration module."""
    from modules.step03__fastmpnndesign.pdb_restoration import (
        extract_remark_lines,
        restore_pdb_features,
        full_mpnn_output_restoration,
        cleanup_final_pdb,
    )
    import inspect

    # Create mock PDB
    mock_pdb_content = '''REMARK 666 MATCH TEMPLATE A HIS 123 MATCH MOTIF A HIS 0 NC 123
REMARK 666 MATCH TEMPLATE A GLU 234 MATCH MOTIF A GLU 1 NC 234
REMARK 666 MATCH TEMPLATE A ASP 345 MATCH MOTIF A ASP 2 NC 345
REMARK   1 Some other remark
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
END
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(mock_pdb_content)
        mock_ref_pdb = f.name

    # Test REMARK 666 extraction
    print("Testing REMARK 666 extraction...")
    remark_lines = extract_remark_lines(mock_ref_pdb, remark_types=['666'], preserve_all=True)
    assert len(remark_lines) == 3
    print("  Extract REMARK 666: PASSED")

    all_remarks = extract_remark_lines(mock_ref_pdb, remark_types=None, preserve_all=True)
    assert len(all_remarks) == 4
    print("  Extract all REMARKs: PASSED")

    # Check function signatures
    print("Testing function signatures...")
    sig = inspect.signature(full_mpnn_output_restoration)
    params = list(sig.parameters.keys())
    assert 'original_ref_pdb' in params
    print("  full_mpnn_output_restoration: PASSED")

    sig = inspect.signature(restore_pdb_features)
    params = list(sig.parameters.keys())
    assert 'preserve_all_remarks' in params
    print("  restore_pdb_features: PASSED")

    os.unlink(mock_ref_pdb)

    # Test cleanup_final_pdb function
    print("Testing cleanup_final_pdb...")
    mock_unclean_pdb = '''REMARK 666 MATCH TEMPLATE B XDW 257 MATCH MOTIF A HIS 13 1 1
HEADER                                            28-JAN-26   XXXX
REMARK 220 EXPERIMENTAL DETAILS
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      5  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
TER
HETATM   20  C1  XDW B 257       2.000   0.000   0.000  1.00  0.00           C
HETATM   21  O1  XDW B 257       3.000   0.000   0.000  1.00  0.00           O
TER
TER
CONECT   20   21
# All scores below are weighted scores, not raw scores.
#BEGIN_POSE_ENERGIES_TABLE test.pdb
total_score 0.0
#END_POSE_ENERGIES_TABLE test.pdb
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(mock_unclean_pdb)
        unclean_pdb = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        cleaned_pdb = f.name

    stats = cleanup_final_pdb(unclean_pdb, cleaned_pdb)

    # Check stats
    assert stats['atoms_renumbered'] == 4  # 2 ATOM + 2 HETATM
    assert stats['conect_lines_removed'] == 1
    assert stats['score_lines_removed'] == 4  # # All scores + #BEGIN + total_score + #END
    assert stats['remarks_reordered'] == True
    print("  cleanup_final_pdb stats: PASSED")

    # Check content
    with open(cleaned_pdb) as f:
        lines = f.readlines()

    # HEADER should come before REMARK 220 which should come before REMARK 666
    header_idx = next(i for i, l in enumerate(lines) if l.startswith('HEADER'))
    r220_idx = next(i for i, l in enumerate(lines) if l.startswith('REMARK 220'))
    r666_idx = next(i for i, l in enumerate(lines) if l.startswith('REMARK 666'))
    assert header_idx < r220_idx < r666_idx
    print("  cleanup_final_pdb remark ordering: PASSED")

    # Check atom numbering is continuous
    atom_lines = [l for l in lines if l.startswith('ATOM') or l.startswith('HETATM')]
    assert len(atom_lines) == 4
    for i, line in enumerate(atom_lines):
        serial = int(line[6:11].strip())
        assert serial == i + 1
    print("  cleanup_final_pdb atom renumbering: PASSED")

    # Check TER lines - should be exactly 2 (one after ATOM, one after HETATM)
    ter_count = sum(1 for l in lines if l.startswith('TER'))
    assert ter_count == 2
    print("  cleanup_final_pdb TER lines: PASSED")

    # Check no CONECT lines
    conect_count = sum(1 for l in lines if l.startswith('CONECT'))
    assert conect_count == 0
    print("  cleanup_final_pdb CONECT removed: PASSED")

    # Check no score table
    score_count = sum(1 for l in lines if '#' in l or 'total_score' in l)
    assert score_count == 0
    print("  cleanup_final_pdb score table removed: PASSED")

    os.unlink(unclean_pdb)
    os.unlink(cleaned_pdb)


def test_protocol_validation():
    """Test protocol validation logic."""
    from modules.step03__fastmpnndesign.protocol_parser import (
        ProtocolParser,
        MPNNStep,
        CartRelaxStep,
    )

    parser = ProtocolParser()

    # Test skip initial cart_relax
    print("Testing skip_initial_cart_relax logic...")
    steps = parser.parse('cart_relax:R2S3 -> mpnn:T0.1:N4 -> torsional_relax:R1S3')
    assert isinstance(steps[0], CartRelaxStep)
    if isinstance(steps[0], CartRelaxStep):
        steps_after_skip = steps[1:]
        assert len(steps_after_skip) == 2
    print("  Skip initial cart_relax: PASSED")

    # Test MPNN-ending detection
    print("Testing MPNN-ending detection...")
    mpnn_ending_steps = parser.parse('cart_relax:R1S2 -> mpnn:T0.1:N4')
    assert isinstance(mpnn_ending_steps[-1], MPNNStep)
    print("  MPNN-ending detection: PASSED")

    # Test consecutive MPNN detection
    print("Testing consecutive MPNN detection...")
    consecutive_mpnn = parser.parse('mpnn:T0.2:N4 -> mpnn:T0.1:N8 -> torsional_relax:R1S3')
    warnings = []
    for i, step in enumerate(consecutive_mpnn[:-1]):
        next_step = consecutive_mpnn[i + 1]
        if isinstance(step, MPNNStep) and isinstance(next_step, MPNNStep):
            warnings.append(f'Consecutive MPNN at {i+1}-{i+2}')
    assert len(warnings) > 0
    print("  Consecutive MPNN detection: PASSED")


def test_fastmpnn_designer_init():
    """Test FastMPNNDesigner initialization."""
    from modules.step03__fastmpnndesign.fastmpnn_design import FastMPNNDesigner
    import logging
    logging.disable(logging.CRITICAL)  # Silence logging for test

    print("Testing FastMPNNDesigner initialization...")
    step02_json = os.path.join(
        PROJECT_ROOT,
        'modules/step02__constrained_cart_relax/test/output_dir/input_pdb_aligned_relaxed_metrics.json'
    )
    params = [os.path.join(
        PROJECT_ROOT,
        'modules/step02__constrained_cart_relax/test/params/XDW.params'
    )]

    designer = FastMPNNDesigner(
        step02_json_path=step02_json,
        params_files=params,
        output_dir='/tmp/test_step03',
        protocol='fast',
        skip_initial_cart_relax=True,
        debug=False,
        test=True,
    )

    assert designer.protocol_str.startswith("protocol:")
    assert designer.skip_initial_cart_relax == True
    print("  FastMPNNDesigner init: PASSED")

    logging.disable(logging.NOTSET)


def test_repack_scope_and_catres_filtering():
    """Test repack scope selection and catres filtering."""
    from modules.step03__fastmpnndesign.fastmpnn_design import FastMPNNDesigner

    class DummyClassifier:
        def get_repack_residues_by_scope(self, scope: str):
            assert scope == "core_shell_flex"
            return ["A:10", "A:11", "B:5"]

    designer = FastMPNNDesigner.__new__(FastMPNNDesigner)
    designer.residue_classifier = DummyClassifier()
    designer.catres_positions = [("A", 10)]
    designer.motif_positions = []

    mobile = designer._get_mobile_residues_for_scope("core_shell_flex", ["dummy.pdb"])
    filtered = designer._filter_out_catres(mobile)

    assert "A:10" not in filtered
    assert "A:11" in filtered
    assert "B:5" in filtered

    # Fallback to PDB parsing when classifier is missing
    designer.residue_classifier = None
    mock_pdb = '''ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      3  N   ALA A   2       1.000   0.000   0.000  1.00  0.00           N
ATOM      4  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C
END
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(mock_pdb)
        pdb_path = f.name

    designer.catres_positions = [("A", 1)]
    designer.motif_positions = []
    mobile = designer._get_nonfixed_residues_from_pdb(pdb_path)
    assert "A:1" not in mobile
    assert "A:2" in mobile
    os.unlink(pdb_path)


def test_global_constraint_weight_scaling():
    """Test stdev scaling for global constraints."""
    from modules.step03__fastmpnndesign.rosetta_relax import compute_scaled_stdev

    # Equal weights -> unchanged
    stdev = compute_scaled_stdev(100.0, 0.5, 100.0)
    assert abs(stdev - 0.5) < 1e-6

    # Lower desired weight -> larger stdev
    stdev = compute_scaled_stdev(100.0, 0.5, 400.0)
    assert abs(stdev - 1.0) < 1e-6

    # Disabled constraints
    assert compute_scaled_stdev(0.0, 0.5, 100.0) is None

    # No scorefunction weight -> return desired stdev
    stdev = compute_scaled_stdev(100.0, 0.5, 0.0)
    assert abs(stdev - 0.5) < 1e-6


def test_subprocess_logging_outputs():
    """Test subprocess logging for MPNN and Rosetta."""
    from modules.step03__fastmpnndesign.mpnn_runner import MPNNRunner, MPNNInput, MPNNResult
    from modules.step03__fastmpnndesign.fastmpnn_design import FastMPNNDesigner

    # --- MPNN logging ---
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "input.pdb")
        with open(pdb_path, "w") as f:
            f.write("ATOM      1  N   ALA A   1       0.0 0.0 0.0  1.00 0.00           N\nEND\n")

        mpnn_out = os.path.join(tmpdir, "mpnn_out")
        mpnn_input = MPNNInput(pdb_path=pdb_path, out_folder=mpnn_out)
        runner = MPNNRunner(mpnn_runner_script="dummy", use_container=False)

        def fake_run(cmd, stdout=None, stderr=None, text=None, timeout=None, cwd=None):
            if stdout:
                stdout.write("mpnn ok\n")
            if stderr:
                stderr.write("mpnn warn\n")
            return SimpleNamespace(returncode=0)

        with patch("modules.step03__fastmpnndesign.mpnn_runner.subprocess.run", fake_run), \
             patch.object(MPNNRunner, "_parse_results", return_value=MPNNResult()):
            runner.run(mpnn_input)

        assert os.path.exists(os.path.join(mpnn_out, "mpnn.stdout"))
        assert os.path.exists(os.path.join(mpnn_out, "mpnn.stderr"))

    # --- Rosetta logging ---
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "input.pdb")
        with open(pdb_path, "w") as f:
            f.write("ATOM      1  N   ALA A   1       0.0 0.0 0.0  1.00 0.00           N\nEND\n")

        designer = FastMPNNDesigner.__new__(FastMPNNDesigner)
        designer.output_dir = Path(tmpdir)
        designer.coord_cst_weight = 750.0
        designer.coord_cst_stdev = 0.01
        designer.global_coord_cst_weight = 0.0
        designer.global_coord_cst_stdev = 0.5
        designer.fa_rep_weight = None
        designer.relax_rounds = 1
        designer.relax_inner_cycles = None
        designer._score_term_overrides = {}
        designer._score_term_overrides_next = None
        designer.constrained_atoms = {}
        designer.his_tautomer_map = {}
        designer.scorefunction_cart = "ref2015_cart"
        designer.scorefunction_torsional = "beta_jan25"
        designer.rosetta_in_process = False
        dummy_img = os.path.join(tmpdir, "pyrosetta.sif")
        with open(dummy_img, "w") as f:
            f.write("dummy")
        designer.pyrosetta_image = dummy_img
        designer.pyrosetta_dir = None
        designer.params_files = []
        designer.step02_pdb = None
        designer.ligand_info = None
        designer.no_container = True
        designer.rosetta_timeout = 5
        designer.start_time = time.time()
        designer.max_runtime = 9999
        designer._pdb_lineage = {}

        def fake_rosetta_run(cmd, stdout=None, stderr=None, text=None, timeout=None, **kwargs):
            if stdout:
                stdout.write("rosetta ok\n")
            if stderr:
                stderr.write("rosetta warn\n")
            if "--output" in cmd:
                out_idx = cmd.index("--output") + 1
                out_path = cmd[out_idx]
                with open(out_path, "w") as f:
                    f.write("ATOM      1  N   ALA A   1       0.0 0.0 0.0  1.00 0.00           N\nEND\n")
            return SimpleNamespace(returncode=0)

        with patch("modules.step03__fastmpnndesign.fastmpnn_design.subprocess.run", fake_rosetta_run):
            outputs = designer._run_rosetta_relax(
                [pdb_path],
                mode="repack",
                repeats=1,
                ramp_stages=1,
            )
        assert len(outputs) == 1
        log_dir = Path(tmpdir) / "logs"
        log_files = list(log_dir.glob("rosetta_*.out")) + list(log_dir.glob("rosetta_*.err"))
        assert len(log_files) >= 2


def test_final_diversify_target_count_resolution():
    """Ensure final_diversify target_count is honored when CLI is not explicit."""
    from modules.step03__fastmpnndesign.fastmpnn_design import FastMPNNDesigner
    from modules.step03__fastmpnndesign.protocol_parser import FinalDiversifyStep

    designer = FastMPNNDesigner.__new__(FastMPNNDesigner)
    designer.num_final_designs = None
    designer.num_final_designs_explicit = False

    step = FinalDiversifyStep(target_count=25)
    assert designer._resolve_final_target_count(step) == 25

    designer.num_final_designs = 5
    designer.num_final_designs_explicit = True
    assert designer._resolve_final_target_count(step) == 5

    designer.num_final_designs = None
    designer.num_final_designs_explicit = False
    step = FinalDiversifyStep(target_count=None)
    assert designer._resolve_final_target_count(step) == 10


def main():
    """Run all tests."""
    print("=" * 60)
    print("Step03 FastMPNN Design - Comprehensive Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Interaction Analyzer", test_interaction_analyzer),
        ("Protocol Parser", test_protocol_parser),
        ("PDB Restoration", test_pdb_restoration),
        ("Protocol Validation", test_protocol_validation),
        ("FastMPNNDesigner Init", test_fastmpnn_designer_init),
        ("Repack Scope + Catres Filtering", test_repack_scope_and_catres_filtering),
        ("Global Constraint Scaling", test_global_constraint_weight_scaling),
        ("Final Diversify Target Count", test_final_diversify_target_count_resolution),
        ("Subprocess Logging", test_subprocess_logging_outputs),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print("=" * 60)
        try:
            test_func()
            passed += 1
            print(f"\n{name}: ALL PASSED")
        except Exception as e:
            failed += 1
            print(f"\n{name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
