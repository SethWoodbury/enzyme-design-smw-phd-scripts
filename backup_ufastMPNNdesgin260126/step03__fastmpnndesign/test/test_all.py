#!/usr/bin/env python3
"""Comprehensive test suite for step03 FastMPNN design module.

Run from project root:
    cd /home/woodbuse/special_scripts/upgraded_fastMPNNdesign
    python -m modules.step03__fastmpnndesign.test.test_all
"""

import os
import sys
import tempfile

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

    # Test preset parsing
    print("Testing preset parsing...")
    fast_steps = parser.parse('fast')
    assert len(fast_steps) > 0
    balanced_steps = parser.parse('balanced')
    assert len(balanced_steps) > 0
    print("  Presets: PASSED")


def test_pdb_restoration():
    """Test PDB restoration module."""
    from modules.step03__fastmpnndesign.pdb_restoration import (
        extract_remark_lines,
        restore_pdb_features,
        full_mpnn_output_restoration,
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
        preset='fast',
        skip_initial_cart_relax=True,
        debug=False,
        test=True,
    )

    assert designer.preset == 'fast'
    assert designer.skip_initial_cart_relax == True
    print("  FastMPNNDesigner init: PASSED")

    logging.disable(logging.NOTSET)


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
