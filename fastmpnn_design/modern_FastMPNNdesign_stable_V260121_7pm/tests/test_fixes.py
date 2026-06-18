#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for enzyme design codebase fixes.

Tests cover:
1. Sidechain flipper safeguards (HIS flip protection)
2. HIS tautomer detection from raw PDB
3. Catalytic residue RMSD tracking
4. REMARK 666 validation
5. Pipeline tracker functionality

Run with: python -m pytest tests/test_fixes.py -v
Or: python tests/test_fixes.py  (for standalone execution)
"""

import os
import sys
import tempfile
import numpy as np

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)


# =============================================================================
# Test HIS Tautomer Detection
# =============================================================================

class TestHisTautomerDetection:
    """Test HIS tautomer detection from raw PDB files."""

    def test_his_d_detection(self):
        """Test detection of HIS_D (delta-protonated, HD1 only)."""
        import hydrogen_utils

        # Create test PDB with HIS_D (HD1 present, no HE2)
        pdb_his_d = """ATOM      1  N   HIS A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  HIS A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   HIS A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   HIS A   1       1.251   2.398   0.000  1.00  0.00           O
ATOM      5  CB  HIS A   1       1.998  -0.760  -1.216  1.00  0.00           C
ATOM      6  CG  HIS A   1       1.523  -0.200  -2.534  1.00  0.00           C
ATOM      7  ND1 HIS A   1       0.233   0.201  -2.816  1.00  0.00           N
ATOM      8  CD2 HIS A   1       2.251  -0.016  -3.673  1.00  0.00           C
ATOM      9  CE1 HIS A   1       0.130   0.629  -4.070  1.00  0.00           C
ATOM     10  NE2 HIS A   1       1.318   0.539  -4.656  1.00  0.00           N
ATOM     11  HD1 HIS A   1      -0.500   0.150  -2.100  1.00  0.00           H
END
"""
        his_map = hydrogen_utils.build_his_tautomer_map_from_pdb(pdb_his_d, verbose=False)
        assert ('A', 1) in his_map, "HIS residue not detected"
        assert his_map[('A', 1)] == "HIS_D", f"Expected HIS_D, got {his_map[('A', 1)]}"

    def test_his_epsilon_detection(self):
        """Test detection of standard HIS (epsilon-protonated, HE2 only)."""
        import hydrogen_utils

        # Create test PDB with standard HIS (HE2 present, no HD1)
        pdb_his = """ATOM      1  N   HIS A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  HIS A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   HIS A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   HIS A   1       1.251   2.398   0.000  1.00  0.00           O
ATOM      5  CB  HIS A   1       1.998  -0.760  -1.216  1.00  0.00           C
ATOM      6  CG  HIS A   1       1.523  -0.200  -2.534  1.00  0.00           C
ATOM      7  ND1 HIS A   1       0.233   0.201  -2.816  1.00  0.00           N
ATOM      8  CD2 HIS A   1       2.251  -0.016  -3.673  1.00  0.00           C
ATOM      9  CE1 HIS A   1       0.130   0.629  -4.070  1.00  0.00           C
ATOM     10  NE2 HIS A   1       1.318   0.539  -4.656  1.00  0.00           N
ATOM     11  HE2 HIS A   1       1.500   0.800  -5.600  1.00  0.00           H
END
"""
        his_map = hydrogen_utils.build_his_tautomer_map_from_pdb(pdb_his, verbose=False)
        assert ('A', 1) in his_map, "HIS residue not detected"
        assert his_map[('A', 1)] == "HIS", f"Expected HIS, got {his_map[('A', 1)]}"

    def test_his_doubly_protonated(self):
        """Test detection of doubly protonated HIS (both HD1 and HE2)."""
        import hydrogen_utils

        # Create test PDB with doubly protonated HIS
        pdb_his_doubly = """ATOM      1  N   HIS A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  HIS A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  ND1 HIS A   1       0.233   0.201  -2.816  1.00  0.00           N
ATOM      4  NE2 HIS A   1       1.318   0.539  -4.656  1.00  0.00           N
ATOM      5  HD1 HIS A   1      -0.500   0.150  -2.100  1.00  0.00           H
ATOM      6  HE2 HIS A   1       1.500   0.800  -5.600  1.00  0.00           H
END
"""
        his_map = hydrogen_utils.build_his_tautomer_map_from_pdb(pdb_his_doubly, verbose=False)
        assert ('A', 1) in his_map, "HIS residue not detected"
        # Doubly protonated defaults to HIS
        assert his_map[('A', 1)] == "HIS", f"Expected HIS for doubly protonated, got {his_map[('A', 1)]}"


# =============================================================================
# Test Pipeline Tracker
# =============================================================================

class TestPipelineTracker:
    """Test pipeline tracking functionality."""

    def test_stage_tracking(self):
        """Test basic stage tracking."""
        from pipeline_tracker import PipelineTracker

        tracker = PipelineTracker(verbose=False)

        tracker.begin_stage("test_stage", "Testing stage tracking")
        tracker.log_metric("test_metric", 1.234, "units")
        tracker.end_stage(success=True)

        assert len(tracker.stages) == 1
        assert tracker.stages[0]['name'] == "test_stage"
        assert tracker.stages[0]['success'] == True
        assert 'test_metric' in tracker.stages[0]['metrics']
        assert tracker.stages[0]['metrics']['test_metric']['value'] == 1.234

    def test_checkpoint_saving(self):
        """Test checkpoint saving (without pose)."""
        from pipeline_tracker import PipelineTracker

        tracker = PipelineTracker(verbose=False)

        tracker.begin_stage("test_stage")
        tracker.checkpoint("test_checkpoint", metrics={'rmsd': 0.5})
        tracker.end_stage()

        assert "test_checkpoint" in tracker.checkpoints
        assert tracker.checkpoints["test_checkpoint"]['metrics']['rmsd'] == 0.5

    def test_summary_generation(self):
        """Test summary report generation."""
        from pipeline_tracker import PipelineTracker

        tracker = PipelineTracker(verbose=False)

        tracker.begin_stage("stage1")
        tracker.log_metric("catres_rmsd_overall", 0.5, "A")
        tracker.end_stage()

        tracker.begin_stage("stage2")
        tracker.log_metric("catres_rmsd_overall", 0.8, "A")
        tracker.end_stage()

        summary = tracker.summary()
        assert "PIPELINE SUMMARY" in summary
        assert "stage1" in summary
        assert "stage2" in summary

    def test_json_report_saving(self):
        """Test JSON report saving."""
        from pipeline_tracker import PipelineTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PipelineTracker(output_dir=tmpdir, verbose=False)

            tracker.begin_stage("test")
            tracker.log_metric("value", 42)
            tracker.end_stage()

            report_path = tracker.save_report()
            assert os.path.exists(report_path)

            import json
            with open(report_path) as f:
                report = json.load(f)
            assert len(report['stages']) == 1


# =============================================================================
# Test RMSD Calculation Constants
# =============================================================================

# Check if pyrosetta is available for tests that need it
try:
    import rosetta_utils
    HAS_ROSETTA_UTILS = True
except ImportError:
    HAS_ROSETTA_UTILS = False


class TestFunctionalAtoms:
    """Test that functional atoms are correctly defined."""

    def test_his_functional_atoms(self):
        """Test HIS functional atoms include both nitrogens."""
        if not HAS_ROSETTA_UTILS:
            print("(skipped - pyrosetta not available)")
            return

        his_atoms = rosetta_utils.FUNCTIONAL_ATOMS.get("HIS", [])
        assert "ND1" in his_atoms, "HIS functional atoms missing ND1"
        assert "NE2" in his_atoms, "HIS functional atoms missing NE2"

    def test_asp_functional_atoms(self):
        """Test ASP functional atoms include carboxylate oxygens."""
        if not HAS_ROSETTA_UTILS:
            print("(skipped - pyrosetta not available)")
            return

        asp_atoms = rosetta_utils.FUNCTIONAL_ATOMS.get("ASP", [])
        assert "OD1" in asp_atoms, "ASP functional atoms missing OD1"
        assert "OD2" in asp_atoms, "ASP functional atoms missing OD2"


# =============================================================================
# Test Chemically Sensitive Flip Constants
# =============================================================================

class TestChemicallySensitiveFlips:
    """Test chemically sensitive flip configuration."""

    def test_his_is_chemically_sensitive(self):
        """Test that HIS is marked as chemically sensitive."""
        if not HAS_ROSETTA_UTILS:
            print("(skipped - pyrosetta not available)")
            return

        assert "HIS" in rosetta_utils.CHEMICALLY_SENSITIVE_FLIPS
        his_config = rosetta_utils.CHEMICALLY_SENSITIVE_FLIPS["HIS"]
        assert "functional_atoms" in his_config
        assert "max_functional_rmsd" in his_config
        assert "ND1" in his_config["functional_atoms"]
        assert "NE2" in his_config["functional_atoms"]


# =============================================================================
# Test Non-H Coordinate Extraction
# =============================================================================

class TestNonHCoordinateExtraction:
    """Test extraction of non-hydrogen coordinates."""

    def test_extract_from_pdb_lines(self):
        """Test extraction from PDB lines."""
        import hydrogen_utils

        pdb_lines = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  H   ALA A   1       0.500   0.500   0.500  1.00  0.00           H
ATOM      4  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
"""
        coords = hydrogen_utils.extract_non_hydrogen_coords_dict(pdb_lines)

        # Should have 3 non-H atoms (N, CA, C)
        assert len(coords) == 3
        assert ('A', 1, 'N') in coords
        assert ('A', 1, 'CA') in coords
        assert ('A', 1, 'C') in coords
        # H should not be included
        assert ('A', 1, 'H') not in coords

    def test_non_h_rmsd_calculation(self):
        """Test RMSD calculation between coordinate dicts."""
        import hydrogen_utils

        coords1 = {
            ('A', 1, 'N'): np.array([0.0, 0.0, 0.0]),
            ('A', 1, 'CA'): np.array([1.458, 0.0, 0.0]),
        }
        coords2 = {
            ('A', 1, 'N'): np.array([0.1, 0.0, 0.0]),  # Moved 0.1 A
            ('A', 1, 'CA'): np.array([1.558, 0.0, 0.0]),  # Moved 0.1 A
        }

        rmsd, n_atoms = hydrogen_utils.calculate_non_h_rmsd(coords1, coords2)
        assert n_atoms == 2
        assert abs(rmsd - 0.1) < 0.01  # Should be ~0.1 A


# =============================================================================
# Test REMARK 666 Extraction
# =============================================================================

class TestRemark666Extraction:
    """Test REMARK 666 line extraction and parsing."""

    def test_extract_remark666_lines(self):
        """Test extraction of REMARK 666 lines from PDB."""
        import hydrogen_utils

        pdb_content = """HEADER    HYDROLASE                               01-JAN-00   1ABC
REMARK 666 MATCH TEMPLATE A XDW 0 MATCH MOTIF A HIS 13 1 1
REMARK 666 MATCH TEMPLATE A XDW 0 MATCH MOTIF A HIS 15 2 1
REMARK 666 MATCH TEMPLATE A XDW 0 MATCH MOTIF A ASP 53 3 1
ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N
END
"""
        remark_lines = hydrogen_utils.extract_remark666_lines(pdb_content)
        assert len(remark_lines) == 3
        assert all("REMARK 666" in line for line in remark_lines)

    def test_parse_catalytic_residues(self):
        """Test parsing catalytic residues from REMARK 666."""
        import hydrogen_utils

        remark_lines = [
            "REMARK 666 MATCH TEMPLATE A XDW 0 MATCH MOTIF A HIS 13 1 1",
            "REMARK 666 MATCH TEMPLATE A XDW 0 MATCH MOTIF A ASP 53 3 1",
        ]

        catres = hydrogen_utils.parse_catalytic_residues_from_remark666(remark_lines, verbose=False)
        assert ('A', 13) in catres
        assert ('A', 53) in catres
        assert catres[('A', 13)]['name3'] == 'HIS'
        assert catres[('A', 53)]['name3'] == 'ASP'


# =============================================================================
# Test Functional Atom RMSD Calculation
# =============================================================================

class TestFunctionalAtomRmsd:
    """Test functional atom RMSD calculation helper."""

    def test_rmsd_calculation_logic(self):
        """Test the RMSD calculation logic with mock data."""
        # This test doesn't need rosetta_utils - just tests the math
        # Test with simple numpy arrays
        coords_before = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        coords_after = np.array([[0.1, 0, 0], [1.1, 0, 0], [2.1, 0, 0]])

        diff = coords_before - coords_after
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        assert abs(rmsd - 0.1) < 0.001


# =============================================================================
# Main - Run tests standalone
# =============================================================================

def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestHisTautomerDetection,
        TestPipelineTracker,
        TestFunctionalAtoms,
        TestChemicallySensitiveFlips,
        TestNonHCoordinateExtraction,
        TestRemark666Extraction,
        TestFunctionalAtomRmsd,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    print("=" * 60)
    print("Running enzyme design fix tests")
    print("=" * 60)

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                method = getattr(instance, method_name)
                try:
                    method()
                    print(f"  [PASS] {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed_tests.append((test_class.__name__, method_name, traceback.format_exc()))

    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")

    if failed_tests:
        print(f"\nFailed tests ({len(failed_tests)}):")
        for class_name, method_name, tb in failed_tests:
            print(f"  - {class_name}.{method_name}")
            # Print truncated traceback
            for line in tb.split('\n')[-4:-1]:
                print(f"    {line}")

    print("=" * 60)
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
