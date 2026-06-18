#!/usr/bin/env python3
"""Test suite for ProtocolFileParser."""

import json
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_parser import (
    ProtocolFileParser,
    ProtocolValidationError,
    ProtocolParser,
    MPNNStep,
    CartRelaxStep,
    TorsionalRelaxStep,
    MinimizeStep,
    RepackStep,
    SelectBestStep,
    StepType,
)


def test_json_protocol_parsing():
    """Test parsing JSON protocol files."""
    print("Testing JSON protocol parsing...")

    json_content = {
        "steps": [
            {"type": "mpnn", "temperature": 0.1, "num_designs": 2, "spheres": ["primary"]},
            {"type": "cart_relax", "repeats": 2, "stages": 3},
            {"type": "mpnn", "temperature": 0.15, "num_designs": 4},
            {"type": "torsional_relax", "repeats": 1, "stages": 3}
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_content, f)
        temp_path = f.name

    try:
        steps = ProtocolFileParser.load_from_file(temp_path)

        assert len(steps) == 4, f"Expected 4 steps, got {len(steps)}"

        # Check first MPNN step
        assert isinstance(steps[0], MPNNStep), f"Expected MPNNStep, got {type(steps[0])}"
        assert steps[0].temperature == 0.1, f"Expected temp 0.1, got {steps[0].temperature}"
        assert steps[0].num_designs == 2, f"Expected 2 designs, got {steps[0].num_designs}"
        assert steps[0].design_spheres == ["primary"], f"Expected primary sphere"

        # Check cart_relax step
        assert isinstance(steps[1], CartRelaxStep), f"Expected CartRelaxStep, got {type(steps[1])}"
        assert steps[1].repeats == 2
        assert steps[1].stages == 3

        # Check second MPNN step
        assert isinstance(steps[2], MPNNStep)
        assert steps[2].temperature == 0.15
        assert steps[2].num_designs == 4

        # Check torsional_relax step
        assert isinstance(steps[3], TorsionalRelaxStep)
        assert steps[3].repeats == 1
        assert steps[3].stages == 3

        print("  JSON parsing: PASSED")

    finally:
        os.unlink(temp_path)


def test_text_protocol_parsing():
    """Test parsing text protocol files."""
    print("Testing text protocol parsing...")

    text_content = """# Comment line
mpnn:T0.2:N3:spheres=primary
cart_relax:R3S4
mpnn:T0.1:N8
torsional_relax:R2S3
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text_content)
        temp_path = f.name

    try:
        steps = ProtocolFileParser.load_from_file(temp_path)

        assert len(steps) == 4, f"Expected 4 steps, got {len(steps)}"

        # Check first MPNN step
        assert isinstance(steps[0], MPNNStep)
        assert steps[0].temperature == 0.2
        assert steps[0].num_designs == 3

        # Check cart_relax step
        assert isinstance(steps[1], CartRelaxStep)
        assert steps[1].repeats == 3
        assert steps[1].stages == 4

        print("  Text parsing: PASSED")

    finally:
        os.unlink(temp_path)


def test_json_validation_errors():
    """Test that invalid JSON protocols raise proper errors."""
    print("Testing JSON validation errors...")

    # Test missing 'steps' key
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"protocol": []}, f)
        temp_path = f.name

    try:
        try:
            ProtocolFileParser.load_from_file(temp_path)
            assert False, "Should have raised ProtocolValidationError"
        except ProtocolValidationError as e:
            assert "must contain a 'steps' array" in str(e)
            print("  Missing 'steps' key: PASSED")
    finally:
        os.unlink(temp_path)

    # Test empty steps array
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"steps": []}, f)
        temp_path = f.name

    try:
        try:
            ProtocolFileParser.load_from_file(temp_path)
            assert False, "Should have raised ProtocolValidationError"
        except ProtocolValidationError as e:
            assert "at least one step" in str(e)
            print("  Empty steps array: PASSED")
    finally:
        os.unlink(temp_path)

    # Test invalid step type
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"steps": [{"type": "invalid_type"}]}, f)
        temp_path = f.name

    try:
        try:
            ProtocolFileParser.load_from_file(temp_path)
            assert False, "Should have raised ProtocolValidationError"
        except ProtocolValidationError as e:
            assert "Unknown step type" in str(e)
            print("  Invalid step type: PASSED")
    finally:
        os.unlink(temp_path)

    # Test missing 'type' field
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"steps": [{"temperature": 0.1}]}, f)
        temp_path = f.name

    try:
        try:
            ProtocolFileParser.load_from_file(temp_path)
            assert False, "Should have raised ProtocolValidationError"
        except ProtocolValidationError as e:
            assert "must have a 'type' field" in str(e)
            print("  Missing 'type' field: PASSED")
    finally:
        os.unlink(temp_path)


def test_all_step_types_json():
    """Test parsing all step types from JSON."""
    print("Testing all step types in JSON format...")

    json_content = {
        "steps": [
            {"type": "mpnn", "temperature": 0.1, "num_designs": 4},
            {"type": "mpnn_primary", "temperature": 0.2, "num_designs": 2},
            {"type": "mpnn_secondary", "temperature": 0.15, "num_designs": 3},
            {"type": "mpnn_2nd_shell", "temperature": 0.1, "num_designs": 2},
            {"type": "cart_relax", "repeats": 2, "stages": 3, "until_converged": True},
            {"type": "torsional_relax", "repeats": 1, "stages": 2},
            {"type": "minimize", "tolerance": 0.001, "max_iter": 500},
            {"type": "repack", "repack_shell": 5.0},
            {"type": "select_best", "n": 2, "metric": "geometry"}
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_content, f)
        temp_path = f.name

    try:
        steps = ProtocolFileParser.load_from_file(temp_path)

        assert len(steps) == 9

        # Check MPNN variants
        assert isinstance(steps[0], MPNNStep) and steps[0].step_type == StepType.MPNN
        assert isinstance(steps[1], MPNNStep) and steps[1].step_type == StepType.MPNN_PRIMARY
        assert isinstance(steps[2], MPNNStep) and steps[2].step_type == StepType.MPNN_SECONDARY
        assert steps[2].design_spheres == ["primary", "secondary"]
        assert isinstance(steps[3], MPNNStep) and steps[3].step_type == StepType.MPNN_2ND_SHELL

        # Check cart_relax with until_converged
        assert isinstance(steps[4], CartRelaxStep)
        assert steps[4].until_converged == True

        # Check torsional_relax
        assert isinstance(steps[5], TorsionalRelaxStep)

        # Check minimize
        assert isinstance(steps[6], MinimizeStep)
        assert steps[6].tolerance == 0.001
        assert steps[6].max_iter == 500

        # Check repack
        assert isinstance(steps[7], RepackStep)
        assert steps[7].repack_shell == 5.0

        # Check select_best
        assert isinstance(steps[8], SelectBestStep)
        assert steps[8].n == 2
        assert steps[8].metric == "geometry"

        print("  All step types: PASSED")

    finally:
        os.unlink(temp_path)


def test_validate_file():
    """Test the validate_file method."""
    print("Testing validate_file method...")

    # Valid JSON file
    json_content = {
        "steps": [{"type": "mpnn", "temperature": 0.1, "num_designs": 4}]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_content, f)
        temp_path = f.name

    try:
        result = ProtocolFileParser.validate_file(temp_path)
        assert result["valid"] == True
        assert result["format"] == "json"
        assert result["num_steps"] == 1
        assert len(result["errors"]) == 0
        print("  Valid file validation: PASSED")
    finally:
        os.unlink(temp_path)

    # Non-existent file
    result = ProtocolFileParser.validate_file("/nonexistent/path.json")
    assert result["valid"] == False
    assert "File not found" in result["errors"][0]
    print("  Non-existent file validation: PASSED")


def test_example_files():
    """Test the example protocol files created in the test directory."""
    print("Testing example protocol files...")

    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Test JSON example
    json_example = os.path.join(test_dir, "example_protocol.json")
    if os.path.exists(json_example):
        steps = ProtocolFileParser.load_from_file(json_example)
        assert len(steps) == 4
        print(f"  {json_example}: PASSED ({len(steps)} steps)")
    else:
        print(f"  {json_example}: SKIPPED (file not found)")

    # Test text example
    txt_example = os.path.join(test_dir, "example_protocol.txt")
    if os.path.exists(txt_example):
        steps = ProtocolFileParser.load_from_file(txt_example)
        assert len(steps) == 4
        print(f"  {txt_example}: PASSED ({len(steps)} steps)")
    else:
        print(f"  {txt_example}: SKIPPED (file not found)")


def test_get_examples():
    """Test the example generators."""
    print("Testing example generators...")

    # Test JSON example
    json_example = ProtocolFileParser.get_example_json()
    data = json.loads(json_example)
    assert "steps" in data
    assert len(data["steps"]) > 0
    print("  get_example_json(): PASSED")

    # Test text example
    text_example = ProtocolFileParser.get_example_text()
    assert "mpnn" in text_example
    assert "#" in text_example  # Has comments
    print("  get_example_text(): PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ProtocolFileParser Test Suite")
    print("=" * 60 + "\n")

    try:
        test_json_protocol_parsing()
        test_text_protocol_parsing()
        test_json_validation_errors()
        test_all_step_types_json()
        test_validate_file()
        test_example_files()
        test_get_examples()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60 + "\n")
        return 0

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
