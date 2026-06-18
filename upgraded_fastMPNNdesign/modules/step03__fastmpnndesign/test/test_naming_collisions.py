#!/usr/bin/env python3
"""Dry checks for filename collision prevention in step03."""
import json
import tempfile
from pathlib import Path

from modules.step03__fastmpnndesign.fastmpnn_design import FastMPNNDesigner


def main() -> int:
    script_dir = Path(__file__).parent.resolve()
    params = script_dir / "params" / "XDW.params"
    if not params.exists():
        print(f"Missing params file: {params}")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        step02_json = tmpdir / "step02_stub.json"
        step02_json.write_text(json.dumps({
            "metadata": {
                "output_pdb": str(script_dir / "step01_outputs" / "input_pdb_aligned.pdb"),
                "input_pdb": str(script_dir / "step01_outputs" / "input_pdb_aligned.pdb"),
                "ref_pdb": str(script_dir / "step01_outputs" / "input_pdb_aligned.pdb"),
            }
        }))

        designer = FastMPNNDesigner(
            step02_json_path=str(step02_json),
            params_files=[str(params)],
            output_dir=str(tmpdir / "out"),
            protocol=None,
            test=True,
        )

        # Two different paths with identical basenames should not collide
        p1 = "/tmp/a/input_for_mpnn_packed_1_1.pdb"
        p2 = "/tmp/b/input_for_mpnn_packed_1_1.pdb"
        out1 = designer._make_tagged_output_path(p1, "repack", out_dir=tmpdir)
        out2 = designer._make_tagged_output_path(p2, "repack", out_dir=tmpdir)
        if out1 == out2:
            print("Collision detected: tagged output paths are identical")
            return 1

        # Packed suffix should be deterministic and unique per call
        idx1, suf1 = designer._next_mpnn_packed_suffix(strategy_tag="s01")
        idx2, suf2 = designer._next_mpnn_packed_suffix(strategy_tag="s02")
        if idx1 == idx2 or suf1 == suf2:
            print("Collision detected: packed suffixes not unique")
            return 1

        # Long stem should be shortened deterministically
        long_stem = "A" * 500
        p3 = f"/tmp/{long_stem}.pdb"
        out3 = designer._make_tagged_output_path(p3, "minimize", out_dir=tmpdir)
        out4 = designer._make_tagged_output_path(p3, "minimize", out_dir=tmpdir)
        if out3 != out4:
            print("Collision detected: shortened names are not deterministic")
            return 1
        if len(Path(out3).name) > 200:
            print("Collision detected: shortened name still too long")
            return 1

    print("OK: naming collision checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
