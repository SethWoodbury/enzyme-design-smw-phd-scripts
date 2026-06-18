"""Pipeline orchestrator using subprocess execution."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from utils.logging_utils import (
    configure_logging,
    is_verbose,
    print_dict_summary,
    print_key_value,
    print_list_item,
    print_section_header,
    print_subsection_header,
)

LOGGER = logging.getLogger(__name__)


def _build_command(python_bin: str, script: str, args: str) -> List[str]:
    cmd = [python_bin, script]
    if args:
        cmd.extend(shlex.split(args))
    return cmd


def _wrap_apptainer(cmd: List[str], image: Optional[str], binds: str, envs: str) -> List[str]:
    if not image:
        return cmd
    apptainer_cmd = ["apptainer", "exec"]
    if binds:
        apptainer_cmd.extend(["--bind", binds])
    if envs:
        for item in envs.split(","):
            item = item.strip()
            if not item:
                continue
            apptainer_cmd.extend(["--env", item])
    apptainer_cmd.append(image)
    apptainer_cmd.extend(cmd)
    return apptainer_cmd


def _run_stage(cmd: List[str], dry_run: bool, stage_name: str = "Stage") -> subprocess.CompletedProcess:
    """Run a pipeline stage with verbose output.

    Args:
        cmd: Command to execute
        dry_run: If True, only print command without executing
        stage_name: Name of the stage for logging

    Returns:
        CompletedProcess result (or empty result for dry_run)
    """
    if is_verbose():
        print_subsection_header(f"Executing {stage_name}")
        print_key_value("Command", " ".join(cmd))
        print()

    LOGGER.info("Running: %s", " ".join(cmd))

    if dry_run:
        print("  [DRY RUN - Command not executed]")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    if is_verbose():
        print(f"  Starting {stage_name}...")
        print("-" * 60)
        start_time = time.time()

    # Run with output streaming so user can see progress
    result = subprocess.run(cmd, check=True)

    if is_verbose():
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"  {stage_name} completed in {elapsed:.2f} seconds")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline stages via subprocesses.")
    parser.add_argument("--input_pdb", type=Path)
    parser.add_argument("--ref_pdb", type=Path)
    parser.add_argument("--params", type=Path, help="Rosetta params file (used in later stages).")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--test", action="store_true", help="Run pipeline using test config.")
    parser.add_argument(
        "--test_config",
        type=Path,
        default=Path("tests/pipeline_test_config.json"),
        help="Path to pipeline test config JSON.",
    )
    parser.add_argument("--catres_subset", type=str, help="Comma-separated block indices for stage 1.")
    parser.add_argument("--stage2", action="store_true", help="Run stage 2.")
    parser.add_argument("--stage1_extra", type=str, default="", help="Extra args for stage 1.")
    parser.add_argument("--stage2_extra", type=str, default="", help="Extra args for stage 2.")
    parser.add_argument(
        "--stage1_image",
        type=str,
        default="/software/containers/universal.sif",
        help="Apptainer image for stage 1.",
    )
    parser.add_argument("--stage2_image", type=str, help="Apptainer image for stage 2.")
    parser.add_argument("--stage1_bind", type=str, default="", help="Bind mounts for stage 1.")
    parser.add_argument("--stage2_bind", type=str, default="", help="Bind mounts for stage 2.")
    parser.add_argument("--stage1_env", type=str, default="", help="Comma-separated KEY=VAL envs for stage 1.")
    parser.add_argument("--stage2_env", type=str, default="", help="Comma-separated KEY=VAL envs for stage 2.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument(
        "--container_python",
        type=str,
        default="python3",
        help="Python executable inside Apptainer image.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    parser.add_argument("--verbose", action="count", default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Enable verbose mode by default for --test, or use explicit --verbose
    if args.test and args.verbose == 0:
        args.verbose = 1

    configure_logging(args.verbose)

    pipeline_start_time = time.time()

    # ==================== PIPELINE INITIALIZATION ====================
    if is_verbose():
        print_section_header("ENZYME FASTMPNN DESIGN PIPELINE")
        print_key_value("Start Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print_key_value("Mode", "TEST MODE" if args.test else "PRODUCTION MODE")
        print_key_value("Dry Run", str(args.dry_run))

    if args.test:
        if not args.test_config.exists():
            raise SystemExit(f"Test config not found: {args.test_config}")
        config = json.loads(args.test_config.read_text(encoding="utf-8"))
        args.input_pdb = Path(config["input_pdb"])
        args.ref_pdb = Path(config["ref_pdb"])
        args.outdir = Path(config.get("outdir", "tests/outdir_pipeline_test"))
        args.catres_subset = config.get("catres_subset")
        if config.get("params"):
            args.params = Path(config["params"])
        LOGGER.info("Loaded test config from %s", args.test_config)

        if is_verbose():
            print_subsection_header("Test Configuration Loaded")
            print_key_value("Config File", str(args.test_config))
            print()
            print("  Configuration Contents:")
            for key, value in config.items():
                print_key_value(key, str(value), indent=4)

    if not args.input_pdb or not args.ref_pdb or not args.outdir:
        raise SystemExit("--input_pdb, --ref_pdb, and --outdir are required (unless --test).")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ==================== PIPELINE INPUTS SUMMARY ====================
    if is_verbose():
        print_section_header("PIPELINE INPUTS")
        print_key_value("Input PDB", str(args.input_pdb))
        print_key_value("  Exists", str(args.input_pdb.exists()))
        if args.input_pdb.exists():
            print_key_value("  Size", f"{args.input_pdb.stat().st_size:,} bytes")

        print_key_value("Reference PDB", str(args.ref_pdb))
        print_key_value("  Exists", str(args.ref_pdb.exists()))
        if args.ref_pdb.exists():
            print_key_value("  Size", f"{args.ref_pdb.stat().st_size:,} bytes")

        print_key_value("Output Directory", str(args.outdir))
        if args.catres_subset:
            print_key_value("Catalytic Residue Subset", args.catres_subset)
        else:
            print_key_value("Catalytic Residue Subset", "All (no filter)")

        if args.params:
            print_key_value("Rosetta Params", str(args.params))

    state: Dict[str, str] = {
        "input_pdb": str(args.input_pdb),
        "ref_pdb": str(args.ref_pdb),
        "outdir": str(args.outdir),
    }
    if args.params:
        state["params"] = str(args.params)
    if args.catres_subset:
        state["catres_subset"] = args.catres_subset

    # ==================== STAGE 1: ACTIVE SITE REMASTERING ====================
    if is_verbose():
        print_section_header("STAGE 1: ACTIVE SITE REMASTERING")
        print_subsection_header("Stage 1 Inputs")
        print_key_value("Input PDB", str(args.input_pdb))
        print_key_value("Reference PDB", str(args.ref_pdb))
        if args.catres_subset:
            print_key_value("Catres Subset Blocks", args.catres_subset)
        print()

    stage1_output_pdb = args.outdir / "stage1_fixed.pdb"
    stage1_output_json = args.outdir / "stage1_catres_catalog.json"

    if is_verbose():
        print_subsection_header("Stage 1 Expected Outputs")
        print_key_value("Modified PDB", str(stage1_output_pdb))
        print_key_value("Catalytic Residue Catalog", str(stage1_output_json))

    catres_arg = f"--catres_subset {args.catres_subset}" if args.catres_subset else ""
    verbose_arg = "--verbose " * args.verbose if args.verbose > 0 else ""
    stage1_args = (
        f"--input_pdb {args.input_pdb} "
        f"--ref_pdb {args.ref_pdb} "
        f"--output_pdb {stage1_output_pdb} "
        f"--output_json {stage1_output_json} "
        f"{catres_arg} "
        f"{verbose_arg} "
        f"{args.stage1_extra}"
    ).strip()

    python_bin = args.container_python if args.stage1_image else args.python
    cmd = _build_command(python_bin, "stages/stage1_activesite_remaster/run.py", stage1_args)

    if is_verbose() and args.stage1_image:
        print_subsection_header("Container Configuration")
        print_key_value("Apptainer Image", args.stage1_image)
        if args.stage1_bind:
            print_key_value("Bind Mounts", args.stage1_bind)
        if args.stage1_env:
            print_key_value("Environment", args.stage1_env)

    cmd = _wrap_apptainer(cmd, args.stage1_image, args.stage1_bind, args.stage1_env)
    _run_stage(cmd, args.dry_run, "Stage 1 (Active Site Remaster)")

    state["stage1_output_pdb"] = str(stage1_output_pdb)
    state["stage1_output_json"] = str(stage1_output_json)

    # ==================== STAGE 1: OUTPUTS SUMMARY ====================
    if is_verbose():
        print_subsection_header("Stage 1 Outputs")

    if not args.dry_run and stage1_output_json.exists():
        try:
            catalog = json.loads(stage1_output_json.read_text(encoding="utf-8"))
            state["stage1_catres_count"] = str(len(catalog))

            if is_verbose():
                print_key_value("Output PDB Written", str(stage1_output_pdb.exists()))
                if stage1_output_pdb.exists():
                    print_key_value("Output PDB Size", f"{stage1_output_pdb.stat().st_size:,} bytes")
                print_key_value("Catalytic Residues Cataloged", str(len(catalog)))
                print()
                print("  Catalytic Residue Summary:")
                for res_id, info in catalog.items():
                    interactions_count = len(info.get("interactions_found", []))
                    component = info.get("important_component", "unknown")
                    print(f"    {res_id} ({info.get('res_type', '???')}): "
                          f"{interactions_count} interactions, important component: {component}")

        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed to read stage1 catalog: %s", exc)
            if is_verbose():
                print_key_value("Warning", f"Could not read catalog: {exc}")
    elif is_verbose() and args.dry_run:
        print("  [DRY RUN - No output files generated]")

    # ==================== STAGE 2: ROSETTA RELAX (Optional) ====================
    if args.stage2:
        if is_verbose():
            print_section_header("STAGE 2: ROSETTA RELAX")
            print_subsection_header("Stage 2 Inputs")
            print_key_value("Stage 1 Output PDB", str(stage1_output_pdb))
            if args.params:
                print_key_value("Rosetta Params", str(args.params))

        stage2_args = args.stage2_extra
        if args.params:
            stage2_args = f"--params {args.params} {stage2_args}".strip()
        python_bin = args.container_python if args.stage2_image else args.python
        cmd = _build_command(python_bin, "run_stage2.py", stage2_args)

        if is_verbose() and args.stage2_image:
            print_subsection_header("Container Configuration")
            print_key_value("Apptainer Image", args.stage2_image)
            if args.stage2_bind:
                print_key_value("Bind Mounts", args.stage2_bind)
            if args.stage2_env:
                print_key_value("Environment", args.stage2_env)

        cmd = _wrap_apptainer(cmd, args.stage2_image, args.stage2_bind, args.stage2_env)
        _run_stage(cmd, args.dry_run, "Stage 2 (Rosetta Relax)")
        state["stage2_ran"] = "true"
    else:
        state["stage2_ran"] = "false"
        if is_verbose():
            print()
            print("  [Stage 2 skipped - use --stage2 to enable]")

    # ==================== PIPELINE COMPLETION ====================
    state_path = args.outdir / "pipeline_state.json"
    if not args.dry_run:
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        LOGGER.info("Wrote pipeline state to %s", state_path)

    if is_verbose():
        print_section_header("PIPELINE COMPLETED")
        elapsed = time.time() - pipeline_start_time
        print_key_value("Total Elapsed Time", f"{elapsed:.2f} seconds")
        print_key_value("Pipeline State File", str(state_path))
        print()
        print("  Final State:")
        for key, value in state.items():
            print_key_value(key, value, indent=4)
        print()
        print("=" * 80)


if __name__ == "__main__":
    main()
