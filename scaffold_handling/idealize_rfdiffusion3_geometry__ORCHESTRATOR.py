#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator wrapper around idealize_rfdiffusion3_geometry__MAIN.py.

Runs MAIN.py N times back-to-back on a single design, feeding stage K's output
PDB as stage K+1's input. Each `--stage` is a quoted string of MAIN.py CLI args
that gets layered on top of the passthrough base for that stage. Use
`--stage ''` for a stage that just uses the base args.

Stage outputs are written to:
    <output_dir>/<basename>_idealized_stage1.pdb       (intermediate)
    <output_dir>/<basename>_idealized_stage2.pdb       (intermediate)
    ...
    <output_dir>/<basename>_idealized.pdb              (final)

The final-stage filename matches MAIN.py's normal `_idealized.pdb` convention
so downstream tools that already look for that name keep working.

Two post-process steps run after the last stage succeeds (both can be opted
out individually):

  1. Strip every `REMARK DESIGN_PATH predesign_cart_relax output <path>` line
     from the final PDB whose path is not the final PDB's own path. The
     intermediate-stage predesign lines get rescued into every later stage's
     output by MAIN.py's REMARK-rescue mechanism, so without this cleanup the
     final PDB would carry one such line per stage.

  2. Replace the final stage's `input_metrics` and `change_from_input` blocks
     with end-to-end versions: `input_metrics` ← stage 1's input baseline
     (= original input pose), `change_from_input` ← (final − original_input).
     Intermediate stages' metrics JSONs keep their LOCAL deltas for
     per-stage debugging.

Example:

    universal.sif idealize_rfdiffusion3_geometry__ORCHESTRATOR.py \\
        --pdb input.pdb --params PBJ.params \\
        --output_dir /scratch/.../ \\
        --corresponding_json_dir /scratch/.../ \\
        --coord_cst_weight 750 --coord_cst_stdev 0.02 \\
        --hbond_conserve_prob 0.8 \\
        --stage '--skip_mpnn --skip_fastrelax' \\
        --stage '' \\
        --stage ''

Convention mirrors process_diffusion3_outputs__ORCHESTRATOR.py: function-based,
argparse.parse_known_args() for passthrough, subprocess.run([sys.executable,
main_script, ...]) for each child invocation (no nested apptainer — the
orchestrator is launched inside the container and child processes inherit it).
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metric_deltas(final_metrics, input_metrics):
    """Numeric-leaf-walk differ. Mirrors the implementation in MAIN.py's
    RFDiffusion3GeometryIdealizer._compute_metric_deltas — duplicated here so
    the orchestrator doesn't need to import (and PyRosetta-init) MAIN.py."""

    def _is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    def _walk(f, i):
        if isinstance(f, dict) and isinstance(i, dict):
            out = {}
            for k in f.keys() & i.keys():
                sub = _walk(f[k], i[k])
                if sub is not None:
                    out[k] = sub
            return out if out else None
        if _is_num(f) and _is_num(i):
            return float(f) - float(i)
        return None

    result = _walk(final_metrics, input_metrics)
    return result if result is not None else {}


def _resolve_original_json(pdb_path: str, json_dir: str = None,
                            explicit_json: str = None) -> str:
    """Mirror MAIN.py's auto-detect logic so we can pass an explicit --json to
    every stage. Order of preference:
      1. explicit_json (user-provided)
      2. <pdb_basename>.json next to pdb_path
      3. <pdb_basename>.json under json_dir
    Returns None if nothing is found (MAIN.py will then warn but proceed).
    """
    if explicit_json:
        return explicit_json
    base = os.path.basename(pdb_path).replace(".pdb", "")
    cand1 = os.path.join(os.path.dirname(os.path.abspath(pdb_path)), f"{base}.json")
    if os.path.isfile(cand1):
        return cand1
    if json_dir:
        cand2 = os.path.join(json_dir, f"{base}.json")
        if os.path.isfile(cand2):
            return cand2
    return None


def _strip_intermediate_design_path_lines(pdb_path: str, final_path: str) -> int:
    """Remove every `REMARK DESIGN_PATH predesign_cart_relax output <path>`
    line whose path (normalized) is not the final output path. Returns the
    number of lines dropped."""
    final_norm = os.path.normpath(final_path)
    with open(pdb_path, "r") as fh:
        lines = fh.readlines()
    out = []
    dropped = 0
    for line in lines:
        if line.startswith("REMARK DESIGN_PATH predesign_cart_relax output"):
            tokens = line.rstrip("\n").split(None, 4)
            if len(tokens) == 5:
                this_path = os.path.normpath(tokens[4].rstrip())
                if this_path != final_norm:
                    dropped += 1
                    continue
        out.append(line)
    with open(pdb_path, "w") as fh:
        fh.writelines(out)
    return dropped


def _metrics_json_for(pdb_path: str) -> str:
    return pdb_path.replace(".pdb", "_metrics.json")


def _replace_endtoend_metrics(final_metrics_path: str,
                              stage1_metrics_path: str) -> dict:
    """Replace the final stage's `input_metrics` + `change_from_input` blocks
    with end-to-end versions (relative to the original input pose, captured in
    stage 1's `input_metrics` block). Returns a small dict summarizing what
    happened (or {'skipped': '<reason>'} if either block is missing)."""
    if not os.path.isfile(final_metrics_path):
        return {'skipped': f'final metrics JSON not found: {final_metrics_path}'}
    if not os.path.isfile(stage1_metrics_path):
        return {'skipped': f'stage 1 metrics JSON not found: {stage1_metrics_path}'}

    with open(final_metrics_path) as fh:
        final_m = json.load(fh)
    with open(stage1_metrics_path) as fh:
        stage1_m = json.load(fh)

    original_input_metrics = stage1_m.get('input_metrics')
    if not original_input_metrics:
        return {'skipped': "stage 1 metrics JSON has no 'input_metrics' block "
                           "(was --no_input_baseline_metrics set?)"}

    # Build the "final pose metrics" view — everything the script considers
    # the final-pose state. We deliberately mirror the top-level structure
    # MAIN.py writes (global_metrics, catalytic_residues, scores, quality_flags,
    # ...) and exclude the previously-written input_metrics / change_from_input
    # so we're diffing the right things.
    final_pose_view = {
        k: v for k, v in final_m.items()
        if k not in ('input_metrics', 'change_from_input',
                     'mpnn', 'metadata', 'declared_covalent_contacts',
                     'geometry', 'constraints', 'timing')
    }
    # input_metrics from stage 1 IS a numeric skeleton already (MAIN.py's
    # _extract_numeric_skeleton ran when it wrote that JSON), so diffing
    # final_pose_view against it yields the end-to-end change.
    endtoend_change = _compute_metric_deltas(final_pose_view, original_input_metrics)

    final_m['input_metrics'] = original_input_metrics
    final_m['change_from_input'] = endtoend_change

    with open(final_metrics_path, "w") as fh:
        json.dump(final_m, fh, indent=2, sort_keys=False)

    return {
        'final_metrics_path': final_metrics_path,
        'stage1_metrics_path': stage1_metrics_path,
        'n_endtoend_deltas': sum(_count_leaves(endtoend_change)),
    }


def _count_leaves(d):
    """Yield 1 for every leaf in a nested dict (used for the verbose summary)."""
    if isinstance(d, dict):
        for v in d.values():
            yield from _count_leaves(v)
    else:
        yield 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Multi-stage wrapper around idealize_rfdiffusion3_geometry__MAIN.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pdb", type=str, required=True,
                        help="Input PDB (passed to stage 1).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where intermediate _stage<K>.pdb files and the "
                             "final _idealized.pdb are written.")
    parser.add_argument("--json", type=str, default=None,
                        help="Explicit JSON metadata path. Auto-detected from --pdb or "
                             "--corresponding_json_dir if omitted. Forwarded explicitly to "
                             "every stage so basename-based auto-detect doesn't break on "
                             "intermediate _stage<K>.pdb files.")
    parser.add_argument("--corresponding_json_dir", type=str, default=None,
                        help="Fallback dir for JSON auto-detect when --json is not given.")

    parser.add_argument("--stage", dest="stages", action="append", default=[],
                        help="Per-stage MAIN.py args (quoted string). Repeatable; one "
                             "stage per occurrence. Use '' for a stage that only uses "
                             "the base passthrough args.")

    parser.add_argument("--keep_intermediate_design_path_remarks", action="store_true",
                        help="Keep all 'REMARK DESIGN_PATH predesign_cart_relax output' "
                             "lines (one per stage) in the final PDB. Default: keep only "
                             "the line whose path is the final PDB's own path.")
    parser.add_argument("--no_endtoend_metrics_replacement", action="store_true",
                        help="Do not replace the final stage's input_metrics / "
                             "change_from_input with end-to-end versions (relative to the "
                             "original input pose).")
    parser.add_argument("--cleanup_intermediates", action="store_true",
                        help="Delete intermediate _stage<K>.pdb and _stage<K>_metrics.json "
                             "files after all stages succeed.")

    parser.add_argument(
        "--main_script", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "idealize_rfdiffusion3_geometry__MAIN.py"),
        help="Path to the underlying idealize_rfdiffusion3_geometry__MAIN.py script.")
    return parser


def main():
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    if not args.stages:
        print("[orchestrator] error: at least one --stage is required.", file=sys.stderr)
        return 2
    if not os.path.isfile(args.main_script):
        print(f"[orchestrator] error: main_script not found: {args.main_script}",
              file=sys.stderr)
        return 2
    if not os.path.isfile(args.pdb):
        print(f"[orchestrator] error: input PDB not found: {args.pdb}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve the original JSON path once and forward it explicitly to every
    # stage. Without this, MAIN.py's basename-based auto-detect would fail on
    # stages 2+ because the input basename has a _stage<K> suffix.
    json_path = _resolve_original_json(
        args.pdb, json_dir=args.corresponding_json_dir, explicit_json=args.json)

    base = os.path.basename(args.pdb).replace(".pdb", "")
    n_stages = len(args.stages)

    print("=" * 70)
    print(f"[orchestrator] idealize wrapper: {n_stages} stage(s)")
    print("=" * 70)
    print(f"  input PDB        : {args.pdb}")
    print(f"  output dir       : {args.output_dir}")
    print(f"  resolved JSON    : {json_path or '(none — MAIN.py will fall back to its own auto-detect)'}")
    print(f"  base passthrough : {passthrough}")
    for i, s in enumerate(args.stages, 1):
        print(f"  stage {i}/{n_stages} args : {s!r}")
    print("=" * 70)

    stage_pdbs = []  # paths to each stage's output PDB
    for i, stage_args_str in enumerate(args.stages, 1):
        is_final = (i == n_stages)
        in_pdb = args.pdb if i == 1 else stage_pdbs[-1]
        out_pdb = os.path.join(
            args.output_dir,
            f"{base}_idealized.pdb" if is_final else f"{base}_idealized_stage{i}.pdb",
        )

        cmd = [sys.executable, args.main_script,
               *passthrough,
               *shlex.split(stage_args_str),
               "--pdb", in_pdb,
               "--output", out_pdb]
        if json_path:
            cmd += ["--json", json_path]

        print("\n" + "=" * 70)
        print(f"[orchestrator] stage {i}/{n_stages}  (final={is_final})")
        print(f"  in  : {in_pdb}")
        print(f"  out : {out_pdb}")
        print(f"  cmd : {' '.join(shlex.quote(c) for c in cmd)}")
        print("=" * 70)
        sys.stdout.flush()

        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f"\n[orchestrator] stage {i} failed with return code {ret.returncode}. "
                  "Aborting; earlier stages' outputs are left in place for inspection.",
                  file=sys.stderr)
            return ret.returncode

        if not os.path.isfile(out_pdb):
            print(f"\n[orchestrator] stage {i} exited 0 but did not produce {out_pdb}. "
                  "Aborting.", file=sys.stderr)
            return 3
        stage_pdbs.append(out_pdb)

    final_pdb = stage_pdbs[-1]
    print("\n" + "=" * 70)
    print(f"[orchestrator] all {n_stages} stage(s) complete.")
    print("=" * 70)

    # ----- Post-process: strip intermediate predesign_cart_relax REMARKs -----
    if not args.keep_intermediate_design_path_remarks:
        dropped = _strip_intermediate_design_path_lines(final_pdb, final_pdb)
        print(f"[orchestrator] final-PDB REMARK cleanup: dropped {dropped} "
              f"intermediate 'REMARK DESIGN_PATH predesign_cart_relax output' line(s) "
              f"(only the one matching the final PDB path is retained).")

    # ----- Post-process: replace end-to-end metrics in the final JSON -----
    if not args.no_endtoend_metrics_replacement and n_stages > 1:
        stage1_metrics = _metrics_json_for(stage_pdbs[0])
        final_metrics = _metrics_json_for(final_pdb)
        result = _replace_endtoend_metrics(final_metrics, stage1_metrics)
        if 'skipped' in result:
            print(f"[orchestrator] end-to-end metrics replacement SKIPPED: {result['skipped']}")
        else:
            print(f"[orchestrator] end-to-end metrics replaced in {final_metrics} "
                  f"({result['n_endtoend_deltas']} delta leaves; baseline = stage 1's "
                  f"input_metrics, i.e. the original input pose).")

    # ----- Optional: clean up intermediate files -----
    if args.cleanup_intermediates and n_stages > 1:
        removed = 0
        for p in stage_pdbs[:-1]:
            for victim in (p, _metrics_json_for(p)):
                if os.path.isfile(victim):
                    os.remove(victim)
                    removed += 1
        print(f"[orchestrator] cleaned up {removed} intermediate file(s).")

    print(f"\n[orchestrator] final PDB     : {final_pdb}")
    print(f"[orchestrator] final metrics : {_metrics_json_for(final_pdb)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
