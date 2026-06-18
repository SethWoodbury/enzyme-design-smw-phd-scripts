#!/usr/bin/env python3
import os
import re
import glob
import json
import time
import argparse
import shutil
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime

DEFAULT_CIF_SIF = "/software/containers/users/sklein89/maxit.sif"
MODEL_DIR_PATTERN = re.compile(r"seed-1_sample-(\d+)$")  # capture multi-digit index at end

def log(msg: str):
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def vlog(verbose: bool, msg: str):
    if verbose:
        log(msg)

def run_cmd(args, verbose=False):
    vlog(verbose, f"RUN: {' '.join(map(str, args))}")
    subprocess.run(args, check=True)

def safe_move(src: Path, dst: Path, verbose=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    vlog(verbose, f"MOVE: {src} → {dst}")
    shutil.move(str(src), str(dst))

def safe_remove(path: Path, verbose=False):
    if path.is_dir():
        vlog(verbose, f"RMTREE: {path}")
        shutil.rmtree(path)
    elif path.exists():
        vlog(verbose, f"RM: {path}")
        path.unlink()

def parse_job_name(job_dir: Path) -> str:
    meta_files = sorted(job_dir.glob("*_data.json"))
    if len(meta_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_data.json in {job_dir}, found {len(meta_files)}"
        )
    with meta_files[0].open("r") as f:
        data = json.load(f)
    if "name" not in data or not data["name"]:
        raise KeyError(f"'name' missing/empty in {meta_files[0]}")
    return data["name"]

def rename_job_dir(job_dir: Path, corrected_name: str, verbose=False) -> Path:
    parent = job_dir.parent
    target = parent / corrected_name
    if job_dir.resolve() == target.resolve():
        vlog(verbose, f"SKIP rename: already {target.name}")
        return job_dir
    if target.exists():
        raise FileExistsError(f"Target directory already exists: {target}")
    vlog(verbose, f"RENAME DIR: {job_dir.name} → {target.name}")
    shutil.move(str(job_dir), str(target))
    return target

def find_model_subdirs(job_root: Path) -> list[Path]:
    subdirs = [p for p in job_root.iterdir() if p.is_dir()]
    matched = []
    for d in subdirs:
        m = MODEL_DIR_PATTERN.search(d.name)
        if m:
            idx = int(m.group(1))
            matched.append((idx, d))
    matched.sort(key=lambda t: t[0])
    return [d for _, d in matched]

def flatten_model(d: Path, corrected_name: str, job_root: Path, verbose=False) -> dict:
    m = MODEL_DIR_PATTERN.search(d.name)
    if not m:
        raise ValueError(f"Cannot parse model index from {d}")
    idx = m.group(1)
    base = f"{corrected_name}_idx_{idx}"

    sources = {
        "confidences": d / "confidences.json",
        "model_cif": d / "model.cif",
        "summary_confidences": d / "summary_confidences.json",
    }
    dests = {
        "confidences": job_root / f"{base}_confidences.json",
        "model_cif": job_root / f"{base}_model.cif",
        "summary_confidences": job_root / f"{base}_summary_confidences.json",
    }

    moved = []
    missing = []
    for key, src in sources.items():
        if src.exists():
            safe_move(src, dests[key], verbose=verbose)
            moved.append(dests[key])
        else:
            log(f"WARNING: Missing expected file: {src}")
            missing.append(src)

    return {"idx": idx, "moved": moved, "missing": missing, "model_dir": d}

def convert_cifs_to_pdbs(job_root: Path, cif_sif: str, verbose=False, include_nonindexed=False) -> list[tuple[Path, Path]]:
    """
    Convert selected CIF files in job_root to PDB using cif_sif.
    By default, only *_idx_<N>_model.cif files are converted.
    If include_nonindexed=True, all *.cif files in the job root are converted.
    """
    # Pattern: match files ending in "_idx_<digits>_model.cif"
    idx_pattern = re.compile(r"_idx_\d+_model\.cif$", re.IGNORECASE)

    all_cifs = sorted(job_root.glob("*.cif"))
    idx_cifs = [c for c in all_cifs if idx_pattern.search(c.name)]
    non_idx_cifs = [c for c in all_cifs if c not in idx_cifs]

    if not include_nonindexed:
        if non_idx_cifs:
            log("WARNING: Skipping non-indexed CIF files: " +
                ", ".join(c.name for c in non_idx_cifs))
        target_cifs = idx_cifs
    else:
        if non_idx_cifs:
            log("Including non-indexed CIF files: " +
                ", ".join(c.name for c in non_idx_cifs))
        target_cifs = all_cifs

    converted = []
    for cif in target_cifs:
        pdb = cif.with_suffix(".pdb")
        args = [cif_sif, "-input", str(cif), "-output", str(pdb), "-o", "2"]
        run_cmd(args, verbose=verbose)
        if pdb.exists():
            converted.append((cif, pdb))
        else:
            log(f"ERROR: Converter reported success but {pdb} was not created.")
    return converted

def cleanup_after_verification(job_root: Path,
                               kept_model_dirs: list[Path],
                               flattened_info: list[dict],
                               converted_pairs: list[tuple[Path, Path]],
                               disable_deletions: bool,
                               verbose=False):
    """
    Delete only after verifying that intended outputs exist.
    - Remove top-level non-model items EXCEPT model dirs and new flattened files and new PDBs
    - Remove original CIFs that were successfully converted
    - Remove model subdirs IF all expected files from that model were flattened
    """
    if disable_deletions:
        log("DEBUG: --disable-deletions set. No deletions will be performed.")
        return

    # Build whitelist of files we must keep (flattened outputs + PDBs)
    keep_files = set()
    for info in flattened_info:
        for p in info["moved"]:
            keep_files.add(p.resolve())
    for cif, pdb in converted_pairs:
        keep_files.add(pdb.resolve())

    # 1) Remove original CIFs that were converted successfully
    for cif, pdb in converted_pairs:
        if pdb.exists() and cif.exists():
            vlog(verbose, f"Verified PDB present → removing CIF: {cif.name}")
            safe_remove(cif, verbose=verbose)

    # 2) Remove per-model dirs that fully flattened (no missing sources)
    for info in flattened_info:
        model_dir: Path = info["model_dir"]
        if len(info["missing"]) == 0:
            vlog(verbose, f"All expected files flattened → removing model dir: {model_dir.name}")
            safe_remove(model_dir, verbose=verbose)
        else:
            log(f"SKIP removing model dir (missing files): {model_dir.name}")

    # 3) Remove unrelated top-level entries EXCEPT items we keep or remaining model dirs
    current_top = list(job_root.iterdir())
    keep_dirs = set(d.resolve() for d in kept_model_dirs if d.exists())  # ones not removed (had missing files)
    for entry in current_top:
        rp = entry.resolve()
        if rp in keep_dirs:
            continue
        if entry.is_file() and rp in keep_files:
            continue
        # We keep *_data.json by default
        if entry.is_file() and entry.name.endswith("_data.json"):
            continue
        # If it's neither a kept dir nor a kept file, it's garbage from prior copies
        vlog(verbose, f"Removing unrelated top-level entry: {entry.name}")
        safe_remove(entry, verbose=verbose)

def process_one(index: int,
                subdir: Path,
                cif_sif: str,
                disable_deletions: bool,
                verbose: bool,
                include_nonindexed_cifs: bool):
    """
    Process a single AF3 subdir. Safe, verbose, and self-validating.
    """
    log(f"[{index+1}] START: {subdir}")
    if (index + 1) in (1, 10, 100, 1000, 10000):
        log(f"PROGRESS: {index+1} subdirectories processed (power of 10).")

    # --- Step 1: read canonical name & rename folder ---
    corrected_name = parse_job_name(subdir)
    job_root = rename_job_dir(subdir, corrected_name, verbose=verbose)

    # --- Step 2: find per-model subdirs ---
    model_dirs = find_model_subdirs(job_root)
    if not model_dirs:
        log(f"WARNING: No model subdirs found in {job_root}. Skipping flatten/convert.")
        return

    # --- Step 3: flatten each model's required files ---
    flattened_info = []
    for d in model_dirs:
        info = flatten_model(d, corrected_name, job_root, verbose=verbose)
        flattened_info.append(info)

    # --- Step 4: convert any remaining CIFs at top level to PDBs ---
    converted_pairs = convert_cifs_to_pdbs(job_root, cif_sif=cif_sif, verbose=verbose, include_nonindexed=include_nonindexed_cifs)
    log(f"Converted {len(converted_pairs)} CIF → PDB in {job_root.name}")

    # --- Step 5: cleanup (guarded & verifiable; can be fully disabled) ---
    cleanup_after_verification(
        job_root=job_root,
        kept_model_dirs=model_dirs,
        flattened_info=flattened_info,
        converted_pairs=converted_pairs,
        disable_deletions=disable_deletions,
        verbose=verbose,
    )

    log(f"[{index+1}] DONE: {job_root}")

def worker(queue, cif_sif: str, disable_deletions: bool, verbose: bool, include_nonindexed_cifs: bool):
    while True:
        item = queue.get()
        if item is None:
            return
        idx, subdir = item
        try:
            process_one(idx, Path(subdir), cif_sif, disable_deletions, verbose, include_nonindexed_cifs)
        except Exception as e:
            log(f"ERROR processing {subdir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Normalize AF3 outputs, flatten models, convert CIF→PDB.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--af3_dir", help="Path of a directory with AF3 outputs (many jobs).")
    src.add_argument("--af3_subdir", help="Path of a single AF3 job subdirectory.")
    parser.add_argument("--cif_sif", default=DEFAULT_CIF_SIF,
                        help=f"Path to CIF→PDB converter (default: {DEFAULT_CIF_SIF}).")
    parser.add_argument("--max_procs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of worker processes for --af3_dir mode.")
    parser.add_argument("--disable-deletions", action="store_true",
                        help="If set, NO deletions are performed (debug mode).")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging.")
    parser.add_argument(
        "--include-nonindexed-cifs",
        action="store_true",
        help="If set, convert non-indexed CIF files in addition to indexed *_idx_<N>_model.cif."
    )
    args = parser.parse_args()

    if args.af3_dir:
        af3_subdirs = sorted([p for p in Path(args.af3_dir).iterdir() if p.is_dir()])
        log(f"Discovered {len(af3_subdirs)} subdirectories under {args.af3_dir}")
        if not af3_subdirs:
            log("Nothing to do.")
            return

        manager = multiprocessing.Manager()
        q = manager.Queue()

        for i, d in enumerate(af3_subdirs):
            q.put((i, str(d)))

        nwork = max(1, args.max_procs)
        log(f"Starting pool with {nwork} workers (deletions {'DISABLED' if args.disable_deletions else 'ENABLED'}).")
        pool = []
        for _ in range(nwork):
            p = multiprocessing.Process(target=worker, args=(q, args.cif_sif, args.disable_deletions, args.verbose, args.include_nonindexed_cifs))
            p.start()
            pool.append(p)

        # Sentinels
        for _ in range(nwork):
            q.put(None)

        # Join
        for p in pool:
            p.join()

        log("All jobs complete.")

    else:
        # Single subdir mode, no extra processes
        process_one(0, Path(args.af3_subdir), args.cif_sif, args.disable_deletions, args.verbose, args.include_nonindexed_cifs)

if __name__ == "__main__":
    main()
