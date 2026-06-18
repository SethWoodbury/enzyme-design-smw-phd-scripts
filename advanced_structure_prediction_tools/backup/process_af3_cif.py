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

#####
# Detect the new AF3 layout: a "samples" dir with files containing "_seed-<d>_sample-<d>_"
NEW_LAYOUT_TOKEN = re.compile(r"_seed-\d+_sample-\d+_")
# Strong classifiers for new layout files
RE_CONF   = re.compile(r"_seed-\d+_sample-\d+_confidences\.json$", re.IGNORECASE)
RE_MODEL  = re.compile(r"_seed-\d+_sample-\d+_model\.cif$", re.IGNORECASE)
RE_SUM    = re.compile(r"_seed-\d+_sample-\d+_summary_confidences\.json$", re.IGNORECASE)

# Detect already-flattened "_idx_<n>_" triplets at the job root
IDX_TOKEN     = re.compile(r"_idx_(\d+)_")
RE_IDX_MODEL  = re.compile(r"_idx_\d+_model\.cif$", re.IGNORECASE)
RE_IDX_CONF   = re.compile(r"_idx_\d+_confidences\.json$", re.IGNORECASE)
RE_IDX_SUM    = re.compile(r"_idx_\d+_summary_confidences\.json$", re.IGNORECASE)

def detect_existing_idx_triplets(job_root: Path, verbose=False) -> list[dict]:
    """
    Scan job_root for already-flattened *_idx_<n>_* files.
    Returns a list of info dicts analogous to flatten_model/flatten_new_layout output:
      { "idx": "<n>", "moved": [Paths that exist], "missing": [("idx<n>", "kind"), ...], "model_dir": job_root }
    """
    files = [p for p in job_root.iterdir() if p.is_file()]
    groups: dict[str, dict[str, Path]] = {}
    for p in files:
        name = p.name
        if RE_IDX_MODEL.search(name) or RE_IDX_CONF.search(name) or RE_IDX_SUM.search(name):
            m = IDX_TOKEN.search(name)
            if not m:
                continue
            idx = m.group(1)
            group = groups.setdefault(idx, {})
            if RE_IDX_MODEL.search(name):
                group["model_cif"] = p
            elif RE_IDX_CONF.search(name):
                group["confidences"] = p
            elif RE_IDX_SUM.search(name):
                group["summary_confidences"] = p

    if not groups:
        return []

    ordered = sorted(groups.items(), key=lambda kv: int(kv[0]))
    info_list: list[dict] = []
    for idx, srcs in ordered:
        # emulate the 'moved' list: what we "have" already at the root
        have = []
        missing: list[tuple[str, str]] = []
        for kind in ("confidences", "model_cif", "summary_confidences"):
            p = srcs.get(kind)
            if p and p.exists():
                have.append(p)
            else:
                missing.append((f"idx_{idx}", kind))
        info_list.append({
            "idx": idx,
            "moved": have,
            "missing": missing,
            "model_dir": None,  # IMPORTANT: was job_root; never point at the root
        })

    vlog(verbose, f"ROBUST: detected {len(info_list)} indexed triplets at root: "
                  f"{', '.join(sorted(groups.keys(), key=lambda s: int(s)))}")
    return info_list

def remove_empty_samples_dir(job_root: Path, verbose=False):
    samples = job_root / "samples"
    if samples.is_dir():
        try:
            contents = list(samples.iterdir())
        except Exception:
            contents = ["<error reading>"]
        if isinstance(contents, list) and len(contents) == 0:
            vlog(verbose, f"Removing empty samples/ directory in {job_root.name}")
            safe_remove(samples, verbose=verbose)

def is_new_layout(job_root: Path) -> bool:
    samples = job_root / "samples"
    if not samples.is_dir():
        return False
    for p in samples.iterdir():
        if p.is_file() and NEW_LAYOUT_TOKEN.search(p.name):
            return True
    return False

def flatten_new_layout(job_root: Path, corrected_name: str, verbose=False) -> list[dict]:
    """
    New-format flattener:
      - Reads from job_root/samples
      - Groups triplets by their '_seed-X_sample-Y_' token
      - Sorts those tokens alphanumerically
      - Renames/moves to job_root as '<corrected_name>_idx_<i>_{confidences,model,summary_confidences}.ext'
      - Returns a list of info dicts similar to flatten_model() output for cleanup compatibility
    """
    samples_dir = job_root / "samples"
    files = [p for p in samples_dir.iterdir() if p.is_file()]
    # Group by the exact seed/sample token (e.g., "_seed-1_sample-3_")
    groups: dict[str, dict[str, Path]] = {}
    for p in files:
        m = NEW_LAYOUT_TOKEN.search(p.name)
        if not m:
            continue
        token = m.group(0)  # includes leading & trailing underscores
        groups.setdefault(token, {})
        # classify file kind (robust)
        if RE_CONF.search(p.name):
            groups[token]["confidences"] = p
        elif RE_MODEL.search(p.name):
            groups[token]["model_cif"] = p
        elif RE_SUM.search(p.name):
            groups[token]["summary_confidences"] = p

    # Alphanumeric order on the token keys, then map to idx_0, idx_1, ...
    ordered_tokens = sorted(groups.keys())

    flattened_info: list[dict] = []
    for i, token in enumerate(ordered_tokens):
        base = f"{corrected_name}_idx_{i}"
        srcs = groups[token]
        dests = {
            "confidences": job_root / f"{base}_confidences.json",
            "model_cif": job_root / f"{base}_model.cif",
            "summary_confidences": job_root / f"{base}_summary_confidences.json",
        }

        moved, missing = [], []
        # move when present; record missing
        for kind, dst in dests.items():
            src = srcs.get(kind, None)
            if src and src.exists():
                safe_move(src, dst, verbose=verbose)
                moved.append(dst)
            else:
                log(f"WARNING: Missing expected file for {token}: {kind}")
                missing.append((token, kind))

        # Mirror the structure returned by flatten_model() to keep cleanup compatible
        flattened_info.append({
            "idx": str(i),
            "moved": moved,
            "missing": missing,
            "model_dir": samples_dir,  # placeholder; we don't have per-model dirs in new layout
        })

    return flattened_info
#####

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

def safe_remove(path: Path, verbose=False, job_root: Path=None):
    # Safety: never remove the job root itself
    if job_root is not None:
        try:
            if path.resolve() == job_root.resolve():
                log(f"SAFETY: refusing to remove job_root: {job_root}")
                return
        except Exception:
            pass

    try:
        if path.is_dir():
            vlog(verbose, f"RMTREE: {path}")
            shutil.rmtree(path)
        elif path.exists():
            vlog(verbose, f"RM: {path}")
            path.unlink()
    except Exception as e:
        log(f"WARNING: failed to remove {path}: {e}")

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
    Delete only after verifying intended outputs exist.
      - Remove original CIFs successfully converted
      - Remove per-model subdirs that fully flattened (but never the job root)
      - Remove unrelated top-level files (keep whitelisted + kept dirs + *_data.json)
    """
    if disable_deletions:
        log("DEBUG: --disable-deletions set. No deletions will be performed.")
        return

    if not job_root.exists():
        log(f"WARNING: job_root missing during cleanup: {job_root}")
        return

    def _is_safe_model_dir(p: Path) -> bool:
        # Only remove directories that are direct children of job_root and match the legacy model pattern
        try:
            return (
                isinstance(p, Path)
                and p.exists()
                and p.is_dir()
                and p.parent.resolve() == job_root.resolve()
                and MODEL_DIR_PATTERN.search(p.name) is not None
            )
        except Exception:
            return False

    # Build whitelist of files we must keep (flattened outputs + PDBs)
    keep_files = set()
    for info in flattened_info or []:
        for p in info.get("moved", []):
            try:
                keep_files.add(Path(p).resolve())
            except Exception:
                pass
    for cif, pdb in converted_pairs or []:
        try:
            keep_files.add(Path(pdb).resolve())
        except Exception:
            pass

    # Always preserve already-flattened idx JSONs at the job root (robust against partial detection)
    for f in job_root.glob("*_idx_*_confidences.json"):
        if RE_IDX_CONF.search(f.name) or RE_IDX_SUM.search(f.name):
            try:
                keep_files.add(f.resolve())
            except Exception:
                pass

    # 1) Remove original CIFs that were converted successfully
    for cif, pdb in converted_pairs or []:
        try:
            if Path(pdb).exists() and Path(cif).exists():
                vlog(verbose, f"Verified PDB present → removing CIF: {Path(cif).name}")
                safe_remove(Path(cif), verbose=verbose)
        except Exception as e:
            log(f"WARNING: could not remove CIF {cif}: {e}")

    # 2) Remove per-model dirs that fully flattened (never remove job_root)
    for info in flattened_info or []:
        model_dir = info.get("model_dir")
        if model_dir is None:
            continue  # robust path: no per-model dir to delete
        model_dir = Path(model_dir)
        if not _is_safe_model_dir(model_dir):
            vlog(verbose, f"Skip removing non-model or unsafe dir: {model_dir}")
            continue
        if len(info.get("missing", [])) == 0:
            vlog(verbose, f"All expected files flattened → removing model dir: {model_dir.name}")
            safe_remove(model_dir, verbose=verbose)
        else:
            log(f"SKIP removing model dir (missing files): {model_dir.name}")

    # 3) Remove unrelated top-level files (never remove directories here)
    if not job_root.exists():
        log(f"WARNING: job_root disappeared mid-cleanup: {job_root}")
        return
    try:
        current_top = list(job_root.iterdir())
    except FileNotFoundError:
        log(f"WARNING: job_root missing when listing: {job_root}")
        return

    keep_dirs = set()
    for d in kept_model_dirs or []:
        try:
            if Path(d).exists():
                keep_dirs.add(Path(d).resolve())
        except Exception:
            pass

    for entry in current_top:
        rp = entry.resolve()
        # Never delete directories here; only files. Model dirs are handled above.
        if entry.is_dir():
            continue
        if rp in keep_files:
            continue
        if entry.name.endswith("_data.json"):
            continue
        vlog(verbose, f"Removing unrelated top-level file: {entry.name}")
        safe_remove(entry, verbose=verbose)

def process_one(index: int,
                subdir: Path,
                cif_sif: str,
                disable_deletions: bool,
                verbose: bool,
                include_nonindexed_cifs: bool,
                robust_mode: bool):
    """
    Process a single AF3 subdir. Safe, verbose, and self-validating.
    """
    log(f"[{index+1}] START: {subdir}")
    if (index + 1) in (1, 10, 100, 1000, 10000):
        log(f"PROGRESS: {index+1} subdirectories processed (power of 10).")

    # --- Step 1: read canonical name & rename folder ---
    corrected_name = parse_job_name(subdir)
    job_root = rename_job_dir(subdir, corrected_name, verbose=verbose)

    # --- NEW LAYOUT BRANCH: samples/ + *_seed-*_sample-*_* files ---
    if is_new_layout(job_root):
        vlog(verbose, f"Detected new layout in: {job_root} (samples/*_seed-*_sample-*_)")
        flattened_info = flatten_new_layout(job_root, corrected_name, verbose=verbose)

        if not flattened_info and robust_mode:
            # samples/ exists but empty → look for already-flattened idx files at root
            log("ROBUST: samples/ appears empty (no groups). Checking for *_idx_* files at root...")
            idx_info = detect_existing_idx_triplets(job_root, verbose=verbose)
            if idx_info:
                converted_pairs = convert_cifs_to_pdbs(
                    job_root, cif_sif=cif_sif, verbose=verbose, include_nonindexed=include_nonindexed_cifs
                )
                log(f"Converted {len(converted_pairs)} CIF → PDB in {job_root.name} (robust idx-root path)")
                cleanup_after_verification(
                    job_root=job_root,
                    kept_model_dirs=[],             # no per-model subdirs here
                    flattened_info=idx_info,        # treat detected idx files as flattened_info
                    converted_pairs=converted_pairs,
                    disable_deletions=disable_deletions,
                    verbose=verbose,
                )
                # Clean stragglers
                for f in job_root.glob("*_data.json"):
                    safe_remove(f, verbose=verbose)
                remove_empty_samples_dir(job_root, verbose=verbose)
                log(f"[{index+1}] DONE (robust: idx-root with empty samples): {job_root}")
                return
            else:
                log("ROBUST: No *_idx_* files at root; cannot proceed via robust path.")

        # Normal new-layout path (unchanged)
        converted_pairs = convert_cifs_to_pdbs(
            job_root, cif_sif=cif_sif, verbose=verbose, include_nonindexed=include_nonindexed_cifs
        )
        log(f"Converted {len(converted_pairs)} CIF → PDB in {job_root.name}")
        cleanup_after_verification(
            job_root=job_root,
            kept_model_dirs=[],
            flattened_info=flattened_info,
            converted_pairs=converted_pairs,
            disable_deletions=disable_deletions,
            verbose=verbose,
        )
        for f in job_root.glob("*_data.json"):
            safe_remove(f, verbose=verbose)
        remove_empty_samples_dir(job_root, verbose=verbose)
        log(f"[{index+1}] DONE (new layout): {job_root}")
        return  # <-- IMPORTANT

    # --- Step 2: find per-model subdirs ---
    model_dirs = find_model_subdirs(job_root)
    if not model_dirs:
        if robust_mode:
            log(f"ROBUST: No model subdirs found in {job_root}. Checking for *_idx_* files at root...")
            idx_info = detect_existing_idx_triplets(job_root, verbose=verbose)
            if idx_info:
                converted_pairs = convert_cifs_to_pdbs(
                    job_root, cif_sif=cif_sif, verbose=verbose, include_nonindexed=include_nonindexed_cifs
                )
                log(f"Converted {len(converted_pairs)} CIF → PDB in {job_root.name} (robust idx-root path)")
                cleanup_after_verification(
                    job_root=job_root,
                    kept_model_dirs=[],      # nothing to remove besides stray files
                    flattened_info=idx_info, # drives whitelist for cleanup
                    converted_pairs=converted_pairs,
                    disable_deletions=disable_deletions,
                    verbose=verbose,
                )
                for f in job_root.glob("*_data.json"):
                    safe_remove(f, verbose=verbose)
                remove_empty_samples_dir(job_root, verbose=verbose)
                log(f"[{index+1}] DONE (robust: idx-root, no model subdirs): {job_root}")
                return
        # Original warning if robust path not available
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

def worker(queue, cif_sif: str, disable_deletions: bool, verbose: bool, include_nonindexed_cifs: bool, robust_mode: bool):
    while True:
        item = queue.get()
        if item is None:
            return
        idx, subdir = item
        try:
            process_one(idx, Path(subdir), cif_sif, disable_deletions, verbose, include_nonindexed_cifs, robust_mode)
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
    parser.add_argument("--include-nonindexed-cifs", action="store_true",
        help="If set, convert non-indexed CIF files in addition to indexed *_idx_<N>_model.cif.")
    parser.add_argument("--robust-mode", action="store_true",
                    help="Proceed if *_idx_<N>_* files exist at job root even when samples/ is empty or model subdirs are missing.")

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
            p = multiprocessing.Process(target=worker, args=(q, args.cif_sif, args.disable_deletions, args.verbose, args.include_nonindexed_cifs, args.robust_mode))
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
        # single-subdir mode
        process_one(
            0,
            Path(args.af3_subdir),
            args.cif_sif,
            args.disable_deletions,
            args.verbose,
            args.include_nonindexed_cifs,
            args.robust_mode,
        )


if __name__ == "__main__":
    main()
