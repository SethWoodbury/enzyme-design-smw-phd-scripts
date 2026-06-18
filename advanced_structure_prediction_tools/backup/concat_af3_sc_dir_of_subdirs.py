#!/usr/bin/env python3
"""
SCRIPT: concat_af3_sc_dir_of_subdirs.py

PURPOSE (CSV-only):
  Given --af3_dir_of_subdirs (a directory containing many subdirectories),
  go 1-level into each subdirectory to find its .sc file (CSV: header + data),
  and concatenate all of them into a single CSV.

  Priority per subdir "<d>":
    1) <d>/<d>.sc
    2) <d>/*.sc (first match)
    3) If none, skip with a note.

DESIGNED FOR MASSIVE SCALE (~100k+ subdirs):
  - Threaded discovery with ETA.
  - Threaded header union (to form global column set) with ETA.
  - Streaming parse → temporary shard CSVs (default 10k rows per shard).
  - Final streaming concat (no big RAM use).
  - Keeps union of all columns; fills missing values with empty string ("").
  - Verbose logging: counts, elapsed, ETA, memory, avg/file, shard details.

USAGE:
  python concat_af3_sc_dir_of_subdirs.py \
      --af3_dir_of_subdirs /path/to/AF3/iteration_dir \
      [--optional_path_for_summary_stats /path/to/out.csv] \
      [--chunk_rows 10000] \
      [--workers 64] \
      [--strict_name_match]

OUTPUT:
  - If --optional_path_for_summary_stats is provided: write there.
  - Else: write zzzzz_af3_analysis_csv_zzzzz.csv inside --af3_dir_of_subdirs.
"""

import os
import csv
import glob
import time
import shutil
import argparse
import tempfile
from typing import List, Dict, Tuple, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# -------------------------
# Utils: timing + memory
# -------------------------
def fmt_secs(s: float) -> str:
    if s < 60: return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60: return f"{m}m {sec}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {sec}s"

def get_mem_used_mb() -> float:
    """Best-effort memory usage in MB (Linux). Falls back gracefully."""
    try:
        import psutil  # optional
        return psutil.Process().memory_info().rss / (1024**2)
    except Exception:
        try:
            with open("/proc/self/statm", "r") as f:
                parts = f.read().strip().split()
                rss_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (rss_pages * page_size) / (1024**2)
        except Exception:
            return float("nan")

# -----------------------------------
# Fast discovery (threads + progress)
# -----------------------------------
def _find_sc_for_subdir(root_dir: str, entry: os.DirEntry, strict: bool = False) -> Optional[Tuple[str, str]]:
    """
    Prefer <subdir>/<subdir>.sc; if strict=False, fallback to first *.sc in subdir.
    """
    dname = entry.name
    dpath = entry.path
    preferred = os.path.join(dpath, f"{dname}.sc")
    if os.path.exists(preferred):
        return (dname, preferred)
    if strict:
        return None
    try:
        with os.scandir(dpath) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".sc"):
                    return (dname, e.path)
    except Exception:
        pass
    return None

def discover_sc_files_threaded(root_dir: str, workers: Optional[int] = None,
                               progress_every: int = 10000, strict: bool = False
                               ) -> Tuple[List[Tuple[str, str]], int, List[str]]:
    """
    Threaded discovery over immediate subdirectories with ETA prints.
    Returns: (list_of_pairs, total_subdirs, missing_subdir_paths)
    """
    t0 = time.time()
    subdir_entries: List[os.DirEntry] = []
    with os.scandir(root_dir) as it:
        for e in it:
            if e.is_dir():
                subdir_entries.append(e)
    total = len(subdir_entries)

    if workers is None:
        import multiprocessing
        workers = min(64, max(4, 2 * multiprocessing.cpu_count()))

    print(f"[Discovery] Scanning {total} subdirectories with {workers} workers…")
    found: List[Tuple[str, str]] = []
    missing: List[str] = []
    milestones = {1, 10, 100, 1000, 10000}
    parsed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fn = partial(_find_sc_for_subdir, root_dir, strict=strict)
        future_map = {}
        for e in subdir_entries:
            fut = ex.submit(fn, e)
            future_map[fut] = e
        for fut in as_completed(future_map):
            res = fut.result()
            entry = future_map[fut]
            if res is not None:
                found.append(res)
            else:
                missing.append(entry.path)
            parsed += 1
            if parsed in milestones or (parsed >= 10000 and parsed % progress_every == 0):
                elapsed = time.time() - t0
                rate = parsed / elapsed if elapsed > 0 else 0.0
                remaining = total - parsed
                eta = remaining / rate if rate > 0 else float("inf")
                mem_mb = get_mem_used_mb()
                print(f"  [Discovery] {parsed}/{total} | elapsed {fmt_secs(elapsed)} "
                      f"| ~{rate:.1f} subdirs/s | ETA {fmt_secs(eta)} | RSS ~{mem_mb:.1f} MB")

    elapsed = time.time() - t0
    print(f"[Discovery] Done: {len(found)} .sc files found, {len(missing)} missing "
          f"(from {total} subdirs) in {fmt_secs(elapsed)}.")
    return found, total, missing

# ------------------------------------------------
# CSV helpers (each file: header row + data row)
# ------------------------------------------------
def read_csv_header(sc_path: str) -> List[str]:
    with open(sc_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and any(cell.strip() for cell in row):
                return [c.strip() for c in row]
    return []

def read_csv_data_row(sc_path: str) -> Optional[List[str]]:
    """
    Return the FIRST non-empty row AFTER the header row; ignore extras.
    """
    with open(sc_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            if not header_seen:
                header_seen = True
                continue
            return [c.strip() for c in row]
    return None

# ---------------------------------------
# Union-of-columns (threaded + progress)
# ---------------------------------------
def union_columns_threaded(sc_files: List[Tuple[str, str]], workers: int) -> List[str]:
    """
    Build global union of columns by reading the header of each CSV in parallel.
    """
    t0 = time.time()
    colset = set()
    milestones = {1, 10, 100, 1000, 10000}
    done = 0
    total = len(sc_files)
    if total == 0:
        return ["subdir", "sc_path"]

    print(f"[Header-Union] Parsing headers from {total} files with {workers} workers…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(read_csv_header, scp) for _, scp in sc_files]
        for fut in as_completed(futures):
            cols = fut.result()
            colset.update(cols)
            done += 1
            if done in milestones or (done >= 10000 and done % 10000 == 0):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else float("inf")
                mem_mb = get_mem_used_mb()
                print(f"  [Header-Union] {done}/{total} | elapsed {fmt_secs(elapsed)} "
                      f"| ~{rate:.1f} files/s | ETA {fmt_secs(eta)} | RSS ~{mem_mb:.1f} MB")

    # Add provenance columns last; sort others (keep 'description' first if present)
    colset.update(["subdir", "sc_path"])
    ordered: List[str] = []
    if "description" in colset:
        ordered.append("description")
        colset.remove("description")
    colset.discard("subdir")
    colset.discard("sc_path")
    ordered += sorted(colset)
    ordered += ["subdir", "sc_path"]

    print(f"[Header-Union] Done in {fmt_secs(time.time()-t0)}. Global union columns = {len(ordered)}.")
    return ordered

# -----------------------
# Shard writing / concat
# -----------------------
def remap_row_to_union(vals: List[str], header: List[str], union_cols: List[str],
                       subdir: str, sc_path: str) -> Dict[str, str]:
    """
    Map a CSV data row to the global union schema; fill missing with "".
    """
    row: Dict[str, str] = {}
    # Map overlapping header -> values
    for i, col in enumerate(header):
        if i < len(vals) and col in union_cols:
            row[col] = vals[i]
    # Fill missing
    for col in union_cols:
        if col not in row:
            row[col] = ""
    # Provenance
    row["subdir"] = subdir
    row["sc_path"] = sc_path
    return row

def shard_write(rows: List[Dict[str, str]], union_cols: List[str], shard_idx: int, tmpdir: str) -> str:
    shard_path = os.path.join(tmpdir, f"shard_{shard_idx:06d}.csv")
    with open(shard_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=union_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [Shard] Wrote {len(rows):>6} rows → {shard_path}")
    return shard_path

def stream_concat_csvs(shard_paths: List[str], out_csv: str):
    if not shard_paths:
        open(out_csv, "w").close()
        return
    with open(out_csv, "w", newline="") as out_f:
        with open(shard_paths[0], "r", newline="") as first:
            shutil.copyfileobj(first, out_f)  # header + data
        for p in shard_paths[1:]:
            with open(p, "r", newline="") as f:
                next(f, None)  # drop header
                shutil.copyfileobj(f, out_f)

# -------------
# Main driver
# -------------
def main():
    parser = argparse.ArgumentParser(description="Concatenate AF3 .sc CSV files (one per subdir) into a single CSV.")
    parser.add_argument("--af3_dir_of_subdirs", required=True,
                        help="Directory whose immediate subdirectories each contain a .sc (CSV: header + data).")
    parser.add_argument("--optional_path_for_summary_stats", default=None,
                        help="Optional final output CSV path; otherwise writes zzzzz_af3_analysis_csv_zzzzz.csv in the root dir.")
    parser.add_argument("--chunk_rows", type=int, default=10000,
                        help="Rows per temporary shard CSV (default 10,000).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Threads for discovery & header scan (default ≈ min(64, 2*CPU)).")
    parser.add_argument("--strict_name_match", action="store_true",
                        help="Only accept <subdir>/<subdir>.sc; skip fallback *.sc scan for speed.")
    parser.add_argument("--find_subdirs_without_viable_sc", action="store_true", help="Only perform discovery and print subdirectories lacking a usable .sc, then exit.")
    args = parser.parse_args()

    root = os.path.abspath(args.af3_dir_of_subdirs.rstrip("/"))
    out_csv = args.optional_path_for_summary_stats or os.path.join(root, "zzzzz_af3_analysis_csv_zzzzz.csv")

    print("############################################")
    print("### AF3 .sc CONCAT — MASSIVE SCALE MODE  ###")
    print("############################################")
    print(f"Root directory        : {root}")
    print(f"Output path           : {out_csv}")
    try:
        import multiprocessing
        print(f"CPU cores (available) : {multiprocessing.cpu_count()}")
    except Exception:
        pass
    print(f"Workers (discovery)   : {args.workers or 'auto'}")
    print(f"Strict name match     : {args.strict_name_match}")
    print(f"Rows per shard        : {args.chunk_rows}")
    print("--------------------------------------------")

    t0 = time.time()

    # PASS 0: Discover candidate .sc files (parallel + ETA)
    workers = args.workers or None
    sc_files, total_subdirs, missing = discover_sc_files_threaded(root, workers=workers, strict=args.strict_name_match)
    print(f"Discovered {total_subdirs} subdirectories.")
    print(f"Found {len(sc_files)} usable .sc files (preferred '<d>.sc' else first '*.sc').")
    print(f"Missing .sc in {len(missing)} subdir(s). Listing them below:")
    for p in missing:
        print(f"  [Missing] {p}")

    # If the user only wants the missing-list, exit early.
    if args.find_subdirs_without_viable_sc:
        print("\n[Exit-by-flag] Completed discovery-only run (--find_subdirs_without_viable_sc).")
        return

    if not sc_files:
        print("[Exit] No .sc files found. Nothing to do.")
        return

    # PASS 1: Compute union-of-columns (header-only, parallel + ETA)
    print("\n[Pass 1] Building global column union from headers…")
    if workers is None:
        import multiprocessing
        workers = min(64, max(4, 2 * multiprocessing.cpu_count()))
    union_cols = union_columns_threaded(sc_files, workers=workers)
    print(f"[Pass 1] Global union columns: {len(union_cols)}")
    if union_cols:
        preview = ", ".join(union_cols[:min(12, len(union_cols))])
        print(f"[Pass 1] Column preview: {preview}{' …' if len(union_cols) > 12 else ''}")

    # Prepare temp shard dir
    tmpdir = tempfile.mkdtemp(prefix=".af3_concat_", dir=root)
    print(f"\n[Temp] Will write shard CSVs under: {tmpdir}")

    # PASS 2: Parse all files → write shard CSVs
    milestones = {1, 10, 100, 1000, 10000}
    def is_milestone(n: int) -> bool:
        return (n in milestones) or (n >= 10000 and n % 10000 == 0)

    parsed_files = 0
    total_rows  = 0
    shard_idx   = 0
    rows_buffer: List[Dict[str, str]] = []
    chunk_rows  = max(1, args.chunk_rows)

    print("\n[Pass 2] Parsing data rows and writing shard CSVs…")
    t2 = time.time()
    for subdir, scp in sc_files:
        header = read_csv_header(scp)
        data_row = read_csv_data_row(scp)
        if not header or data_row is None:
            # Malformed or empty; skip but log
            print(f"  [Skip] No usable data in: {scp}")
            parsed_files += 1
            continue

        # Normalize length mismatch (pad with empty if data shorter)
        if len(data_row) < len(header):
            data_row = data_row + [""] * (len(header) - len(data_row))

        row = remap_row_to_union(data_row, header, union_cols, subdir, scp)
        rows_buffer.append(row)

        if len(rows_buffer) >= chunk_rows:
            shard_write(rows_buffer, union_cols, shard_idx, tmpdir)
            shard_idx += 1
            total_rows += len(rows_buffer)
            rows_buffer.clear()

        parsed_files += 1
        if is_milestone(parsed_files):
            elapsed = time.time() - t2
            avg_per_file = elapsed / max(1, parsed_files)
            mem_mb = get_mem_used_mb()
            print(f"  [Pass 2] Parsed {parsed_files} files | Elapsed: {fmt_secs(elapsed)} "
                  f"| Avg/file: {avg_per_file:.4f}s | Shards: {shard_idx} "
                  f"| Rows (written so far): {total_rows} | RSS ~{mem_mb:.1f} MB")

    # Flush any remaining rows
    if rows_buffer:
        shard_write(rows_buffer, union_cols, shard_idx, tmpdir)
        shard_idx += 1
        total_rows += len(rows_buffer)
        rows_buffer.clear()

    # FINAL: Concatenate shards
    print("\n[Final] Concatenating shard CSVs into final output…")
    shard_paths = sorted(glob.glob(os.path.join(tmpdir, "shard_*.csv")))
    print(f"[Final] {len(shard_paths)} shard(s) to merge.")
    t_concat0 = time.time()
    stream_concat_csvs(shard_paths, out_csv)
    print(f"[Final] Concatenation done in {fmt_secs(time.time() - t_concat0)}.")

    # Clean up shards
    try:
        shutil.rmtree(tmpdir)
        print(f"[Temp] Removed temporary shard directory: {tmpdir}")
    except Exception as e:
        print(f"[Temp] Could not remove {tmpdir}: {e}")

    # SUMMARY
    elapsed = time.time() - t0
    # Count final rows/cols quickly by peeking header + counting lines
    n_cols = 0
    n_rows = 0
    try:
        with open(out_csv, "r", newline="") as f:
            reader = csv.reader(f)
            header_line = next(reader, [])
            n_cols = len(header_line)
            for n_rows, _ in enumerate(f, start=0):
                pass  # counts data lines after the first newline from csv reader
    except Exception:
        n_rows = total_rows
        n_cols = len(union_cols)

    print("\n============================================")
    print("                  SUMMARY")
    print("============================================")
    print(f"Total subdirectories     : {total_rows if total_rows > 0 else len(sc_files)} (rows ~ files parsed)")
    print(f"Total .sc files parsed   : {parsed_files}")
    print(f"Total rows in output     : {n_rows}")
    print(f"Total columns in output  : {n_cols}")
    print(f"Output CSV               : {out_csv}")
    print(f"Total elapsed            : {fmt_secs(elapsed)}")
    try:
        import multiprocessing
        print(f"CPU cores (available)    : {multiprocessing.cpu_count()}")
    except Exception:
        pass
    print("============================================")

if __name__ == "__main__":
    main()
