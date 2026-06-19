#!/usr/bin/env python3
"""
SCRIPT: af2_parse_processing_multi.py

PURPOSE (CSV-only):
  Concatenate the per-prediction `.sc` scorefiles produced by
  `process_af2_pdb.py` (which writes one flat `<prediction>.sc` per AF2/superfold
  prediction) into a single summary CSV.

  This is the AF2 analogue of
  `advanced_structure_prediction_tools/concat_af3_sc_dir_of_subdirs.py`, but AF2
  output is FLAT (many `<pred>.sc` files in one directory) rather than one `.sc`
  per subdirectory — so discovery is a flat glob instead of a subdir walk. The
  rest of the engine (column union, bounded-memory shard streaming) is the same.

  Each `.sc` is a CSV: a header row followed by one (occasionally more) data
  row(s). NOTE: these are CSV files, NOT Rosetta whitespace `SCORE:` files.

DESIGNED FOR SCALE (10k-1M+ files):
  - Threaded discovery + threaded header-union with a bounded in-flight window
    (peak RAM is O(max_inflight + chunk_rows), flat regardless of file count).
  - Streaming parse -> temporary shard CSVs -> final streaming concat.
  - Union of all columns; missing values filled with "".
  - Multi-row `.sc` files are supported (every data row is emitted), so pointing
    this at a combined CSV does not silently drop rows.
  - DETERMINISTIC by default: files are processed in sorted order, so the output
    row order is stable run-to-run (good for diffs/joins). Use --threaded_unordered
    for the faster completion-order pipeline at extreme scale.

USAGE:
  python af2_parse_processing_multi.py \
      --af2_sc_dir /path/to/dir/of/.sc/files \
      [--optional_path_for_summary_stats /path/to/out.csv] \
      [--recursive] [--glob_pattern '*.sc'] \
      [--workers 64] [--chunk_rows 10000] [--max_inflight 2000] \
      [--threaded_unordered] [--find_files_without_viable_data]

OUTPUT:
  - If --optional_path_for_summary_stats is given: written there.
  - Else: zzzzz_af2_analysis_csv_zzzzz.csv inside --af2_sc_dir.
"""

import os
import csv
import glob
import time
import shutil
import fnmatch
import argparse
import tempfile
from typing import List, Dict, Tuple, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

# csv fields can be long (wide per-catres tables); lift the limit defensively.
try:
    csv.field_size_limit(2**27)
except Exception:
    pass

# Sentinel output names we must never read back in as inputs.
DEFAULT_OUTPUT_NAME = "zzzzz_af2_analysis_csv_zzzzz.csv"
SHARD_TMP_PREFIX = ".af2_concat_"


# -------------------------
# Utils: timing + memory
# -------------------------
def fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m {sec}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {sec}s"


def get_mem_used_mb() -> float:
    """Best-effort RSS in MB (Linux). Falls back gracefully."""
    try:
        import psutil  # optional
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        try:
            with open("/proc/self/statm", "r") as f:
                rss_pages = int(f.read().split()[1])
            return (rss_pages * os.sysconf("SC_PAGE_SIZE")) / (1024 ** 2)
        except Exception:
            return float("nan")


def bounded_imap_unordered(fn, items, workers, max_inflight):
    """Apply fn over items on a thread pool, keeping <= max_inflight futures live.
    Yields (item, future) in completion order; memory is O(max_inflight)."""
    it = iter(items)
    future_item = {}
    pending = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        def _submit_next() -> bool:
            try:
                item = next(it)
            except StopIteration:
                return False
            fut = ex.submit(fn, item)
            future_item[fut] = item
            pending.add(fut)
            return True

        for _ in range(max(1, max_inflight)):
            if not _submit_next():
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                item = future_item.pop(fut)
                _submit_next()
                yield item, fut


# -----------------------------------
# Flat discovery (one dir or recursive)
# -----------------------------------
def discover_sc_files_flat(root_dir: str, recursive: bool, pattern: str,
                           exclude_names: Iterable[str]) -> List[Tuple[str, str]]:
    """Find `.sc` (or --glob_pattern) files directly in root_dir (or recursively).

    Returns a PATH-SORTED list of (label, sc_path), where label = basename without
    the trailing '.sc'. The summary-output file and any temp shard dirs are excluded
    so a re-run never ingests its own output.
    """
    t0 = time.time()
    exclude = set(exclude_names)
    found: List[Tuple[str, str]] = []

    def _consider(dirpath: str, name: str):
        if name in exclude:
            return
        if not fnmatch.fnmatch(name, pattern):
            return
        label = name[:-3] if name.endswith(".sc") else os.path.splitext(name)[0]
        found.append((label, os.path.join(dirpath, name)))

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # never descend into our own temp shard dirs
            dirnames[:] = [d for d in dirnames if not d.startswith(SHARD_TMP_PREFIX)]
            for name in filenames:
                _consider(dirpath, name)
    else:
        with os.scandir(root_dir) as it:
            for e in it:
                if e.is_file():
                    _consider(root_dir, e.name)

    found.sort(key=lambda pair: pair[1])  # deterministic order
    print(f"[Discovery] {len(found)} '{pattern}' files under {root_dir} "
          f"({'recursive' if recursive else 'flat'}) in {fmt_secs(time.time()-t0)}.")
    return found


# ------------------------------------------------
# CSV helpers (header + one-or-more data rows)
# ------------------------------------------------
def read_csv_header(sc_path: str) -> List[str]:
    with open(sc_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.reader(f):
            if row and any(cell.strip() for cell in row):
                return [c.strip() for c in row]
    return []


def read_header_and_rows(sc_path: str) -> Tuple[List[str], List[List[str]]]:
    """Return (header, [data_row, ...]) — ALL non-empty data rows after the header.
    Normal per-prediction `.sc` files have exactly one; combined CSVs have many
    (we keep them all rather than silently dropping)."""
    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    with open(sc_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.reader(f):
            if not row or not any(cell.strip() for cell in row):
                continue
            cells = [c.strip() for c in row]
            if header is None:
                header = cells
            else:
                rows.append(cells)
    return header or [], rows


# ---------------------------------------
# Column ordering + row remap
# ---------------------------------------
def compute_union_from_headers(colset: set, first_col: str = "description") -> List[str]:
    """Order: first_col (or 'description') first, sorted middle, then sc_path last."""
    colset = set(colset)
    colset.add("sc_path")
    ordered: List[str] = []
    if first_col in colset:
        ordered.append(first_col)
        colset.remove(first_col)
    elif "description" in colset:
        ordered.append("description")
        colset.remove("description")
    colset.discard("sc_path")
    ordered += sorted(colset)
    ordered += ["sc_path"]
    return ordered


def union_columns(sc_files: List[Tuple[str, str]], workers: int,
                  first_col: str, max_inflight: int) -> List[str]:
    """Threaded, bounded header-union (order-independent)."""
    t0 = time.time()
    colset: set = set()
    total = len(sc_files)
    if total == 0:
        return ["sc_path"]
    done = 0
    for _item, fut in bounded_imap_unordered(
            lambda pair: read_csv_header(pair[1]), sc_files, workers, max_inflight):
        try:
            colset.update(fut.result())
        except Exception:
            pass
        done += 1
        if done in (1, 100, 1000, 10000) or (done >= 10000 and done % 10000 == 0):
            print(f"  [Header-Union] {done}/{total} | {fmt_secs(time.time()-t0)} "
                  f"| RSS ~{get_mem_used_mb():.0f} MB")
    ordered = compute_union_from_headers(colset, first_col=first_col)
    print(f"[Header-Union] {len(ordered)} columns in {fmt_secs(time.time()-t0)}.")
    return ordered


def remap_row_to_union(vals: List[str], header: List[str], union_cols: List[str],
                       sc_path: str) -> Dict[str, str]:
    row: Dict[str, str] = {}
    for i, col in enumerate(header):
        if col in union_cols and i < len(vals):
            row[col] = vals[i]
    for col in union_cols:
        row.setdefault(col, "")
    row["sc_path"] = sc_path
    return row


# -----------------------
# Shard writing / concat
# -----------------------
def shard_write(rows: List[Dict[str, str]], union_cols: List[str], idx: int, tmpdir: str) -> str:
    p = os.path.join(tmpdir, f"shard_{idx:06d}.csv")
    with open(p, "w", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=union_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  [Shard] {len(rows):>6} rows -> {p}")
    return p


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
    ap = argparse.ArgumentParser(
        description="Concatenate per-prediction AF2 .sc CSV files (from process_af2_pdb.py) into one CSV.")
    ap.add_argument("--af2_sc_dir", required=True,
                    help="Directory containing the flat per-prediction *.sc files.")
    ap.add_argument("--optional_path_for_summary_stats", default=None,
                    help=f"Output CSV path; default {DEFAULT_OUTPUT_NAME} inside --af2_sc_dir.")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories (os.walk) instead of a single flat scandir.")
    ap.add_argument("--glob_pattern", default="*.sc",
                    help="Filename pattern to include (default '*.sc').")
    ap.add_argument("--chunk_rows", type=int, default=10000, help="Rows per temp shard (default 10000).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Threads for header-union (default min(64, 2*CPU)).")
    ap.add_argument("--max_inflight", type=int, default=None,
                    help="Max files read concurrently in header-union (default max(4*workers, 2000)).")
    ap.add_argument("--threaded_unordered", action="store_true",
                    help="Use the faster threaded completion-order data pass (non-deterministic row order). "
                         "Default is the deterministic sorted serial pass.")
    ap.add_argument("--first_col", default="description",
                    help="Column placed first in the output (default 'description').")
    ap.add_argument("--find_files_without_viable_data", action="store_true",
                    help="Discovery-only: list .sc files with no usable data row, then exit.")
    args = ap.parse_args()

    root = os.path.abspath(args.af2_sc_dir.rstrip("/"))
    out_csv = args.optional_path_for_summary_stats or os.path.join(root, DEFAULT_OUTPUT_NAME)
    out_basename = os.path.basename(out_csv)

    workers = args.workers or min(64, max(4, 2 * (os.cpu_count() or 4)))
    max_inflight = args.max_inflight or max(4 * workers, 2000)

    print("############################################")
    print("###       AF2 .sc CONCAT (flat dir)      ###")
    print("############################################")
    print(f"Root           : {root}")
    print(f"Output         : {out_csv}")
    print(f"Pattern        : {args.glob_pattern} | recursive={args.recursive}")
    print(f"Workers        : {workers} | max_inflight {max_inflight} | chunk_rows {args.chunk_rows}")
    print(f"Data pass      : {'threaded (completion-order)' if args.threaded_unordered else 'serial (sorted, deterministic)'}")
    print("--------------------------------------------")
    t0 = time.time()

    sc_files = discover_sc_files_flat(
        root, recursive=args.recursive, pattern=args.glob_pattern,
        exclude_names={DEFAULT_OUTPUT_NAME, out_basename})

    if args.find_files_without_viable_data:
        bad = [p for _l, p in sc_files if not read_header_and_rows(p)[1]]
        print(f"\n[Discovery-only] {len(bad)} file(s) without a usable data row:")
        for p in bad:
            print(f"  [NoData] {p}")
        return

    if not sc_files:
        print("[Exit] No .sc files found. Nothing to do.")
        return

    # PASS 1 — column union (threaded, order-independent)
    print(f"\n[Pass 1] Column union over {len(sc_files)} files…")
    union_cols = union_columns(sc_files, workers=workers,
                               first_col=args.first_col, max_inflight=max_inflight)
    preview = ", ".join(union_cols[:min(12, len(union_cols))])
    print(f"[Pass 1] {len(union_cols)} cols. Preview: {preview}{' …' if len(union_cols) > 12 else ''}")

    # PASS 2 — read rows -> shards
    tmpdir = tempfile.mkdtemp(prefix=SHARD_TMP_PREFIX, dir=root)
    print(f"\n[Pass 2] Reading rows -> shards in {tmpdir}")
    chunk_rows = max(1, args.chunk_rows)
    buf: List[Dict[str, str]] = []
    shard_idx = total_rows = parsed = skipped = multirow = 0
    shard_paths: List[str] = []
    t2 = time.time()

    def _handle(label: str, scp: str, header: List[str], rows: List[List[str]]):
        nonlocal buf, shard_idx, total_rows, skipped, multirow
        if not header or not rows:
            print(f"  [Skip] No usable data: {scp}")
            skipped += 1
            return
        if len(rows) > 1:
            multirow += 1
        for data_row in rows:
            if len(data_row) < len(header):
                data_row = data_row + [""] * (len(header) - len(data_row))
            buf.append(remap_row_to_union(data_row, header, union_cols, scp))
            if len(buf) >= chunk_rows:
                shard_paths.append(shard_write(buf, union_cols, shard_idx, tmpdir))
                shard_idx += 1
                total_rows += len(buf)
                buf = []

    if args.threaded_unordered:
        for (label, scp), fut in bounded_imap_unordered(
                lambda pair: read_header_and_rows(pair[1]), sc_files, workers, max_inflight):
            try:
                header, rows = fut.result()
            except Exception as exc:
                print(f"  [Error] {scp}: {exc}")
                skipped += 1
                parsed += 1
                continue
            _handle(label, scp, header, rows)
            parsed += 1
            if parsed in (1, 1000, 10000) or (parsed >= 10000 and parsed % 10000 == 0):
                print(f"  [Pass 2] {parsed}/{len(sc_files)} | {fmt_secs(time.time()-t2)} "
                      f"| shards {shard_idx} | rows {total_rows} | RSS ~{get_mem_used_mb():.0f} MB")
    else:
        for label, scp in sc_files:  # sorted order -> deterministic output
            try:
                header, rows = read_header_and_rows(scp)
            except Exception as exc:
                print(f"  [Error] {scp}: {exc}")
                skipped += 1
                parsed += 1
                continue
            _handle(label, scp, header, rows)
            parsed += 1
            if parsed in (1, 1000, 10000) or (parsed >= 10000 and parsed % 10000 == 0):
                print(f"  [Pass 2] {parsed}/{len(sc_files)} | {fmt_secs(time.time()-t2)} "
                      f"| shards {shard_idx} | rows {total_rows} | RSS ~{get_mem_used_mb():.0f} MB")

    if buf:
        shard_paths.append(shard_write(buf, union_cols, shard_idx, tmpdir))
        shard_idx += 1
        total_rows += len(buf)
        buf = []

    # FINAL — concat shards (shard order = sorted/serial = deterministic by default)
    print(f"\n[Final] Concatenating {len(shard_paths)} shard(s)…")
    stream_concat_csvs(sorted(shard_paths), out_csv)
    try:
        shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"[Temp] Could not remove {tmpdir}: {e}")

    # summary
    n_cols = n_rows = 0
    try:
        with open(out_csv, "r", newline="") as f:
            hl = f.readline()
            n_cols = hl.count(",") + 1 if hl.strip() else 0
            n_rows = sum(1 for ln in f if ln.strip())
    except Exception:
        n_rows, n_cols = total_rows, len(union_cols)

    print("\n============================================")
    print("                  SUMMARY")
    print("============================================")
    print(f"Files discovered : {len(sc_files)}")
    print(f"Files parsed     : {parsed} (skipped {skipped}, multi-row {multirow})")
    print(f"Rows in output   : {n_rows}")
    print(f"Columns          : {n_cols}")
    print(f"Output CSV       : {out_csv}")
    print(f"Total elapsed    : {fmt_secs(time.time()-t0)}")
    print("============================================")


if __name__ == "__main__":
    main()
