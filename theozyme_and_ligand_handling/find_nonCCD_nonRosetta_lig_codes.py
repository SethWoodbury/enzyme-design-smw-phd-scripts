#!/usr/bin/env python3
"""
find_nonCCD_nonRosetta_lig_codes.py

Fusion of:
  - find_nonCCD_lig_codes.py                       (parallel CCD search)
  - check_if_ligand_3string_code_exists_in_rosetta.py  (Rosetta scan)

Two mutually-exclusive modes:

  SEARCH mode  (--class NAME:PATTERN ...):
      Find N "available" 3-char codes per class, where a code is available
      iff it is absent from every selected database.

  CHECK mode   (--code CODE [CODE ...]):
      Report, per code, whether it exists in CCD and/or Rosetta, plus the
      matching Rosetta lines. Exit 0 unless a hard error occurs.

Database selection: --check {ccd,rosetta,both}  (default: both)

Efficiency: the Rosetta residue_types.txt is read ONCE into memory and
scanned locally (free). The CCD check is an HTTP request. Each code is
Rosetta-checked first; the CCD HTTP request is only made if still needed
(and skipped entirely for --check rosetta). Since the Rosetta residue set
is far smaller than the CCD, this ordering minimizes network traffic.

Examples
--------
  # SEARCH: 1000 codes matching S?? unused in BOTH CCD and Rosetta
  python find_nonCCD_nonRosetta_lig_codes.py \
      --class CLASS_0:S?? --target-per-class 1000 --max-workers 64

  # CHECK: is CO2 / ZN1 / ZZ9 already taken?
  python find_nonCCD_nonRosetta_lig_codes.py --code CO2 ZN1 ZZ9

  # CHECK against Rosetta only (no network needed)
  python find_nonCCD_nonRosetta_lig_codes.py --code CO2 --check rosetta
"""

import argparse
import itertools
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

BASE_URL = "https://files.rcsb.org/ligands/view/{code}.cif"
DEFAULT_ROSETTA_TXT = repo_paths.ROSETTA_RESIDUE_TYPES


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Search for / check 3-letter ligand codes not present in "
                    "the PDB CCD and/or the Rosetta residue set."
    )
    # exactly one of these two:
    p.add_argument(
        "--class", dest="classes", action="append", default=None,
        help="SEARCH mode. NAME:PATTERN where PATTERN is 3 chars of A-Z/0-9 "
             "and '?' wildcards, e.g. CLASS_0:S?? . Repeatable.",
    )
    p.add_argument(
        "--code", dest="codes", nargs="+", default=None,
        help="CHECK mode. One or more explicit codes to test, e.g. CO2 ZN1.",
    )

    p.add_argument(
        "--check", choices=("ccd", "rosetta", "both"), default="both",
        help="Which database(s) to consult (default: both).",
    )
    p.add_argument(
        "--rosetta-txt", default=DEFAULT_ROSETTA_TXT,
        help=f"Path to Rosetta residue_types.txt (default: {DEFAULT_ROSETTA_TXT}).",
    )

    # search-only knobs (ignored in check mode)
    p.add_argument(
        "--letters", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        help="Characters used to expand '?' wildcards (default: A-Z0-9).",
    )
    p.add_argument(
        "--target-per-class", type=int, default=9,
        help="Available codes to find per class before stopping (default: 9).",
    )
    p.add_argument(
        "--max-workers", type=int, default=32,
        help="Max parallel HTTP workers (default: 32).",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="Candidate codes per batch per class (default: 64).",
    )
    p.add_argument(
        "--timeout", type=float, default=4.0,
        help="HTTP timeout in seconds per CCD query (default: 4.0).",
    )
    p.add_argument(
        "--sleep-between-classes", type=float, default=0.0,
        help="Optional sleep (s) between classes.",
    )
    p.add_argument(
        "--log-every", type=int, default=500,
        help="Progress message every N codes checked per class (default: 500).",
    )
    p.add_argument(
        "--no-verify-ssl", action="store_true",
        help="Disable SSL verification (not recommended).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Pattern / class helpers (from find_nonCCD_lig_codes.py)
# ---------------------------------------------------------------------------

def parse_class_spec(spec):
    """Parse NAME:PATTERN into (name, pattern_list with None for '?')."""
    if ":" not in spec:
        raise ValueError(f"Class spec '{spec}' must be of the form NAME:PATTERN")
    name, pattern = spec.split(":", 1)
    name = name.strip()
    pattern = pattern.strip().upper()
    if len(pattern) != 3:
        raise ValueError(
            f"Pattern '{pattern}' in '{spec}' must be exactly 3 characters.")
    for ch in pattern:
        if not (ch.isalnum() or ch == "?"):
            raise ValueError(
                f"Invalid character '{ch}' in pattern '{pattern}'. "
                "Use A-Z, 0-9, or '?' as wildcard.")
    return name, [None if c == "?" else c for c in pattern]


def codes_from_pattern(pattern, letters):
    """Expand a pattern like ['S', None, None] over `letters`."""
    slots = [letters if ch is None else ch for ch in pattern]
    for combo in itertools.product(*slots):
        yield "".join(combo)


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# CCD check (HTTP) — from find_nonCCD_lig_codes.py
# ---------------------------------------------------------------------------

def code_ccd_state(code, timeout=4.0, verify=True):
    """Return (state, url) where state is:
        "YES" -> exists in CCD (HTTP 200)
        "no"  -> not in CCD (HTTP 404)
        "ERR" -> request failed / unexpected status (treated conservatively
                 as 'exists', but reported explicitly so the user knows the
                 answer is uncertain)."""
    url = BASE_URL.format(code=code)
    try:
        resp = requests.get(url, timeout=timeout, verify=verify)
        if resp.status_code == 200:
            return "YES", url
        if resp.status_code == 404:
            return "no", url
        print(f"[WARN] Unexpected status {resp.status_code} for code {code}; "
              f"treating as 'exists' (uncertain).")
        return "ERR", url
    except requests.RequestException as e:
        print(f"[WARN] Request error for {code}: {e}; "
              f"treating as 'exists' (uncertain).")
        return "ERR", url


def code_in_ccd(code, timeout=4.0, verify=True):
    """True if `code` exists in the CCD, False if 404. On any other status /
    request error, conservatively return True (unchanged search behavior)."""
    state, _ = code_ccd_state(code, timeout, verify)
    return state != "no"


# ---------------------------------------------------------------------------
# Rosetta check (local scan) — ported from
# check_if_ligand_3string_code_exists_in_rosetta.py
# ---------------------------------------------------------------------------

def load_rosetta_lines(path):
    """Read the Rosetta residue_types.txt once. Raises FileNotFoundError."""
    with open(path, "r") as f:
        return f.read().splitlines()


def code_in_rosetta(code, rosetta_lines):
    """Scan pre-loaded Rosetta lines for `code`.

    Returns (found: bool, matches: list[str]) where each match is a
    "[category] Line N: ..." string, mirroring the original check script:
      [filename]   exact CODE.params
      [standalone] word-boundary CODE not part of .params
      [warning]    a .params line that contains CODE somewhere
    """
    standalone_pattern = re.compile(rf"\b{re.escape(code)}\b")
    filename_pattern = re.compile(rf"{re.escape(code)}\.params\b")
    matches = []
    for lineno, raw in enumerate(rosetta_lines, start=1):
        has_params = ".params" in raw
        exact_fn = bool(filename_pattern.search(raw))
        standalone = bool(standalone_pattern.search(raw)) and not exact_fn
        any_code = code in raw
        if exact_fn:
            matches.append(f"[filename]   Line {lineno}: {raw}")
        if standalone:
            matches.append(f"[standalone] Line {lineno}: {raw}")
        if has_params and any_code and not exact_fn:
            matches.append(f"[warning]    Line {lineno}: {raw}")
    return (len(matches) > 0, matches)


# ---------------------------------------------------------------------------
# SEARCH mode
# ---------------------------------------------------------------------------

def search_class(name, pattern, letters, target_per_class, max_workers,
                 batch_size, timeout, verify_ssl, log_every,
                 checks, rosetta_lines):
    """Find up to `target_per_class` codes absent from all selected DBs."""
    print(f"[INFO] Searching {name} pattern {pattern} "
          f"(checks={sorted(checks)})...")
    codes_iter = codes_from_pattern(pattern, letters)
    use_ccd = "ccd" in checks
    use_ros = "rosetta" in checks

    found = []
    checked = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch in batched(codes_iter, batch_size):
            # Cheap local Rosetta filter first; only HTTP the survivors.
            ccd_futures = {}
            for code in batch:
                if use_ros and code_in_rosetta(code, rosetta_lines)[0]:
                    checked += 1
                    continue  # taken by Rosetta -> not available, no HTTP
                if use_ccd:
                    ccd_futures[executor.submit(
                        code_in_ccd, code, timeout, verify_ssl)] = code
                else:
                    # rosetta-only mode and not in Rosetta -> available
                    checked += 1
                    found.append(code)
                    print(f"  -> {name}: found available code {code}")
                    if len(found) >= target_per_class:
                        print(f"[INFO] Done {name}: reached target "
                              f"{target_per_class} after {checked} checked.")
                        return found

            for future in as_completed(ccd_futures):
                code = ccd_futures[future]
                checked += 1
                if not future.result():  # not in CCD, and (already) not in Rosetta
                    found.append(code)
                    print(f"  -> {name}: found available code {code}")
                    if len(found) >= target_per_class:
                        print(f"[INFO] Done {name}: reached target "
                              f"{target_per_class} after {checked} checked.")
                        return found
                if log_every and checked % log_every == 0:
                    print(f"[INFO] {name}: checked {checked}, "
                          f"found {len(found)} available.")

    print(f"[INFO] Exhausted {name} without reaching target "
          f"({len(found)} found, {target_per_class} requested).")
    return found


# ---------------------------------------------------------------------------
# CHECK mode
# ---------------------------------------------------------------------------

def run_check(codes, checks, rosetta_lines, timeout, verify_ssl,
              rosetta_txt=None):
    use_ccd = "ccd" in checks
    use_ros = "rosetta" in checks

    print("=== CODE CHECK ===")
    print(f"[INFO] Checks: {', '.join(sorted(checks))}")
    if use_ros:
        print(f"[INFO] Rosetta txt: {rosetta_txt} "
              f"({len(rosetta_lines)} lines)")
    if use_ccd:
        print(f"[INFO] CCD endpoint: {BASE_URL.format(code='<CODE>')} "
              f"(timeout {timeout}s, ssl_verify={verify_ssl})")
    print(f"[INFO] Codes to check: {', '.join(codes)}")
    print("")
    available, in_use, uncertain = [], [], []
    for code in codes:
        ccd_state = "-"
        ros_state = "-"
        ros_matches = []
        ccd_url = None

        if use_ros:
            r_found, ros_matches = code_in_rosetta(code, rosetta_lines)
            ros_state = "YES" if r_found else "no"
        if use_ccd:
            ccd_state, ccd_url = code_ccd_state(code, timeout, verify_ssl)

        where = []
        if ccd_state == "YES":
            where.append("CCD")
        if ros_state == "YES":
            where.append("Rosetta")

        if where:
            verdict = "IN USE"
            tail = (f"  *** WARNING: '{code}' already exists "
                    f"(in {' + '.join(where)}) -- NOT recommended, "
                    f"choose a different code ***")
            in_use.append(code)
        elif ccd_state == "ERR":
            verdict = "UNCERTAIN"
            tail = (f"  *** WARNING: CCD lookup for '{code}' FAILED; "
                    f"membership unknown -- treat as IN USE / verify "
                    f"manually ***")
            uncertain.append(code)
        else:
            verdict = "AVAILABLE"
            tail = "  (free to use)"
            available.append(code)

        print(f"{code:<4s}  CCD={ccd_state:<3s}  ROSETTA={ros_state:<3s}  "
              f"-> {verdict}{tail}")
        if ccd_url is not None:
            print(f"    CCD: {ccd_url}")
        for m in ros_matches[:8]:
            print(f"    {m}")
        if len(ros_matches) > 8:
            print(f"    ... (+{len(ros_matches) - 8} more Rosetta lines)")

    print("\n=== SUMMARY ===")
    print(f"AVAILABLE ({len(available)}): "
          f"{', '.join(available) if available else '(none)'}")
    print(f"IN USE    ({len(in_use)}): "
          f"{', '.join(in_use) if in_use else '(none)'}")
    if uncertain:
        print(f"UNCERTAIN ({len(uncertain)}): {', '.join(uncertain)}  "
              f"(CCD lookup failed -- re-run or check manually)")
    if in_use or uncertain:
        print("\n[RECOMMENDATION] Do NOT reuse IN USE/UNCERTAIN codes; "
              "pick from AVAILABLE or run SEARCH mode (--class) to find "
              "free codes.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    if bool(args.classes) == bool(args.codes):
        print("[ERROR] Provide exactly one of --class (SEARCH mode) or "
              "--code (CHECK mode).", file=sys.stderr)
        sys.exit(2)

    checks = {"ccd", "rosetta"} if args.check == "both" else {args.check}

    rosetta_lines = []
    if "rosetta" in checks:
        try:
            rosetta_lines = load_rosetta_lines(args.rosetta_txt)
        except FileNotFoundError:
            print(f"[ERROR] Rosetta file not found: {args.rosetta_txt}\n"
                  f"        Pass --rosetta-txt PATH, or use --check ccd to "
                  f"skip the Rosetta check.", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            print(f"[ERROR] Could not read {args.rosetta_txt}: {e}",
                  file=sys.stderr)
            sys.exit(3)

    verify_ssl = not args.no_verify_ssl

    # ---------------- CHECK mode ----------------
    if args.codes:
        codes = [c.strip().upper() for c in args.codes]
        run_check(codes, checks, rosetta_lines, args.timeout, verify_ssl,
                  rosetta_txt=args.rosetta_txt)
        sys.exit(0)

    # ---------------- SEARCH mode ----------------
    try:
        class_specs = [parse_class_spec(s) for s in args.classes]
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    letters = args.letters.upper()
    if not letters or any(not ch.isalnum() for ch in letters):
        print("[ERROR] --letters must be non-empty and only A-Z/0-9.",
              file=sys.stderr)
        sys.exit(1)

    print("[INFO] Configuration:")
    print(f"  Mode: SEARCH   Checks: {sorted(checks)}")
    print(f"  Classes: {[n for n, _ in class_specs]}")
    print(f"  Letters for '?': {letters}")
    print(f"  Target per class: {args.target_per_class}")
    print(f"  Max workers: {args.max_workers}  Batch: {args.batch_size}")
    print(f"  Timeout: {args.timeout}s  SSL verify: {verify_ssl}")
    if "rosetta" in checks:
        print(f"  Rosetta txt: {args.rosetta_txt} "
              f"({len(rosetta_lines)} lines)")
    print("")

    results = {}
    for i, (name, pattern) in enumerate(class_specs, start=1):
        results[name] = search_class(
            name=name, pattern=pattern, letters=letters,
            target_per_class=args.target_per_class,
            max_workers=args.max_workers, batch_size=args.batch_size,
            timeout=args.timeout, verify_ssl=verify_ssl,
            log_every=args.log_every, checks=checks,
            rosetta_lines=rosetta_lines,
        )
        if i < len(class_specs) and args.sleep_between_classes > 0:
            time.sleep(args.sleep_between_classes)

    print("\n=== SUMMARY OF SUGGESTED CODES ===")
    for name, codes in results.items():
        print(f"{name}: {', '.join(codes) if codes else '(none found)'}")


if __name__ == "__main__":
    main()
