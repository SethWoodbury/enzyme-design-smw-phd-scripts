#!/usr/bin/env python3
"""
SCRIPT NAME:
    setup_special_scripts.py

PURPOSE:
    This script scans through all files in the `special_scripts` directory (including subdirectories)
    and performs two independent, text-only replacements:

    (A) Base directory replacement (enabled by default):
        Replace every instance of a target path string (default):
            /home/woodbuse/special_scripts/
        with a new base directory string:
            1) By default — the directory where THIS script resides.
            2) Optionally — a custom directory path via --new_path.
        Notes: Trailing slashes are normalized so the replacement ends with exactly one "/".
               You can override the target string via --old_path.

    (B) Open Babel executable replacement (optional; only if --obabel_replacement_path is given):
        Replace every instance of an Open Babel executable path (default):
            /home/woodbuse/conda_envs/openbabel_env/bin/obabel
        with a custom executable path via --obabel_replacement_path.
        Notes: You can override the target string to search for via --obabel_path_to_replace.
               No trailing slash normalization is applied to executable paths.

USAGE:
    Default behavior (A only; replace default OLD_PATH with this script's directory):
        python /path/to/special_scripts_dir/replace_special_scripts_path.py

    Specify a custom replacement path for (A):
        python replace_special_scripts_path.py --new_path /home/user/custom_dir

    Specify a custom old base path to search for in (A):
        python replace_special_scripts_path.py --old_path /old/location/to/replace/

    Enable (B) and replace obabel path:
        python replace_special_scripts_path.py --obabel_replacement_path /usr/local/bin/obabel

    Customize the obabel path to search for in (B):
        python replace_special_scripts_path.py \
            --obabel_replacement_path /opt/conda/bin/obabel \
            --obabel_path_to_replace /some/other/obabel

NOTES:
    - Only modifies text files. Binary files are skipped.
    - Prints a summary of how many files were changed and how many replacements were made (for A and B).
"""

import os
import argparse

### ARGUMENT PARSER ###
parser = argparse.ArgumentParser(
    description="Replace a target base path and (optionally) an obabel executable path in all special_scripts files."
)
# (A) Base path replacement controls
parser.add_argument(
    "--new_path",
    type=str,
    help="Optional: custom directory path to replace with for the base path (A). Trailing slash handled automatically."
)
parser.add_argument(
    "--old_path",
    type=str,
    default="/home/woodbuse/special_scripts/",
    help="Optional: base path string to search for in (A). Default: /home/woodbuse/special_scripts/"
)
parser.add_argument(
    "--use_mnt",
    action="store_true",
    help="Optional: keep the /mnt prefix in the resolved script directory. By default /mnt is stripped."
)
# (B) Open Babel replacement controls (only active if --obabel_replacement_path is provided)
parser.add_argument(
    "--obabel_replacement_path",
    type=str,
    help="Optional: if provided, enable (B) and replace obabel executable path with this value."
)
parser.add_argument(
    "--obabel_path_to_replace",
    type=str,
    default="/home/woodbuse/conda_envs/openbabel_env/bin/obabel",
    help="Optional: executable path string to search for in (B). Default: /home/woodbuse/conda_envs/openbabel_env/bin/obabel"
)
args = parser.parse_args()

### CONSTANTS & NORMALIZATION ###
# (A) Normalize base path strings to have exactly one trailing slash.
OLD_PATH = args.old_path.rstrip("/") + "/"

script_dir = os.path.dirname(os.path.abspath(__file__))
if args.new_path:
    replacement_path = args.new_path.rstrip("/") + "/"
else:
    rpath = script_dir
    if not args.use_mnt and rpath.startswith("/mnt"):
        rpath = rpath[len("/mnt"):]
    replacement_path = rpath.rstrip("/") + "/"

# (B) Obabel paths: no trailing slash normalization (executable paths usually have none).
OBABEL_OLD = args.obabel_path_to_replace
OBABEL_NEW = args.obabel_replacement_path  # may be None if not provided

### INFO BANNER ###
print(f"[INFO] Searching for base path (A): {OLD_PATH}")
print(f"[INFO] Base path replacement set to: {replacement_path}")
if OBABEL_NEW:
    print(f"[INFO] Obabel replacement enabled (B)")
    print(f"[INFO]   Search obabel path: {OBABEL_OLD}")
    print(f"[INFO]   Replace with     : {OBABEL_NEW}")
else:
    print(f"[INFO] Obabel replacement (B) not enabled (pass --obabel_replacement_path to enable).")

### FIND THE ROOT special_scripts DIR ###
special_scripts_root = script_dir
if not os.path.exists(special_scripts_root):
    print(f"[ERROR] Could not find special_scripts directory at: {special_scripts_root}")
    exit(1)

### COUNTERS ###
files_scanned = 0
files_changed = 0
# Per-category counts
base_replacements_total = 0   # (A)
obabel_replacements_total = 0 # (B)

### UTILS ###
def replace_all(content: str, old: str, new: str) -> tuple[str, int]:
    """Return (new_content, count) after replacing occurrences of old with new."""
    if not old or old not in content:
        return content, 0
    count = content.count(old)
    return content.replace(old, new), count

### WALK THROUGH FILES ###
for root, dirs, files in os.walk(special_scripts_root):
    for fname in files:
        fpath = os.path.join(root, fname)
        files_scanned += 1

        # Read as text; skip binaries
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            continue  # skip binary-ish files
        except Exception as e:
            print(f"[WARN] Could not read file: {fpath} ({e})")
            continue

        # Perform replacements
        changed_here = False
        base_count = 0
        obabel_count = 0

        # (A) Base path replacement (always active)
        new_content, base_count = replace_all(content, OLD_PATH, replacement_path)

        # (B) Obabel replacement (only if OBABEL_NEW is provided)
        if OBABEL_NEW:
            newer_content, obabel_count = replace_all(new_content, OBABEL_OLD, OBABEL_NEW)
        else:
            newer_content, obabel_count = new_content, 0

        # If anything changed, write back
        if base_count > 0 or obabel_count > 0:
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(newer_content)
                changed_here = True
            except Exception as e:
                print(f"[WARN] Could not write file: {fpath} ({e})")
                continue

        if changed_here:
            files_changed += 1
            base_replacements_total += base_count
            obabel_replacements_total += obabel_count
            # File-level change log
            details = []
            if base_count:
                details.append(f"base:{base_count}")
            if obabel_count:
                details.append(f"obabel:{obabel_count}")
            print(f"[CHANGED] {fpath} — " + ", ".join(details))

### SUMMARY ###
print("\n========== SUMMARY ==========")
print(f"Files scanned:                {files_scanned}")
print(f"Files changed:                {files_changed}")
print(f"Base path replacements (A):   {base_replacements_total}")
print(f"Obabel replacements (B):      {obabel_replacements_total}")
print(f"Total replacements:           {base_replacements_total + obabel_replacements_total}")
print("=============================\n")

# Reconstruct run command echo
run_cmd = f"python {os.path.basename(__file__)}"
if args.old_path and args.old_path != "/home/woodbuse/special_scripts/":
    run_cmd += f" --old_path {args.old_path}"
if args.new_path:
    run_cmd += f" --new_path {args.new_path}"
if OBABEL_NEW:
    run_cmd += f" --obabel_replacement_path {OBABEL_NEW}"
    if args.obabel_path_to_replace != "/home/woodbuse/conda_envs/openbabel_env/bin/obabel":
        run_cmd += f" --obabel_path_to_replace {OBABEL_OLD}"
print(f"[DONE] Run command was: {run_cmd}")