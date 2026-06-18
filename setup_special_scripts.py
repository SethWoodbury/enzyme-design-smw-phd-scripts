#!/usr/bin/env python3
"""
setup_special_scripts.py  —  DEPRECATED (kept only as a friendly pointer).

This script used to find-and-replace a hardcoded base path
(``/home/woodbuse/special_scripts/``) and the Open Babel executable path
throughout every file in this repository. That approach is no longer used —
and is actively harmful for a shared git repo, because rewriting tracked files
makes every clone show spurious local modifications and causes merge conflicts.

HOW PATHS WORK NOW (no setup step required)
-------------------------------------------
* INTERNAL paths (other scripts / data inside this repo) resolve themselves at
  runtime from each script's own location via ``pathlib.Path(__file__)``. Nothing
  to edit when you clone the repo or run it as a different user.

* EXTERNAL paths (shared cluster software, containers, model weights, databases,
  Open Babel, ...) live as named constants in ``repo_paths.py`` at the repo root.
  On the IPD cluster the defaults work as-is. To point one elsewhere, set an
  environment variable of the same name, e.g.::

      export OBABEL=/path/to/obabel
      export UNIVERSAL_SIF=/path/to/universal.sif

Running this file just prints the resolved configuration (same as
``python repo_paths.py``) so you can confirm what a fresh checkout will use.
"""

import os
import sys
from pathlib import Path

# Make repo_paths importable regardless of where this is run from.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

try:
    import repo_paths  # noqa: E402
except Exception as exc:  # pragma: no cover
    print(f"[setup_special_scripts] could not import repo_paths.py: {exc}")
    sys.exit(1)

print(__doc__)
print("=" * 70)
print("Resolved configuration (defaults unless overridden by an env var):\n")
print(f"REPO_ROOT = {repo_paths.REPO_ROOT}")
for _k, _v in sorted(vars(repo_paths).items()):
    if _k.isupper() and isinstance(_v, str):
        _src = "env" if _k in os.environ else "default"
        print(f"  {_k:24s} = {_v}   [{_src}]")
print("\nNothing was modified. (This script no longer edits any files.)")
