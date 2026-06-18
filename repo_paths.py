"""
repo_paths.py — single source of truth for paths used across these scripts.

WHY THIS EXISTS
---------------
Scripts here are run from the CLI and submitted to SLURM on the IPD / Baker Lab
cluster. They reference two kinds of path:

  1. INTERNAL paths — other scripts / data that live *inside this repository*.
     These are resolved at runtime from each script's own location
     (``pathlib.Path(__file__).resolve().parent``), so nothing needs editing
     when the repo is cloned to a new location or by a different user.

  2. EXTERNAL paths — shared cluster software, containers, model weights, and
     databases that live *outside this repository*. Those are collected here as
     named constants. On the IPD cluster the defaults below "just work". Off
     the cluster (or if a tool moves), override any of them with an environment
     variable of the same name — e.g.  ``export UNIVERSAL_SIF=/my/universal.sif``.

The defaults are the exact paths the scripts used historically, so importing a
constant instead of hard-coding the string changes nothing on the cluster.

HOW SCRIPTS USE IT
------------------
Because scripts are launched from many sub-directories, add this small bootstrap
near the top of a script that needs an EXTERNAL path, then use the constants:

    # --- locate repo root and import shared external paths ---
    import sys as _sys
    from pathlib import Path as _Path
    for _anc in _Path(__file__).resolve().parents:
        if (_anc / "repo_paths.py").is_file():
            _sys.path.insert(0, str(_anc)); break
    import repo_paths

    subprocess.run([repo_paths.CRISPY_SIF, "python", ...])

INTERNAL (in-repo) paths do NOT need this module — use ``__file__`` directly:

    from pathlib import Path
    HERE = Path(__file__).resolve().parent
    STEP1 = HERE / "contact_counter__STEP1_calculate_contacts.py"
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# =============================================================================
#  Repository root (auto-detected — this file lives at the repo root)
# =============================================================================
REPO_ROOT: Path = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    """Return ``$name`` if set in the environment, else ``default``."""
    return os.environ.get(name, default)


def find_repo_root(start: str | os.PathLike) -> Path:
    """Walk up from ``start`` (usually ``__file__``) to the repo root.

    Useful if a script is symlinked or copied elsewhere but still wants the
    repo it belongs to. Falls back to this file's directory.
    """
    p = Path(start).resolve()
    for anc in [p, *p.parents]:
        if (anc / "repo_paths.py").is_file():
            return anc
    return REPO_ROOT


# =============================================================================
#  Cluster software roots (IPD).  NOTE: the old /software -> /net/software symlink
#  has been REMOVED, so every software path uses /net/software directly. (Some
#  scripts historically wrote /software/...; those now route through these
#  constants, so this is the one place to change if the layout moves again.)
# =============================================================================
NET_SOFTWARE = _env("NET_SOFTWARE", "/net/software")
SOFTWARE     = _env("SOFTWARE",     NET_SOFTWARE)         # back-compat alias; /software symlink is gone
DATABASES    = _env("DATABASES",    "/databases")
LAB_SCRIPTS  = _env("LAB_SCRIPTS",  f"{NET_SOFTWARE}/lab/scripts/enzyme_design")

# =============================================================================
#  Apptainer / Singularity containers (*.sif)
# =============================================================================
CONTAINER_DIR = _env("CONTAINER_DIR", f"{NET_SOFTWARE}/containers")
UNIVERSAL_SIF = _env("UNIVERSAL_SIF", f"{CONTAINER_DIR}/universal.sif")
CRISPY_SIF    = _env("CRISPY_SIF",    f"{CONTAINER_DIR}/crispy.sif")
PYROSETTA_SIF = _env("PYROSETTA_SIF", f"{CONTAINER_DIR}/pyrosetta.sif")
MLFOLD_SIF    = _env("MLFOLD_SIF",    f"{CONTAINER_DIR}/mlfold.sif")
MAXIT_SIF     = _env("MAXIT_SIF",     f"{NET_SOFTWARE}/containers/users/sklein89/maxit.sif")
ESMC_SIF      = _env("ESMC_SIF",      "/net/software/containers/users/woodbuse/esmc.sif")

# =============================================================================
#  MPNN: fused_mpnn engine (external dependency, NOT vendored) + model weights
# =============================================================================
# Shared cluster copy of fused_mpnn_api (maintained in the lab software tree).
FUSED_MPNN_DIR = _env("FUSED_MPNN_DIR", f"{LAB_SCRIPTS}/fused_mpnn_api")
# Lab fused_mpnn install root and Seth's working run.py (used by some pipelines).
FUSED_MPNN_ROOT = _env("FUSED_MPNN_ROOT", "/net/software/lab/fused_mpnn")
FUSED_MPNN_RUN = _env("FUSED_MPNN_RUN", f"{FUSED_MPNN_ROOT}/seth_temp/run.py")
# ProteinMPNN / LigandMPNN model-weight directory and notable weight files.
MPNN_WEIGHTS = _env("MPNN_WEIGHTS", f"{DATABASES}/mpnn")
LIGANDMPNN_WEIGHTS = _env("LIGANDMPNN_WEIGHTS", f"{MPNN_WEIGHTS}/ligand_mpnn_model_weights")
VANILLA_MPNN_WEIGHTS = _env("VANILLA_MPNN_WEIGHTS", f"{MPNN_WEIGHTS}/vanilla_model_weights")
SOLUBLE_MPNN_MODELS = _env("SOLUBLE_MPNN_MODELS", "/projects/ml/struc2seq")
ESMFOLD_DB = _env("ESMFOLD_DB", f"{DATABASES}/esmfold")

# =============================================================================
#  Rosetta / PyRosetta
# =============================================================================
ROSETTA           = _env("ROSETTA",           "/net/software/rosetta/main")
ROSETTA_LATEST    = _env("ROSETTA_LATEST",     f"{NET_SOFTWARE}/rosetta/latest")
ROSETTA_DB        = _env("ROSETTA_DB",         f"{ROSETTA}/database")
ROSETTA_RESIDUE_TYPES = _env(
    "ROSETTA_RESIDUE_TYPES",
    f"{ROSETTA}/database/chemical/residue_type_sets/fa_standard/residue_types.txt",
)
MOLFILE_TO_PARAMS = _env(
    "MOLFILE_TO_PARAMS",
    "/net/software/rosetta/main/source/scripts/python/public/molfile_to_params.py",
)
PYROSETTA         = _env("PYROSETTA",          f"{NET_SOFTWARE}/pyrosetta/latest")
DALPHABALL        = _env("DALPHABALL",         f"{LAB_SCRIPTS}/DAlphaBall.gcc")

# =============================================================================
#  Other tools / databases
# =============================================================================
ENZYME_DESIGN_DIR      = _env("ENZYME_DESIGN_DIR",      f"{NET_SOFTWARE}/scripts/enzyme_design")
ENZYME_DESIGN_UTILS    = _env("ENZYME_DESIGN_UTILS",    f"{ENZYME_DESIGN_DIR}/utils")
ENZYME_DESIGN_FASTMPNN = _env("ENZYME_DESIGN_FASTMPNN", f"{ENZYME_DESIGN_DIR}/FastMPNNDesign")
INVROTZYME_UTILS       = _env("INVROTZYME_UTILS",       "/net/software/scripts/enzyme_design/invrotzyme/utils")
CHAI_RUN            = _env("CHAI_RUN",            "/net/software/lab/chai/chai-lab/run_chai.sh")
RFDIFFUSION_AA      = _env("RFDIFFUSION_AA",      "/net/software/lab/rf_diffusion_aa")
HHSUITE             = _env("HHSUITE",             f"{NET_SOFTWARE}/hhsuite")
HUGGINGFACE         = _env("HUGGINGFACE",         "/net/databases/huggingface")
FOLDSEEK            = _env("FOLDSEEK",            "/net/software/foldseek")
TMALIGN             = _env("TMALIGN",             f"{NET_SOFTWARE}/utils/TMalign")
SIGNALP             = _env("SIGNALP",             "/net/software/signalp/bin/signalp")
IPD_BLOCKS          = _env("IPD_BLOCKS",          "/net/shared/IPDblocks")


# =============================================================================
#  Open Babel — smart default
#  Order of preference:
#    1. $OBABEL                    (explicit user override; always wins)
#    2. obabel on $PATH            (the user's own active environment)
#    3. a lab-shared install       (if/when built under /net/software/lab)
#    4. Seth's conda obabel 3.1.0  (fallback that has historically been used)
#    5. bare "obabel"              (resolved from PATH at call time)
#  Note: /net/software/openbabel is Open Babel 2.4.1 (2018) and is intentionally
#  NOT used. Latest upstream is 3.2.0; the historical default here is 3.1.0.
# =============================================================================
def _detect_obabel() -> str:
    explicit = os.environ.get("OBABEL")
    if explicit:
        return explicit
    on_path = shutil.which("obabel")
    if on_path:
        return on_path
    for cand in (
        "/net/software/lab/openbabel/bin/obabel",            # lab-shared (if installed)
        "/home/woodbuse/conda_envs/openbabel_env/bin/obabel",  # Seth's 3.1.0
    ):
        if Path(cand).is_file():
            return cand
    return "obabel"


OBABEL = _detect_obabel()


if __name__ == "__main__":
    # `python repo_paths.py` prints the resolved configuration — handy for
    # checking what a fresh checkout / a labmate's environment will use.
    print(f"REPO_ROOT = {REPO_ROOT}")
    for _k, _v in sorted(globals().items()):
        if _k.isupper() and isinstance(_v, str):
            _src = "env" if _k in os.environ else "default"
            print(f"{_k:22s} = {_v}   [{_src}]")
