"""SaProt input-prep helpers.

SaProt's input is a "SA token" sequence: per residue we concatenate the
amino-acid letter (uppercase) and the foldseek 3Di letter (lowercase), e.g.
ubiquitin's first residues might tokenize as ['Md', 'Qa', 'If', ...].

This module handles:
  - running foldseek's structureto3didescriptor on a PDB,
  - aligning the 3Di string to the input AA sequence,
  - producing a string SaProt's tokenizer can consume.
"""

import os
import subprocess
import tempfile
from pathlib import Path

FOLDSEEK_BIN = os.environ.get("FOLDSEEK_BIN", "foldseek")


def run_foldseek_3di(pdb_path: str, chain: str | None = None) -> tuple[str, str]:
    """Return (aa_sequence, 3di_sequence) for the requested chain (or first chain).

    Uses `foldseek structureto3didescriptor` which outputs a tab-separated row
    per chain: query<tab>chain<tab>aa_seq<tab>3di_seq<tab>...
    """
    pdb_path = str(Path(pdb_path).resolve())
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.tsv")
        subprocess.run(
            [FOLDSEEK_BIN, "structureto3didescriptor", pdb_path, out],
            check=True,
            capture_output=True,
        )
        with open(out) as fh:
            rows = [line.rstrip("\n").split("\t") for line in fh if line.strip()]

    if not rows:
        raise RuntimeError(f"foldseek produced no output for {pdb_path}")

    # Pick chain. structureto3didescriptor's first column is "filename_CHAIN".
    if chain is not None:
        rows = [r for r in rows if r[0].endswith(f"_{chain}")]
        if not rows:
            raise ValueError(f"chain {chain!r} not found in {pdb_path}")

    aa_seq = rows[0][1]
    di_seq = rows[0][2]
    if len(aa_seq) != len(di_seq):
        raise RuntimeError(
            f"AA/3Di length mismatch for {pdb_path}: {len(aa_seq)} vs {len(di_seq)}"
        )
    return aa_seq, di_seq


SAPROT_3DI_ALPHABET = set("acdefghiklmnpqrstvwy")  # 20 lowercase 3Di letters


def make_sa_tokens(aa_seq: str, di_seq: str) -> str:
    """Interleave AA (uppercase) + 3Di (lowercase) -> 'MaQbIc...' style string.

    Foldseek emits 3Di letters in *uppercase*; SaProt's vocab uses lowercase,
    so we lowercase here. Anything outside the 20-letter 3Di alphabet (e.g. 'X'
    for un-assigned residues) maps to '#', SaProt's 'no structure' token.
    """
    if len(aa_seq) != len(di_seq):
        raise ValueError("aa_seq and di_seq must be same length")
    out = []
    for aa, di in zip(aa_seq, di_seq):
        di_lc = di.lower()
        di_lc = di_lc if di_lc in SAPROT_3DI_ALPHABET else "#"
        out.append(f"{aa}{di_lc}")
    return "".join(out)


def sa_tokens_from_pdb(pdb_path: str, chain: str | None = None) -> str:
    aa, di = run_foldseek_3di(pdb_path, chain=chain)
    return make_sa_tokens(aa, di)
