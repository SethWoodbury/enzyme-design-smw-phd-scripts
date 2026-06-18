"""Download SaProt weights into /net/databases/huggingface/saprot.

Downloads only the artifacts we actually need:
  - For SaProt_1.3B_AFDB_OMG_NCBI: safetensors (skip duplicate pytorch_model.bin).
  - For 650M and 35M: pytorch_model.bin (no safetensors uploaded).
  - Tokenizer / config files for all.
"""

import os
import sys

from huggingface_hub import snapshot_download

CACHE = "/net/databases/huggingface/saprot"
os.environ["HF_HOME"] = CACHE
os.environ["HF_HUB_CACHE"] = f"{CACHE}/hub"

JOBS = [
    # repo_id, allow_patterns, ignore_patterns
    (
        "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI",
        None,
        ["pytorch_model.bin", "*.pt"],  # keep safetensors only
    ),
    (
        "westlake-repl/SaProt_650M_PDB",
        None,
        ["*.pt"],  # only pytorch_model.bin available, skip duplicate .pt
    ),
    (
        "westlake-repl/SaProt_35M_AF2",
        None,
        ["*.pt"],
    ),
]


def main():
    for repo, allow, ignore in JOBS:
        print(f"=== {repo} ===", flush=True)
        path = snapshot_download(
            repo_id=repo,
            cache_dir=f"{CACHE}/hub",
            allow_patterns=allow,
            ignore_patterns=ignore,
        )
        print(f"  -> {path}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
