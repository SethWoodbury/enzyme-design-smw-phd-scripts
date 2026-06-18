"""Smoke test: load each SaProt model, tokenize test_pte.pdb, forward pass.

Runs the three downloaded SaProt models (35M / 650M / 1.3B) on the same
PDB and prints embedding/logits stats so we can confirm everything works
end-to-end inside esmc.sif.
"""

import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Local helper (sibling of this file)
sys.path.insert(0, str(Path(__file__).parent))
from saprot_utils import sa_tokens_from_pdb  # noqa: E402

import torch  # noqa: E402
from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: E402


PDB = str(_HERE / "test_pte.pdb")
MODELS = [
    "westlake-repl/SaProt_35M_AF2",
    "westlake-repl/SaProt_650M_PDB",
    "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI",
]


def load_one(repo_id: str, device: str):
    print(f"\n=== {repo_id} ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = EsmForMaskedLM.from_pretrained(repo_id, torch_dtype=torch.float32).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    return tok, model


def main():
    print(f"HF_HOME    = {os.environ.get('HF_HOME')}")
    print(f"HF_HUB_CACHE = {os.environ.get('HF_HUB_CACHE')}")
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")

    sa_str = sa_tokens_from_pdb(PDB)
    print(f"PDB:        {PDB}")
    print(f"SA length:  {len(sa_str) // 2} residues")
    print(f"SA prefix:  {sa_str[:80]}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"device:     {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")

    for repo in MODELS:
        tok, model = load_one(repo, device)
        # Tokenizer expects spaces between SA tokens (each SA token = 2 chars)
        spaced = " ".join(sa_str[i:i + 2] for i in range(0, len(sa_str), 2))
        inputs = tok(spaced, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        emb = out.hidden_states[-1]
        logits = out.logits
        print(f"  emb:    shape={tuple(emb.shape)} mean={emb.mean().item():+.4f} std={emb.std().item():.4f}")
        print(f"  logits: shape={tuple(logits.shape)}")
        # GPU mem
        if torch.cuda.is_available():
            print(f"  vram:   alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB "
                  f"max={torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        del model, tok, out, emb, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nOK")


if __name__ == "__main__":
    main()
