"""Quick smoke test for ESM-C in our esmc.sif.

What it does:
  1. Loads esmc_300m on the GPU (downloads weights to HF_HOME on first run).
  2. Runs a short protein through it.
  3. Prints embedding shape + a sanity check on the mask-prediction logits.

Run via the slurm wrapper test_esmc.sbatch, or directly:
  /net/software/containers/users/woodbuse/esmc.sif \
      /home/woodbuse/special_scripts/ESM/test_esmc.py
"""

import os
import time

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


# A short, well-known protein: ubiquitin (76 aa)
SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


def main():
    print(f"HF_HOME = {os.environ.get('HF_HOME', '<unset>')}")
    print(f"torch = {torch.__version__}, cuda available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device = {torch.cuda.get_device_name(0)}")
        print(f"capability = {torch.cuda.get_device_capability(0)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    model = ESMC.from_pretrained("esmc_300m").to(device).eval()
    print(f"loaded esmc_300m in {time.time() - t0:.1f}s")

    protein = ESMProtein(sequence=SEQUENCE)
    protein_t = model.encode(protein)
    input_ids = protein_t.sequence  # 1-D tensor of token ids
    print(f"tokens shape: {tuple(input_ids.shape)}, len(seq)={len(SEQUENCE)}")

    with torch.no_grad():
        out = model.logits(
            protein_t,
            LogitsConfig(sequence=True, return_embeddings=True),
        )

    emb = out.embeddings
    logits = out.logits.sequence
    print(f"embeddings: shape={tuple(emb.shape)} dtype={emb.dtype} "
          f"mean={emb.float().mean().item():.4f} std={emb.float().std().item():.4f}")
    print(f"logits:     shape={tuple(logits.shape)}")

    # Sanity: the model's argmax token at each position should usually match
    # the input residue (no masks here, so this is just confirming things flow).
    pred_ids = logits.argmax(dim=-1).squeeze(0)
    match = (pred_ids[1:-1] == input_ids[1:-1]).float().mean().item()
    print(f"argmax-matches-input fraction (no masking): {match:.2%}")

    # Pre-warm 600M weights into the HF cache so the second model is also
    # downloaded as part of this test job (no GPU load — just disk).
    print("warming esmc_600m weights into HF cache...")
    t0 = time.time()
    ESMC.from_pretrained("esmc_600m")  # download only, stays on CPU
    print(f"esmc_600m cached in {time.time() - t0:.1f}s")

    print("OK")


if __name__ == "__main__":
    main()
