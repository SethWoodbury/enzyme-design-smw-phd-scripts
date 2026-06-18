import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import esm

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths


ESM_MODEL_NAMES = [
    "esm2_t48_15B_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t6_8M_UR50D",
]

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
FIVE_CHAR_RESNAMES = {"HIS_D", "HIS_E", "CYS_D"}

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common non-canonical or aliases
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "HIS_D": "H",
    "HIS_E": "H",
    "CYS_D": "C",
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suggest amino-acid mutations using ESM2 masked language model."
    )
    parser.add_argument("--input_fasta", type=str, default=None, help="FASTA input file.")
    parser.add_argument("--input_pdb", type=str, default=None, help="PDB input file.")
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Chain ID to score from PDB (default: all protein chains).",
    )
    parser.add_argument(
        "--esm_model",
        choices=ESM_MODEL_NAMES,
        default=None,
        help="ESM2 model to use. Default: 650M on GPU, 8M on CPU.",
    )
    parser.add_argument(
        "--max_tokens_forward_pass",
        type=int,
        default=100000,
        help="Max tokens per forward pass (reduce if OOM).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top substitutions to show per position.",
    )
    parser.add_argument(
        "--max_positions",
        type=int,
        default=20,
        help="Max positions to report per chain (0 for all).",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Minimum ΔlogP to report a substitution.",
    )
    parser.add_argument(
        "--include_catalytic",
        action="store_true",
        help="Include catalytic residues from REMARK 666 in suggestions.",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=repo_paths.ESMFOLD_DB,
        help="Directory with local ESM2 .pt weights (fallback to download if missing).",
    )
    parser.add_argument(
        "--nterm",
        type=str,
        default="",
        help="N-terminal sequence to prepend if not already present (e.g., MSG).",
    )
    parser.add_argument(
        "--cterm",
        type=str,
        default="",
        help="C-terminal sequence to append if not already present.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g. cpu, cuda:0).",
    )
    return parser


def parse_fasta(filename: str) -> Dict[str, str]:
    with open(filename, "r") as handle:
        filestring = handle.read()
    sequence_blocks = [x.strip() for x in filestring.split(">") if x.strip()]
    sequences: Dict[str, str] = {}
    for sequence_block in sequence_blocks:
        lines = sequence_block.splitlines()
        name = lines[0].strip()
        sequence = "".join(x.strip() for x in lines[1:] if x.strip())
        sequences[name] = sequence
    return sequences


def parse_remark_666(lines: Iterable[str]) -> List[Dict]:
    entries: List[Dict] = []
    for line in lines:
        if not line.startswith("REMARK 666"):
            continue
        tokens = line.split()
        if "MOTIF" not in tokens or "MATCH" not in tokens:
            continue
        motif_idx = tokens.index("MOTIF")
        try:
            motif_chain = tokens[motif_idx + 1]
            motif_resname = tokens[motif_idx + 2]
            motif_resno = int(tokens[motif_idx + 3])
            block_index = int(tokens[-2])
            block_variant = int(tokens[-1])
        except (IndexError, ValueError):
            continue
        entry = {
            "line": line.rstrip(),
            "motif_chain": motif_chain,
            "motif_resname": motif_resname,
            "motif_resno": motif_resno,
            "block_index": block_index,
            "block_variant": block_variant,
        }
        entries.append(entry)
    return entries


def parse_pdb_sequences(pdb_path: str) -> Tuple[Dict[str, str], Dict[str, List[Dict]], List[Dict]]:
    sequences: Dict[str, str] = {}
    residues_by_chain: Dict[str, List[Dict]] = {}
    seen_keys: Dict[str, set] = {}
    with open(pdb_path, "r") as handle:
        lines = handle.readlines()

    remark_entries = parse_remark_666(lines)

    for line in lines:
        if not line.startswith("ATOM"):
            continue
        # 5-character resname handling
        resname_5char = line[16:21].strip()
        if resname_5char in FIVE_CHAR_RESNAMES:
            resname = resname_5char
        else:
            resname = line[17:20].strip()
        chain = line[21:22]
        resno_str = line[22:26].strip()
        if not resno_str:
            continue
        resno = int(resno_str)
        icode = line[26:27].strip()
        key = (chain, resno, icode)
        if chain not in residues_by_chain:
            residues_by_chain[chain] = []
            seen_keys[chain] = set()
        if key in seen_keys[chain]:
            continue
        seen_keys[chain].add(key)
        aa = THREE_TO_ONE.get(resname, "X")
        residues_by_chain[chain].append(
            {"chain": chain, "resno": resno, "icode": icode, "resname": resname, "aa": aa}
        )

    for chain, residues in residues_by_chain.items():
        sequences[chain] = "".join(r["aa"] for r in residues)

    return sequences, residues_by_chain, remark_entries


def resolve_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_model(model_name: str, device: str, weights_dir: Optional[str] = None):
    weights_path = None
    if weights_dir:
        candidate = Path(weights_dir) / f"{model_name}.pt"
        if candidate.is_file():
            weights_path = candidate
    if weights_path is not None:
        try:
            # Allowlist argparse.Namespace for older ESM checkpoints under PyTorch >= 2.6
            try:
                import argparse as _argparse

                if hasattr(torch, "serialization") and hasattr(
                    torch.serialization, "add_safe_globals"
                ):
                    torch.serialization.add_safe_globals([_argparse.Namespace])
            except Exception:
                pass
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(
                str(weights_path)
            )
            print(f"Loaded weights from {weights_path}")
        except Exception as exc:
            print(f"Local weights load failed ({exc}); falling back to download.")
            model, alphabet = getattr(esm.pretrained, model_name)()
    else:
        model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model.to(device)
    return model, alphabet


def create_masked_token_matrix(sequence: str, alphabet, device: str):
    sequence_length = len(sequence)
    data = [(None, sequence)]
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    true_tokens = batch_tokens.repeat(sequence_length, 1)
    masked_tokens = true_tokens.clone()
    padded_eye_mask = torch.eye(
        sequence_length + 1, sequence_length + 2, dtype=bool, device=device
    )
    padded_eye_mask = padded_eye_mask[1:]
    masked_tokens[padded_eye_mask] = alphabet.mask_idx
    return masked_tokens, true_tokens


def get_model_logits(
    model,
    batched_tokens: torch.Tensor,
    device: str,
    max_tokens_forward_pass: int,
):
    total_seqs, seq_len = batched_tokens.shape
    maximum_batch_size = max(1, int(max_tokens_forward_pass / seq_len))
    logits_list = []
    with torch.no_grad():
        for index in range(0, total_seqs, maximum_batch_size):
            results = model(batched_tokens[index : index + maximum_batch_size])
            logits_list.append(results["logits"].to(device))
    return torch.cat(logits_list, dim=0)


def compute_log_probs(
    model,
    alphabet,
    sequence: str,
    device: str,
    max_tokens_forward_pass: int,
):
    masked_tokens, _ = create_masked_token_matrix(sequence, alphabet, device)
    logits = get_model_logits(model, masked_tokens, device, max_tokens_forward_pass)
    seq_len = len(sequence)
    token_positions = torch.arange(seq_len, device=device) + 1  # +1 for BOS
    logits_masked = logits[torch.arange(seq_len, device=device), token_positions]
    return torch.log_softmax(logits_masked, dim=-1)


def build_catalytic_index_map(
    residues_by_chain: Dict[str, List[Dict]],
    remark_entries: List[Dict],
) -> Tuple[Dict[str, set], Dict[str, List[int]]]:
    catalytic_positions: Dict[str, set] = {}
    missing: Dict[str, List[int]] = {}

    resno_to_index: Dict[str, Dict[int, int]] = {}
    for chain, residues in residues_by_chain.items():
        resno_to_index[chain] = {}
        for idx, res in enumerate(residues):
            # ignore insertion code for REMARK 666 matching
            if res["resno"] not in resno_to_index[chain]:
                resno_to_index[chain][res["resno"]] = idx

    for entry in remark_entries:
        chain = entry["motif_chain"]
        resno = entry["motif_resno"]
        if chain not in catalytic_positions:
            catalytic_positions[chain] = set()
        if chain in resno_to_index and resno in resno_to_index[chain]:
            catalytic_positions[chain].add(resno_to_index[chain][resno])
        else:
            missing.setdefault(chain, []).append(resno)

    return catalytic_positions, missing


def apply_terminal_additions(
    sequence: str,
    nterm: str,
    cterm: str,
    residue_numbers: Optional[List[Optional[int]]] = None,
    catalytic_indices: Optional[set] = None,
) -> Tuple[str, Optional[List[Optional[int]]], Optional[set], List[str], int, int]:
    notes: List[str] = []
    new_sequence = sequence
    new_residue_numbers: Optional[List[Optional[int]]] = None
    new_catalytic_indices: Optional[set] = None
    n_added = 0
    c_added = 0

    if residue_numbers is not None:
        new_residue_numbers = [int(x) for x in residue_numbers]

    if nterm:
        if not new_sequence.startswith(nterm):
            new_sequence = nterm + new_sequence
            n_added = len(nterm)
            notes.append(f"Prepended N-term '{nterm}'")
            if new_residue_numbers is not None:
                new_residue_numbers = [None] * n_added + new_residue_numbers
        else:
            notes.append(f"N-term '{nterm}' already present")

    if catalytic_indices is not None:
        new_catalytic_indices = set(catalytic_indices)
        if n_added:
            new_catalytic_indices = {i + n_added for i in new_catalytic_indices}

    if cterm:
        if not new_sequence.endswith(cterm):
            new_sequence = new_sequence + cterm
            c_added = len(cterm)
            notes.append(f"Appended C-term '{cterm}'")
            if new_residue_numbers is not None:
                new_residue_numbers = new_residue_numbers + [None] * len(cterm)
        else:
            notes.append(f"C-term '{cterm}' already present")

    return new_sequence, new_residue_numbers, new_catalytic_indices, notes, n_added, c_added


def format_substitutions(subs: List[Dict], top_k: int) -> List[str]:
    lines = []
    for s in subs[:top_k]:
        lines.append(
            f"  {s['wt']}->{s['mut']}  dlogP={s['delta']:+.3f}  P={s['prob']:.3f}"
        )
    return lines


def format_catalytic_list(
    indices: List[int],
    sequence: str,
    residue_numbers: Optional[List[Optional[int]]],
    chain: Optional[str],
) -> List[str]:
    labels = []
    for idx in indices:
        wt = sequence[idx] if idx < len(sequence) else "?"
        resno = residue_numbers[idx] if residue_numbers else idx + 1
        resno_label = str(resno) if resno is not None else f"pos{idx + 1}"
        if chain:
            labels.append(f"{chain}:{resno_label}{wt}")
        else:
            labels.append(f"{resno_label}{wt}")
    # Wrap into multiple lines for readability
    wrapped = []
    chunk_size = 8
    for i in range(0, len(labels), chunk_size):
        wrapped.append(", ".join(labels[i : i + chunk_size]))
    return wrapped


def suggest_mutations_for_sequence(
    name: str,
    sequence: str,
    model,
    alphabet,
    device: str,
    max_tokens_forward_pass: int,
    catalytic_indices: Optional[set],
    include_catalytic: bool,
    min_delta: float,
    top_k: int,
    max_positions: int,
    residue_numbers: Optional[List[Optional[int]]] = None,
    chain: Optional[str] = None,
    nterm_added: int = 0,
    original_length: Optional[int] = None,
):
    start = time.perf_counter()
    log_probs = compute_log_probs(
        model, alphabet, sequence, device, max_tokens_forward_pass
    )
    seq_len = len(sequence)
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in STANDARD_AAS}
    suggestions = []

    for i in range(seq_len):
        if not include_catalytic and catalytic_indices and i in catalytic_indices:
            continue
        wt = sequence[i]
        if wt not in aa_to_idx:
            continue
        wt_idx = aa_to_idx[wt]
        wt_logp = log_probs[i, wt_idx].item()
        subs = []
        for aa in STANDARD_AAS:
            if aa == wt:
                continue
            idx = aa_to_idx[aa]
            logp = log_probs[i, idx].item()
            delta = logp - wt_logp
            if delta < min_delta:
                continue
            subs.append({
                "wt": wt,
                "mut": aa,
                "logp": logp,
                "delta": delta,
                "prob": math.exp(logp),
            })
        if not subs:
            continue
        subs.sort(key=lambda x: x["delta"], reverse=True)
        best_delta = subs[0]["delta"]
        suggestions.append({
            "pos": i + 1,
            "wt": wt,
            "best_delta": best_delta,
            "subs": subs,
            "resno": residue_numbers[i] if residue_numbers else None,
        })

    suggestions.sort(key=lambda x: x["best_delta"], reverse=True)
    if max_positions > 0:
        suggestions = suggestions[:max_positions]

    elapsed = time.perf_counter() - start

    chain_label = f" chain {chain}" if chain else ""
    print(f"\n=== {name}{chain_label} (len={seq_len}) ===")
    print(f"Device: {device}")
    print(f"Computed log-probabilities in {elapsed:.2f}s")
    if nterm_added:
        print(f"N-term added length: {nterm_added} (aug pos = orig pos + {nterm_added})")
    if catalytic_indices:
        cat_indices_sorted = sorted(catalytic_indices)
        print(f"Catalytic residues from REMARK 666: {len(cat_indices_sorted)}")
        for line in format_catalytic_list(
            cat_indices_sorted, sequence, residue_numbers, chain
        ):
            print(f"  {line}")
        if not include_catalytic:
            print("Catalytic residues excluded from suggestions.")

    if not suggestions:
        print("No substitutions passed filters.")
        return

    for s in suggestions:
        if s["resno"] is not None:
            if nterm_added:
                orig_pos = s["pos"] - nterm_added
                pos_label = (
                    f"pos {s['pos']} (orig pos {orig_pos}, resno {s['resno']})"
                )
            else:
                pos_label = f"pos {s['pos']} (resno {s['resno']})"
        else:
            if nterm_added and s["pos"] <= nterm_added:
                pos_label = f"pos {s['pos']} (added N-term)"
            elif original_length is not None and s["pos"] > nterm_added + original_length:
                pos_label = f"pos {s['pos']} (added C-term)"
            else:
                pos_label = f"pos {s['pos']} (added term)"
        print(f"{pos_label} WT={s['wt']} best dlogP={s['best_delta']:+.3f}")
        for line in format_substitutions(s["subs"], top_k):
            print(line)


def main():
    parser = create_parser()
    args = parser.parse_args()

    if (args.input_fasta is None) == (args.input_pdb is None):
        raise SystemExit("Specify exactly one of --input_fasta or --input_pdb.")

    device = resolve_device(args.device)
    esm_model = args.esm_model
    if esm_model is None:
        esm_model = "esm2_t33_650M_UR50D" if device.startswith("cuda") else "esm2_t6_8M_UR50D"

    t0 = time.perf_counter()
    model, alphabet = get_model(esm_model, device, args.weights_dir)
    print(f"Loaded model {esm_model} on {device} in {time.perf_counter() - t0:.2f}s")

    if args.input_fasta:
        sequences = parse_fasta(args.input_fasta)
        for name, seq in sequences.items():
            orig_len = len(seq)
            seq, _, _, notes, n_added, _ = apply_terminal_additions(
                seq, args.nterm, args.cterm, None, None
            )
            if notes:
                print(f"\n{name}: " + " | ".join(notes))
            suggest_mutations_for_sequence(
                name=name,
                sequence=seq,
                model=model,
                alphabet=alphabet,
                device=device,
                max_tokens_forward_pass=args.max_tokens_forward_pass,
                catalytic_indices=None,
                include_catalytic=True,
                min_delta=args.min_delta,
                top_k=args.top_k,
                max_positions=args.max_positions,
                nterm_added=n_added,
                original_length=orig_len,
            )
        return

    sequences, residues_by_chain, remark_entries = parse_pdb_sequences(args.input_pdb)
    catalytic_indices, missing = build_catalytic_index_map(residues_by_chain, remark_entries)

    if missing:
        for chain, resnos in missing.items():
            print(f"Warning: REMARK 666 residues not found in chain {chain}: {sorted(resnos)}")

    chains = [args.chain] if args.chain else sorted(sequences.keys())
    for chain in chains:
        if chain not in sequences:
            print(f"Skipping chain {chain}: no protein residues found")
            continue
        residues = residues_by_chain[chain]
        residue_numbers = [r["resno"] for r in residues]
        orig_len = len(sequences[chain])
        seq, residue_numbers, cat_indices, notes, n_added, _ = apply_terminal_additions(
            sequences[chain],
            args.nterm,
            args.cterm,
            residue_numbers,
            catalytic_indices.get(chain),
        )
        if notes:
            print(f"\n{Path(args.input_pdb).name} chain {chain}: " + " | ".join(notes))
        suggest_mutations_for_sequence(
            name=Path(args.input_pdb).name,
            sequence=seq,
            model=model,
            alphabet=alphabet,
            device=device,
            max_tokens_forward_pass=args.max_tokens_forward_pass,
            catalytic_indices=cat_indices,
            include_catalytic=args.include_catalytic,
            min_delta=args.min_delta,
            top_k=args.top_k,
            max_positions=args.max_positions,
            residue_numbers=residue_numbers,
            chain=chain,
            nterm_added=n_added,
            original_length=orig_len,
        )


if __name__ == "__main__":
    main()
