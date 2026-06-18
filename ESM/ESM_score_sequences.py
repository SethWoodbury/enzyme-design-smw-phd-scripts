import torch
import pandas as pd
import argparse
import submitit
from typing import Dict, Optional, List
from pathlib import Path
from functools import partial

import esm
from esm.model.esm2 import ESM2
from esm.data import Alphabet


ESM_MODEL_NAMES = [
    "esm2_t48_15B_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t6_8M_UR50D",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scores a set of sequences via ESMFold2."
    )
    parser.add_argument(
        "--input_fasta_file",
        type=str,
        default=None,
        help="Set to a fasta file to predict scores for all sequences in that file.",
    )
    parser.add_argument(
        "--input_fasta_dir",
        type=str,
        default=None,
        help="Set to a directory to predict scores for ALL sequences in ALL fasta files in that directory.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="esm_scores.csv",
        help="Set to a file path to output scores in csv format to that file. Defaults to `esm_scores.csv`",
    )
    parser.add_argument(
        "--esm_model",
        choices=ESM_MODEL_NAMES,
        default="esm2_t33_650M_UR50D",
        help="Which ESM2 model to use. There are a family of models in the ESM2 paper, from 8 million parameters to 15 billion. I generally recommend the 650M parameter model, which is what this script defaults to. See https://github.com/facebookresearch/esm#available-models for a complete list of available models.",
    )
    parser.add_argument(
        "--max_tokens_forward_pass",
        type=int,
        default=100000,
        help="The maximum number of tokens that you can fit in a forward pass on your GPU. If you are running out of memory running this script, particularly on smaller GPUs, lower this number. Defaults to 100000.",
    )
    return parser


def add_slurm_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Submits a job to the digs cluster via submitit"
        )
    parser.add_argument(
        "--slurm_log_path",
        default="slurm_logs",
        type=str,
        help="Path where slurm logs will go. Defaults to `slurm_logs`",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Set to true to run locally rather than submitting to slurm. This is useful for testing purposes.",
    )
    parser.add_argument(
        f"--slurm_partition",
        type=str,
        default="gpu",
        help="Slurm partition to run job on. Defaults to `gpu`, and probably should never be changed.",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="a6000",
        help="Which gpus to run on, slurm gres constraint. Defaults to a6000. Language models benefit quite a bit from large batch sizes, so bigger gpus are better.",
    )
    parser.add_argument(
        "--cpu_memory",
        type=int,
        default=16,
        help="Amount of cpu job memory to request for slurm submission. Defaults to 16.",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=2,
        help="Number of cpu cores to request for slurm submission",
    )
    parser.add_argument(
        "--timeout_min",
        type=int,
        default=120,
        help="Maximum number of minutes for slurm job to run. Defaults to 120.",
    )
    parser.add_argument(
        "--max_slurm_jobs_at_once",
        type=int,
        default=16,
        help="Maximum number of array jobs to run at once.",
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes to submit to."
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="score_esm_sequences",
        help="A string to indicate what the job should be called, when submitting to slurm.",
    )
    return parser


def create_executor(args: Dict) -> submitit.AutoExecutor:
    log_folder = Path(args.slurm_log_path)
    log_folder.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_partition=args.slurm_partition,
        slurm_mem=f"{args.cpu_memory}gb",
        slurm_job_name=args.job_name,
        cpus_per_task=args.cpus_per_task,
        slurm_ntasks_per_node=1,
        slurm_array_parallelism=args.max_slurm_jobs_at_once,
        nodes=args.nodes,
        timeout_min=args.timeout_min,
    )
    if args.gpu_type != "none":
        executor.update_parameters(
            slurm_gres=f"gpu:{args.gpu_type}:1",
        )

    return executor


def parse_fasta(filename: str):
    with open(filename, "r") as handle:
        filestring = handle.read()

    sequence_blocks = [x.strip() for x in filestring.split(">") if x.strip()]
    name = None
    sequences = {}
    for sequence_block in sequence_blocks:
        sequence_block = sequence_block.splitlines()
        name = sequence_block[0].strip()
        sequence = [x.strip() for x in sequence_block[1:] if x.strip()]
        sequence = "".join(sequence)
        sequences[name] = sequence
    return sequences


def get_model(model_name: str, device: str = "cuda:0"):
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model.to(device)
    return model, alphabet


def create_masked_token_matrix(
    sequence: str, alphabet: Alphabet, device: str = "cuda:0"
):
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
    return masked_tokens, true_tokens, padded_eye_mask


def get_model_logits(
    model: ESM2,
    batched_tokens: torch.Tensor,
    device: str = "cuda:0",
    max_tokens_forward_pass: int = 100000,
):
    total_seqs, seq_len = batched_tokens.shape
    maximum_batch_size = int(max_tokens_forward_pass / seq_len)
    logits_list = []

    with torch.no_grad():
        for index in range(0, total_seqs, maximum_batch_size):
            results = model(batched_tokens[index : index + maximum_batch_size])
            logits_list.append(results["logits"].to(device))

    logits = torch.cat(logits_list, dim=0)
    return logits


def get_pseudo_perplexity(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    padded_eye_mask: torch.Tensor,
    cross_entropy_fn: torch.nn.CrossEntropyLoss,
):
    normalization_constant = tokens.shape[0]

    negative_log_likelihoods = cross_entropy_fn(
        logits.permute(1, 2, 0), tokens.permute(1, 0)
    ).permute(1, 0)
    summed_diagonal_likelihoods = torch.sum(negative_log_likelihoods[padded_eye_mask])
    return torch.exp(summed_diagonal_likelihoods / normalization_constant).item()


def run_on_fastas(
    fasta_list: List[Path],
    model: ESM2,
    alphabet: Alphabet,
    cross_entropy_fn: torch.nn.CrossEntropyLoss,
    device: str = "cuda:0",
    args: Optional[argparse.Namespace] = None,
):
    fasta_files = []
    sequence_headers = []
    sequence_indices = []
    perplexities = []
    for fasta_file in fasta_list:
        print(f"Scoring sequences in fasta file {fasta_file}".center(75, "="))
        fasta_sequence_dictionary = parse_fasta(fasta_file)
        num_sequences = len(fasta_sequence_dictionary)

        for index, (name, sequence) in enumerate(fasta_sequence_dictionary.items()):
            print(f"Scoring sequence >{name},{sequence[:10]}... {index}/{num_sequences}", end="\r")
            masked_tokens, true_tokens, padded_eye_mask = create_masked_token_matrix(
                sequence, alphabet, device
            )
            logits = get_model_logits(
                model, masked_tokens, device, args.max_tokens_forward_pass
            )

            pseudo_perplexity = get_pseudo_perplexity(
                true_tokens, logits, padded_eye_mask, cross_entropy_fn
            )

            fasta_files.append(fasta_file)
            sequence_headers.append(name)
            sequence_indices.append(index)
            perplexities.append(pseudo_perplexity)

    data = pd.DataFrame(
        {
            "fasta_file": fasta_files,
            "sequence_header": sequence_headers,
            "sequence_index": sequence_indices,
            "esm_perplexity": perplexities,
        }
    )
    data.to_csv(args.output_file, index=False)


def run_all(args: argparse.Namespace):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, alphabet = get_model(args.esm_model, device=device)
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction="none").to(device)

    if args.input_fasta_file is None:
        print(args.input_fasta_dir)
        fasta_files = list(Path(args.input_fasta_dir).glob("*.fasta"))
        fasta_files += list(Path(args.input_fasta_dir).glob("*.fa"))
    else:
        fasta_files = [args.input_fasta_file]

    assert (
        len(fasta_files) > 0
    ), f"I couldn't find any fasta files in user provided directory {args.input_fasta_dir}. Note that your fasta files have to end in .fasta or .fa for them to be recognized."

    run_on_fastas(fasta_files, model, alphabet, cross_entropy_fn, device, args)


def main():
    parser = create_parser()
    parser = add_slurm_args(parser)
    args = parser.parse_args()

    assert args.input_fasta_file is not None or args.input_fasta_dir is not None, "You must specify either an input fasta file via input_fasta_file or an input fasta directory via input_fasta_dir to run this script."
    assert args.input_fasta_file is None or args.input_fasta_dir is None, "You cannot specify both input_fasta_file and input_fasta_dir. Pick one!"

    run_fn = partial(run_all, args=args)

    if args.local:
        run_fn()
    else:
        executor = create_executor(args)
        job = executor.submit(run_fn)
        print(f"Submitted job {job.job_id} with name {args.job_name} to slurm. Logs will appear in {args.slurm_log_path}.")


if __name__ == "__main__":
    main()
