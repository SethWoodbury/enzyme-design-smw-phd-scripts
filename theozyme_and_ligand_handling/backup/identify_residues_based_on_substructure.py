#!/usr/bin/env python3

import sys

# Custom scoring parameters and debugging flag
DEBUGGING = False  # Set to False to disable debugging output
MATCH_SCORE = 2       # Reward for an exact atom match
MISMATCH_SCORE = -1   # Penalty for atom mismatch
GAP_PENALTY = -2      # Penalty for alignment gaps

def align_smiles(fragment, candidate):
    """Align fragment SMILES with candidate SMILES using a custom scoring algorithm."""
    n = len(fragment)
    m = len(candidate)

    # Initialize scoring matrix
    score_matrix = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill scoring matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score_matrix[i - 1][j - 1] + (MATCH_SCORE if fragment[i - 1] == candidate[j - 1] else MISMATCH_SCORE)
            delete = score_matrix[i - 1][j] + GAP_PENALTY
            insert = score_matrix[i][j - 1] + GAP_PENALTY
            score_matrix[i][j] = max(match, delete, insert)

    # Traceback to find best alignment score
    best_score = score_matrix[n][m]
    
    if DEBUGGING:
        # Print the alignment matrix and scores for debugging
        print(f"\n--- Alignment Matrix for Fragment: {fragment} and Candidate: {candidate} ---")
        for row in score_matrix:
            print(' '.join(f"{val:4}" for val in row))
        print(f"\nBest alignment score between '{fragment}' and '{candidate}': {best_score}")

    return best_score

def find_best_substructure_match(fragment_smiles, candidate_amino_acids):
    """Find the best substructure match for a fragment SMILES among candidate amino acids."""
    best_match = "unknown"
    highest_score = float('-inf')

    if DEBUGGING:
        print(f"\n### STARTING SUBSTRUCTURE MATCHING ###")
        print(f"Fragment SMILES: {fragment_smiles}")
        print(f"Candidate amino acids for matching: {candidate_amino_acids}\n")

    for aa_name, aa_smiles in candidate_amino_acids.items():
        # Calculate alignment score between fragment and candidate
        alignment_score = align_smiles(fragment_smiles, aa_smiles)
        
        if DEBUGGING:
            print(f"\nCalculated alignment score for candidate {aa_name} with SMILES '{aa_smiles}': {alignment_score}")

        if alignment_score > highest_score:
            highest_score = alignment_score
            best_match = aa_name

    if DEBUGGING:
        print(f"\n### MATCHING SUMMARY ###")
        print(f"Best match for fragment SMILES '{fragment_smiles}' is '{best_match}' with a score of {highest_score}")
    else:
        # Only print the best match in non-debugging mode
        print(f"{best_match}")

    return best_match

if __name__ == "__main__":
    # Parse inputs
    fragment_smiles = sys.argv[1]
    candidate_amino_acids = {name: smiles for name, smiles in zip(sys.argv[2::2], sys.argv[3::2])}

    # Output best match (along with detailed debugging output if DEBUGGING is True)
    best_match = find_best_substructure_match(fragment_smiles, candidate_amino_acids)
