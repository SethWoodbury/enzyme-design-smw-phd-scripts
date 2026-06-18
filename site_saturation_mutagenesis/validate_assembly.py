#!/usr/bin/env python3
"""
Validate BsaI Golden Gate assembly of SSM fragment oligos.

For every possible single-mutation assembly (287 total), this script:
  1. Takes the mutant fragment + 3 WT fragments
  2. Strips primers (first 20bp and last 20bp)
  3. Strips BsaI adapters (GGTCTCA from left, CGAGACC from right)
  4. Verifies sticky-end compatibility for 4-fragment assembly
  5. Concatenates coding regions (stripping overlapping OHs)
  6. Translates assembled coding DNA
  7. Verifies exactly 1 AA difference from WT at the expected position
  8. Verifies the mutant AA matches the mutation name
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FASTA_PATH = (
    "/home/woodbuse/projects/organophosphatase/pxn/"
    "zapp_p1d1_i1__SSM_and_opt/DESIGN_ORDER/golden_gate_ssm_i1/"
    "260327_ZAPP_i1_p1D1__GG_SSM__IPDblock_Order.fasta"
)

WT_PROTEIN = (
    "ITEEQYRAILEGLREKQELIRKGEIPGAHPEVYELYEEIVRTLEEKGPTEEAII"
    "EAVKVYLKKAKEIVEKLADEEFEAPTGTKVTLAEHALEHPAIALKAAGVELPPEL"
    "KAALEKFDEIARKLHPGLDALRLHLAGIADDPLFVELAREFGLGEDVERARASGF"
    "RISAAHGKGAFAVVFLFLYAIRKGYEDLIIEELK"
)

# Expected sticky-end junctions
EXPECTED_OHS = {
    "F1_F2": "ATCA",
    "F2_F3": "GAAC",
    "F3_F4": "CTGG",
}

PRIMER_LEN = 20
BSAI_ADAPTER_LEFT = "GGTCTCA"   # 7 bp
BSAI_ADAPTER_RIGHT = "CGAGACC"  # 7 bp
OH_LEN = 4

# Standard codon table
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def read_fasta(path):
    """Parse a FASTA file into a list of (header, sequence) tuples."""
    records = []
    header = None
    seq_parts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts).upper()))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts).upper()))
    return records


def parse_header(header):
    """Extract frag_group and mutation from a FASTA header (clean or detailed format)."""
    # Try detailed format first: "...  frag_group=F1 mutation=E11D ..."
    fg_match = re.search(r"frag_group=(\S+)", header)
    mut_match = re.search(r"mutation=(\S+)", header)
    if fg_match and mut_match:
        return fg_match.group(1), mut_match.group(1)
    # Fall back to clean format: "...__F1_E11D" or "...__F1_WT"
    # The last segment after __ contains FN_MUTATION
    parts = header.split("__")
    last = parts[-1].strip()  # e.g. "F1_E11D" or "F4_WT"
    m = re.match(r"(F\d+)_(.+)", last)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse header: {header}")


def strip_primers(seq):
    """Remove first and last 20 bp (primer binding sites)."""
    return seq[PRIMER_LEN:-PRIMER_LEN]


def strip_bsai_adapters(inner_seq):
    """
    Remove BsaI adapter sequences.
    Left:  GGTCTCA (7 bp)
    Right: CGAGACC (7 bp)
    Returns the insert: [OH_left][coding][OH_right]
    """
    assert inner_seq[:7] == BSAI_ADAPTER_LEFT, (
        f"Expected left adapter {BSAI_ADAPTER_LEFT}, got {inner_seq[:7]}"
    )
    assert inner_seq[-7:] == BSAI_ADAPTER_RIGHT, (
        f"Expected right adapter {BSAI_ADAPTER_RIGHT}, got {inner_seq[-7:]}"
    )
    return inner_seq[7:-7]


def get_insert(oligo_seq):
    """
    Extract the insert from an oligo sequence.
    Auto-detects format:
      - insert_only: sequence IS the insert (starts with OH, no adapters/primers)
      - full_oligo: has primers + adapters wrapping the insert
    """
    upper = oligo_seq.upper()
    # Check if it starts with a known primer (20bp) followed by GGTCTCA
    if len(upper) > 27 and upper[20:27] == BSAI_ADAPTER_LEFT:
        # full_oligo mode
        inner = strip_primers(upper)
        insert = strip_bsai_adapters(inner)
        return insert
    elif BSAI_ADAPTER_LEFT in upper[:10]:
        # Has adapters but no primers
        insert = strip_bsai_adapters(upper)
        return insert
    else:
        # insert_only mode — the sequence IS the insert
        return upper


def translate(dna):
    """Translate a DNA sequence to protein (reading frame starts at pos 0)."""
    protein = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i + 3]
        aa = CODON_TABLE.get(codon, "?")
        protein.append(aa)
    return "".join(protein)


def assemble_fragments(inserts):
    """
    Given a dict {F1: insert, F2: insert, F3: insert, F4: insert},
    verify sticky ends and assemble the coding DNA.

    Each insert = [OH_left (4bp)][coding][OH_right (4bp)]

    Assembly order: F1 -- F2 -- F3 -- F4
    Overlaps:
      F1 right OH == F2 left OH  (ATCA)
      F2 right OH == F3 left OH  (GAAC)
      F3 right OH == F4 left OH  (CTGG)

    Returns (assembled_dna, oh_errors) where oh_errors is a list of mismatch descriptions.
    assembled_dna is the full insert including flanking vector OHs.
    """
    oh_errors = []

    f1, f2, f3, f4 = inserts["F1"], inserts["F2"], inserts["F3"], inserts["F4"]

    # Check sticky ends
    f1_right_oh = f1[-OH_LEN:]
    f2_left_oh = f2[:OH_LEN]
    f2_right_oh = f2[-OH_LEN:]
    f3_left_oh = f3[:OH_LEN]
    f3_right_oh = f3[-OH_LEN:]
    f4_left_oh = f4[:OH_LEN]

    if f1_right_oh != f2_left_oh:
        oh_errors.append(
            f"F1 right OH ({f1_right_oh}) != F2 left OH ({f2_left_oh}); "
            f"expected {EXPECTED_OHS['F1_F2']}"
        )
    elif f1_right_oh != EXPECTED_OHS["F1_F2"]:
        oh_errors.append(
            f"F1-F2 junction OH is {f1_right_oh}, expected {EXPECTED_OHS['F1_F2']}"
        )

    if f2_right_oh != f3_left_oh:
        oh_errors.append(
            f"F2 right OH ({f2_right_oh}) != F3 left OH ({f3_left_oh}); "
            f"expected {EXPECTED_OHS['F2_F3']}"
        )
    elif f2_right_oh != EXPECTED_OHS["F2_F3"]:
        oh_errors.append(
            f"F2-F3 junction OH is {f2_right_oh}, expected {EXPECTED_OHS['F2_F3']}"
        )

    if f3_right_oh != f4_left_oh:
        oh_errors.append(
            f"F3 right OH ({f3_right_oh}) != F4 left OH ({f4_left_oh}); "
            f"expected {EXPECTED_OHS['F3_F4']}"
        )
    elif f3_right_oh != EXPECTED_OHS["F3_F4"]:
        oh_errors.append(
            f"F3-F4 junction OH is {f3_right_oh}, expected {EXPECTED_OHS['F3_F4']}"
        )

    # Concatenate: full F1 insert + F2 without left OH + F3 without left OH + F4 without left OH
    assembled = f1 + f2[OH_LEN:] + f3[OH_LEN:] + f4[OH_LEN:]

    return assembled, oh_errors


def parse_mutation_name(mutation_str):
    """
    Parse a mutation string like 'E11D' into (wt_aa, position_1indexed, mut_aa).
    """
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mutation_str)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("BsaI Golden Gate Assembly Validation for SSM Fragment Oligos")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # 1. Read FASTA
    # ------------------------------------------------------------------
    records = read_fasta(FASTA_PATH)
    print(f"Read {len(records)} records from FASTA file.")

    # Organise records by frag_group and mutation
    frag_data = {}  # (frag_group, mutation) -> sequence
    wt_frags = {}   # frag_group -> sequence (WT oligos)
    mutant_frags = defaultdict(list)  # frag_group -> [(mutation, sequence), ...]

    for header, seq in records:
        fg, mut = parse_header(header)
        frag_data[(fg, mut)] = seq
        if mut == "WT":
            wt_frags[fg] = seq
        else:
            mutant_frags[fg].append((mut, seq))

    print(f"WT fragments found: {sorted(wt_frags.keys())}")
    for fg in sorted(mutant_frags.keys()):
        print(f"  {fg}: {len(mutant_frags[fg])} mutant fragments")
    total_mutants = sum(len(v) for v in mutant_frags.values())
    print(f"Total mutant fragments: {total_mutants}")
    print()

    # ------------------------------------------------------------------
    # 2. Verify WT assembly
    # ------------------------------------------------------------------
    print("-" * 80)
    print("STEP 1: Verify WT-only assembly")
    print("-" * 80)

    wt_inserts = {}
    for fg in ["F1", "F2", "F3", "F4"]:
        wt_inserts[fg] = get_insert(wt_frags[fg])
        print(f"  {fg} WT insert length: {len(wt_inserts[fg])} bp")
        print(f"    Left OH:  {wt_inserts[fg][:OH_LEN]}")
        print(f"    Right OH: {wt_inserts[fg][-OH_LEN:]}")

    assembled_wt, wt_oh_errors = assemble_fragments(wt_inserts)
    if wt_oh_errors:
        print(f"  WT ASSEMBLY OH ERRORS: {wt_oh_errors}")
    else:
        print("  Sticky ends: ALL MATCH")

    # Strip flanking vector OHs (4bp on each side) to get coding DNA
    wt_coding = assembled_wt[OH_LEN:-OH_LEN]
    # Translate - use only full codons (594 bp = 198 codons)
    wt_protein = translate(wt_coding)

    print(f"  Assembled coding DNA length: {len(wt_coding)} bp")
    print(f"  Translated protein length:   {len(wt_protein)} AA")
    print(f"  Translated protein: {wt_protein}")

    if wt_protein == WT_PROTEIN:
        print("  >>> WT ASSEMBLY: PASS (protein matches expected WT)")
    else:
        print("  >>> WT ASSEMBLY: FAIL")
        # Show differences
        for i, (a, b) in enumerate(zip(wt_protein, WT_PROTEIN)):
            if a != b:
                print(f"      Position {i+1}: got {a}, expected {b}")
        if len(wt_protein) != len(WT_PROTEIN):
            print(f"      Length mismatch: got {len(wt_protein)}, expected {len(WT_PROTEIN)}")
    print()

    # ------------------------------------------------------------------
    # 3. Validate all 287 single-mutation assemblies
    # ------------------------------------------------------------------
    print("-" * 80)
    print("STEP 2: Validate all single-mutation assemblies")
    print("-" * 80)
    print()

    pass_count = 0
    fail_count = 0
    failures = []

    for fg in ["F1", "F2", "F3", "F4"]:
        for mutation, mut_seq in mutant_frags[fg]:
            errors = []

            # Build the 4-fragment set: mutant replaces its frag_group, others are WT
            inserts = {}
            for frag in ["F1", "F2", "F3", "F4"]:
                if frag == fg:
                    inserts[frag] = get_insert(mut_seq)
                else:
                    inserts[frag] = wt_inserts[frag]

            # Verify sticky ends
            assembled, oh_errors = assemble_fragments(inserts)
            if oh_errors:
                errors.extend(oh_errors)

            # Get coding DNA (strip vector OHs)
            coding_dna = assembled[OH_LEN:-OH_LEN]

            # Translate
            protein = translate(coding_dna)

            # Check protein length
            if len(protein) != len(WT_PROTEIN):
                errors.append(
                    f"Protein length {len(protein)} != expected {len(WT_PROTEIN)}"
                )

            # Find AA differences from WT
            diffs = []
            for i in range(min(len(protein), len(WT_PROTEIN))):
                if protein[i] != WT_PROTEIN[i]:
                    diffs.append((i + 1, WT_PROTEIN[i], protein[i]))  # 1-indexed

            # Parse the expected mutation
            parsed = parse_mutation_name(mutation)
            if parsed is None:
                errors.append(f"Could not parse mutation name: {mutation}")
            else:
                exp_wt_aa, exp_pos, exp_mut_aa = parsed

                # Verify exactly 1 difference
                if len(diffs) != 1:
                    errors.append(
                        f"Expected exactly 1 AA difference, found {len(diffs)}: {diffs}"
                    )
                else:
                    pos, wt_aa, mut_aa = diffs[0]

                    # Verify position matches
                    if pos != exp_pos:
                        errors.append(
                            f"Mutation at position {pos}, expected position {exp_pos}"
                        )

                    # Verify WT AA at that position matches
                    if wt_aa != exp_wt_aa:
                        errors.append(
                            f"WT AA at position {pos} is {wt_aa}, "
                            f"mutation name says {exp_wt_aa}"
                        )

                    # Verify mutant AA matches
                    if mut_aa != exp_mut_aa:
                        errors.append(
                            f"Mutant AA is {mut_aa}, mutation name says {exp_mut_aa}"
                        )

            if errors:
                fail_count += 1
                failures.append((fg, mutation, errors))
            else:
                pass_count += 1

    # ------------------------------------------------------------------
    # 4. Print summary
    # ------------------------------------------------------------------
    print("-" * 80)
    print("RESULTS SUMMARY")
    print("-" * 80)
    print()
    print(f"Total single-mutation assemblies tested: {pass_count + fail_count}")
    print(f"  PASSED: {pass_count}")
    print(f"  FAILED: {fail_count}")
    print()

    if failures:
        print("FAILURES:")
        print("-" * 40)
        for fg, mutation, errors in failures:
            print(f"  {fg} {mutation}:")
            for err in errors:
                print(f"    - {err}")
        print()
    else:
        print("All assemblies passed validation!")
        print()

    # Per-fragment-group summary
    print("Per-fragment-group breakdown:")
    for fg in ["F1", "F2", "F3", "F4"]:
        n_mut = len(mutant_frags[fg])
        n_fail = sum(1 for f, m, e in failures if f == fg)
        n_pass = n_mut - n_fail
        print(f"  {fg}: {n_pass}/{n_mut} passed")
    print()

    # Final verdict
    if fail_count == 0 and wt_protein == WT_PROTEIN and not wt_oh_errors:
        print("=" * 80)
        print("OVERALL VERDICT: ALL VALIDATIONS PASSED")
        print("=" * 80)
        return 0
    else:
        print("=" * 80)
        print("OVERALL VERDICT: SOME VALIDATIONS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
