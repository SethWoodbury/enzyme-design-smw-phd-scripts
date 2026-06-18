#!/usr/bin/env python3
"""
Validate BsaI Golden Gate assembly of SSM fragment oligos -- directly from CSV.

Reads /home/woodbuse/Downloads/ipdblocks-order-6802.csv and performs full
assembly validation on all 287 single-mutation assemblies plus the WT assembly.

For every assembly:
  1. Takes the mutant fragment + 3 WT fragments (insert-only format)
  2. Verifies sticky-end compatibility at junctions
  3. Strips overlapping OHs, concatenates coding regions
  4. Translates assembled coding DNA
  5. Verifies protein is exactly 198 AA
  6. Verifies exactly 1 AA difference from WT at the expected position
  7. Verifies the mutation position and amino acid match the design name
"""

import csv
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_PATH = "/home/woodbuse/Downloads/ipdblocks-order-6802.csv"

WT_PROTEIN = (
    "ITEEQYRAILEGLREKQELIRKGEIPGAHPEVYELYEEIVRTLEEKGPTEEAII"
    "EAVKVYLKKAKEIVEKLADEEFEAPTGTKVTLAEHALEHPAIALKAAGVELPPEL"
    "KAALEKFDEIARKLHPGLDALRLHLAGIADDPLFVELAREFGLGEDVERARASGF"
    "RISAAHGKGAFAVVFLFLYAIRKGYEDLIIEELK"
)

# Expected sticky-end junctions (4 bp overhangs)
EXPECTED_OHS = {
    "F1_F2": "ATCA",
    "F2_F3": "GAAC",
    "F3_F4": "CTGG",
}

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
def parse_design_name(design_name):
    """
    Parse design_name like 'zapp_i1_p1D1__ssm_frag_0001__plate1__A1__384well__F1_E11D'
    Returns (frag_group, mutation) e.g. ('F1', 'E11D') or ('F1', 'WT').
    """
    # The fragment and mutation info is in the last '__'-delimited segment
    parts = design_name.split("__")
    last = parts[-1].strip()  # e.g. "F1_E11D" or "F4_WT"
    m = re.match(r"(F\d+)_(.+)", last)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse design_name: {design_name}")


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

    Returns (assembled_dna, oh_errors).
    assembled_dna includes the flanking vector OHs on each end.
    """
    oh_errors = []

    f1, f2, f3, f4 = inserts["F1"], inserts["F2"], inserts["F3"], inserts["F4"]

    # Extract sticky ends
    f1_right_oh = f1[-OH_LEN:]
    f2_left_oh  = f2[:OH_LEN]
    f2_right_oh = f2[-OH_LEN:]
    f3_left_oh  = f3[:OH_LEN]
    f3_right_oh = f3[-OH_LEN:]
    f4_left_oh  = f4[:OH_LEN]

    # Check F1-F2 junction
    if f1_right_oh != f2_left_oh:
        oh_errors.append(
            f"F1 right OH ({f1_right_oh}) != F2 left OH ({f2_left_oh}); "
            f"expected {EXPECTED_OHS['F1_F2']}"
        )
    elif f1_right_oh != EXPECTED_OHS["F1_F2"]:
        oh_errors.append(
            f"F1-F2 junction OH is {f1_right_oh}, expected {EXPECTED_OHS['F1_F2']}"
        )

    # Check F2-F3 junction
    if f2_right_oh != f3_left_oh:
        oh_errors.append(
            f"F2 right OH ({f2_right_oh}) != F3 left OH ({f3_left_oh}); "
            f"expected {EXPECTED_OHS['F2_F3']}"
        )
    elif f2_right_oh != EXPECTED_OHS["F2_F3"]:
        oh_errors.append(
            f"F2-F3 junction OH is {f2_right_oh}, expected {EXPECTED_OHS['F2_F3']}"
        )

    # Check F3-F4 junction
    if f3_right_oh != f4_left_oh:
        oh_errors.append(
            f"F3 right OH ({f3_right_oh}) != F4 left OH ({f4_left_oh}); "
            f"expected {EXPECTED_OHS['F3_F4']}"
        )
    elif f3_right_oh != EXPECTED_OHS["F3_F4"]:
        oh_errors.append(
            f"F3-F4 junction OH is {f3_right_oh}, expected {EXPECTED_OHS['F3_F4']}"
        )

    # Concatenate: full F1 + F2 without left OH + F3 without left OH + F4 without left OH
    assembled = f1 + f2[OH_LEN:] + f3[OH_LEN:] + f4[OH_LEN:]

    return assembled, oh_errors


def parse_mutation_name(mutation_str):
    """
    Parse a mutation string like 'E11D' into (wt_aa, position_1indexed, mut_aa).
    Returns None if it cannot be parsed.
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
    print("CSV-Based Assembly Validation for SSM Fragment Oligos")
    print(f"Source: {CSV_PATH}")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # 1. Read CSV and extract all 291 DNA sequences
    # ------------------------------------------------------------------
    rows = []
    with open(CSV_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows from CSV.")
    print()

    # ------------------------------------------------------------------
    # 2. Parse and group by fragment
    # ------------------------------------------------------------------
    wt_inserts = {}          # frag_group -> insert sequence (uppercase)
    mutant_frags = defaultdict(list)  # frag_group -> [(mutation, insert_seq), ...]

    for row in rows:
        design_name = row["design_name"]
        seq = row["sequence"].strip().upper()
        fg, mut = parse_design_name(design_name)

        if mut == "WT":
            wt_inserts[fg] = seq
        else:
            mutant_frags[fg].append((mut, seq))

    print(f"WT fragments found: {sorted(wt_inserts.keys())}")
    for fg in ["F1", "F2", "F3", "F4"]:
        n = len(mutant_frags.get(fg, []))
        print(f"  {fg}: {n} mutant fragments")
    total_mutants = sum(len(v) for v in mutant_frags.values())
    print(f"Total mutant fragments: {total_mutants}")
    print(f"Total sequences: {len(rows)}  (expected 291 = 287 mutants + 4 WT)")
    print()

    # ------------------------------------------------------------------
    # 3. Display insert structure for each WT fragment
    # ------------------------------------------------------------------
    print("-" * 80)
    print("STEP 1: Verify WT-only assembly")
    print("-" * 80)

    for fg in ["F1", "F2", "F3", "F4"]:
        ins = wt_inserts[fg]
        print(f"  {fg} WT insert length: {len(ins)} bp")
        print(f"    Left OH  (first 4bp): {ins[:OH_LEN]}")
        print(f"    Right OH (last  4bp): {ins[-OH_LEN:]}")

    # Assemble WT
    assembled_wt, wt_oh_errors = assemble_fragments(wt_inserts)
    if wt_oh_errors:
        print(f"  WT ASSEMBLY OH ERRORS: {wt_oh_errors}")
    else:
        print("  Sticky ends: ALL MATCH")

    # Strip flanking vector OHs to get coding DNA
    wt_coding = assembled_wt[OH_LEN:-OH_LEN]
    wt_protein = translate(wt_coding)

    print(f"  Assembled coding DNA length: {len(wt_coding)} bp")
    print(f"  Translated protein length:   {len(wt_protein)} AA")
    print(f"  Translated protein: {wt_protein}")

    wt_pass = (wt_protein == WT_PROTEIN)
    if wt_pass:
        print("  >>> WT ASSEMBLY: PASS (protein matches expected WT)")
    else:
        print("  >>> WT ASSEMBLY: FAIL")
        for i, (a, b) in enumerate(zip(wt_protein, WT_PROTEIN)):
            if a != b:
                print(f"      Position {i+1}: got {a}, expected {b}")
        if len(wt_protein) != len(WT_PROTEIN):
            print(f"      Length mismatch: got {len(wt_protein)}, expected {len(WT_PROTEIN)}")
    print()

    # ------------------------------------------------------------------
    # 4. Validate all 287 single-mutation assemblies
    # ------------------------------------------------------------------
    print("-" * 80)
    print("STEP 2: Validate all single-mutation assemblies")
    print("-" * 80)
    print()

    pass_count = 0
    fail_count = 0
    failures = []  # (frag_group, mutation, [error_strings])

    for fg in ["F1", "F2", "F3", "F4"]:
        fg_pass = 0
        fg_fail = 0
        fg_failures = []

        for mutation, mut_seq in mutant_frags[fg]:
            errors = []

            # Build 4-fragment set: mutant replaces its frag_group, others WT
            inserts = {}
            for frag in ["F1", "F2", "F3", "F4"]:
                if frag == fg:
                    inserts[frag] = mut_seq
                else:
                    inserts[frag] = wt_inserts[frag]

            # a) Verify sticky ends match at junctions
            assembled, oh_errors = assemble_fragments(inserts)
            if oh_errors:
                errors.extend(oh_errors)

            # b) Get coding DNA (strip vector OHs)
            coding_dna = assembled[OH_LEN:-OH_LEN]

            # c) Translate
            protein = translate(coding_dna)

            # d) Verify protein is exactly 198 AA
            if len(protein) != 198:
                errors.append(
                    f"Protein length {len(protein)} != expected 198"
                )

            if len(protein) != len(WT_PROTEIN):
                errors.append(
                    f"Protein length {len(protein)} != WT length {len(WT_PROTEIN)}"
                )

            # e) Find AA differences from WT
            diffs = []
            for i in range(min(len(protein), len(WT_PROTEIN))):
                if protein[i] != WT_PROTEIN[i]:
                    diffs.append((i + 1, WT_PROTEIN[i], protein[i]))

            # f) Parse the expected mutation from name
            parsed = parse_mutation_name(mutation)
            if parsed is None:
                errors.append(f"Could not parse mutation name: {mutation}")
            else:
                exp_wt_aa, exp_pos, exp_mut_aa = parsed

                # g) Verify exactly 1 AA difference
                if len(diffs) != 1:
                    errors.append(
                        f"Expected exactly 1 AA difference, found {len(diffs)}: {diffs}"
                    )
                else:
                    pos, wt_aa, mut_aa = diffs[0]

                    # h) Verify position matches
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
                fg_fail += 1
                failures.append((fg, mutation, errors))
                fg_failures.append((mutation, errors))
            else:
                pass_count += 1
                fg_pass += 1

        # Per-fragment-group results
        print(f"  {fg}: {fg_pass}/{fg_pass + fg_fail} passed, {fg_fail} failed")
        if fg_failures:
            for mutation, errs in fg_failures:
                print(f"    FAIL {mutation}:")
                for e in errs:
                    print(f"      - {e}")

    print()

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print("-" * 80)
    print("RESULTS SUMMARY")
    print("-" * 80)
    print()
    print(f"Total sequences in CSV:                {len(rows)}")
    print(f"  WT fragments:                        {len(wt_inserts)}")
    print(f"  Mutant fragments:                    {total_mutants}")
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
        print("All 287 single-mutation assemblies passed validation!")
        print()

    # Per-fragment-group summary
    print("Per-fragment-group breakdown:")
    for fg in ["F1", "F2", "F3", "F4"]:
        n_mut = len(mutant_frags[fg])
        n_fail = sum(1 for f, m, e in failures if f == fg)
        n_pass = n_mut - n_fail
        status = "PASS" if n_fail == 0 else "FAIL"
        print(f"  {fg}: {n_pass}/{n_mut} passed  [{status}]")
    print()

    # WT verdict
    wt_verdict = "PASS" if (wt_pass and not wt_oh_errors) else "FAIL"
    print(f"WT assembly: {wt_verdict}")
    print()

    # Final verdict
    if fail_count == 0 and wt_pass and not wt_oh_errors:
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
