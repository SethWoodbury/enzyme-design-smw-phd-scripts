#!/usr/bin/env python3
"""
Golden Gate Assembly Utilities for SSM Fragment Design (IPD Blocks compatible)

Generates BsaI-compatible DNA fragments for site-saturation mutagenesis
combinatorial libraries on Twist chips. Uses a WT DNA template and only swaps
codons at mutated positions, avoiding full reverse translation.

Compatible with:
  - CF_1 vector (BsaI, 5'-sticky=AGGA, 3'-sticky=GGTTCC → 4nt OH=TTCC)
  - IPD Blocks chip ordering (oopzs primer system, 300bp max oligo)
  - BsaI-HFv2 fidelity matrix for overhang validation

Oligo structure on chip (5'->3'):
  fw_primer(20bp) + GGTCTC + a + [insert] + c + GAGACC + RC(rv_primer)(20bp)

Insert structure:
  [OH_left][coding_DNA][OH_right]
  Where OH regions are 4nt at internal junctions (from coding sequence),
  or vector sticky ends at the first/last fragment boundaries.

Author: Seth M. Woodbury (with Claude Code assistance)
"""

import re
import csv
from collections import OrderedDict

# --- locate repo root + shared external paths ---
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "repo_paths.py").is_file():
        _sys.path.insert(0, str(_anc)); break
import repo_paths

# ============================================================================
# Constants
# ============================================================================

BSAI_FWD = "GGTCTC"
BSAI_REV = "GAGACC"
FW_ADAPTER = "ggtctca"   # BsaI_fwd + 1nt spacer 'a'

# ============================================================================
# Vector Registry (from https://lab.ipd.uw.edu/ipdblocks/backbones)
# ============================================================================

VECTOR_REGISTRY = {
    # BsaI vectors with standard stickies (agga / ggttcc)
    "LM0627":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "GSGSHHWGSTHHHHHH",  "desc": "C-term SNAC-His",              "bioregistry": 14663},
    "LM0670":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "",           "cterm": "GSHHHHHH",           "desc": "C-term His",                   "bioregistry": 14664},
    "LM1369":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "GSAWSHPQFEK",        "desc": "C-term StrepII tag",           "bioregistry": 14658},
    "LM1371":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MHHHHHHSG",  "cterm": "",                   "desc": "N-term His",                   "bioregistry": 42924},
    "LM1425":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "GSGSGLNDIFEAQKIEWHEHHHHHH", "desc": "C-term AviTag + His",  "bioregistry": 60080},
    "NMB001":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "GSGSHHWGSTHHHHHHSRLEEELRRRLTE", "desc": "C-term SNAC-His-ALFA", "bioregistry": 14671},
    "pCFCB20": {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "",           "cterm": "",                   "desc": "Cell-free, no tag",            "bioregistry": 14660},
    "pCFCB21": {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "",           "cterm": "",                   "desc": "Cell-free + mScarlet-i3",      "bioregistry": 14661},
    "CF_1":    {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSKIKSGGSG", "cterm": "",                   "desc": "Cell-free combinatorial + mScarlet", "bioregistry": 122896},
    "CF_2":    {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "",           "cterm": "",                   "desc": "pETCON adapters + cell-free",  "bioregistry": 42922},
    "pCOOL1":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "",                   "desc": "C-term SNAC-mScarlet-His",     "bioregistry": 60081},
    "FP0016":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "",           "cterm": "",                   "desc": "N-term His + Trp + TEV",       "bioregistry": 110563},
    "GB1":     {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "",           "cterm": "",                   "desc": "N-term His + GB1 solubility",  "bioregistry": 122895},
    "AB0010":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "",                   "desc": "C-term mStayGold-His",         "bioregistry": 136057},
    "AB0011":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "",                   "desc": "C-term mScarlet3-His",         "bioregistry": 136058},
    "AS0064":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "",           "cterm": "",                   "desc": "Cell-free + Electra2",         "bioregistry": 234645},
    "pMSCH1":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "GSGSHHWGSTHHHHHH",  "desc": "C-term SNAC-His",              "bioregistry": 268539},
    "HK005":   {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "",           "cterm": "",                   "desc": "N-term Secrecon, C-term Fc+His", "bioregistry": 268540},
    "MDL002":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "",                   "desc": "C-term linker1-I53-50A",       "bioregistry": 314678},
    "MDL003":  {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "",                   "desc": "C-term linker2-I53-50A",       "bioregistry": 314679},
    "CK943":   {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "MSG",        "cterm": "",                   "desc": "C-term GB1-Avitag-His",        "bioregistry": 314703},
    "JW_FRA_0009": {"enzyme": "BsaI", "5_sticky": "agga", "3_sticky": "ggttcc", "nterm": "",       "cterm": "",                   "desc": "Cell-free + SpyCatcher",       "bioregistry": 14662},
    # BsaI vectors with NON-STANDARD stickies
    "ZBJ101":  {"enzyme": "BsaI", "5_sticky": "AGGTGCCACC", "3_sticky": "TAATAGCCTA", "nterm": "", "cterm": "",  "desc": "Mammalian CMV, user provides ATG", "bioregistry": 42923},
    "E33":     {"enzyme": "BsaI", "5_sticky": "tatg", "3_sticky": "tagt", "nterm": "",             "cterm": "",  "desc": "Yeast TRP1 promoter",              "bioregistry": 85016},
    "E37":     {"enzyme": "BsaI", "5_sticky": "tatg", "3_sticky": "tagt", "nterm": "",             "cterm": "",  "desc": "Yeast His3 promoter",              "bioregistry": 136059},
    "GG3.2":   {"enzyme": "BsaI", "5_sticky": "aggttgccacc", "3_sticky": "taaggccta", "nterm": "", "cterm": "",  "desc": "Mammalian production, no tags",    "bioregistry": 294842},
    "P142-WC": {"enzyme": "BsaI", "5_sticky": "catg", "3_sticky": "ggat", "nterm": "",             "cterm": "",  "desc": "Mammalian GCN4-VP48-mCherry",      "bioregistry": 314704},
    # Non-BsaI vectors
    "CX1":     {"enzyme": "BbsI", "5_sticky": "atcc", "3_sticky": "ggcg", "nterm": "",             "cterm": "",  "desc": "Mammalian Bxb1 landing pad",       "bioregistry": 14659},
    "CX_attBbxb1": {"enzyme": "BbsI", "5_sticky": "cgaa", "3_sticky": "gcgg", "nterm": "",         "cterm": "",  "desc": "Mammalian Bxb1 landing pad",       "bioregistry": 251936},
    "MA0006":  {"enzyme": "SapI", "5_sticky": "aaa", "3_sticky": "tga", "nterm": "",               "cterm": "",  "desc": "SapI vector",                      "bioregistry": 268547},
}


def get_vector(name):
    """
    Look up a vector by name from the registry.
    Returns a dict with keys: enzyme, 5_sticky, 3_sticky, nterm, cterm, desc, bioregistry.
    Raises KeyError if not found.
    """
    name_upper = name.upper().replace(" ", "")
    # Try exact match first
    if name in VECTOR_REGISTRY:
        return VECTOR_REGISTRY[name].copy()
    # Try case-insensitive
    for k, v in VECTOR_REGISTRY.items():
        if k.upper().replace(" ", "") == name_upper:
            return v.copy()
    raise KeyError(f"Vector '{name}' not found in registry. "
                   f"Available: {', '.join(sorted(VECTOR_REGISTRY.keys()))}")


def list_vectors(enzyme_filter=None):
    """Print all available vectors, optionally filtered by enzyme."""
    print(f"{'Name':<15s} {'Enzyme':<6s} {'5-sticky':<12s} {'3-sticky':<12s} {'Description'}")
    print("-" * 80)
    for name, v in VECTOR_REGISTRY.items():
        if enzyme_filter and v['enzyme'] != enzyme_filter:
            continue
        print(f"{name:<15s} {v['enzyme']:<6s} {v['5_sticky']:<12s} {v['3_sticky']:<12s} {v['desc']}")
RV_ADAPTER = "cgagacc"   # 1nt spacer 'c' + BsaI_rev

AVOID_SEQUENCES = [
    "GGTCTC", "GAGACC",     # BsaI sites
    "GGAGG",                 # E. coli Shine-Dalgarno
    "TAAGGAG",               # strong RBS
    "GCTGGTGG",              # Chi site
    "AAAAA", "TTTTT",       # polyA/T terminators
    "CCCCCC", "GGGGGG",     # polyC/G runs
]

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

ECOLI_PREFERRED_CODONS = {
    'A': ['GCG', 'GCC', 'GCA', 'GCT'],
    'R': ['CGT', 'CGC', 'CGG', 'CGA', 'AGG', 'AGA'],
    'N': ['AAC', 'AAT'],
    'D': ['GAT', 'GAC'],
    'C': ['TGC', 'TGT'],
    'E': ['GAA', 'GAG'],
    'Q': ['CAG', 'CAA'],
    'G': ['GGC', 'GGT', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'K': ['AAA', 'AAG'],
    'L': ['CTG', 'CTT', 'CTC', 'CTA', 'TTG', 'TTA'],
    'M': ['ATG'],
    'F': ['TTC', 'TTT'],
    'P': ['CCG', 'CCA', 'CCT', 'CCC'],
    'S': ['AGC', 'TCT', 'TCC', 'TCG', 'TCA', 'AGT'],
    'T': ['ACC', 'ACG', 'ACT', 'ACA'],
    'W': ['TGG'],
    'V': ['GTG', 'GTT', 'GTC', 'GTA'],
    'Y': ['TAT', 'TAC'],
}


# ============================================================================
# Basic utilities
# ============================================================================

def translate(dna_seq):
    """Translate a DNA sequence to protein."""
    protein = []
    seq = dna_seq.upper()
    if len(seq) % 3 != 0:
        print(f"  WARNING: sequence length {len(seq)} is not a multiple of 3")
    for i in range(0, len(seq) - 2, 3):
        aa = CODON_TABLE.get(seq[i:i+3], '?')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    return ''.join(comp.get(c, c) for c in reversed(seq))


def has_bsai_site(dna_seq):
    """Return positions of BsaI recognition sites in a DNA sequence."""
    s = dna_seq.upper()
    return ([m.start() for m in re.finditer(BSAI_FWD, s)] +
            [m.start() for m in re.finditer(BSAI_REV, s)])


def check_avoid_sequences(dna_seq):
    """Return list of (pattern, position) for problematic sequences found."""
    issues = []
    s = dna_seq.upper()
    for pattern in AVOID_SEQUENCES:
        for m in re.finditer(pattern, s):
            issues.append((pattern, m.start()))
    return issues


def is_palindrome(seq):
    """Check if a 4nt overhang is self-complementary (palindromic)."""
    return seq.upper() == reverse_complement(seq).upper()


def get_overhang(coding_dna, cut_after_aa):
    """Get the 4nt junction overhang when cutting after a given AA position."""
    nt_pos = cut_after_aa * 3
    return coding_dna[nt_pos:nt_pos + 4].upper()


def find_hairpins(dna_seq, min_stem=8):
    """
    Find potential hairpin structures in a DNA sequence.
    Returns list of (pos1, pos2, stem_len, loop_len, stem_seq) tuples.
    """
    s = dna_seq.upper()
    rc = reverse_complement(s).upper()
    n = len(s)
    hairpins = []
    for stem_len in range(min_stem, n // 2 + 1):
        for i in range(n - stem_len + 1):
            kmer = s[i:i + stem_len]
            kmer_rc = reverse_complement(kmer).upper()
            # Search for the RC downstream (with a loop of at least 3nt)
            search_start = i + stem_len + 3
            j = s.find(kmer_rc, search_start)
            while j != -1:
                loop_len = j - (i + stem_len)
                hairpins.append((i, j, stem_len, loop_len, kmer))
                j = s.find(kmer_rc, j + 1)
        if hairpins:
            break  # report only the shortest stem found
    return hairpins


def check_primer_overlaps(dna_seq, primers, min_overlap=15):
    """
    Check for primer sequence overlaps within a DNA insert.
    Returns list of (primer_name, overlap_len, position) for matches >= min_overlap.
    """
    s = dna_seq.upper()
    issues = []
    for name, primer in primers.items():
        p = primer.upper()
        p_rc = reverse_complement(p).upper()
        # Check all substrings of length min_overlap to len(primer)
        for length in range(len(p), min_overlap - 1, -1):
            for start in range(len(p) - length + 1):
                sub = p[start:start + length]
                if sub in s:
                    issues.append((name, length, s.index(sub)))
                    break
                sub_rc = p_rc[start:start + length]
                if sub_rc in s:
                    issues.append((f"{name}_RC", length, s.index(sub_rc)))
                    break
            else:
                continue
            break  # found a match for this primer, move to next
    return issues


# ============================================================================
# Fidelity checking
# ============================================================================

def check_overhang_fidelity(overhangs, fidelity_df, max_off_target=0):
    """
    Check all overhangs + their RCs for pairwise off-target ligation.
    Returns list of (oh_a, oh_b, score) for violations.
    """
    all_seqs = list(overhangs) + [reverse_complement(oh).upper() for oh in overhangs]
    issues = []
    for i in range(len(all_seqs)):
        for j in range(i + 1, len(all_seqs)):
            a, b = all_seqs[i].upper(), all_seqs[j].upper()
            if a == reverse_complement(b).upper():
                continue  # intended pair
            if a not in fidelity_df.index or b not in fidelity_df.columns:
                issues.append((a, b, -1))  # missing from matrix
                continue
            val = fidelity_df.loc[a, b]
            if val > max_off_target:
                issues.append((a, b, int(val)))
    return issues


def compute_gga_efficiency(overhangs, fidelity_df):
    """
    Compute estimated Golden Gate Assembly efficiency for a set of overhangs.
    Returns (list of (overhang, efficiency_pct), overall_efficiency_pct).
    """
    all_ohs = [oh.upper() for oh in overhangs]
    all_rcs = [reverse_complement(oh).upper() for oh in overhangs]
    pool = list(set(all_ohs + all_rcs))

    junction_effs = []
    for oh in all_ohs:
        rc = reverse_complement(oh).upper()
        try:
            intended = float(fidelity_df.loc[oh, rc])
        except KeyError:
            junction_effs.append((oh, 0.0))
            continue
        if intended == 0:
            junction_effs.append((oh, 0.0))
            continue
        total = sum(float(fidelity_df.loc[oh, p]) for p in pool
                    if oh in fidelity_df.index and p in fidelity_df.columns)
        junction_effs.append((oh, (intended / total * 100) if total else 0.0))

    overall = 1.0
    for _, eff in junction_effs:
        overall *= (eff / 100.0)
    overall *= 100.0
    return junction_effs, overall


def print_gga_efficiency(overhangs, fidelity_df):
    """Compute and print GGA efficiency estimate."""
    junction_effs, overall = compute_gga_efficiency(overhangs, fidelity_df)
    print("\n  --- GGA Efficiency Estimate ---")
    for oh, eff in junction_effs:
        rc = reverse_complement(oh).upper()
        print(f"    Junction {oh} x {rc}: {eff:.1f}% correct")
    print(f"    Overall assembly efficiency: {overall:.1f}%")
    if overall >= 95:
        print(f"    Assessment: EXCELLENT (>= 95%)")
    elif overall >= 85:
        print(f"    Assessment: GOOD (>= 85%)")
    elif overall >= 70:
        print(f"    Assessment: ACCEPTABLE (>= 70%)")
    else:
        print(f"    Assessment: LOW (< 70%) -- consider different overhangs")
    return overall


# ============================================================================
# Codon selection
# ============================================================================

def pick_codon(target_aa, context_left, context_right, wt_codon=None):
    """
    Pick the best E. coli codon for target_aa that doesn't create avoid
    sequences when placed in the given context.
    """
    max_ctx = max(len(p) for p in AVOID_SEQUENCES) - 1  # = 7
    candidates = ECOLI_PREFERRED_CODONS.get(target_aa.upper())
    if not candidates:
        raise ValueError(f"Unknown amino acid: {target_aa}")

    # If WT codon already encodes this AA and is safe, use it
    if wt_codon and CODON_TABLE.get(wt_codon.upper()) == target_aa.upper():
        test = (context_left[-max_ctx:] + wt_codon + context_right[:max_ctx]).upper()
        if not check_avoid_sequences(test):
            return wt_codon.upper()

    # Try codons in E. coli preference order
    for codon in candidates:
        test = (context_left[-max_ctx:] + codon + context_right[:max_ctx]).upper()
        if not check_avoid_sequences(test):
            return codon

    # Fallback: just avoid BsaI sites (relax other constraints)
    for codon in candidates:
        test = (context_left[-max_ctx:] + codon + context_right[:max_ctx]).upper()
        if not has_bsai_site(test):
            return codon

    raise ValueError(f"All codons for {target_aa} create BsaI sites in context")


# ============================================================================
# Fragment construction
# ============================================================================

def _compute_insert(coding_dna, cuts, frag_index, n_frags,
                    vec_oh_5, vec_oh_3_full):
    """
    Compute the INSERT region for a fragment (between BsaI adapters).

    The insert is the DNA that remains after BsaI digestion:
      [OH_left] + [unique_coding] + [OH_right]

    For internal junctions, OH comes from the coding DNA (shared between
    adjacent fragments — each fragment contains the 4nt OH, but after
    ligation only one copy exists in the final product).

    Returns:
        insert (str), oh_left (str), oh_right (str)
    """
    n_aa = len(coding_dna) // 3

    # Determine coding DNA slice boundaries
    if frag_index == 0:
        # First fragment: starts at nt 0
        coding_start = 0
    else:
        # Non-first: starts at the left junction OH (= end of previous fragment's coding)
        coding_start = cuts[frag_index - 1] * 3

    if frag_index < n_frags - 1:
        # Non-last: ends at the right junction OH (includes 4nt into next fragment)
        coding_end = cuts[frag_index] * 3 + 4
    else:
        # Last fragment: ends at the end of coding
        coding_end = n_aa * 3

    frag_slice = coding_dna[coding_start:coding_end].upper()

    # Add non-coding vector stickies
    if frag_index == 0:
        insert = vec_oh_5.upper() + frag_slice
        oh_left = vec_oh_5.upper()
    else:
        insert = frag_slice
        oh_left = frag_slice[:4]

    if frag_index == n_frags - 1:
        insert = insert + vec_oh_3_full.upper()
        oh_right = vec_oh_3_full.upper()
    else:
        oh_right = frag_slice[-4:]

    return insert, oh_left, oh_right


def build_fragment_dna(coding_dna, cuts, frag_index, n_frags,
                       vec_oh_5, vec_oh_3_full):
    """
    Build the ordered DNA for one Golden Gate fragment (without primers).
    Structure: FW_ADAPTER + [insert] + RV_ADAPTER
    """
    insert, _, _ = _compute_insert(coding_dna, cuts, frag_index, n_frags,
                                   vec_oh_5, vec_oh_3_full)
    return (FW_ADAPTER + insert + RV_ADAPTER).upper()


def add_primers(fragment_dna, fw_primer_seq, rv_primer_seq):
    """
    Add amplification primers to a fragment for chip ordering.

    Args:
        fragment_dna: the fragment (FW_ADAPTER + insert + RV_ADAPTER)
        fw_primer_seq: forward primer sequence (appears as-is at 5' of oligo)
        rv_primer_seq: reverse primer sequence (its RC appears at 3' of oligo)

    The chip oligo is single-stranded. For PCR:
      - fw_primer anneals to the bottom strand → its sequence is at the 5' end of the oligo
      - rv_primer anneals to the top strand (the oligo itself) → RC(rv_primer) is at the 3' end
    """
    return (fw_primer_seq + fragment_dna + reverse_complement(rv_primer_seq)).upper()


def _codon_offset_in_insert(coding_dna, cuts, frag_index, n_frags,
                            vec_oh_5, mut_aa_pos):
    """Compute the offset of a mutation's codon within the fragment insert."""
    if frag_index == 0:
        coding_start = 0
        prefix_len = len(vec_oh_5)
    else:
        coding_start = cuts[frag_index - 1] * 3
        prefix_len = 0  # no non-coding prefix; coding starts at position 0

    codon_in_coding = (mut_aa_pos - 1) * 3  # 0-indexed position in full coding DNA
    return prefix_len + (codon_in_coding - coding_start)


def swap_codon_in_fragment(wt_frag_no_primers, coding_dna, cuts, frag_index,
                           n_frags, vec_oh_5, mut_aa_pos, mut_aa):
    """
    Create a mutant variant by swapping one codon in a fragment.
    Input: fragment WITHOUT primers (FW_ADAPTER + insert + RV_ADAPTER).
    """
    ap = FW_ADAPTER.upper()
    asp = RV_ADAPTER.upper()
    insert = wt_frag_no_primers[len(ap):-len(asp)]

    offset = _codon_offset_in_insert(coding_dna, cuts, frag_index, n_frags,
                                     vec_oh_5, mut_aa_pos)

    context_left = insert[:offset]
    wt_codon = insert[offset:offset + 3]
    context_right = insert[offset + 3:]

    new_codon = pick_codon(mut_aa, context_left, context_right, wt_codon)
    mutant_insert = insert[:offset] + new_codon + insert[offset + 3:]
    return ap + mutant_insert + asp


# ============================================================================
# Main generation pipeline
# ============================================================================

def check_mutation_in_junction(cuts, mut_aa_pos):
    """
    Check if a mutation falls within a junction overhang region (4nt = ~1.3 codons).
    Returns the problematic cut position, or None if safe.
    """
    for cut in cuts:
        oh_nt_start = cut * 3
        oh_nt_end = oh_nt_start + 4
        codon_start = (mut_aa_pos - 1) * 3
        codon_end = codon_start + 3
        if codon_start < oh_nt_end and codon_end > oh_nt_start:
            return cut
    return None


def generate_all_fragments(coding_dna, cuts, mutations_by_position,
                           vec_oh_5="AGGA", vec_oh_3_full="GGTTCC",
                           fw_primers=None, rv_primers=None,
                           output_mode="insert_only"):
    """
    Generate all WT and mutant fragment DNA sequences.

    Args:
        coding_dna: full WT coding DNA (uppercase, no adapters)
        cuts: list of cut-after-AA positions (sorted)
        mutations_by_position: dict {aa_pos: [(mutation_name, target_aa), ...]}
        vec_oh_5: 5' vector overhang (4nt, e.g. 'AGGA' for CF_1)
        vec_oh_3_full: full 3' vector sticky (6nt, e.g. 'GGTTCC' for CF_1)
        fw_primers: list of forward primer SEQUENCES per fragment (or None)
        rv_primers: list of reverse primer SEQUENCES per fragment (or None)
        output_mode: "ipdblocks" (default) = adapter + insert + adapter
                       for IPD Blocks with "None" backbone (system adds primers + buffer)
                     "insert_only" = just [OH_left][coding][OH_right]
                       bare insert, no adapters or primers
                     "full_oligo" = primer + adapter + insert + adapter + primer
                       for manual chip ordering (you provide everything)

    Returns:
        OrderedDict of {name: dna_sequence},
        list of fragment_info dicts
    """
    if output_mode not in ("ipdblocks", "insert_only", "full_oligo"):
        raise ValueError(f"output_mode must be 'ipdblocks', 'insert_only', or 'full_oligo', got '{output_mode}'")

    if output_mode == "full_oligo" and not (fw_primers and rv_primers):
        raise ValueError("full_oligo mode requires fw_primers and rv_primers")

    n_aa = len(coding_dna) // 3
    boundaries = [0] + list(cuts) + [n_aa]
    n_frags = len(boundaries) - 1

    results = OrderedDict()
    assembly_info = []

    for fi in range(n_frags):
        frag_start = boundaries[fi] + 1
        frag_end = boundaries[fi + 1]
        frag_label = f"F{fi + 1}"

        # Build WT fragment (with BsaI adapters, no primers)
        wt_frag = build_fragment_dna(coding_dna, cuts, fi, n_frags,
                                     vec_oh_5, vec_oh_3_full)
        insert, oh_left, oh_right = _compute_insert(coding_dna, cuts, fi, n_frags,
                                                    vec_oh_5, vec_oh_3_full)

        # Determine what to store based on output_mode
        if output_mode == "ipdblocks":
            wt_ordered = wt_frag.upper()  # adapter + insert + adapter (system adds primers)
        elif output_mode == "insert_only":
            wt_ordered = insert.upper()   # just [OH_left][coding][OH_right]
        elif output_mode == "full_oligo":
            wt_ordered = add_primers(wt_frag, fw_primers[fi], rv_primers[fi])

        results[f"{frag_label}_WT"] = wt_ordered

        # Find mutations in this fragment (exclude junction-overlapping ones)
        frag_mutations = {}
        junction_warnings = []
        for pos in range(frag_start, frag_end + 1):
            if pos not in mutations_by_position:
                continue
            conflict = check_mutation_in_junction(cuts, pos)
            if conflict is not None:
                for mut_name, _ in mutations_by_position[pos]:
                    junction_warnings.append(
                        f"{mut_name} at pos {pos} overlaps junction OH at cut {conflict}")
            else:
                frag_mutations[pos] = mutations_by_position[pos]

        info = {
            'fragment': frag_label,
            'aa_range': f"{frag_start}-{frag_end}",
            'size_aa': frag_end - frag_start + 1,
            'oh_left': oh_left,
            'oh_right': oh_right,
            'n_mutation_sites': len(frag_mutations),
            'n_variants': sum(len(v) for v in frag_mutations.values()),
            'ordered_dna_len': len(wt_ordered),
            'output_mode': output_mode,
            'junction_warnings': junction_warnings,
        }
        assembly_info.append(info)

        # Generate mutant variants
        for pos in sorted(frag_mutations.keys()):
            for mut_name, target_aa in frag_mutations[pos]:
                mut_frag = swap_codon_in_fragment(
                    wt_frag, coding_dna, cuts, fi, n_frags, vec_oh_5,
                    pos, target_aa
                )
                if output_mode == "ipdblocks":
                    mut_ordered = mut_frag.upper()  # adapter + insert + adapter
                elif output_mode == "insert_only":
                    ap = FW_ADAPTER.upper()
                    asp = RV_ADAPTER.upper()
                    mut_ordered = mut_frag[len(ap):-len(asp)].upper()
                elif output_mode == "full_oligo":
                    mut_ordered = add_primers(mut_frag, fw_primers[fi], rv_primers[fi])
                else:
                    mut_ordered = mut_frag
                results[f"{frag_label}_{mut_name}"] = mut_ordered

    return results, assembly_info


# ============================================================================
# Validation
# ============================================================================

def validate_all(results, fidelity_df=None, overhangs=None,
                 max_oligo_len=300, primer_len=20, output_mode="insert_only",
                 primers=None, verbose=True):
    """
    Comprehensive validation:
      1. No internal BsaI sites in the insert
      2. Overhang fidelity
      3. Oligo/insert length within limit
      4. Avoid sequences (polyA/T, Shine-Dalgarno, Chi, BsaI)
      5. Hairpin detection (>= 8bp stem)
      6. Primer overlap check (>= 15bp match)
    """
    warnings_list = []
    hairpin_flags = []
    avoid_flags = []

    for name, dna in results.items():
        upper = dna.upper()

        # Get the insert region to check (strip adapters/primers, keep just insert)
        if output_mode == "insert_only":
            check_region = upper
        elif output_mode == "ipdblocks":
            check_region = upper[len(FW_ADAPTER):-len(RV_ADAPTER)]
        else:  # full_oligo
            adapter_overhead = primer_len + len(FW_ADAPTER)
            check_region = upper[adapter_overhead:-adapter_overhead]

        # 1. Length check
        if len(upper) > max_oligo_len:
            warnings_list.append(f"{name}: {len(upper)}bp > {max_oligo_len}bp limit")

        # 2. Internal BsaI check
        for pat in [BSAI_FWD, BSAI_REV]:
            sites = [m.start() for m in re.finditer(pat, check_region)]
            if sites:
                warnings_list.append(
                    f"{name}: internal {pat} at positions {sites}")

        # 3. Avoid sequence check
        avoids = check_avoid_sequences(check_region)
        if avoids:
            for pattern, pos in avoids:
                avoid_flags.append((name, pattern, pos))

        # 4. Hairpin check
        hairpins = find_hairpins(check_region, min_stem=8)
        if hairpins:
            for hp in hairpins:
                hairpin_flags.append((name, hp[2], hp[3], hp[4]))  # name, stem, loop, seq

        # 5. Primer overlap check
        if primers:
            overlaps = check_primer_overlaps(check_region, primers, min_overlap=15)
            for pname, length, pos in overlaps:
                warnings_list.append(
                    f"{name}: primer {pname} has {length}bp overlap at position {pos}")

    # 6. Fidelity check
    if fidelity_df is not None and overhangs is not None:
        for a, b, score in check_overhang_fidelity(overhangs, fidelity_df):
            warnings_list.append(f"Fidelity: {a} x {b} = {score} (must be 0)")

    # Print summary of non-blocking flags
    if verbose and (avoid_flags or hairpin_flags):
        print(f"\n  --- Non-blocking flags ---")
        if avoid_flags:
            from collections import Counter
            pattern_counts = Counter(pat for _, pat, _ in avoid_flags)
            print(f"  Avoid sequences: {len(avoid_flags)} flags in {len(set(n for n,_,_ in avoid_flags))} sequences")
            for pat, count in pattern_counts.most_common():
                examples = [n for n, p, _ in avoid_flags if p == pat][:3]
                print(f"    {pat}: {count} occurrences ({', '.join(examples)}{'...' if count > 3 else ''})")
        if hairpin_flags:
            print(f"  Hairpins (>= 8bp stem): {len(hairpin_flags)} flags in {len(set(n for n,_,_,_ in hairpin_flags))} sequences")
            for name, stem, loop, seq in hairpin_flags[:5]:
                print(f"    {name}: {stem}bp stem, {loop}bp loop ({seq})")

    return warnings_list


def verify_assembly(results, coding_dna, cuts, n_frags,
                    vec_oh_5, vec_oh_3_full, primer_len=20,
                    output_mode="insert_only"):
    """
    Verify that WT fragments, when assembled, reconstruct the original coding DNA.
    Handles both insert_only and full_oligo output modes.
    Returns (success: bool, message: str).
    """
    adapter_len = len(FW_ADAPTER)
    assembled = ""

    for fi in range(n_frags):
        wt_name = f"F{fi + 1}_WT"
        if wt_name not in results:
            return False, f"Missing {wt_name}"
        oligo = results[wt_name]

        # Get the insert depending on output mode
        if output_mode == "insert_only":
            # The stored sequence IS the insert: [OH_left][coding][OH_right]
            insert = oligo.upper()
        elif output_mode == "ipdblocks":
            # adapter + insert + adapter: strip adapters only
            insert = oligo[adapter_len:-adapter_len].upper()
        else:
            # full_oligo: strip primers + adapters
            if primer_len > 0:
                insert = oligo[primer_len + adapter_len:-(primer_len + adapter_len)]
            else:
                insert = oligo[adapter_len:-adapter_len]

        # Strip non-coding prefixes/suffixes
        if fi == 0:
            insert = insert[len(vec_oh_5):]
        if fi == n_frags - 1:
            insert = insert[:-len(vec_oh_3_full)]

        # For non-first fragments, skip left junction OH (already counted from prev frag)
        if fi > 0:
            insert = insert[4:]

        assembled += insert

    if assembled == coding_dna:
        return True, "WT fragments reconstruct original coding DNA"
    else:
        # Find first mismatch
        for i, (a, b) in enumerate(zip(assembled, coding_dna)):
            if a != b:
                return False, (f"Mismatch at nt {i}: got {a}, expected {b} "
                               f"(assembled {len(assembled)} nt, expected {len(coding_dna)} nt)")
        return False, f"Length mismatch: assembled {len(assembled)}, expected {len(coding_dna)}"


# ============================================================================
# Ordering instructions
# ============================================================================

def print_ordering_instructions(output_mode, n_sequences, max_len, fasta_path=None):
    """Print ordering instructions specific to the output mode."""
    print()
    print("=" * 70)
    print("  ORDERING INSTRUCTIONS")
    print("=" * 70)

    if output_mode == "ipdblocks":
        print(f"""
  Mode: ipdblocks (adapter + insert + adapter)

  Your sequences include BsaI adapters (ggtctca / cgagacc) flanking the
  insert. The IPD Blocks system will add only oopzs primers + buffer
  padding to reach the standard chip oligo length.

  IPD Blocks order form settings:
    Order Name:        [your order name]
    Vector/Backbone:   None          <-- NOT "None-BsaI"
    DNA Input:         CHECKED
    Deliver Source Plate: CHECKED
    Pooled Library:    unchecked

  Constraints:
    Max sequence length: 290 bp (yours: {max_len} bp -- {'OK' if max_len <= 290 else 'OVER LIMIT!'})
    Sequences:         {n_sequences}

  NOTE: Do NOT use "None-BsaI" — that is for long sequences (>290bp) where
  the system splits your DNA into fragments and adds its own BsaI sites.
  It would conflict with our custom junction design.

  After receiving the source plate:
    1. PCR amplify each fragment group using oopzs primers assigned by system
    2. Golden Gate assemble into your vector (e.g., LM1369) with BsaI-HFv2 + T4 ligase
    3. The BsaI adapters will be cut off, leaving your inserts with custom sticky ends""")

    elif output_mode == "insert_only":
        print(f"""
  Mode: insert_only (bare insert, no adapters or primers)

  Your sequences contain ONLY the insert: [OH_left][coding][OH_right]
  No BsaI adapters, no primers.

  *** WARNING: These are NOT ready for direct IPD Blocks submission! ***

  To submit to IPD Blocks, switch to output_mode="ipdblocks" which adds
  the required BsaI adapters. Or manually wrap each sequence:
    ggtctca + [your sequence] + cgagacc

  This mode is useful for:
    - Archival / reference (just the biological insert)
    - Feeding into other pipelines that add their own adapters
    - Cross-checking / debugging

  Sequences: {n_sequences}, length range up to {max_len} bp""")

    elif output_mode == "full_oligo":
        print(f"""
  Mode: full_oligo (primer + adapter + insert + adapter + primer)

  Your sequences include EVERYTHING: oopzs primers + BsaI adapters + insert.

  *** WARNING: Do NOT submit these to IPD Blocks! ***
  The IPD Blocks system would add ADDITIONAL primers on top of yours,
  resulting in double primers. These are for manual Twist chip ordering only.

  If you want to order through IPD Blocks, switch to output_mode="ipdblocks".

  This mode is useful for:
    - Ordering directly from Twist Bioscience (not through IPD Blocks)
    - Cases where you manage your own primer assignments
    - Backup if the IPD Blocks system doesn't work for your use case

  Sequences: {n_sequences}, length range up to {max_len} bp""")

    if fasta_path:
        print(f"\n  FASTA file: {fasta_path}")
    print("=" * 70)
    print()


VECTOR_FASTA_DIR = "/net/software/lab/johnbercow/entry_vectors/"

# Map vector names to their FASTA filenames on disk
VECTOR_FASTA_FILES = {
    "LM0627": "LM0627_MSG-AGGA-promoter-ccdb-TTCC-SNAC-his.fa",
    "LM1369": "LM1369_ccdb_strepii_ass.fa",
    "LM0670": "LM0670_MSG-AGGA-promoter-ccdb-TTCC-GS-his.fa",
    "LM1371": "LM1371_MSG-AGGA-promoter-ccdb-TTCC-his.fa",
    "CF_1":   "CF_1.fa",
    "pCOOL1": "pCOOL1.fa",
}


def load_vector_sequence(vector_name):
    """Load a vector's full plasmid sequence from the local FASTA database."""
    import os
    fname = VECTOR_FASTA_FILES.get(vector_name)
    if fname:
        path = os.path.join(VECTOR_FASTA_DIR, fname)
    else:
        # Try to find it by glob
        import glob
        matches = glob.glob(os.path.join(VECTOR_FASTA_DIR, f"*{vector_name}*"))
        if matches:
            path = matches[0]
        else:
            return None, None

    if not os.path.isfile(path):
        return None, path

    with open(path) as f:
        header, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    return ''.join(seq).upper(), path


def print_neb_verification_guide(results, coding_dna, vec_oh_5, vec_oh_3_full,
                                 cuts, output_mode="ipdblocks",
                                 vector_name=None):
    """Print instructions for manual verification using NEB's Golden Gate tool."""
    n_frags = len(cuts) + 1
    internal_ohs = [get_overhang(coding_dna, c) for c in cuts]
    oh_chain = " -> ".join([vec_oh_5] + internal_ohs + [vec_oh_3_full[-4:]])

    # Extract WT inserts and wrap with adapters for NEB tool input
    # NEB tool needs the full fragment WITH BsaI sites so it can simulate the digest
    wt_fragments = {}
    for fi in range(n_frags):
        name = f"F{fi+1}_WT"
        if name not in results:
            continue
        dna = results[name]
        if output_mode == "ipdblocks":
            # Already has adapters
            wt_fragments[f"F{fi+1}"] = dna.upper()
        elif output_mode == "insert_only":
            # Add adapters
            wt_fragments[f"F{fi+1}"] = (FW_ADAPTER + dna + RV_ADAPTER).upper()
        else:  # full_oligo — strip primers, keep adapters
            wt_fragments[f"F{fi+1}"] = dna[20:-20].upper()

    print()
    print("=" * 70)
    print("  OPTIONAL: NEB GOLDEN GATE TOOL VERIFICATION")
    print("=" * 70)
    print("""
  For an independent third-party check, use NEB's online tool:
    https://goldengate.neb.com/#!/

  Steps:
    1. Open the link above in Chrome
    2. Select enzyme: BsaI-HFv2 (or BsaI)
    3. Under "Destination Vector", paste your linearized vector sequence
       (LM1369 or whichever vector you are cloning into)
    4. Add fragments — paste each WT fragment below as a separate insert.
       These include the BsaI adapter sites so the tool can simulate digestion:
""")

    for frag_name, dna in wt_fragments.items():
        print(f"  --- {frag_name} ({len(dna)} bp, with BsaI adapters) ---")
        print(f"  {dna}")
        print()

    # Load and print vector sequence if available
    if vector_name:
        vec_seq, vec_path = load_vector_sequence(vector_name)
        if vec_seq:
            print(f"  --- Destination Vector: {vector_name} ({len(vec_seq)} bp) ---")
            print(f"  Source: {vec_path}")
            print(f"  {vec_seq}")
            print()
        else:
            print(f"  --- Destination Vector: {vector_name} ---")
            print(f"  Sequence not found on disk. Paste your vector manually.")
            print(f"  Tried: {vec_path or VECTOR_FASTA_DIR + '*' + vector_name + '*'}")
            print()

    print(f"""  5. Click "Assemble" or "Analyze"
  6. Expected result if everything is correct:
     - All {n_frags} fragments assemble in order: {' -> '.join(f'F{i+1}' for i in range(n_frags))}
     - Overhang chain: {oh_chain}
     - Zero off-target or mismatched ligations
     - No BsaI sites remaining in the assembled product
     - Assembled insert = {len(coding_dna)} bp coding DNA

  If the tool shows errors, compare against our computational validation
  which simulated all assemblies and confirmed correct proteins.
""")
    print("=" * 70)
    print()


# ============================================================================
# Well assignment & naming
# ============================================================================

def well_id(index, plate_format=384):
    """
    Convert a 0-indexed sequence number to a well ID (e.g., A1, A2, A3...).
    Fills row-first (A1, A2, A3, ...A24, B1, B2, ...) matching IPD Blocks ordering.
    """
    if plate_format == 384:
        n_rows, n_cols = 16, 24
        row_labels = "ABCDEFGHIJKLMNOP"
    elif plate_format == 96:
        n_rows, n_cols = 8, 12
        row_labels = "ABCDEFGH"
    else:
        raise ValueError(f"Unsupported plate format: {plate_format}")

    wells_per_plate = n_rows * n_cols
    plate_num = index // wells_per_plate + 1
    local_idx = index % wells_per_plate
    row = local_idx // n_cols
    col = local_idx % n_cols + 1
    return f"{row_labels[row]}{col}", plate_num


def build_order_table(results, n_frags, design_prefix="frag",
                      plate_format=384):
    """
    Build a sorted, well-assigned order table from generated fragments.

    Sorting order (Echo-friendly):
      1. F1 variants (sorted by position, then by AA)
      2. F2 variants
      3. F3 variants
      4. F4 variants
      5. WT fragments (at the end — can also be ordered as G-blocks)

    Returns:
        list of dicts with keys: seq_id, well, plate, frag_group, mutation,
        description, dna, oligo_len
    """
    # Separate variants and WT
    variants = []
    wt_entries = []
    for name, dna in results.items():
        frag_group = name.split("_")[0]  # e.g., "F1"
        mut_label = "_".join(name.split("_")[1:])  # e.g., "E38T" or "WT"
        is_wt = mut_label == "WT"

        # Parse position for sorting
        import re as _re
        m = _re.match(r"([A-Z])(\d+)([A-Z])", mut_label)
        sort_pos = int(m.group(2)) if m else 9999
        sort_aa = m.group(3) if m else ""

        entry = {
            'name': name,
            'frag_group': frag_group,
            'mutation': mut_label,
            'sort_key': (int(frag_group[1:]), sort_pos, sort_aa),
            'dna': dna,
            'oligo_len': len(dna),
        }
        if is_wt:
            wt_entries.append(entry)
        else:
            variants.append(entry)

    # Sort: variants by (frag_group, position, AA), then WT at end
    variants.sort(key=lambda e: e['sort_key'])
    wt_entries.sort(key=lambda e: e['sort_key'])
    all_entries = variants + wt_entries

    # Assign well IDs and sequence numbers
    table = []
    for idx, entry in enumerate(all_entries):
        well, plate = well_id(idx, plate_format)
        seq_num = f"{idx + 1:04d}"
        frag_g = entry['frag_group']
        mut = entry['mutation']

        # Build systematic name:
        # frag_0001__plate1__A1__384well__F1_E38T
        sys_name = (f"{design_prefix}_{seq_num}__plate{plate}__{well}"
                    f"__{plate_format}well__{frag_g}_{mut}")

        # Description for FASTA header
        desc = f"frag_group={frag_g} mutation={mut} well={well} plate={plate} len={entry['oligo_len']}bp"

        table.append({
            'seq_id': sys_name,
            'seq_num': seq_num,
            'well': well,
            'plate': plate,
            'frag_group': frag_g,
            'mutation': mut,
            'is_wt': mut == "WT",
            'description': desc,
            'dna': entry['dna'],
            'oligo_len': entry['oligo_len'],
            'original_name': entry['name'],
        })

    return table


def build_wt_gblocks(coding_dna, cuts, n_frags, vec_oh_5, vec_oh_3_full,
                     gblock_fw_primer, gblock_rv_primer):
    """
    Build WT fragment G-block sequences with specified primers.
    These are for separate ordering (e.g., IDT gBlocks) for amplification controls.

    Returns:
        OrderedDict {name: dna_sequence}
    """
    gblocks = OrderedDict()
    for fi in range(n_frags):
        wt_frag = build_fragment_dna(coding_dna, cuts, fi, n_frags,
                                     vec_oh_5, vec_oh_3_full)
        # Add G-block primers (same pair for all WT fragments)
        gblock = add_primers(wt_frag, gblock_fw_primer, gblock_rv_primer)
        gblocks[f"WT_gblock_F{fi + 1}"] = gblock
    return gblocks


# ============================================================================
# Output writers
# ============================================================================

def write_chip_fasta(table, filepath):
    """Write chip-order FASTA with name only (no description). DNA on single line."""
    with open(filepath, 'w') as fh:
        for row in table:
            fh.write(f">{row['seq_id']}\n")
            fh.write(f"{row['dna']}\n")


def write_chip_fasta_detailed(table, filepath):
    """Write detailed FASTA with name + description. DNA on single line."""
    with open(filepath, 'w') as fh:
        for row in table:
            fh.write(f">{row['seq_id']}  {row['description']}\n")
            fh.write(f"{row['dna']}\n")


def write_gblock_fasta(gblocks, filepath):
    """Write WT G-block FASTA. DNA on single line."""
    with open(filepath, 'w') as fh:
        for name, dna in gblocks.items():
            fh.write(f">{name}  len={len(dna)}bp\n")
            fh.write(f"{dna}\n")


def write_order_csv(table, filepath):
    """Write full order CSV (for Echo, plate maps, etc.)."""
    with open(filepath, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'Seq_ID', 'Seq_Num', 'Well', 'Plate', 'Frag_Group', 'Mutation',
            'Is_WT', 'Oligo_Length', 'Description', 'DNA_Sequence'
        ])
        for row in table:
            writer.writerow([
                row['seq_id'], row['seq_num'], row['well'], row['plate'],
                row['frag_group'], row['mutation'], row['is_wt'],
                row['oligo_len'], row['description'], row['dna'],
            ])


def write_assembly_csv(assembly_info, results, filepath):
    """Write fragment assembly summary CSV."""
    with open(filepath, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'Fragment', 'AA_Range', 'Size_AA', 'OH_Left', 'OH_Right',
            'N_Mutation_Sites', 'N_Variants', 'Ordered_DNA_Length',
            'Total_Sequences', 'Junction_Warnings'
        ])
        for info in assembly_info:
            frag = info['fragment']
            n_seqs = sum(1 for name in results if name.startswith(frag + '_'))
            writer.writerow([
                info['fragment'], info['aa_range'], info['size_aa'],
                info['oh_left'], info['oh_right'],
                info['n_mutation_sites'], info['n_variants'],
                info['ordered_dna_len'], n_seqs,
                '; '.join(info.get('junction_warnings', [])) or 'none',
            ])


def write_echo_picklist(table, n_frags, filepath):
    """
    Write an Echo-compatible CSV picklist for assembling SSM libraries.
    Each row = one transfer. Groups variants by fragment for easy pooling.
    """
    with open(filepath, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'Source_Plate', 'Source_Well', 'Fragment_Group',
            'Mutation', 'Seq_ID', 'Is_WT', 'Notes'
        ])
        for row in table:
            writer.writerow([
                f"plate{row['plate']}", row['well'], row['frag_group'],
                row['mutation'], row['seq_id'], row['is_wt'],
                'WT_control' if row['is_wt'] else '',
            ])


def print_run_parameters(params):
    """Print a formatted summary of all parameters used in a run."""
    print()
    print("=" * 70)
    print("  RUN PARAMETERS")
    print("=" * 70)
    for section, items in params.items():
        print(f"\n  [{section}]")
        for key, val in items.items():
            print(f"    {key:30s} = {val}")
    print("=" * 70)
    print()


# ============================================================================
# CLI entry point
# ============================================================================

def parse_mutations_fasta(fasta_path):
    """Parse a mutations FASTA file into {position: [(name, target_aa), ...]}."""
    mutations = {}
    wt_seq = None
    with open(fasta_path) as fh:
        header, seq = None, []
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if header and seq:
                    full_seq = ''.join(seq)
                    if 'WT_control' in header:
                        wt_seq = full_seq
                    else:
                        name = header.split()[0]
                        m = re.match(r"([A-Z])(\d+)([A-Z])", name)
                        if m:
                            mutations.setdefault(int(m.group(2)), []).append(
                                (name, m.group(3)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header and seq:
            full_seq = ''.join(seq)
            if 'WT_control' in header:
                wt_seq = full_seq
            else:
                name = header.split()[0]
                m = re.match(r"([A-Z])(\d+)([A-Z])", name)
                if m:
                    mutations.setdefault(int(m.group(2)), []).append(
                        (name, m.group(3)))
    return mutations, wt_seq


def parse_wt_dna_from_construct(construct_seq):
    """Extract coding DNA and vector stickies from a JohnBercow-style construct."""
    cs = next(i for i, c in enumerate(construct_seq) if c.isupper())
    ce = len(construct_seq) - next(
        i for i, c in enumerate(reversed(construct_seq)) if c.isupper())
    coding = construct_seq[cs:ce].upper()
    adapter_5 = construct_seq[:cs]
    adapter_3 = construct_seq[ce:]
    # Parse stickies (assumes BsaI adapter format)
    vec_5 = adapter_5[len("atactacggtctca"):].upper()
    vec_3 = adapter_3[:len(adapter_3) - len("cgagaccgtaatgc")].upper()
    return coding, vec_5, vec_3


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Golden Gate SSM Fragment Generator for IPD Blocks chip ordering.\n"
                    "Generates BsaI-compatible fragments from a WT DNA template,\n"
                    "swapping only mutated codons (E. coli preferred).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect vector from input DNA):
  python golden_gate_utils.py \\
    --wt_dna_fasta wt_construct.fasta \\
    --mutations_fasta passing_mutants.fasta \\
    --cuts 52 103 151 \\
    -o output_dir/

  # Override vector stickies for a different plasmid:
  python golden_gate_utils.py \\
    --wt_dna_fasta wt_construct.fasta \\
    --mutations_fasta passing_mutants.fasta \\
    --cuts 52 103 151 \\
    --vec_5_sticky AGGA --vec_3_sticky GGTTCC \\
    --vector_name CF_1 \\
    --nterm_tag MSKIKSGGSG --cterm_tag "" \\
    -o output_dir/

  # Change plate format and primers:
  python golden_gate_utils.py \\
    --wt_dna_fasta wt.fasta --mutations_fasta muts.fasta \\
    --cuts 52 103 151 --plate_format 96 \\
    --chip_primers oopzs001 oopzs002 oopzs003 oopzs004 oopzs005 oopzs006 oopzs007 oopzs008 \\
    --gblock_fw_primer AATATCACGCAAAAGCACCG \\
    --gblock_rv_primer AATGCAAAGCTATTAGCGCG \\
    -o output_dir/
""")

    # Required
    parser.add_argument('--wt_dna_fasta', required=True,
                        help='FASTA with the WT construct (full oligo with adapters, OR just coding DNA)')
    parser.add_argument('--mutations_fasta', required=True,
                        help='FASTA of passing mutant protein sequences (from AF3 filtering)')
    parser.add_argument('--cuts', required=True, type=int, nargs='+',
                        help='Cut positions (AA numbers to cut after, e.g., 52 103 151)')

    # Output
    parser.add_argument('-o', '--output_dir', default='./',
                        help='Output directory (default: ./)')
    parser.add_argument('--design_prefix', default='frag',
                        help='Prefix for systematic names (default: frag)')

    # Vector
    parser.add_argument('--vec_5_sticky', default=None,
                        help='Override 5\' vector sticky end (e.g., AGGA). Default: auto-detect from input')
    parser.add_argument('--vec_3_sticky', default=None,
                        help='Override 3\' vector sticky end (e.g., GGTTCC). Default: auto-detect from input')
    parser.add_argument('--vector_name', default='LM0627',
                        help='Vector name for output labels (default: LM0627)')
    parser.add_argument('--nterm_tag', default='MSG',
                        help='N-terminal tag AA sequence from vector (default: MSG)')
    parser.add_argument('--cterm_tag', default='GSGSHHWGSTHHHHHH',
                        help='C-terminal tag AA sequence from vector (default: GSGSHHWGSTHHHHHH)')

    # Plate & primers
    parser.add_argument('--plate_format', type=int, default=384, choices=[96, 384],
                        help='Well plate format (default: 384)')
    parser.add_argument('--max_oligo_len', type=int, default=300,
                        help='Maximum oligo length for chip (default: 300)')
    parser.add_argument('--chip_primers', nargs='+', default=None,
                        help='8 primer sequences for chip oligos (fw1 rv1 fw2 rv2 fw3 rv3 fw4 rv4). '
                             'Default: first 8 oopzs primers')
    parser.add_argument('--gblock_fw_primer', default='AATATCACGCAAAAGCACCG',
                        help='Forward primer for WT G-blocks (default: oopzs016)')
    parser.add_argument('--gblock_rv_primer', default='AATGCAAAGCTATTAGCGCG',
                        help='Reverse primer for WT G-blocks (default: oopzs018)')
    parser.add_argument('--oopzs_csv', default=f'{repo_paths.IPD_BLOCKS}/files/oopzs_primers_rev_comp.csv',
                        help='Path to oopzs primers CSV (for default chip primer assignment)')

    # Fidelity
    parser.add_argument('--fidelity_csv',
                        default=f'{repo_paths.IPD_BLOCKS}/files/b4-BsaI-HFv2-37_16_cycling.table-overhang_matrix.csv',
                        help='Path to BsaI fidelity matrix CSV')

    args = parser.parse_args()

    # ── Load inputs ──
    # WT DNA
    wt_records = []
    with open(args.wt_dna_fasta) as fh:
        hdr, seq = None, []
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if hdr and seq:
                    wt_records.append((hdr, ''.join(seq)))
                hdr = line[1:]
                seq = []
            else:
                seq.append(line)
        if hdr and seq:
            wt_records.append((hdr, ''.join(seq)))

    if not wt_records:
        print("ERROR: No sequences found in --wt_dna_fasta")
        return 1

    wt_full = wt_records[0][1]
    # Check if it has adapters (mixed case) or is pure coding (all uppercase)
    has_adapters = any(c.islower() for c in wt_full)
    if has_adapters:
        coding_dna, auto_v5, auto_v3 = parse_wt_dna_from_construct(wt_full)
        print(f"Parsed construct with adapters: {len(coding_dna)} nt coding")
    else:
        coding_dna = wt_full.upper()
        auto_v5, auto_v3 = "AGGA", "GGTTCC"
        print(f"Pure coding DNA input: {len(coding_dna)} nt")

    vec_5 = (args.vec_5_sticky or auto_v5).upper()
    vec_3 = (args.vec_3_sticky or auto_v3).upper()

    # Mutations
    mutations, wt_protein_fasta = parse_mutations_fasta(args.mutations_fasta)
    wt_protein = translate(coding_dna)
    if wt_protein_fasta and wt_protein_fasta != wt_protein:
        print(f"WARNING: FASTA WT protein differs from DNA translation!")

    n_muts = sum(len(v) for v in mutations.values())
    print(f"Loaded {n_muts} mutations at {len(mutations)} positions")

    # Primers
    n_frags = len(args.cuts) + 1
    if args.chip_primers:
        if len(args.chip_primers) != n_frags * 2:
            print(f"ERROR: --chip_primers needs {n_frags * 2} sequences "
                  f"({n_frags} fw + {n_frags} rv), got {len(args.chip_primers)}")
            return 1
        fw_primers = [args.chip_primers[i * 2].upper() for i in range(n_frags)]
        rv_primers = [args.chip_primers[i * 2 + 1].upper() for i in range(n_frags)]
    else:
        import pandas as pd
        oopzs = pd.read_csv(args.oopzs_csv)
        fw_primers = [oopzs.iloc[i * 2]['Sequence'].upper() for i in range(n_frags)]
        rv_primers = [oopzs.iloc[i * 2 + 1]['Sequence'].upper() for i in range(n_frags)]

    # Fidelity matrix
    import pandas as pd
    fidelity_df = pd.read_csv(args.fidelity_csv, index_col='Overhang')

    # ── Print parameters ──
    params = {
        "Input": {
            "WT DNA": args.wt_dna_fasta,
            "Mutations FASTA": args.mutations_fasta,
            "Coding DNA length": f"{len(coding_dna)} nt ({len(coding_dna)//3} aa)",
            "Mutations": f"{n_muts} at {len(mutations)} positions",
        },
        "Fragmentation": {
            "N fragments": n_frags,
            "Cuts (after AA)": str(args.cuts),
            "5' vector sticky": vec_5,
            "3' vector sticky": vec_3,
        },
        "Vector": {
            "Name": args.vector_name,
            "N-term tag": args.nterm_tag or "(none)",
            "C-term tag": args.cterm_tag or "(none)",
        },
        "Chip": {
            "Plate format": f"{args.plate_format}-well",
            "Max oligo length": f"{args.max_oligo_len} bp",
            "Design prefix": args.design_prefix,
        },
        "Primers (chip)": {
            f"F{i+1} fw": fw_primers[i] for i in range(n_frags)
        },
        "Primers (chip) rv": {
            f"F{i+1} rv": rv_primers[i] for i in range(n_frags)
        },
        "Primers (G-block)": {
            "fw (oopzs016)": args.gblock_fw_primer,
            "rv (oopzs018)": args.gblock_rv_primer,
        },
        "Output": {
            "Directory": os.path.abspath(args.output_dir),
        },
    }
    print_run_parameters(params)

    # ── Generate ──
    results, assembly_info = generate_all_fragments(
        coding_dna, args.cuts, mutations,
        vec_oh_5=vec_5, vec_oh_3_full=vec_3,
        fw_primers=fw_primers, rv_primers=rv_primers,
    )

    order_table = build_order_table(results, n_frags,
                                    design_prefix=args.design_prefix,
                                    plate_format=args.plate_format)

    wt_gblocks = build_wt_gblocks(coding_dna, args.cuts, n_frags, vec_5, vec_3,
                                  args.gblock_fw_primer, args.gblock_rv_primer)

    # ── Validate ──
    internal_ohs = [get_overhang(coding_dna, c) for c in args.cuts]
    all_ohs = [vec_5] + internal_ohs + [vec_3[-4:]]

    warnings = validate_all(results, fidelity_df, all_ohs,
                           args.max_oligo_len, primer_len=len(fw_primers[0]))
    ok, msg = verify_assembly(results, coding_dna, args.cuts, n_frags,
                              vec_5, vec_3, primer_len=len(fw_primers[0]))

    print(f"Validation: {len(warnings)} warnings")
    for w in warnings:
        print(f"  {w}")
    print(f"Assembly: {'PASS' if ok else 'FAIL'} — {msg}")

    # ── Write outputs ──
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, "golden_gate_ssm")

    write_chip_fasta(order_table, f"{prefix}_chip_order.fasta")
    write_gblock_fasta(wt_gblocks, f"{prefix}_wt_gblocks.fasta")
    write_order_csv(order_table, f"{prefix}_order_table.csv")
    write_assembly_csv(assembly_info, results, f"{prefix}_assembly_map.csv")
    write_echo_picklist(order_table, n_frags, f"{prefix}_echo_picklist.csv")

    # ── Summary ──
    lens = [len(v) for v in results.values()]
    combo = 1
    for fi in range(n_frags):
        combo *= sum(1 for r in order_table if r['frag_group'] == f"F{fi+1}")

    wt_expressed = (args.nterm_tag or '') + wt_protein + (args.cterm_tag or '')

    print(f"\n{'='*60}")
    print(f"  DONE — {len(order_table)} chip oligos + {len(wt_gblocks)} WT G-blocks")
    print(f"  Oligo range: {min(lens)}-{max(lens)} bp")
    print(f"  Plates: {max(r['plate'] for r in order_table)}")
    print(f"  Combinatorial: {combo:,} assemblies")
    print(f"  Expressed: {args.nterm_tag}-[insert]-{args.cterm_tag} ({len(wt_expressed)} aa)")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
