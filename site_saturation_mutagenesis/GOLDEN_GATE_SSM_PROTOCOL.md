# Golden Gate SSM Combinatorial Fragment Assembly Protocol

## Overview

This protocol generates BsaI-compatible DNA fragments for combinatorial testing
of site-saturation mutagenesis (SSM) mutations using Golden Gate assembly on a
Twist chip. It preserves the original WT codon-optimized DNA and only swaps the
single codon at each mutation position.

**System**: LM0627 / CF_1 vector (BsaI, identical sticky ends) / BsaI-HFv2 / E. coli / IPD Blocks chip

**Vector compatibility**: LM0627 (C-term SNAC-His) and CF_1 (combinatorial libraries) share
identical BsaI sticky ends (`agga` / `ggttcc`). Fragments generated here work with either vector.
The original WT DNA was reverse-translated by JohnBercow.py with `--gg_vector LM0627`.

**Inputs**:
- WT DNA: from original order Excel (JohnBercow output, with BsaI adapters + sticky ends)
- Mutations: from AF3 ensemble metrics passing FASTA (Section III.F output)

**Key principle**: The WT codon-optimized DNA is preserved as the template. Only the single
mutated codon is swapped (using E. coli preferred codons). No full reverse translation needed.

## Design Strategy

```
                    Cut 1 (aa 52)    Cut 2 (aa 103)   Cut 3 (aa 151)
                         |                |                |
Protein:  [----F1(52aa)----][---F2(51aa)---][--F3(48aa)---][--F4(47aa)--]
          mutations 11-40   mutations 66-97  muts 111-141   muts 160-181

Overhangs: AGGA -- ATCA -- GAAC -- CTGG -- TTCC
           vec5'   J1      J2      J3      vec3'
```

Each fragment variant differs from WT by exactly **1 codon**. Fragments are
assembled combinatorially: pick one variant per fragment group, Golden Gate
assembles them into a full gene in the CF_1 vector.

## Oligo Structure on Twist Chip

Each oligo (5'->3'):

```
fw_primer(20bp) + GGTCTC + a + [OH_left][coding][OH_right] + c + GAGACC + RC(rv_primer)(20bp)
|_______________|   |_BsaI_| sp |_______insert_____________| sp |_BsaI_| |__________________|
   amplification    5' adapter       after BsaI digest           3' adapter    amplification
```

- **fw_primer**: oopzs orthogonal primer (unique per fragment group) — for selective PCR from chip pool
- **rv_primer**: different oopzs primer (unique per fragment group) — RC appended at 3' end
- **BsaI adapters**: `ggtctca` (5') and `cgagacc` (3') — cleaved during Golden Gate assembly
- **OH_left / OH_right**: 4nt sticky ends from coding sequence at junctions; 5' vector sticky (`AGGA`) for F1 left; full 3' vector sticky (`GGTTCC`) for F4 right

## Overhangs & Fidelity

All 5 overhangs validated against BsaI-HFv2 37C 16-cycle fidelity matrix:

| Position | Overhang | RC     | Self-ligation | Off-target |
|----------|----------|--------|---------------|------------|
| Vec 5'   | AGGA     | TCCT   | high          | 0          |
| J1 (aa52)| ATCA     | TGAT   | 555           | 0          |
| J2 (aa103)| GAAC    | GTTC   | 552           | 0          |
| J3 (aa151)| CTGG    | CCAG   | 538           | 0          |
| Vec 3'   | TTCC     | GGAA   | high          | 0          |

- No palindromic overhangs
- Zero pairwise off-target ligation between any overhang and any non-intended partner
- Cut positions chosen in mutation-free gaps (no passing mutations overlap junction OHs)

## Experimental Workflow

### 1. Chip Ordering
- Order all ~291 oligos on a Twist 300bp chip
- Each fragment group shares the same primer pair for selective amplification
- 8 orthogonal oopzs primers used (2 per fragment group)

### 2. Selective PCR Amplification
- PCR each fragment group separately using its fw/rv primer pair
- This amplifies all variants of that fragment from the chip pool
- 4 separate PCR reactions (one per fragment group)

### 3. BsaI Golden Gate Assembly
- Pool one variant from each fragment group (or use the full amplified pool for libraries)
- Add CF_1 vector backbone (linearized)
- BsaI-HFv2 digestion + T4 ligase (standard Golden Gate cycling: 37C/16C, 30+ cycles)
- BsaI cuts release the insert from primers + adapters
- 4nt sticky ends direct ordered assembly: F1 + F2 + F3 + F4 into vector

### 4. Testing Modes
- **Individual mutations**: Assemble 3 WT fragments + 1 variant fragment
- **Combinatorial libraries**: Mix amplified pools → screen for synergistic combinations
- **Targeted combos**: Cherry-pick specific variants per fragment for designed combinations

### 5. Transformation & Screening
- Transform assembled products into E. coli
- Cell-free expression (CF_1 compatible) or plating
- Screen for activity (paraoxonase assay)

## Files

| File | Description |
|------|-------------|
| `golden_gate_utils.py` | Core utility module (codon swap, fragment generation, validation) |
| `golden_gate_ssm_fragments.fasta` | All oligo sequences ready for Twist chip ordering |
| `golden_gate_ssm_assembly_map.csv` | Fragment map with sizes, overhangs, variant counts |
| `af3_ssm_i1_passing_mutants.fasta` | Input: protein sequences of AF3-passing mutants |

## Key Design Decisions

1. **4 fragments** (not 3): 18.4M combinatorial assemblies vs ~815K with 3 fragments, at the cost of 1 extra junction (negligible impact on GG efficiency)
2. **Cuts at aa 52, 103, 151**: Placed in the 3 largest mutation-free gaps (aa 41-65, 98-110, 142-159), validated for fidelity
3. **WT DNA preserved**: Only the mutated codon changes; no full reverse translation. Minimizes synthesis errors and unexpected expression issues
4. **E. coli preferred codons**: Mutant codons chosen from highest-frequency E. coli codons, with automatic fallback if the preferred codon creates a BsaI site or other problematic sequence
5. **3' vector sticky is 6nt** (`GGTTCC`): The CF_1 vector design places 2 extra nt (`GG`) that remain double-stranded after BsaI digest, followed by 4nt overhang (`TTCC`). This is how the vector was designed and must be preserved.

## Avoid Sequences

Codons are checked against these patterns (from IPD Blocks Domesticator):
- `GGTCTC` / `GAGACC` — BsaI recognition sites
- `GGAGG` — E. coli Shine-Dalgarno sequence
- `TAAGGAG` — strong ribosome binding site
- `GCTGGTGG` — Chi recombination site
- `AAAAA` / `TTTTT` — polymerase slippage / terminators
- `CCCCCC` / `GGGGGG` — synthesis-problematic runs

## Troubleshooting

- **"All codons create BsaI sites"**: The mutation context inherently creates a BsaI site regardless of codon choice. Consider moving the fragment boundary or excluding this mutation.
- **Junction conflict**: A mutation's codon overlaps with a 4nt junction overhang. The script automatically detects and excludes these — they cannot be ordered as single-codon swaps.
- **Oligo > 300bp**: Fragment too large for chip. Reduce fragment size by adjusting cut positions.
- **Low self-ligation score**: The chosen overhang has weak intended ligation. Consider alternative cut positions within the same gap.
