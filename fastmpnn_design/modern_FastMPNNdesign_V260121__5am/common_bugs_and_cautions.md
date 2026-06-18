## Bug 1: HIS Tautomer Mutation with Metal Pseudobonds

**Error:**
```
[HIS Tautomer] Mutating A13 from HIS:MP-NE2-connect to HIS_D
core.conformation.Residue: [ ERROR ] Unable to handle change in the number of residue connections in the presence of pseudobonds!
```

**Root Cause:**
- PyRosetta cannot mutate residues (change residue type) when pseudobonds exist
- Metal-coordinating HIS residues have pseudobonds to zinc atoms
- **CRITICAL**: Pseudobonds are created from GEOMETRY detection, not just REMARK 666 lines!
- When zinc atoms are present near HIS residues, PyRosetta auto-creates pseudobonds during `pose_from_file()`

**Attempted Fixes (FAILED):**
1. First attempt: Skip metal-coordinating HIS - REJECTED because this leaves wrong tautomers
2. Second attempt: Strip REMARK 666 lines before loading - FAILED because PyRosetta detects metals from geometry
3. Third attempt: Remove duplicate tautomer call - FAILED because pseudobonds still created from geometry

**What We Learned:**
- Stripping REMARK 666 is NOT enough - PyRosetta detects metal coordination from atomic coordinates
- The only way to prevent pseudobond creation is to load the PDB WITHOUT metal atoms (HETATM)
- Tautomers must be applied to a protein-only pose, then metals can be added back

**WORKING FIX (in `setup_pose_from_mpnn_output()`):**

The solution is to load the protein separately from HETATM (metals/ligands):

```python
# STEP 1: Separate protein from HETATM
protein_lines = [line for line in pdb if line.startswith('ATOM') or line.startswith('TER')]
hetatm_lines = [line for line in pdb if line.startswith('HETATM')]

# STEP 2: Load PROTEIN-ONLY (no metals = no pseudobonds!)
protein_only_pdb = '\n'.join(protein_lines) + '\nEND\n'
pose = pyr.pose_from_file(protein_only_temp_path)  # PyRosetta adds H here

# STEP 3: Apply HIS tautomers (SAFE - no metals, no pseudobonds)
apply_his_tautomers_to_pose(pose, his_tautomer_map)

# STEP 4: Export protein with correct tautomers, merge back HETATM
protein_with_H = pyr.distributed.io.to_pdbstring(pose)
full_pdb = headers + remark666 + protein_with_H + hetatm_lines + END

# STEP 5: Load full PDB (now pseudobonds are created with CORRECT tautomers)
pose = pyr.pose_from_file(full_temp_path)
```

**Debug output files (when --debug is used):**
- `pose{n}_step1_protein_only.pdb` - Protein without HETATM (no H yet)
- `pose{n}_step2_after_tautomers.pdb` - After tautomer correction
- `pose{n}_step3_full_pdb_fixed_tautomers.pdb` - Full PDB before loading
- `pose{n}_step4_pose_with_hetatm.pdb` - After loading with metals
- `pose{n}_step5_final_pose.pdb` - Final pose after all setup

## Bug 2: LYS 174 SidechainConjugation Warning

**Warning:**
```
core.conformation.Residue: [ WARNING ] Residue::inter_residue_connection_partner: Invalid residue connection, returning BOGUS ID: this_rsd= LYS:SidechainConjugation 174 connid= 3 partner_seqpos= 0
```

**What This Means:**
- LYS 174 has a sidechain conjugation patch but no partner residue
- This suggests a covalent bond that wasn't properly established

## General Cautions

1. **Metal detection from geometry**: PyRosetta detects metal coordination from atomic coordinates, not just REMARK 666 lines. If zinc atoms are within coordination distance of HIS/CYS/ASP/GLU, pseudobonds will be created.

2. **HETATM must be separated**: To avoid premature pseudobond creation, load protein-only first, then add HETATM back.

3. **Order of operations for HIS tautomers**:
   1. Load protein-only (no metals)
   2. Apply tautomer corrections (safe - no pseudobonds)
   3. Export protein with correct H atoms
   4. Merge back HETATM + headers
   5. Load full structure (pseudobonds now created with correct tautomers)

4. **MPNN output**: Raw MPNN output has no hydrogens on protein (only HETATM). Hydrogens are added by PyRosetta during `pose_from_file()`.

5. **MutateResidue limitations**: Cannot change residue types when pseudobonds exist. This includes HIS↔HIS_D tautomer changes.

6. **Inspired by**: The working approach is based on `/home/woodbuse/special_scripts/general_utils/add_remark666_lines_AND_rosetta_hydrogens_to_pdb_OPTIMIZED.py` which uses the same strategy of loading protein-only, applying mutations, then merging back.
