#!/usr/bin/env python3
"""Analyze 3D geometry around vanadium center for vanadate ester SMILES variants.
Uses distance geometry (ETKDG) since UFF/MMFF don't parameterize vanadium well."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

def angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_a = np.clip(cos_a, -1, 1)
    return np.degrees(np.arccos(cos_a))

def identify_ligand(atom, v_idx):
    """Identify what a V-neighbor oxygen is bonded to."""
    n_neighbors = [a for a in atom.GetNeighbors() if a.GetIdx() != v_idx]
    if atom.GetSymbol() != 'O':
        return atom.GetSymbol()
    if len(n_neighbors) == 0:
        charge = atom.GetFormalCharge()
        total_h = atom.GetTotalNumHs()
        if charge == -1:
            return "O^-"
        elif total_h > 0:
            return "OH"
        else:
            return "O="
    attached = n_neighbors[0]
    if attached.GetIsAromatic():
        return "O-PNP"
    elif attached.GetSymbol() == 'C':
        return "O-Et"
    elif attached.GetSymbol() == 'H':
        return "OH"
    else:
        return f"O-{attached.GetSymbol()}"

def analyze_vanadium_geometry(smiles, label=""):
    print(f"\n{'='*70}")
    print(f"SMILES: {smiles}")
    if label:
        print(f"Label:  {label}")
    print(f"{'='*70}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("  ERROR: Could not parse SMILES")
        return

    mol = Chem.AddHs(mol)

    # Find vanadium
    v_idx = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'V':
            v_idx = atom.GetIdx()
            break
    if v_idx is None:
        print("  ERROR: No vanadium atom found")
        return

    v_atom = mol.GetAtomWithIdx(v_idx)
    neighbors = list(v_atom.GetNeighbors())
    coord_num = len(neighbors)
    print(f"\n  V coordination number: {coord_num}")
    for n in neighbors:
        desc = identify_ligand(n, v_idx)
        print(f"    idx={n.GetIdx():3d}  {desc}")

    # Generate conformers with ETKDG (no force field needed)
    n_confs = 200
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    params.maxIterations = 5000
    params.pruneRmsThresh = 0.5

    result = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(result) == 0:
        params.useRandomCoords = True
        params.forceTol = 0.1
        result = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if len(result) == 0:
            print("  ERROR: Could not generate conformers")
            return
        print(f"  (Used random coords fallback)")

    print(f"  Generated {len(result)} conformers")

    # Tally geometry across ALL conformers
    axial_pair_counts = {}  # track which pair is axial most often

    neighbor_idxs = [n.GetIdx() for n in neighbors]
    neighbor_descs = [identify_ligand(n, v_idx) for n in neighbors]

    for conf_id in range(len(result)):
        conf = mol.GetConformer(conf_id)
        v_pos = np.array(conf.GetAtomPosition(v_idx))

        vecs = []
        for n in neighbors:
            n_pos = np.array(conf.GetAtomPosition(n.GetIdx()))
            vec = n_pos - v_pos
            vecs.append(vec / np.linalg.norm(vec))

        # Find the pair with largest angle (axial)
        max_angle = 0
        axial_pair = None
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                ang = angle_between(vecs[i], vecs[j])
                if ang > max_angle:
                    max_angle = ang
                    axial_pair = (i, j)

        if axial_pair:
            key = tuple(sorted([neighbor_descs[axial_pair[0]], neighbor_descs[axial_pair[1]]]))
            axial_pair_counts[key] = axial_pair_counts.get(key, 0) + 1

    # Report statistics
    total = sum(axial_pair_counts.values())
    print(f"\n  AXIAL PAIR STATISTICS across {total} conformers:")
    print(f"  {'Axial pair':<35s} {'Count':>6s} {'%':>7s}")
    print(f"  {'-'*50}")
    for pair, count in sorted(axial_pair_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        marker = " <-- DESIRED" if set(pair) == {"OH", "O-PNP"} or set(pair) == {"O=", "O-PNP"} or set(pair) == {"O^-", "O-PNP"} else ""
        print(f"  {pair[0] + ' --- V --- ' + pair[1]:<35s} {count:>6d} {pct:>6.1f}%{marker}")

    # Detailed analysis of the most common geometry
    print(f"\n  DETAILED ANALYSIS of a representative conformer:")
    # Pick the conformer closest to the most common axial assignment
    most_common_pair = max(axial_pair_counts, key=axial_pair_counts.get)

    for conf_id in range(len(result)):
        conf = mol.GetConformer(conf_id)
        v_pos = np.array(conf.GetAtomPosition(v_idx))

        ligand_info = []
        for i, n in enumerate(neighbors):
            n_pos = np.array(conf.GetAtomPosition(n.GetIdx()))
            vec = n_pos - v_pos
            dist = np.linalg.norm(vec)
            ligand_info.append({
                'desc': neighbor_descs[i],
                'vec': vec / np.linalg.norm(vec),
                'dist': dist,
            })

        # Check if this conformer matches the most common assignment
        max_angle = 0
        axial = None
        for i in range(len(ligand_info)):
            for j in range(i+1, len(ligand_info)):
                ang = angle_between(ligand_info[i]['vec'], ligand_info[j]['vec'])
                if ang > max_angle:
                    max_angle = ang
                    axial = (i, j)

        pair_key = tuple(sorted([ligand_info[axial[0]]['desc'], ligand_info[axial[1]]['desc']]))
        if pair_key == most_common_pair:
            # Print this conformer
            print(f"  (Conformer {conf_id})")
            for li in ligand_info:
                print(f"    {li['desc']:12s}  V-O = {li['dist']:.3f} A")

            print(f"\n    All L-V-L angles:")
            for i in range(len(ligand_info)):
                for j in range(i+1, len(ligand_info)):
                    ang = angle_between(ligand_info[i]['vec'], ligand_info[j]['vec'])
                    label_str = ""
                    if (i, j) == axial or (j, i) == axial:
                        label_str = " [AXIAL]"
                    print(f"      {ligand_info[i]['desc']:8s} - V - {ligand_info[j]['desc']:8s} = {ang:6.1f}°{label_str}")

            equatorial = [k for k in range(len(ligand_info)) if k not in axial]
            print(f"\n    AXIAL:      {ligand_info[axial[0]]['desc']} --- V --- {ligand_info[axial[1]]['desc']}  ({max_angle:.1f}°)")
            print(f"    EQUATORIAL: {', '.join(ligand_info[k]['desc'] for k in equatorial)}")

            eq_angles = []
            for i in range(len(equatorial)):
                for j in range(i+1, len(equatorial)):
                    ang = angle_between(ligand_info[equatorial[i]]['vec'], ligand_info[equatorial[j]]['vec'])
                    eq_angles.append(ang)
            if eq_angles:
                print(f"    Eq-V-Eq angles: {', '.join(f'{a:.1f}°' for a in eq_angles)}")
                if len(equatorial) == 3 and all(abs(a - 120) < 25 for a in eq_angles):
                    print(f"    => TRIGONAL BIPYRAMIDAL geometry")
                elif len(equatorial) == 3:
                    print(f"    => Distorted geometry (ideal TBP would have 120° equatorial)")
            break


# ============================================================================
smiles_variants = [
    ("O[V](Oc1ccc([N+](=O)[O-])cc1)([O-])(OCC)(OCC)",
     "Original: OH, O-PNP, O^-, 2x O-Et"),

    ("O=[V](Oc1ccc([N+](=O)[O-])cc1)([O-])(OCC)(OCC)",
     "V=O oxo instead of V-OH"),

    ("[O-][V](O)(Oc1ccc([N+](=O)[O-])cc1)(OCC)(OCC)",
     "Rearranged atom order"),

    ("O=[V](Oc1ccc([N+](=O)[O-])cc1)(OCC)(OCC)",
     "4-coordinate tetrahedral: V=O, O-PNP, 2x O-Et"),
]

for smiles, label in smiles_variants:
    analyze_vanadium_geometry(smiles, label)

print("\n" + "="*70)
print("SUMMARY - DESIRED GEOMETRY (trigonal bipyramidal)")
print("="*70)
print("""
  AXIAL:      OH (or O=) and O-PNP
  EQUATORIAL: O^-, O-Et, O-Et  (ethoxides + oxide on same plane)

  Ideal TBP angles:
    Axial-V-Axial:           180°
    Equatorial-V-Equatorial: 120°
    Axial-V-Equatorial:       90°
""")
