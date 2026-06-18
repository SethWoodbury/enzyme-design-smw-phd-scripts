#!/usr/bin/env python3
"""Analyze 3D geometry around pentacoordinate phosphorus center (phosphorane).
Same ligand set as the vanadate complex but with P instead of V."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def angle_between(v1, v2):
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_a = np.clip(cos_a, -1, 1)
    return np.degrees(np.arccos(cos_a))

def identify_ligand(atom, center_idx):
    n_neighbors = [a for a in atom.GetNeighbors() if a.GetIdx() != center_idx]
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

def analyze_geometry(smiles, label=""):
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

    # Find P center
    center_idx = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'P':
            center_idx = atom.GetIdx()
            break
    if center_idx is None:
        print("  ERROR: No phosphorus atom found")
        return

    center_atom = mol.GetAtomWithIdx(center_idx)
    neighbors = list(center_atom.GetNeighbors())
    coord_num = len(neighbors)
    neighbor_descs = [identify_ligand(n, center_idx) for n in neighbors]

    print(f"\n  P coordination number: {coord_num}")
    for n, desc in zip(neighbors, neighbor_descs):
        print(f"    idx={n.GetIdx():3d}  {desc}")

    # Generate conformers
    n_confs = 200
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    params.maxIterations = 5000
    params.pruneRmsThresh = 0.5

    result = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(result) == 0:
        params.useRandomCoords = True
        result = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if len(result) == 0:
            print("  ERROR: Could not generate conformers")
            return
        print(f"  (Used random coords fallback)")

    print(f"  Generated {len(result)} conformers")

    # Try UFF optimization since P is supported
    optimized = 0
    energies = []
    for conf_id in range(len(result)):
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=2000)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff:
                energies.append((conf_id, ff.CalcEnergy()))
                optimized += 1
        except:
            pass

    if energies:
        energies.sort(key=lambda x: x[1])
        print(f"  UFF optimized {optimized} conformers")
        print(f"  Energy range: {energies[0][1]:.1f} - {energies[-1][1]:.1f} kcal/mol")
        conf_ids_to_analyze = [cid for cid, _ in energies]
    else:
        print("  UFF optimization failed - using raw ETKDG conformers")
        conf_ids_to_analyze = list(range(len(result)))

    # Tally axial pairs across all conformers
    axial_pair_counts = {}

    for conf_id in conf_ids_to_analyze:
        conf = mol.GetConformer(conf_id)
        center_pos = np.array(conf.GetAtomPosition(center_idx))

        vecs = []
        for n in neighbors:
            n_pos = np.array(conf.GetAtomPosition(n.GetIdx()))
            vec = n_pos - center_pos
            vecs.append(vec / np.linalg.norm(vec))

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

    total = sum(axial_pair_counts.values())
    print(f"\n  AXIAL PAIR STATISTICS across {total} conformers:")
    print(f"  {'Axial pair':<35s} {'Count':>6s} {'%':>7s}")
    print(f"  {'-'*50}")
    for pair, count in sorted(axial_pair_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        marker = ""
        if "O-PNP" in pair and ("OH" in pair or "O=" in pair or "O^-" in pair):
            marker = " <-- DESIRED"
        print(f"  {pair[0] + ' --- P --- ' + pair[1]:<35s} {count:>6d} {pct:>6.1f}%{marker}")

    # Detailed analysis of top 5 lowest-energy (or first 5) conformers
    analyze_ids = [cid for cid, _ in energies[:5]] if energies else conf_ids_to_analyze[:5]

    for rank, conf_id in enumerate(analyze_ids):
        conf = mol.GetConformer(conf_id)
        center_pos = np.array(conf.GetAtomPosition(center_idx))

        energy_str = ""
        if energies:
            e = next((e for cid, e in energies if cid == conf_id), None)
            if e is not None:
                energy_str = f", E={e:.1f}"

        ligand_info = []
        for i, n in enumerate(neighbors):
            n_pos = np.array(conf.GetAtomPosition(n.GetIdx()))
            vec = n_pos - center_pos
            dist = np.linalg.norm(vec)
            ligand_info.append({
                'desc': neighbor_descs[i],
                'vec': vec / np.linalg.norm(vec),
                'dist': dist,
            })

        max_angle = 0
        axial = None
        for i in range(len(ligand_info)):
            for j in range(i+1, len(ligand_info)):
                ang = angle_between(ligand_info[i]['vec'], ligand_info[j]['vec'])
                if ang > max_angle:
                    max_angle = ang
                    axial = (i, j)

        equatorial = [k for k in range(len(ligand_info)) if k not in axial]

        print(f"\n  --- Rank {rank+1} (conf {conf_id}{energy_str}) ---")
        for li in ligand_info:
            print(f"    {li['desc']:12s}  P-O = {li['dist']:.3f} A")

        print(f"\n    All L-P-L angles:")
        for i in range(len(ligand_info)):
            for j in range(i+1, len(ligand_info)):
                ang = angle_between(ligand_info[i]['vec'], ligand_info[j]['vec'])
                tag = " [AXIAL]" if (i, j) == axial or (j, i) == axial else ""
                print(f"      {ligand_info[i]['desc']:8s} - P - {ligand_info[j]['desc']:8s} = {ang:6.1f}°{tag}")

        print(f"\n    AXIAL:      {ligand_info[axial[0]]['desc']} --- P --- {ligand_info[axial[1]]['desc']}  ({max_angle:.1f}°)")
        print(f"    EQUATORIAL: {', '.join(ligand_info[k]['desc'] for k in equatorial)}")

        eq_angles = []
        for i in range(len(equatorial)):
            for j in range(i+1, len(equatorial)):
                ang = angle_between(ligand_info[equatorial[i]]['vec'], ligand_info[equatorial[j]]['vec'])
                eq_angles.append(ang)
        if eq_angles:
            print(f"    Eq-P-Eq angles: {', '.join(f'{a:.1f}°' for a in eq_angles)}")
            if len(equatorial) == 3 and all(abs(a - 120) < 20 for a in eq_angles):
                print(f"    => TRIGONAL BIPYRAMIDAL")
            elif len(equatorial) == 3:
                print(f"    => Distorted from ideal TBP")


# ============================================================================
smiles_variants = [
    ("O[P](Oc1ccc([N+](=O)[O-])cc1)([O-])(OCC)(OCC)",
     "5-coord: OH, O-PNP, O^-, 2x O-Et"),

    ("O=[P](Oc1ccc([N+](=O)[O-])cc1)([O-])(OCC)(OCC)",
     "5-coord: P=O oxo, O-PNP, O^-, 2x O-Et"),

    ("[O-][P](O)(Oc1ccc([N+](=O)[O-])cc1)(OCC)(OCC)",
     "5-coord: rearranged order"),

    # Also the actual substrate for reference (tetrahedral)
    ("CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1",
     "4-coord SUBSTRATE (paraoxon): P=O, 2x O-Et, O-PNP"),
]

for smiles, label in smiles_variants:
    analyze_geometry(smiles, label)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
  For the 5-coordinate phosphorane (transition state analog):
    DESIRED AXIAL:      OH (or O=) and O-PNP
    DESIRED EQUATORIAL: O^-, O-Et, O-Et

  Apicophilicity rules for TBP phosphorus:
    - Most electronegative / best pi-acceptor groups prefer AXIAL
    - PNP (aryloxide) and OH/O= are good axial candidates
    - Ethoxides are less electronegative -> equatorial
""")
