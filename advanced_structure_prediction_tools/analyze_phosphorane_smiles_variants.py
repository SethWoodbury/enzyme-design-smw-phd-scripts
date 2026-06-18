#!/usr/bin/env python3
"""Test many SMILES variants for the phosphorane transition state analog.
Goal: find which SMILES most reliably gives OH-P-PNP axial in TBP."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)  # suppress warnings

def angle_between(v1, v2):
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

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
    elif attached.GetSymbol() == 'P':
        return "O (bridging)"
    else:
        return f"O-{attached.GetSymbol()}"

def analyze(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  {label:55s} | FAILED TO PARSE")
        return None

    mol = Chem.AddHs(mol)

    # Find P
    p_idx = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'P':
            p_idx = atom.GetIdx()
            break
    if p_idx is None:
        print(f"  {label:55s} | NO PHOSPHORUS FOUND")
        return None

    p_atom = mol.GetAtomWithIdx(p_idx)
    neighbors = list(p_atom.GetNeighbors())
    coord_num = len(neighbors)
    neighbor_descs = [identify_ligand(n, p_idx) for n in neighbors]

    # Generate + optimize conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    params.maxIterations = 5000
    params.pruneRmsThresh = 0.3
    n_confs = 300

    result = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(result) == 0:
        params.useRandomCoords = True
        result = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(result) == 0:
        print(f"  {label:55s} | NO CONFORMERS")
        return None

    # UFF optimize
    energies = []
    for conf_id in range(len(result)):
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=2000)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff:
                energies.append((conf_id, ff.CalcEnergy()))
        except:
            energies.append((conf_id, None))

    valid_energies = [(c, e) for c, e in energies if e is not None]
    if valid_energies:
        valid_energies.sort(key=lambda x: x[1])
        conf_order = [c for c, _ in valid_energies]
        best_e = valid_energies[0][1]
    else:
        conf_order = list(range(len(result)))
        best_e = None

    # Tally axial pairs
    axial_counts = {}
    axial_angles = {}  # track actual angles per pair
    all_conf_data = []

    for conf_id in conf_order:
        conf = mol.GetConformer(conf_id)
        p_pos = np.array(conf.GetAtomPosition(p_idx))

        vecs = []
        dists = []
        for n in neighbors:
            n_pos = np.array(conf.GetAtomPosition(n.GetIdx()))
            vec = n_pos - p_pos
            d = np.linalg.norm(vec)
            dists.append(d)
            vecs.append(vec / d)

        max_ang = 0
        axial = None
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                ang = angle_between(vecs[i], vecs[j])
                if ang > max_ang:
                    max_ang = ang
                    axial = (i, j)

        if axial:
            key = tuple(sorted([neighbor_descs[axial[0]], neighbor_descs[axial[1]]]))
            axial_counts[key] = axial_counts.get(key, 0) + 1
            axial_angles.setdefault(key, []).append(max_ang)

            equatorial = [k for k in range(len(vecs)) if k not in axial]
            eq_angs = []
            for i in range(len(equatorial)):
                for j in range(i+1, len(equatorial)):
                    eq_angs.append(angle_between(vecs[equatorial[i]], vecs[equatorial[j]]))

            all_conf_data.append({
                'conf_id': conf_id,
                'axial_key': key,
                'axial_angle': max_ang,
                'eq_angles': eq_angs,
                'dists': dict(zip(neighbor_descs, dists)),
            })

    total = sum(axial_counts.values())

    # Check desired pairs
    desired_keys = [k for k in axial_counts if "O-PNP" in k and any(x in k for x in ("OH", "O=", "O^-"))]
    desired_count = sum(axial_counts.get(k, 0) for k in desired_keys)
    desired_pct = 100 * desired_count / total if total else 0

    # Get best conformer info
    best_conf = all_conf_data[0] if all_conf_data else None
    best_axial_str = f"{best_conf['axial_key'][0]}--P--{best_conf['axial_key'][1]}" if best_conf else "N/A"
    best_axial_ang = f"{best_conf['axial_angle']:.1f}" if best_conf else "N/A"

    # Is best conformer's geometry desired?
    best_is_desired = best_conf and best_conf['axial_key'] in [tuple(sorted(k)) for k in desired_keys] if desired_keys else False

    # Check TBP quality of best conformer
    tbp_quality = "N/A"
    if best_conf and len(best_conf['eq_angles']) == 3:
        if best_conf['axial_angle'] > 170 and all(abs(a - 120) < 15 for a in best_conf['eq_angles']):
            tbp_quality = "excellent"
        elif best_conf['axial_angle'] > 160 and all(abs(a - 120) < 25 for a in best_conf['eq_angles']):
            tbp_quality = "good"
        else:
            tbp_quality = "distorted"
    elif best_conf and coord_num == 4:
        tbp_quality = "tetrahedral"

    desired_mark = "***" if best_is_desired and tbp_quality in ("excellent", "good") else ""

    return {
        'label': label,
        'smiles': smiles,
        'coord_num': coord_num,
        'ligands': neighbor_descs,
        'n_confs': len(result),
        'n_optimized': len(valid_energies),
        'best_energy': best_e,
        'total_confs': total,
        'axial_counts': axial_counts,
        'axial_angles': axial_angles,
        'desired_pct': desired_pct,
        'best_axial': best_axial_str,
        'best_axial_ang': best_axial_ang,
        'best_is_desired': best_is_desired,
        'tbp_quality': tbp_quality,
        'desired_mark': desired_mark,
        'all_conf_data': all_conf_data,
    }


# ============================================================================
# SMILES VARIANTS
# ============================================================================
variants = [
    # --- BONDED 5-COORDINATE ---
    ("O[P](Oc1ccc([N+](=O)[O-])cc1)([O-])(OCC)(OCC)",
     "A) Original bonded 5-coord"),

    ("[O-][P](O)(Oc1ccc([N+](=O)[O-])cc1)(OCC)(OCC)",
     "B) Rearranged: O^- first"),

    ("O[P](OCC)(OCC)([O-])Oc1ccc([N+](=O)[O-])cc1",
     "C) Rearranged: OH first, PNP last"),

    ("Oc1ccc([N+](=O)[O-])cc1[P](O)([O-])(OCC)(OCC)",
     "D) PNP ring written first, then P"),

    # --- DISCONNECTED: OH as separate water ---
    ("O.[P](Oc1ccc([N+](=O)[O-])cc1)([O-])(OCC)(OCC)",
     "E) Disconnected: water + 4-coord P(O^-)"),

    ("[OH2].[P](Oc1ccc([N+](=O)[O-])cc1)([O-])(OCC)(OCC)",
     "F) Disconnected: explicit H2O + 4-coord"),

    # --- DISCONNECTED: hydroxide ion ---
    ("[OH-].[P](Oc1ccc([N+](=O)[O-])cc1)(OCC)(OCC)",
     "G) Disconnected: OH^- + 3-coord P (no O^-)"),

    ("[OH-].[P+](Oc1ccc([N+](=O)[O-])cc1)([O-])(OCC)(OCC)",
     "H) Disconnected: OH^- + 4-coord P+ with O^-"),

    # --- DISCONNECTED: PNP as leaving group ---
    ("Oc1ccc([N+](=O)[O-])cc1.O[P]([O-])(OCC)(OCC)",
     "I) Disconnected: free PNP-OH + 4-coord P"),

    ("[O-]c1ccc([N+](=O)[O-])cc1.O[P]([O-])(OCC)(OCC)",
     "J) Disconnected: PNP-O^- + 4-coord P"),

    # --- CHARGE VARIANTS ---
    ("[O][P]([O-])(Oc1ccc([N+](=O)[O-])cc1)(OCC)(OCC)",
     "K) Radical O on P (no H, no charge)"),

    ("[OH][P]([O-])(Oc1ccc([N+](=O)[O-])cc1)(OCC)(OCC)",
     "L) Explicit [OH] bracket on P"),

    # --- DIFFERENT O^- PLACEMENT ---
    ("O[P](Oc1ccc([N+](=O)[O-])cc1)(O)(OCC)(OCC)",
     "M) Two OH instead of OH + O^-"),

    ("O[P](Oc1ccc([N+](=O)[O-])cc1)([O-])([O-]CC)(OCC)",
     "N) One ethoxide as alkoxide O^-CC"),

    # --- 4-COORDINATE (tetrahedral, no TBP possible) ---
    ("CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1",
     "O) 4-coord substrate paraoxon (reference)"),

    # --- BOTH AXIAL LIGANDS DISCONNECTED (substrate + nucleophile) ---
    ("O.[P](OCC)(OCC)([O-]).Oc1ccc([N+](=O)[O-])cc1",
     "P) Fully disconnected: H2O + P(OEt)2(O^-) + PNP-OH"),
]

print("="*100)
print("PHOSPHORANE SMILES VARIANT ANALYSIS")
print("Desired: TBP geometry with OH (or O=/O^-) and O-PNP as AXIAL pair")
print("="*100)

results = []
for smiles, label in variants:
    r = analyze(smiles, label)
    if r:
        results.append(r)

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)
print(f"  {'Label':<55s} | {'Coord':>5s} | {'Confs':>5s} | {'Best axial pair':<25s} | {'Ang':>6s} | {'TBP':>10s} | {'%Desired':>8s} | {'Note':>3s}")
print(f"  {'-'*55}-+-{'-'*5}-+-{'-'*5}-+-{'-'*25}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*3}")

for r in results:
    print(f"  {r['label']:<55s} | {r['coord_num']:>5d} | {r['n_confs']:>5d} | {r['best_axial']:<25s} | {r['best_axial_ang']:>6s} | {r['tbp_quality']:>10s} | {r['desired_pct']:>7.1f}% | {r['desired_mark']:>3s}")

# ============================================================================
# DETAILED BREAKDOWN FOR TOP CANDIDATES
# ============================================================================
print("\n" + "="*100)
print("DETAILED AXIAL PAIR DISTRIBUTIONS")
print("="*100)

for r in results:
    if r['coord_num'] < 5:
        continue
    print(f"\n  {r['label']}")
    print(f"  SMILES: {r['smiles']}")
    print(f"  Ligands: {', '.join(r['ligands'])}")
    print(f"  Conformers: {r['n_confs']} generated, {r['n_optimized']} optimized")
    if r['best_energy']:
        print(f"  Best energy: {r['best_energy']:.1f} kcal/mol")
    print(f"  {'Axial pair':<35s} {'Count':>6s} {'%':>7s} {'Mean angle':>11s}")
    print(f"  {'-'*62}")
    for pair, count in sorted(r['axial_counts'].items(), key=lambda x: -x[1]):
        pct = 100 * count / r['total_confs']
        mean_ang = np.mean(r['axial_angles'][pair])
        marker = " <-- DESIRED" if "O-PNP" in pair and any(x in pair for x in ("OH", "O=", "O^-")) else ""
        print(f"  {pair[0] + ' --- P --- ' + pair[1]:<35s} {count:>6d} {pct:>6.1f}% {mean_ang:>10.1f}°{marker}")

    # Show best conformer details
    if r['all_conf_data']:
        best = r['all_conf_data'][0]
        print(f"\n  Lowest-energy conformer (conf {best['conf_id']}):")
        print(f"    Axial: {best['axial_key'][0]} --- P --- {best['axial_key'][1]}  ({best['axial_angle']:.1f}°)")
        if best['eq_angles']:
            print(f"    Equatorial angles: {', '.join(f'{a:.1f}°' for a in best['eq_angles'])}")
        print(f"    Distances: {', '.join(f'{k}={v:.3f}A' for k,v in best['dists'].items())}")

print("\n" + "="*100)
print("RECOMMENDATION")
print("="*100)

# Find best candidate
best_candidates = [r for r in results if r['best_is_desired'] and r['tbp_quality'] in ('excellent', 'good')]
if best_candidates:
    best = max(best_candidates, key=lambda r: r['desired_pct'])
    print(f"\n  BEST SMILES: {best['smiles']}")
    print(f"  Label: {best['label']}")
    print(f"  Desired axial (OH/O^-/O= + PNP): {best['desired_pct']:.1f}% of conformers")
    print(f"  Lowest-energy conformer: {best['best_axial']} at {best['best_axial_ang']}° ({best['tbp_quality']} TBP)")
else:
    # Fallback: best desired_pct among 5-coord
    five_coord = [r for r in results if r['coord_num'] == 5]
    if five_coord:
        best = max(five_coord, key=lambda r: r['desired_pct'])
        print(f"\n  No variant has desired axial as lowest-energy conformer.")
        print(f"  Highest desired %: {best['smiles']}")
        print(f"  ({best['desired_pct']:.1f}% of conformers have desired axial pair)")

print("""
  NOTE: SMILES atom ordering does NOT affect RDKit's distance geometry or
  UFF optimization. The 3D geometry is determined by the molecular graph
  (connectivity + charges), not the SMILES string order. So rearranging
  the same molecule gives identical conformer distributions.

  What DOES matter:
  1. Bonded vs disconnected (different molecules!)
  2. Charge distribution (O^- vs OH vs O=)
  3. Coordination number (4 vs 5)
""")
