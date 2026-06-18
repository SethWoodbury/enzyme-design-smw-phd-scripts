#!/usr/bin/env python3
"""For the original 5-coord phosphorane, show axial assignment as a function
of energy rank - is the desired geometry actually preferred at low energy?"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

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
    else:
        return f"O-{attached.GetSymbol()}"

smiles = "O[P](Oc1ccc([N+](=O)[O-])cc1)([O-])(OCC)(OCC)"
print(f"SMILES: {smiles}\n")

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

p_idx = None
for atom in mol.GetAtoms():
    if atom.GetSymbol() == 'P':
        p_idx = atom.GetIdx()
        break

p_atom = mol.GetAtomWithIdx(p_idx)
neighbors = list(p_atom.GetNeighbors())
neighbor_descs = [identify_ligand(n, p_idx) for n in neighbors]

# Generate lots of conformers
params = AllChem.ETKDGv3()
params.randomSeed = 42
params.numThreads = 1
params.maxIterations = 5000
params.pruneRmsThresh = 0.2
n_confs = 500

result = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
print(f"Generated {len(result)} conformers")

# Optimize all
data = []
for conf_id in range(len(result)):
    try:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=2000)
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        if ff is None:
            continue
        energy = ff.CalcEnergy()
    except:
        continue

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

    # Find axial pair
    max_ang = 0
    axial = None
    all_angles = {}
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            ang = angle_between(vecs[i], vecs[j])
            all_angles[(i,j)] = ang
            if ang > max_ang:
                max_ang = ang
                axial = (i, j)

    axial_key = tuple(sorted([neighbor_descs[axial[0]], neighbor_descs[axial[1]]]))
    equatorial = [k for k in range(len(vecs)) if k not in axial]
    eq_descs = tuple(sorted([neighbor_descs[k] for k in equatorial]))

    eq_angles = []
    for i in range(len(equatorial)):
        for j in range(i+1, len(equatorial)):
            eq_angles.append(all_angles.get((min(equatorial[i],equatorial[j]), max(equatorial[i],equatorial[j])),
                             angle_between(vecs[equatorial[i]], vecs[equatorial[j]])))

    # Is this the desired geometry? PNP axial with a non-ethanol
    has_pnp_axial = "O-PNP" in axial_key
    has_nonet_partner = any(x in axial_key for x in ("OH", "O^-", "O="))
    is_desired = has_pnp_axial and has_nonet_partner

    # Are ethoxides equatorial?
    et_equatorial = eq_descs.count("O-Et") == 2

    data.append({
        'conf_id': conf_id,
        'energy': energy,
        'axial_key': axial_key,
        'axial_angle': max_ang,
        'eq_descs': eq_descs,
        'eq_angles': eq_angles,
        'is_desired': is_desired,
        'et_equatorial': et_equatorial,
        'dists': dict(zip(neighbor_descs, dists)),
    })

data.sort(key=lambda x: x['energy'])
min_e = data[0]['energy']

print(f"Optimized {len(data)} conformers")
print(f"Energy range: {min_e:.2f} - {data[-1]['energy']:.2f} kcal/mol\n")

# ============================================================================
# SHOW EVERY CONFORMER IN ENERGY ORDER
# ============================================================================
print("="*110)
print(f"{'Rank':>4s} {'E (kcal/mol)':>12s} {'dE':>6s} | {'Axial pair':<30s} {'Ang':>6s} | {'Equatorial':30s} | {'Desired?':>8s} | {'EtEq?':>5s}")
print(f"{'':>4s} {'':>12s} {'':>6s} | {'':30s} {'':>6s} | {'':30s} | {'':>8s} | {'':>5s}")
print("-"*110)

for i, d in enumerate(data[:50]):  # top 50
    de = d['energy'] - min_e
    ax_str = f"{d['axial_key'][0]} --- P --- {d['axial_key'][1]}"
    eq_str = ", ".join(d['eq_descs'])
    desired = "YES" if d['is_desired'] else ""
    et_eq = "yes" if d['et_equatorial'] else "no"

    print(f"{i+1:>4d} {d['energy']:>12.2f} {de:>+5.1f} | {ax_str:<30s} {d['axial_angle']:>5.1f}° | {eq_str:<30s} | {desired:>8s} | {et_eq:>5s}")

# ============================================================================
# BINNED STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STATISTICS BY ENERGY WINDOW")
print("="*80)

bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 999)]
for lo, hi in bins:
    subset = [d for d in data if lo <= (d['energy'] - min_e) < hi]
    if not subset:
        continue
    n = len(subset)
    n_desired = sum(1 for d in subset if d['is_desired'])
    n_et_eq = sum(1 for d in subset if d['et_equatorial'])

    # Count specific axial pairs
    pair_counts = {}
    for d in subset:
        pair_counts[d['axial_key']] = pair_counts.get(d['axial_key'], 0) + 1

    print(f"\n  dE = {lo:.1f} - {hi:.1f} kcal/mol  ({n} conformers)")
    print(f"  Desired (PNP + non-Et axial): {n_desired}/{n} = {100*n_desired/n:.1f}%")
    print(f"  Both ethoxides equatorial:    {n_et_eq}/{n} = {100*n_et_eq/n:.1f}%")
    print(f"  Axial pair breakdown:")
    for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
        marker = " <--" if "O-PNP" in pair and any(x in pair for x in ("OH", "O^-", "O=")) else ""
        print(f"    {pair[0]} --- P --- {pair[1]:<12s}: {count:>3d} ({100*count/n:>5.1f}%){marker}")

# ============================================================================
# KEY QUESTION: What % of the time are both ethoxides equatorial?
# ============================================================================
print("\n" + "="*80)
print("KEY QUESTION: How often are both ethoxides on the same plane (equatorial)?")
print("="*80)
n_et_eq_total = sum(1 for d in data if d['et_equatorial'])
print(f"\n  Overall: {n_et_eq_total}/{len(data)} = {100*n_et_eq_total/len(data):.1f}%")

top20 = data[:20]
n_et_eq_top20 = sum(1 for d in top20 if d['et_equatorial'])
print(f"  Top 20 lowest-energy: {n_et_eq_top20}/{len(top20)} = {100*n_et_eq_top20/len(top20):.1f}%")

top10 = data[:10]
n_et_eq_top10 = sum(1 for d in top10 if d['et_equatorial'])
print(f"  Top 10 lowest-energy: {n_et_eq_top10}/{len(top10)} = {100*n_et_eq_top10/len(top10):.1f}%")

top5 = data[:5]
n_et_eq_top5 = sum(1 for d in top5 if d['et_equatorial'])
print(f"  Top  5 lowest-energy: {n_et_eq_top5}/{len(top5)} = {100*n_et_eq_top5/len(top5):.1f}%")
