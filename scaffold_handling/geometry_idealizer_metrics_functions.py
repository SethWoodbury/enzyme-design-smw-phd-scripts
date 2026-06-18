# Comprehensive metrics functions to replace in idealize_rfdiffusion3_geometry.py

def calculate_comprehensive_metrics(self, pose, sfxn, fixed_residues, ligand_residues):
    """
    Calculate comprehensive validation metrics for geometry quality.

    Returns:
        dict: Nested dictionary of metrics (JSON-serializable)
    """
    metrics = {
        'metadata': {},
        'scores': {},
        'geometry': {},
        'constraints': {},
        'catalytic_residues': {},
        'quality_flags': {},
        'timing': self.timings.copy()
    }

    # === METADATA ===
    metrics['metadata']['structure_name'] = self.pdbname
    metrics['metadata']['timestamp'] = datetime.now().isoformat()
    metrics['metadata']['num_residues'] = pose.size()
    metrics['metadata']['num_fixed_residues'] = len(fixed_residues)
    metrics['metadata']['num_ligands'] = len(ligand_residues)
    metrics['metadata']['mobile_region_size'] = len(self.mobile_residues)

    # === ENERGY SCORES ===
    metrics['scores']['total_score'] = float(sfxn(pose))
    metrics['scores']['cart_bonded'] = float(pose.energies().total_energies()[ScoreType.cart_bonded])
    metrics['scores']['coordinate_constraint'] = float(pose.energies().total_energies()[ScoreType.coordinate_constraint])

    try:
        metrics['scores']['fa_rep'] = float(pose.energies().total_energies()[ScoreType.fa_rep])
        metrics['scores']['fa_atr'] = float(pose.energies().total_energies()[ScoreType.fa_atr])
        metrics['scores']['fa_sol'] = float(pose.energies().total_energies()[ScoreType.fa_sol])
        metrics['scores']['fa_elec'] = float(pose.energies().total_energies()[ScoreType.fa_elec])
        metrics['scores']['hbond_sc'] = float(pose.energies().total_energies()[ScoreType.hbond_sc])
        metrics['scores']['hbond_bb_sc'] = float(pose.energies().total_energies()[ScoreType.hbond_bb_sc])
    except:
        pass

    # === GEOMETRY QUALITY ===
    # Chain breaks
    chain_breaks = self.detect_chain_breaks(pose)
    metrics['geometry']['num_chain_breaks'] = len(chain_breaks)
    metrics['geometry']['chain_break_residues'] = [int(x) for x in chain_breaks]

    # Clashing residues
    clash_threshold = 10.0  # fa_rep per residue
    clashing_residues = []
    for i in range(1, pose.size() + 1):
        try:
            res_energy = pose.energies().residue_total_energies(i)[ScoreType.fa_rep]
            if res_energy > clash_threshold:
                clashing_residues.append({
                    'residue': int(i),
                    'pdb_id': f"{pose.pdb_info().chain(i)}{pose.pdb_info().number(i)}",
                    'resname': pose.residue(i).name3(),
                    'fa_rep': float(res_energy)
                })
        except:
            continue

    metrics['geometry']['num_clashing_residues'] = len(clashing_residues)
    metrics['geometry']['clashing_residues'] = clashing_residues[:20]  # Top 20

    # === CONSTRAINT SATISFACTION ===
    # Fixed atom displacement (CORRECTED - superimpose first!)
    if hasattr(self, 'original_pose'):
        # Superimpose on CA atoms to remove rigid-body motion
        from pyrosetta.rosetta.core.scoring import CA_rmsd

        # Calculate CA-RMSD (overall alignment quality)
        ca_rmsd = CA_rmsd(self.original_pose, pose)
        metrics['constraints']['ca_rmsd_overall'] = float(ca_rmsd)

        # Per-residue displacement for fixed atoms (after superposition)
        fixed_res_displacements = []

        for res_idx in list(fixed_residues.keys()) + ligand_residues:
            if res_idx > pose.size():
                continue

            try:
                res_orig = self.original_pose.residue(res_idx)
                res_final = pose.residue(res_idx)

                pdb_id = f"{pose.pdb_info().chain(res_idx)}{pose.pdb_info().number(res_idx)}"
                resname = res_final.name3()

                # Calculate displacement for constrained atoms only
                atom_spec = fixed_residues.get(res_idx, "ALL")

                if atom_spec == "ALL":
                    atom_names = [res_final.atom_name(i).strip()
                                 for i in range(1, res_final.natoms() + 1)
                                 if not res_final.atom_is_hydrogen(i)]
                else:
                    atom_names = [name.strip() for name in atom_spec.split(',')]

                sq_dev = []
                for atom_name in atom_names:
                    if not res_orig.has(atom_name) or not res_final.has(atom_name):
                        continue
                    dist = res_orig.xyz(atom_name).distance(res_final.xyz(atom_name))
                    sq_dev.append(dist**2)

                if sq_dev:
                    rmsd = float(np.sqrt(np.mean(sq_dev)))
                    max_dev = float(np.sqrt(max(sq_dev)))

                    fixed_res_displacements.append({
                        'residue': int(res_idx),
                        'pdb_id': pdb_id,
                        'resname': resname,
                        'atom_spec': atom_spec,
                        'num_atoms': len(sq_dev),
                        'rmsd': rmsd,
                        'max_displacement': max_dev
                    })
            except Exception as e:
                continue

        metrics['constraints']['fixed_residue_displacements'] = fixed_res_displacements

        # Summary statistics
        if fixed_res_displacements:
            all_rmsds = [x['rmsd'] for x in fixed_res_displacements]
            all_max_disps = [x['max_displacement'] for x in fixed_res_displacements]
            metrics['constraints']['mean_fixed_rmsd'] = float(np.mean(all_rmsds))
            metrics['constraints']['max_fixed_rmsd'] = float(np.max(all_rmsds))
            metrics['constraints']['mean_max_displacement'] = float(np.mean(all_max_disps))
            metrics['constraints']['max_displacement_overall'] = float(np.max(all_max_disps))
        else:
            metrics['constraints']['mean_fixed_rmsd'] = 0.0
            metrics['constraints']['max_fixed_rmsd'] = 0.0

    # === CATALYTIC RESIDUE DETAILS ===
    cat_res_details = []

    for res_idx, atom_spec in fixed_residues.items():
        if res_idx > pose.size():
            continue

        residue = pose.residue(res_idx)
        pdb_id = f"{pose.pdb_info().chain(res_idx)}{pose.pdb_info().number(res_idx)}"

        # Per-residue cart_bonded score
        try:
            res_cart_bonded = float(pose.energies().residue_total_energies(res_idx)[ScoreType.cart_bonded])
        except:
            res_cart_bonded = 0.0

        # Per-residue fa_rep (clashes)
        try:
            res_fa_rep = float(pose.energies().residue_total_energies(res_idx)[ScoreType.fa_rep])
        except:
            res_fa_rep = 0.0

        cat_res_details.append({
            'residue': int(res_idx),
            'pdb_id': pdb_id,
            'resname': residue.name3(),
            'atom_spec': atom_spec,
            'cart_bonded': res_cart_bonded,
            'fa_rep': res_fa_rep,
            'is_clashing': res_fa_rep > clash_threshold
        })

    metrics['catalytic_residues']['details'] = cat_res_details

    # Summary
    if cat_res_details:
        metrics['catalytic_residues']['mean_cart_bonded'] = float(np.mean([x['cart_bonded'] for x in cat_res_details]))
        metrics['catalytic_residues']['max_cart_bonded'] = float(np.max([x['cart_bonded'] for x in cat_res_details]))
        metrics['catalytic_residues']['num_clashing'] = sum(1 for x in cat_res_details if x['is_clashing'])

    # === QUALITY FLAGS & EXPLANATIONS ===
    flags = []
    reasons_failed = []

    # Check chain breaks
    if metrics['geometry']['num_chain_breaks'] == 0:
        flags.append('no_chain_breaks')
    else:
        reasons_failed.append(f"{metrics['geometry']['num_chain_breaks']} chain breaks detected")

    # Check clashes
    if metrics['geometry']['num_clashing_residues'] < 5:
        flags.append('low_clashes')
    else:
        reasons_failed.append(f"{metrics['geometry']['num_clashing_residues']} clashing residues (threshold: <5)")

    # Check fixed atom displacement
    if 'max_fixed_rmsd' in metrics['constraints']:
        if metrics['constraints']['max_fixed_rmsd'] < 0.1:
            flags.append('tight_constraints')
        else:
            reasons_failed.append(f"Fixed atoms moved {metrics['constraints']['max_fixed_rmsd']:.3f}Å (threshold: <0.1Å)")

    # Overall geometry acceptable
    geometry_acceptable = (
        metrics['geometry']['num_chain_breaks'] == 0 and
        metrics['geometry']['num_clashing_residues'] < 5 and
        metrics['constraints'].get('max_fixed_rmsd', 1.0) < 0.1
    )

    metrics['quality_flags']['geometry_acceptable'] = geometry_acceptable
    metrics['quality_flags']['passed_checks'] = flags
    metrics['quality_flags']['failed_checks'] = reasons_failed
    metrics['quality_flags']['explanation'] = (
        "All checks passed" if geometry_acceptable
        else "Failed: " + "; ".join(reasons_failed)
    )

    return metrics


def write_metrics_json(self, metrics, output_path):
    """
    Write validation metrics to JSON file (human-readable, nested structure).

    Args:
        metrics: Dictionary of metrics
        output_path: Path for output JSON (will use PDB basename)
    """
    # Determine JSON path (same basename as PDB)
    json_path = output_path.replace('.pdb', '_metrics.json')

    # Write to JSON with nice formatting
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=False)

    print(f"✓ Metrics written to: {json_path}")

    return json_path
