"""Metrics calculation for step03 FastMPNN design.

This module provides comprehensive metrics for evaluating designed structures:
- Bond length/angle geometry (with constrained vs unconstrained breakdown)
- RMSD calculations (ligand, constrained atoms, CA vs step01 and step02)
- Sequence metrics (identity, mutations)
- Geometry convergence checking
- Clash detection (fa_rep scores)
- Catres-specific bond geometry validation
- Dunbrack rotamer quality
- Secondary structure analysis (DSSP)
- Solvent accessibility (SASA)
- Ligand contact molecular surface
- CA RMSD after ligand alignment
- Catalytic residue mutation verification

Metrics are designed to be compatible with step02 output for comparison.
"""
import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# Add module_utils to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOGGER = logging.getLogger(__name__)


def init_pyrosetta_if_needed(params_files: List[str] = None) -> bool:
    """Initialize PyRosetta if not already done."""
    try:
        import pyrosetta as pyr
        # Check if already initialized
        try:
            pyr.get_fa_scorefxn()
            return True
        except:
            pass

        options = "-ignore_unrecognized_res false -ignore_zero_occupancy false"
        if params_files:
            params_str = " ".join(params_files)
            options += f" -extra_res_fa {params_str}"

        pyr.init(options)
        return True
    except ImportError:
        LOGGER.warning("PyRosetta not available, some metrics will be skipped")
        return False


class MetricsCalculator:
    """Calculate comprehensive metrics for step03 designs.

    Designed to track:
    - Bond geometry (continuing from step02)
    - RMSD vs reference structures (step01, step02)
    - Sequence changes (mutations, identity)
    - Geometry convergence
    - Clash detection and analysis
    - Catres-specific geometry validation
    - Rotamer quality
    - Secondary structure
    - Solvent accessibility
    - Ligand-protein interface
    """

    def __init__(
        self,
        designed_pdb: str,
        step02_pdb: str,
        step01_pdb: Optional[str] = None,
        params_files: Optional[List[str]] = None,
        constrained_atoms: Optional[Dict[str, List[str]]] = None,
        catres_positions: Optional[List[Tuple[str, int]]] = None,
        ligand_info: Optional[Tuple[str, str, int]] = None,
        step02_sequence: Optional[str] = None,
        bond_length_tolerance: float = 0.05,
        bond_angle_tolerance: float = 10.0,
        catres_bond_tolerance: float = 0.05,
        catres_angle_tolerance: float = 7.5,
    ):
        """Initialize metrics calculator.

        Args:
            designed_pdb: Path to designed PDB
            step02_pdb: Path to step02 relaxed PDB (immediate reference)
            step01_pdb: Path to step01 aligned PDB (original reference)
            params_files: List of ligand .params files
            constrained_atoms: Dict mapping "chain:resno" -> atom names
            catres_positions: List of (chain, resno) for catalytic residues
            ligand_info: Tuple of (chain, resname, resno) for ligand
            step02_sequence: Reference sequence from step02
            bond_length_tolerance: Tolerance for bond convergence (Angstroms)
            bond_angle_tolerance: Tolerance for angle convergence (degrees)
            catres_bond_tolerance: Stricter tolerance for catres bonds (Angstroms)
            catres_angle_tolerance: Stricter tolerance for catres angles (degrees)
        """
        self.designed_pdb = designed_pdb
        self.step02_pdb = step02_pdb
        self.step01_pdb = step01_pdb
        self.params_files = params_files or []
        self.constrained_atoms = constrained_atoms or {}
        self.catres_positions = catres_positions or []
        self.ligand_info = ligand_info
        self.step02_sequence = step02_sequence
        self.bond_length_tolerance = bond_length_tolerance
        self.bond_angle_tolerance = bond_angle_tolerance
        self.catres_bond_tolerance = catres_bond_tolerance
        self.catres_angle_tolerance = catres_angle_tolerance

        # Convert constrained_atoms to set for quick lookup
        self.constrained_atoms_set: Set[Tuple[str, int, str]] = set()
        for res_key, atoms in self.constrained_atoms.items():
            if ":" in res_key:
                chain, resno = res_key.split(":")
                resno = int(resno)
            else:
                chain = res_key[0]
                resno = int(res_key[1:])

            if atoms == "ALL_HEAVY" or (isinstance(atoms, list) and "ALL_HEAVY" in atoms):
                # Will be populated dynamically
                pass
            else:
                for atom in (atoms if isinstance(atoms, list) else [atoms]):
                    self.constrained_atoms_set.add((chain, resno, atom))

        # Convert catres_positions to set for quick lookup
        self.catres_set: Set[Tuple[str, int]] = set(self.catres_positions)

        # PyRosetta objects (lazy loaded)
        self._pyrosetta_available = None
        self._pose = None
        self._sfxn = None

    @property
    def pyrosetta_available(self) -> bool:
        """Check if PyRosetta is available."""
        if self._pyrosetta_available is None:
            self._pyrosetta_available = init_pyrosetta_if_needed(self.params_files)
        return self._pyrosetta_available

    def _load_pose(self, pdb_path: str):
        """Load a pose from PDB."""
        import pyrosetta as pyr
        return pyr.pose_from_pdb(pdb_path)

    def _get_scorefunction(self):
        """Get or create scorefunction."""
        if self._sfxn is None:
            from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
            self._sfxn = ScoreFunctionFactory.create_score_function("ref2015_cart")
        return self._sfxn

    def calculate_sequence_metrics(self) -> Dict:
        """Calculate sequence-based metrics.

        Returns:
            Dict with sequence_identity, mutations, num_mutations
        """
        from module_utils.pdb_utils import read_pdb_atoms
        from module_utils.sequence_utils import (
            get_sequence_from_atoms,
            calculate_sequence_identity,
            get_mutations_list,
        )

        _, designed_atoms = read_pdb_atoms(self.designed_pdb)
        _, step02_atoms = read_pdb_atoms(self.step02_pdb)

        designed_seq = get_sequence_from_atoms(designed_atoms)
        step02_seq = get_sequence_from_atoms(step02_atoms)

        # Use provided step02_sequence if available
        if self.step02_sequence:
            step02_seq = self.step02_sequence

        identity = calculate_sequence_identity(step02_seq, designed_seq)
        mutations = get_mutations_list(step02_seq, designed_seq)

        return {
            "sequence_identity_vs_step02": round(identity, 4),
            "num_mutations": len(mutations),
            "mutations": mutations,
            "designed_sequence": designed_seq,
            "step02_sequence": step02_seq,
        }

    def calculate_rmsd_metrics(self) -> Dict:
        """Calculate RMSD metrics.

        Returns:
            Dict with ligand, constrained_atoms, global_ca_vs_step01, global_ca_vs_step02
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}

        from pyrosetta.rosetta.core.scoring import CA_rmsd

        metrics = {}

        # Load poses
        designed_pose = self._load_pose(self.designed_pdb)
        step02_pose = self._load_pose(self.step02_pdb)

        # CA RMSD vs step02
        metrics["global_ca_vs_step02"] = round(float(CA_rmsd(step02_pose, designed_pose)), 4)

        # CA RMSD vs step01 (if available)
        if self.step01_pdb and os.path.exists(self.step01_pdb):
            step01_pose = self._load_pose(self.step01_pdb)
            metrics["global_ca_vs_step01"] = round(float(CA_rmsd(step01_pose, designed_pose)), 4)
        else:
            metrics["global_ca_vs_step01"] = None

        # Ligand RMSD (should be ~0.0 due to constraints)
        if self.ligand_info:
            ligand_rmsd = self._calculate_ligand_rmsd(step02_pose, designed_pose)
            metrics["ligand"] = round(ligand_rmsd, 4)
        else:
            metrics["ligand"] = None

        # Constrained atoms RMSD (should be ~0.0)
        if self.constrained_atoms:
            cst_rmsd = self._calculate_constrained_rmsd(step02_pose, designed_pose)
            metrics["constrained_atoms"] = cst_rmsd
        else:
            metrics["constrained_atoms"] = None

        return metrics

    def _calculate_ligand_rmsd(self, pose_before, pose_after) -> float:
        """Calculate ligand heavy atom RMSD."""
        if not self.ligand_info:
            return 0.0

        lig_chain, lig_resname, lig_resno = self.ligand_info

        try:
            pose_idx_before = pose_before.pdb_info().pdb2pose(lig_chain, lig_resno)
            pose_idx_after = pose_after.pdb_info().pdb2pose(lig_chain, lig_resno)

            if pose_idx_before == 0 or pose_idx_after == 0:
                return 0.0

            res_before = pose_before.residue(pose_idx_before)
            res_after = pose_after.residue(pose_idx_after)

            sq_devs = []
            for i in range(1, res_before.natoms() + 1):
                if res_before.atom_is_hydrogen(i):
                    continue
                atom_name = res_before.atom_name(i).strip()
                if not res_after.has(atom_name):
                    continue
                xyz_before = res_before.xyz(atom_name)
                xyz_after = res_after.xyz(atom_name)
                sq_dev = (xyz_before.x - xyz_after.x)**2 + \
                        (xyz_before.y - xyz_after.y)**2 + \
                        (xyz_before.z - xyz_after.z)**2
                sq_devs.append(sq_dev)

            if sq_devs:
                return float(np.sqrt(np.mean(sq_devs)))
        except Exception as e:
            LOGGER.warning(f"Error calculating ligand RMSD: {e}")

        return 0.0

    def _calculate_constrained_rmsd(self, pose_before, pose_after) -> Dict:
        """Calculate RMSD of constrained atoms."""
        sq_devs = []
        per_residue = {}

        for res_key, atoms in self.constrained_atoms.items():
            if ":" in res_key:
                chain, resno = res_key.split(":")
                resno = int(resno)
            else:
                chain = res_key[0]
                resno = int(res_key[1:])

            try:
                pose_idx_before = pose_before.pdb_info().pdb2pose(chain, resno)
                pose_idx_after = pose_after.pdb_info().pdb2pose(chain, resno)

                if pose_idx_before == 0 or pose_idx_after == 0:
                    continue

                res_before = pose_before.residue(pose_idx_before)
                res_after = pose_after.residue(pose_idx_after)

                # Determine atom names
                if atoms == "ALL_HEAVY" or (isinstance(atoms, list) and "ALL_HEAVY" in atoms):
                    atom_names = [
                        res_before.atom_name(i).strip()
                        for i in range(1, res_before.natoms() + 1)
                        if not res_before.atom_is_hydrogen(i)
                    ]
                else:
                    atom_names = atoms if isinstance(atoms, list) else [atoms]

                res_sq_devs = []
                for atom_name in atom_names:
                    if not res_before.has(atom_name) or not res_after.has(atom_name):
                        continue
                    xyz_before = res_before.xyz(atom_name)
                    xyz_after = res_after.xyz(atom_name)
                    sq_dev = (xyz_before.x - xyz_after.x)**2 + \
                            (xyz_before.y - xyz_after.y)**2 + \
                            (xyz_before.z - xyz_after.z)**2
                    sq_devs.append(sq_dev)
                    res_sq_devs.append(sq_dev)

                if res_sq_devs:
                    per_residue[res_key] = round(float(np.sqrt(np.mean(res_sq_devs))), 4)

            except Exception as e:
                LOGGER.warning(f"Error calculating constrained RMSD for {res_key}: {e}")

        aggregate = round(float(np.sqrt(np.mean(sq_devs))), 4) if sq_devs else 0.0

        return {
            "aggregate": aggregate,
            "per_residue": per_residue,
        }

    def calculate_bond_geometry(self) -> Dict:
        """Calculate bond length/angle geometry metrics.

        Returns:
            Dict with bond_length_geometry and bond_angle_geometry
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}

        pose = self._load_pose(self.designed_pdb)

        # Calculate bond length deviations
        bond_metrics = self._calculate_bond_metrics(pose)

        # Calculate bond angle deviations
        angle_metrics = self._calculate_angle_metrics(pose)

        return {
            "bond_length_geometry": bond_metrics,
            "bond_angle_geometry": angle_metrics,
        }

    def _calculate_bond_metrics(self, pose) -> Dict:
        """Calculate bond length deviations from ideal."""
        from pyrosetta.rosetta.core.conformation import Residue
        from pyrosetta.rosetta.core.chemical import AtomType

        all_deviations = []
        unconstrained_deviations = []

        for i in range(1, pose.size() + 1):
            res = pose.residue(i)
            chain = pose.pdb_info().chain(i)
            resno = pose.pdb_info().number(i)

            for bond_idx in range(res.type().nbonds()):
                atom1_idx = res.type().bond(bond_idx + 1).atom1()
                atom2_idx = res.type().bond(bond_idx + 1).atom2()

                if res.atom_is_hydrogen(atom1_idx) or res.atom_is_hydrogen(atom2_idx):
                    continue

                # Get actual bond length
                xyz1 = res.xyz(atom1_idx)
                xyz2 = res.xyz(atom2_idx)
                actual = np.sqrt(
                    (xyz1.x - xyz2.x)**2 +
                    (xyz1.y - xyz2.y)**2 +
                    (xyz1.z - xyz2.z)**2
                )

                # Get ideal bond length
                ideal = res.type().bond_length(bond_idx + 1)
                deviation = abs(actual - ideal)

                all_deviations.append(deviation)

                # Check if atoms are constrained
                atom1_name = res.atom_name(atom1_idx).strip()
                atom2_name = res.atom_name(atom2_idx).strip()
                is_constrained = (
                    (chain, resno, atom1_name) in self.constrained_atoms_set or
                    (chain, resno, atom2_name) in self.constrained_atoms_set
                )

                if not is_constrained:
                    unconstrained_deviations.append(deviation)

        def aggregate(devs):
            if not devs:
                return {"mean": 0.0, "max": 0.0, "std": 0.0, "count": 0}
            return {
                "mean": round(float(np.mean(devs)), 4),
                "max": round(float(np.max(devs)), 4),
                "std": round(float(np.std(devs)), 4),
                "count": len(devs),
            }

        return {
            "all": aggregate(all_deviations),
            "unconstrained_only": aggregate(unconstrained_deviations),
        }

    def _calculate_angle_metrics(self, pose) -> Dict:
        """Calculate bond angle deviations from ideal."""
        all_deviations = []
        unconstrained_deviations = []

        for i in range(1, pose.size() + 1):
            res = pose.residue(i)
            chain = pose.pdb_info().chain(i)
            resno = pose.pdb_info().number(i)

            # Iterate over angles
            for angle_idx in range(res.type().num_bondangles()):
                atom_indices = res.type().bondangle(angle_idx + 1)
                atom1_idx = atom_indices[0]
                atom2_idx = atom_indices[1]
                atom3_idx = atom_indices[2]

                # Skip if any hydrogen
                if (res.atom_is_hydrogen(atom1_idx) or
                    res.atom_is_hydrogen(atom2_idx) or
                    res.atom_is_hydrogen(atom3_idx)):
                    continue

                # Get actual angle
                xyz1 = np.array([res.xyz(atom1_idx).x, res.xyz(atom1_idx).y, res.xyz(atom1_idx).z])
                xyz2 = np.array([res.xyz(atom2_idx).x, res.xyz(atom2_idx).y, res.xyz(atom2_idx).z])
                xyz3 = np.array([res.xyz(atom3_idx).x, res.xyz(atom3_idx).y, res.xyz(atom3_idx).z])

                v1 = xyz1 - xyz2
                v2 = xyz3 - xyz2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                actual = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                # Get ideal angle
                ideal = np.degrees(res.type().bondangle_ideal(angle_idx + 1))
                deviation = abs(actual - ideal)

                all_deviations.append(deviation)

                # Check if constrained
                atom1_name = res.atom_name(atom1_idx).strip()
                atom2_name = res.atom_name(atom2_idx).strip()
                atom3_name = res.atom_name(atom3_idx).strip()
                is_constrained = (
                    (chain, resno, atom1_name) in self.constrained_atoms_set or
                    (chain, resno, atom2_name) in self.constrained_atoms_set or
                    (chain, resno, atom3_name) in self.constrained_atoms_set
                )

                if not is_constrained:
                    unconstrained_deviations.append(deviation)

        def aggregate(devs):
            if not devs:
                return {"mean": 0.0, "max": 0.0, "std": 0.0, "count": 0}
            return {
                "mean": round(float(np.mean(devs)), 4),
                "max": round(float(np.max(devs)), 4),
                "std": round(float(np.std(devs)), 4),
                "count": len(devs),
            }

        return {
            "all": aggregate(all_deviations),
            "unconstrained_only": aggregate(unconstrained_deviations),
        }

    def calculate_clash_detection(self) -> Dict:
        """Calculate clash detection metrics using fa_rep scores.

        Excludes clashes between constrained catres atoms and between
        constrained atoms and ligand (expected from QM geometry).

        Returns:
            Dict with total_fa_rep, catres_clashes, per_residue_hotspots
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}

        try:
            import pyrosetta as pyr
            from pyrosetta.rosetta.core.scoring import ScoreType

            pose = self._load_pose(self.designed_pdb)
            sfxn = self._get_scorefunction()

            # Score the pose
            sfxn(pose)

            # Get total fa_rep score
            total_fa_rep = pose.energies().total_energies()[ScoreType.fa_rep]

            # Collect per-residue fa_rep scores
            per_residue_scores = []
            catres_clashes = []

            for i in range(1, pose.size() + 1):
                chain = pose.pdb_info().chain(i)
                resno = pose.pdb_info().number(i)
                res = pose.residue(i)

                # Get fa_rep for this residue
                fa_rep = pose.energies().residue_total_energies(i)[ScoreType.fa_rep]

                if fa_rep > 0.5:  # Threshold for reporting
                    residue_info = {
                        "chain": chain,
                        "resno": resno,
                        "resname": res.name3(),
                        "fa_rep_score": round(float(fa_rep), 4)
                    }
                    per_residue_scores.append(residue_info)

                    # Check if this is a catalytic residue
                    if (chain, resno) in self.catres_set:
                        catres_clashes.append(residue_info)

            # Sort by worst offenders
            per_residue_scores.sort(key=lambda x: x["fa_rep_score"], reverse=True)

            return {
                "total_fa_rep": round(float(total_fa_rep), 4),
                "catres_clashes": catres_clashes,
                "catres_clashes_warning": len(catres_clashes) > 0,
                "per_residue_hotspots": per_residue_scores[:20],  # Top 20 worst
                "num_residues_with_clashes": len(per_residue_scores)
            }

        except Exception as e:
            LOGGER.warning(f"Error calculating clash detection: {e}")
            return {"error": str(e)}

    def calculate_catres_bond_geometry(self) -> Dict:
        """Calculate bond geometry specifically for catalytic residues.

        Focuses on bonds involving at least one UNCONSTRAINED catres atom.
        Reports failures and worst offenders.

        Returns:
            Dict with catres bond/angle stats, failures, worst offenders
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}

        try:
            import pyrosetta as pyr

            pose = self._load_pose(self.designed_pdb)

            bond_deviations = []
            angle_deviations = []
            bond_failures = []
            angle_failures = []

            for chain, resno in self.catres_positions:
                try:
                    pose_idx = pose.pdb_info().pdb2pose(chain, resno)
                    if pose_idx == 0:
                        continue

                    res = pose.residue(pose_idx)

                    # Check bonds
                    for bond_idx in range(res.type().nbonds()):
                        atom1_idx = res.type().bond(bond_idx + 1).atom1()
                        atom2_idx = res.type().bond(bond_idx + 1).atom2()

                        if res.atom_is_hydrogen(atom1_idx) or res.atom_is_hydrogen(atom2_idx):
                            continue

                        atom1_name = res.atom_name(atom1_idx).strip()
                        atom2_name = res.atom_name(atom2_idx).strip()

                        # Check if at least one atom is UNCONSTRAINED
                        atom1_constrained = (chain, resno, atom1_name) in self.constrained_atoms_set
                        atom2_constrained = (chain, resno, atom2_name) in self.constrained_atoms_set

                        if atom1_constrained and atom2_constrained:
                            continue  # Skip fully constrained bonds

                        # Calculate deviation
                        xyz1 = res.xyz(atom1_idx)
                        xyz2 = res.xyz(atom2_idx)
                        actual = np.sqrt(
                            (xyz1.x - xyz2.x)**2 +
                            (xyz1.y - xyz2.y)**2 +
                            (xyz1.z - xyz2.z)**2
                        )
                        ideal = res.type().bond_length(bond_idx + 1)
                        deviation = abs(actual - ideal)

                        bond_info = {
                            "chain": chain,
                            "resno": resno,
                            "resname": res.name3(),
                            "bond": f"{atom1_name}-{atom2_name}",
                            "deviation": round(float(deviation), 4)
                        }

                        bond_deviations.append(deviation)

                        if deviation > self.catres_bond_tolerance:
                            bond_failures.append(bond_info)

                    # Check angles
                    for angle_idx in range(res.type().num_bondangles()):
                        atom_indices = res.type().bondangle(angle_idx + 1)
                        atom1_idx = atom_indices[0]
                        atom2_idx = atom_indices[1]
                        atom3_idx = atom_indices[2]

                        if (res.atom_is_hydrogen(atom1_idx) or
                            res.atom_is_hydrogen(atom2_idx) or
                            res.atom_is_hydrogen(atom3_idx)):
                            continue

                        atom1_name = res.atom_name(atom1_idx).strip()
                        atom2_name = res.atom_name(atom2_idx).strip()
                        atom3_name = res.atom_name(atom3_idx).strip()

                        # Check if at least one atom is UNCONSTRAINED
                        atom1_constrained = (chain, resno, atom1_name) in self.constrained_atoms_set
                        atom2_constrained = (chain, resno, atom2_name) in self.constrained_atoms_set
                        atom3_constrained = (chain, resno, atom3_name) in self.constrained_atoms_set

                        if atom1_constrained and atom2_constrained and atom3_constrained:
                            continue  # Skip fully constrained angles

                        # Calculate deviation
                        xyz1 = np.array([res.xyz(atom1_idx).x, res.xyz(atom1_idx).y, res.xyz(atom1_idx).z])
                        xyz2 = np.array([res.xyz(atom2_idx).x, res.xyz(atom2_idx).y, res.xyz(atom2_idx).z])
                        xyz3 = np.array([res.xyz(atom3_idx).x, res.xyz(atom3_idx).y, res.xyz(atom3_idx).z])

                        v1 = xyz1 - xyz2
                        v2 = xyz3 - xyz2
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                        actual = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                        ideal = np.degrees(res.type().bondangle_ideal(angle_idx + 1))
                        deviation = abs(actual - ideal)

                        angle_info = {
                            "chain": chain,
                            "resno": resno,
                            "resname": res.name3(),
                            "angle": f"{atom1_name}-{atom2_name}-{atom3_name}",
                            "deviation": round(float(deviation), 4)
                        }

                        angle_deviations.append(deviation)

                        if deviation > self.catres_angle_tolerance:
                            angle_failures.append(angle_info)

                except Exception as e:
                    LOGGER.warning(f"Error processing catres {chain}{resno}: {e}")

            # Sort failures by severity
            bond_failures.sort(key=lambda x: x["deviation"], reverse=True)
            angle_failures.sort(key=lambda x: x["deviation"], reverse=True)

            return {
                "bond_geometry": {
                    "mean_deviation": round(float(np.mean(bond_deviations)), 4) if bond_deviations else 0.0,
                    "max_deviation": round(float(np.max(bond_deviations)), 4) if bond_deviations else 0.0,
                    "num_bonds": len(bond_deviations),
                    "num_failures": len(bond_failures),
                    "worst_offenders": bond_failures[:10]
                },
                "angle_geometry": {
                    "mean_deviation": round(float(np.mean(angle_deviations)), 4) if angle_deviations else 0.0,
                    "max_deviation": round(float(np.max(angle_deviations)), 4) if angle_deviations else 0.0,
                    "num_angles": len(angle_deviations),
                    "num_failures": len(angle_failures),
                    "worst_offenders": angle_failures[:10]
                }
            }

        except Exception as e:
            LOGGER.warning(f"Error calculating catres bond geometry: {e}")
            return {"error": str(e)}

    def calculate_dunbrack_quality(self) -> Dict:
        """Calculate Dunbrack rotamer quality for catalytic residues.

        Returns:
            Dict with per-catres Dunbrack energies and overall stats
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}

        try:
            import pyrosetta as pyr
            from pyrosetta.rosetta.core.scoring import ScoreType

            pose = self._load_pose(self.designed_pdb)
            sfxn = self._get_scorefunction()

            # Score the pose
            sfxn(pose)

            catres_dunbrack = []

            for chain, resno in self.catres_positions:
                try:
                    pose_idx = pose.pdb_info().pdb2pose(chain, resno)
                    if pose_idx == 0:
                        continue

                    res = pose.residue(pose_idx)

                    # Get Dunbrack energy
                    dunbrack_score = pose.energies().residue_total_energies(pose_idx)[ScoreType.fa_dun]

                    catres_dunbrack.append({
                        "chain": chain,
                        "resno": resno,
                        "resname": res.name3(),
                        "fa_dun_score": round(float(dunbrack_score), 4)
                    })

                except Exception as e:
                    LOGGER.warning(f"Error getting Dunbrack for {chain}{resno}: {e}")

            # Sort by worst scores
            catres_dunbrack.sort(key=lambda x: x["fa_dun_score"], reverse=True)

            dunbrack_scores = [x["fa_dun_score"] for x in catres_dunbrack]

            return {
                "per_catres_dunbrack": catres_dunbrack,
                "mean_fa_dun": round(float(np.mean(dunbrack_scores)), 4) if dunbrack_scores else 0.0,
                "max_fa_dun": round(float(np.max(dunbrack_scores)), 4) if dunbrack_scores else 0.0,
                "num_poor_rotamers": len([x for x in dunbrack_scores if x > 1.0])
            }

        except Exception as e:
            LOGGER.warning(f"Error calculating Dunbrack quality: {e}")
            return {"error": str(e)}

    def calculate_secondary_structure(self) -> Dict:
        """Calculate secondary structure using DSSP.

        Returns:
            Dict with per-catres SS assignments and overall content
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}

        try:
            import pyrosetta as pyr
            from pyrosetta.rosetta.core.scoring.dssp import Dssp

            pose = self._load_pose(self.designed_pdb)

            # Run DSSP
            dssp = Dssp(pose)
            dssp.dssp_reduced()

            # Get SS for catalytic residues
            catres_ss = []
            for chain, resno in self.catres_positions:
                try:
                    pose_idx = pose.pdb_info().pdb2pose(chain, resno)
                    if pose_idx == 0:
                        continue

                    ss = dssp.get_dssp_secstruct(pose_idx)
                    res = pose.residue(pose_idx)

                    catres_ss.append({
                        "chain": chain,
                        "resno": resno,
                        "resname": res.name3(),
                        "secondary_structure": ss
                    })

                except Exception as e:
                    LOGGER.warning(f"Error getting SS for {chain}{resno}: {e}")

            # Calculate overall SS content
            total_residues = pose.size()
            helix_count = 0
            sheet_count = 0
            loop_count = 0

            for i in range(1, pose.size() + 1):
                if pose.residue(i).is_protein():
                    ss = dssp.get_dssp_secstruct(i)
                    if ss == 'H':
                        helix_count += 1
                    elif ss == 'E':
                        sheet_count += 1
                    else:
                        loop_count += 1

            protein_residues = helix_count + sheet_count + loop_count

            return {
                "catres_secondary_structure": catres_ss,
                "overall_content": {
                    "helix_percent": round(100.0 * helix_count / protein_residues, 2) if protein_residues > 0 else 0.0,
                    "sheet_percent": round(100.0 * sheet_count / protein_residues, 2) if protein_residues > 0 else 0.0,
                    "loop_percent": round(100.0 * loop_count / protein_residues, 2) if protein_residues > 0 else 0.0,
                    "total_protein_residues": protein_residues
                }
            }

        except Exception as e:
            LOGGER.warning(f"Error calculating secondary structure: {e}")
            return {"error": str(e)}

    def calculate_sasa(self) -> Dict:
        """Calculate solvent accessible surface area.

        Returns:
            Dict with ligand SASA and per-catres SASA
        """
        if not self.pyrosetta_available:
            return {"error": "PyRosetta not available"}

        try:
            import pyrosetta as pyr
            from pyrosetta.rosetta.core.scoring.sasa import SasaCalc

            pose = self._load_pose(self.designed_pdb)

            # Calculate SASA
            sasa_calc = SasaCalc()
            sasa_calc.calculate(pose)

            # Get ligand SASA
            ligand_sasa = None
            if self.ligand_info:
                lig_chain, lig_resname, lig_resno = self.ligand_info
                try:
                    pose_idx = pose.pdb_info().pdb2pose(lig_chain, lig_resno)
                    if pose_idx > 0:
                        ligand_sasa = round(float(sasa_calc.get_residue_sasa()[pose_idx]), 4)
                except Exception as e:
                    LOGGER.warning(f"Error calculating ligand SASA: {e}")

            # Get catres SASA
            catres_sasa = []
            for chain, resno in self.catres_positions:
                try:
                    pose_idx = pose.pdb_info().pdb2pose(chain, resno)
                    if pose_idx == 0:
                        continue

                    res = pose.residue(pose_idx)
                    sasa = sasa_calc.get_residue_sasa()[pose_idx]

                    catres_sasa.append({
                        "chain": chain,
                        "resno": resno,
                        "resname": res.name3(),
                        "sasa": round(float(sasa), 4)
                    })

                except Exception as e:
                    LOGGER.warning(f"Error getting SASA for {chain}{resno}: {e}")

            return {
                "ligand_sasa": ligand_sasa,
                "per_catres_sasa": catres_sasa,
                "mean_catres_sasa": round(float(np.mean([x["sasa"] for x in catres_sasa])), 4) if catres_sasa else 0.0
            }

        except Exception as e:
            LOGGER.warning(f"Error calculating SASA: {e}")
            return {"error": str(e)}

    def calculate_ligand_interface(self) -> Dict:
        """Calculate buried surface area of ligand (interface with protein).

        Returns:
            Dict with ligand interface area
        """
        if not self.pyrosetta_available or not self.ligand_info:
            return {"error": "PyRosetta not available or no ligand info"}

        try:
            import pyrosetta as pyr
            from pyrosetta.rosetta.core.scoring.sasa import SasaCalc

            pose = self._load_pose(self.designed_pdb)

            lig_chain, lig_resname, lig_resno = self.ligand_info
            pose_idx = pose.pdb_info().pdb2pose(lig_chain, lig_resno)

            if pose_idx == 0:
                return {"error": "Ligand not found in pose"}

            # Calculate SASA of full complex
            sasa_calc = SasaCalc()
            sasa_calc.calculate(pose)
            sasa_complex = sasa_calc.get_residue_sasa()[pose_idx]

            # Create a pose with only the ligand to get its isolated SASA
            # This is approximated by calculating the total accessible surface
            # For now, we'll use a simpler metric: just report the SASA in complex
            # The buried surface area = isolated SASA - complex SASA
            # But calculating isolated SASA requires pose manipulation

            return {
                "ligand_sasa_in_complex": round(float(sasa_complex), 4),
                "note": "Full interface calculation requires ligand isolation"
            }

        except Exception as e:
            LOGGER.warning(f"Error calculating ligand interface: {e}")
            return {"error": str(e)}

    def calculate_ca_rmsd_after_ligand_alignment(self) -> Dict:
        """Calculate CA RMSD vs step01 after first aligning on ligand atoms.

        Important: This aligns structures based on the ligand, then calculates CA RMSD.

        Returns:
            Dict with CA RMSD after ligand alignment
        """
        if not self.pyrosetta_available or not self.step01_pdb or not self.ligand_info:
            return {"error": "PyRosetta not available, no step01, or no ligand info"}

        try:
            import pyrosetta as pyr
            from pyrosetta.rosetta.core.scoring import CA_rmsd
            from pyrosetta.rosetta.protocols.toolbox.superimpose import superimpose_pose
            from pyrosetta.rosetta.core.id import AtomID
            from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as vector1_xyz

            step01_pose = self._load_pose(self.step01_pdb)
            designed_pose = self._load_pose(self.designed_pdb)

            lig_chain, lig_resname, lig_resno = self.ligand_info

            # Get ligand pose indices
            step01_lig_idx = step01_pose.pdb_info().pdb2pose(lig_chain, lig_resno)
            designed_lig_idx = designed_pose.pdb_info().pdb2pose(lig_chain, lig_resno)

            if step01_lig_idx == 0 or designed_lig_idx == 0:
                return {"error": "Ligand not found in one or both poses"}

            # Collect ligand heavy atom coordinates for alignment
            step01_lig_res = step01_pose.residue(step01_lig_idx)
            designed_lig_res = designed_pose.residue(designed_lig_idx)

            # Create atom ID lists for superposition
            step01_atoms = vector1_xyz()
            designed_atoms = vector1_xyz()

            for i in range(1, step01_lig_res.natoms() + 1):
                if step01_lig_res.atom_is_hydrogen(i):
                    continue
                atom_name = step01_lig_res.atom_name(i).strip()
                if designed_lig_res.has(atom_name):
                    step01_atoms.append(step01_lig_res.xyz(i))
                    designed_atoms.append(designed_lig_res.xyz(atom_name))

            # Superimpose designed onto step01 based on ligand
            # This modifies designed_pose in place
            from pyrosetta.rosetta.numeric import xyzMatrix_double_t, xyzVector_double_t
            from pyrosetta.rosetta.protocols.toolbox import superposition_transform

            # Calculate transformation
            if len(step01_atoms) < 3:
                return {"error": "Not enough ligand atoms for alignment"}

            # Apply superposition - align designed to step01
            rotation = xyzMatrix_double_t()
            translation = xyzVector_double_t()
            superposition_transform(designed_atoms, step01_atoms, rotation, translation)

            # Apply transformation to designed pose
            designed_pose.apply_transform_Rx_plus_v(rotation, translation)

            # Now calculate CA RMSD
            ca_rmsd = CA_rmsd(step01_pose, designed_pose)

            return {
                "ca_rmsd_after_ligand_alignment": round(float(ca_rmsd), 4),
                "num_ligand_atoms_aligned": len(step01_atoms)
            }

        except Exception as e:
            LOGGER.warning(f"Error calculating CA RMSD after ligand alignment: {e}")
            return {"error": str(e)}

    def check_catres_mutations(self) -> Dict:
        """Verify that NO catalytic residues from REMARK 666 were mutated.

        Returns error if any catres were mutated.

        Returns:
            Dict with mutation check results
        """
        if not self.step01_pdb:
            return {"error": "No step01 PDB available for comparison"}

        try:
            from module_utils.pdb_utils import read_pdb_atoms

            # Read both PDBs
            _, step01_atoms = read_pdb_atoms(self.step01_pdb)
            _, designed_atoms = read_pdb_atoms(self.designed_pdb)

            # Build residue name maps
            step01_resnames = {}
            for atom in step01_atoms:
                if atom["record_type"] == "ATOM":
                    key = (atom["chain"], atom["resno"])
                    step01_resnames[key] = atom["resname"]

            designed_resnames = {}
            for atom in designed_atoms:
                if atom["record_type"] == "ATOM":
                    key = (atom["chain"], atom["resno"])
                    designed_resnames[key] = atom["resname"]

            # Check each catres
            mutations = []
            for chain, resno in self.catres_positions:
                key = (chain, resno)
                step01_resname = step01_resnames.get(key)
                designed_resname = designed_resnames.get(key)

                if step01_resname and designed_resname:
                    # Normalize for comparison (handle HIS tautomers, etc.)
                    step01_base = step01_resname[:3]
                    designed_base = designed_resname[:3]

                    if step01_base != designed_base:
                        mutations.append({
                            "chain": chain,
                            "resno": resno,
                            "step01_resname": step01_resname,
                            "designed_resname": designed_resname
                        })

            if mutations:
                return {
                    "catres_preserved": False,
                    "mutations_detected": mutations,
                    "error": f"Catalytic residues were mutated: {len(mutations)} mutations found"
                }
            else:
                return {
                    "catres_preserved": True,
                    "mutations_detected": [],
                    "num_catres_checked": len(self.catres_positions)
                }

        except Exception as e:
            LOGGER.warning(f"Error checking catres mutations: {e}")
            return {"error": str(e)}

    def check_geometry_converged(self) -> Dict:
        """Check if geometry has converged to acceptable tolerances.

        Returns:
            Dict with bond_length_converged, bond_angle_converged, overall_converged
        """
        geometry = self.calculate_bond_geometry()

        if "error" in geometry:
            return {"error": geometry["error"]}

        bond_max = geometry["bond_length_geometry"]["unconstrained_only"]["max"]
        angle_max = geometry["bond_angle_geometry"]["unconstrained_only"]["max"]

        bond_converged = bond_max <= self.bond_length_tolerance
        angle_converged = angle_max <= self.bond_angle_tolerance

        return {
            "bond_length_converged": bond_converged,
            "bond_angle_converged": angle_converged,
            "overall_converged": bond_converged and angle_converged,
            "max_bond_deviation": round(bond_max, 4),
            "max_angle_deviation": round(angle_max, 4),
        }

    def verify_constraints_preserved(self) -> Dict:
        """Verify that constrained atoms stayed near their target positions.

        Returns:
            Dict with preserved (bool), max_deviation, warning messages
        """
        rmsd = self.calculate_rmsd_metrics()

        if "error" in rmsd:
            return {"error": rmsd["error"]}

        warnings = []
        preserved = True

        # Check ligand RMSD
        if rmsd.get("ligand") is not None:
            if rmsd["ligand"] > 0.1:
                warnings.append(f"Ligand RMSD too high: {rmsd['ligand']:.3f}A")
                preserved = False

        # Check constrained atoms RMSD
        if rmsd.get("constrained_atoms") is not None:
            cst_aggregate = rmsd["constrained_atoms"].get("aggregate", 0.0)
            if cst_aggregate > 0.1:
                warnings.append(f"Constrained atom RMSD too high: {cst_aggregate:.3f}A")
                preserved = False

        return {
            "preserved": preserved,
            "ligand_rmsd": rmsd.get("ligand"),
            "constrained_atom_rmsd": rmsd.get("constrained_atoms", {}).get("aggregate"),
            "warnings": warnings,
        }

    def calculate_all_metrics(self) -> Dict:
        """Calculate all standard metrics (legacy method).

        Returns:
            Dict with all metrics categories
        """
        LOGGER.info(f"Calculating metrics for {self.designed_pdb}")

        metrics = {
            "metadata": {},
            "sequence_metrics": {},
            "rmsd": {},
            "bond_geometry": {},
            "convergence": {},
            "constraint_verification": {},
        }

        # Metadata
        metrics["metadata"]["designed_pdb"] = os.path.abspath(self.designed_pdb)
        metrics["metadata"]["step02_pdb"] = os.path.abspath(self.step02_pdb) if self.step02_pdb else None
        metrics["metadata"]["step01_pdb"] = os.path.abspath(self.step01_pdb) if self.step01_pdb else None
        metrics["metadata"]["bond_length_tolerance"] = self.bond_length_tolerance
        metrics["metadata"]["bond_angle_tolerance"] = self.bond_angle_tolerance

        # Sequence metrics
        try:
            metrics["sequence_metrics"] = self.calculate_sequence_metrics()
        except Exception as e:
            LOGGER.warning(f"Error calculating sequence metrics: {e}")
            metrics["sequence_metrics"] = {"error": str(e)}

        # RMSD metrics
        try:
            metrics["rmsd"] = self.calculate_rmsd_metrics()
        except Exception as e:
            LOGGER.warning(f"Error calculating RMSD metrics: {e}")
            metrics["rmsd"] = {"error": str(e)}

        # Bond geometry
        try:
            metrics["bond_geometry"] = self.calculate_bond_geometry()
        except Exception as e:
            LOGGER.warning(f"Error calculating bond geometry: {e}")
            metrics["bond_geometry"] = {"error": str(e)}

        # Convergence check
        try:
            metrics["convergence"] = self.check_geometry_converged()
        except Exception as e:
            LOGGER.warning(f"Error checking convergence: {e}")
            metrics["convergence"] = {"error": str(e)}

        # Constraint verification
        try:
            metrics["constraint_verification"] = self.verify_constraints_preserved()
        except Exception as e:
            LOGGER.warning(f"Error verifying constraints: {e}")
            metrics["constraint_verification"] = {"error": str(e)}

        return metrics

    def calculate_comprehensive_metrics(self) -> Dict:
        """Calculate all comprehensive metrics including new advanced metrics.

        Returns:
            Dict with all metrics categories including:
            - Standard metrics (sequence, RMSD, bond geometry, convergence, constraints)
            - Clash detection
            - Catres-specific bond geometry
            - Dunbrack rotamer quality
            - Secondary structure
            - SASA (solvent accessibility)
            - Ligand interface
            - CA RMSD after ligand alignment
            - Catres mutation check
        """
        LOGGER.info(f"Calculating comprehensive metrics for {self.designed_pdb}")

        metrics = {
            "metadata": {},
            "sequence_metrics": {},
            "rmsd": {},
            "bond_geometry": {},
            "convergence": {},
            "constraint_verification": {},
            "clash_detection": {},
            "catres_bond_geometry": {},
            "dunbrack_quality": {},
            "secondary_structure": {},
            "sasa": {},
            "ligand_interface": {},
            "ca_rmsd_ligand_aligned": {},
            "catres_mutation_check": {},
        }

        # Metadata
        metrics["metadata"]["designed_pdb"] = os.path.abspath(self.designed_pdb)
        metrics["metadata"]["step02_pdb"] = os.path.abspath(self.step02_pdb) if self.step02_pdb else None
        metrics["metadata"]["step01_pdb"] = os.path.abspath(self.step01_pdb) if self.step01_pdb else None
        metrics["metadata"]["bond_length_tolerance"] = self.bond_length_tolerance
        metrics["metadata"]["bond_angle_tolerance"] = self.bond_angle_tolerance
        metrics["metadata"]["catres_bond_tolerance"] = self.catres_bond_tolerance
        metrics["metadata"]["catres_angle_tolerance"] = self.catres_angle_tolerance
        metrics["metadata"]["num_catres"] = len(self.catres_positions)
        metrics["metadata"]["num_constrained_residues"] = len(self.constrained_atoms)

        # Sequence metrics
        try:
            metrics["sequence_metrics"] = self.calculate_sequence_metrics()
        except Exception as e:
            LOGGER.warning(f"Error calculating sequence metrics: {e}")
            metrics["sequence_metrics"] = {"error": str(e)}

        # RMSD metrics
        try:
            metrics["rmsd"] = self.calculate_rmsd_metrics()
        except Exception as e:
            LOGGER.warning(f"Error calculating RMSD metrics: {e}")
            metrics["rmsd"] = {"error": str(e)}

        # Bond geometry
        try:
            metrics["bond_geometry"] = self.calculate_bond_geometry()
        except Exception as e:
            LOGGER.warning(f"Error calculating bond geometry: {e}")
            metrics["bond_geometry"] = {"error": str(e)}

        # Convergence check
        try:
            metrics["convergence"] = self.check_geometry_converged()
        except Exception as e:
            LOGGER.warning(f"Error checking convergence: {e}")
            metrics["convergence"] = {"error": str(e)}

        # Constraint verification
        try:
            metrics["constraint_verification"] = self.verify_constraints_preserved()
        except Exception as e:
            LOGGER.warning(f"Error verifying constraints: {e}")
            metrics["constraint_verification"] = {"error": str(e)}

        # Clash detection
        try:
            metrics["clash_detection"] = self.calculate_clash_detection()
        except Exception as e:
            LOGGER.warning(f"Error calculating clash detection: {e}")
            metrics["clash_detection"] = {"error": str(e)}

        # Catres-specific bond geometry
        try:
            metrics["catres_bond_geometry"] = self.calculate_catres_bond_geometry()
        except Exception as e:
            LOGGER.warning(f"Error calculating catres bond geometry: {e}")
            metrics["catres_bond_geometry"] = {"error": str(e)}

        # Dunbrack quality
        try:
            metrics["dunbrack_quality"] = self.calculate_dunbrack_quality()
        except Exception as e:
            LOGGER.warning(f"Error calculating Dunbrack quality: {e}")
            metrics["dunbrack_quality"] = {"error": str(e)}

        # Secondary structure
        try:
            metrics["secondary_structure"] = self.calculate_secondary_structure()
        except Exception as e:
            LOGGER.warning(f"Error calculating secondary structure: {e}")
            metrics["secondary_structure"] = {"error": str(e)}

        # SASA
        try:
            metrics["sasa"] = self.calculate_sasa()
        except Exception as e:
            LOGGER.warning(f"Error calculating SASA: {e}")
            metrics["sasa"] = {"error": str(e)}

        # Ligand interface
        try:
            metrics["ligand_interface"] = self.calculate_ligand_interface()
        except Exception as e:
            LOGGER.warning(f"Error calculating ligand interface: {e}")
            metrics["ligand_interface"] = {"error": str(e)}

        # CA RMSD after ligand alignment
        try:
            metrics["ca_rmsd_ligand_aligned"] = self.calculate_ca_rmsd_after_ligand_alignment()
        except Exception as e:
            LOGGER.warning(f"Error calculating CA RMSD after ligand alignment: {e}")
            metrics["ca_rmsd_ligand_aligned"] = {"error": str(e)}

        # Catres mutation check
        try:
            metrics["catres_mutation_check"] = self.check_catres_mutations()
        except Exception as e:
            LOGGER.warning(f"Error checking catres mutations: {e}")
            metrics["catres_mutation_check"] = {"error": str(e)}

        return metrics


def round_metrics(obj, decimals: int = 4):
    """Recursively round all float values in a nested structure."""
    if isinstance(obj, dict):
        return {k: round_metrics(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_metrics(item, decimals) for item in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    else:
        return obj
