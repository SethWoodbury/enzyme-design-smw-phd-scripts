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
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

LOGGER = logging.getLogger(__name__)


def _pyrosetta_is_initialized(pyr) -> bool:
    """Best-effort check for PyRosetta initialization without triggering asserts."""
    # Common flags in various PyRosetta versions
    for attr_name in ("_is_init", "_is_initialized"):
        if hasattr(pyr, attr_name):
            try:
                return bool(getattr(pyr, attr_name))
            except Exception:
                pass
    for func_name in ("is_initialized", "pyrosetta_initialized"):
        if hasattr(pyr, func_name):
            try:
                return bool(getattr(pyr, func_name)())
            except Exception:
                pass
    try:
        return bool(pyr.rosetta.basic.is_initialized())
    except Exception:
        pass
    try:
        return bool(pyr.rosetta.basic.was_init_called())
    except Exception:
        pass
    return False


def init_pyrosetta_if_needed(params_files: List[str] = None) -> bool:
    """Initialize PyRosetta if not already done."""
    try:
        try:
            import pyrosetta as pyr
        except ImportError:
            from ..module_utils.pyrosetta_utils import try_import_pyrosetta
            if not try_import_pyrosetta():
                LOGGER.warning("PyRosetta not available, some metrics will be skipped")
                return False
            import pyrosetta as pyr

        if _pyrosetta_is_initialized(pyr):
            return True

        options = "-ignore_unrecognized_res false -ignore_zero_occupancy false"
        if params_files:
            params_str = " ".join(params_files)
            options += f" -extra_res_fa {params_str}"

        try:
            pyr.init(options)
            return True
        except Exception as e:
            msg = str(e).lower()
            if "already" in msg and "initialize" in msg:
                return True
            LOGGER.warning(f"PyRosetta init failed: {e}")
            return False
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
        motif_positions: Optional[List[Tuple[str, int]]] = None,
        ligand_info: Optional[Tuple[str, str, int]] = None,
        step02_sequence: Optional[str] = None,
        bond_length_tolerance: float = 0.05,
        bond_angle_tolerance: float = 10.0,
        catres_bond_tolerance: float = 0.05,
        catres_angle_tolerance: float = 7.5,
        pyrosetta_image: Optional[str] = None,
        use_container_fallback: bool = True,
        container_timeout: Optional[int] = None,
    ):
        """Initialize metrics calculator.

        Args:
            designed_pdb: Path to designed PDB
            step02_pdb: Path to step02 relaxed PDB (immediate reference)
            step01_pdb: Path to step01 aligned PDB (original reference)
            params_files: List of ligand .params files
            constrained_atoms: Dict mapping "chain:resno" -> atom names
            catres_positions: List of (chain, resno) for catalytic residues (catres_subset)
            motif_positions: List of (chain, resno) for all REMARK 666 motifs (catres + conserved)
            ligand_info: Tuple of (chain, resname, resno) for ligand
            step02_sequence: Reference sequence from step02
            bond_length_tolerance: Tolerance for bond convergence (Angstroms)
            bond_angle_tolerance: Tolerance for angle convergence (degrees)
            catres_bond_tolerance: Stricter tolerance for catres bonds (Angstroms)
            catres_angle_tolerance: Stricter tolerance for catres angles (degrees)
            pyrosetta_image: Path to PyRosetta apptainer image (for container fallback)
            use_container_fallback: If True, run metrics in container when host PyRosetta is unavailable
            container_timeout: Timeout for container metrics runs (seconds)
        """
        self.designed_pdb = designed_pdb
        self.step02_pdb = step02_pdb
        self.step01_pdb = step01_pdb
        self.params_files = params_files or []
        self.constrained_atoms = constrained_atoms or {}
        self.catres_positions = catres_positions or []
        self.motif_positions = motif_positions or []
        # Normalize catres positions to (chain, resno) tuples
        normalized_catres = []
        for item in self.catres_positions:
            try:
                chain, resno = item
                normalized_catres.append((str(chain), int(resno)))
            except Exception:
                continue
        self.catres_positions = normalized_catres

        normalized_motifs = []
        for item in self.motif_positions:
            try:
                chain, resno = item
                normalized_motifs.append((str(chain), int(resno)))
            except Exception:
                continue
        self.motif_positions = normalized_motifs
        self.ligand_info = ligand_info
        self.step02_sequence = step02_sequence
        self.bond_length_tolerance = bond_length_tolerance
        self.bond_angle_tolerance = bond_angle_tolerance
        self.catres_bond_tolerance = catres_bond_tolerance
        self.catres_angle_tolerance = catres_angle_tolerance

        # Container fallback for metrics when host PyRosetta is unavailable
        if pyrosetta_image is None:
            try:
                from ..module_utils.constants import DEFAULT_PYROSETTA_IMAGE
            except Exception:
                DEFAULT_PYROSETTA_IMAGE = None
            pyrosetta_image = DEFAULT_PYROSETTA_IMAGE

        self.pyrosetta_image = pyrosetta_image
        self.use_container_fallback = use_container_fallback
        self.container_timeout = container_timeout
        self._container_metrics_cache: Optional[Dict] = None
        self._container_metrics_mode: Optional[str] = None
        self._disable_container_fallback = bool(os.environ.get("FASTMPNN_METRICS_IN_CONTAINER"))

        # Convert explicitly listed constrained atoms to a base set for quick lookup.
        # ALL_ATOMS / ALL_HEAVY entries are expanded dynamically per pose.
        self._explicit_constrained_atoms_set: Set[Tuple[str, int, str]] = set()
        for res_key, atoms in self.constrained_atoms.items():
            if str(res_key).lower() == "ligand":
                if not self.ligand_info:
                    continue
                lig_chain, _lig_resname, lig_resno = self.ligand_info
                pose_idx = pose.pdb_info().pdb2pose(lig_chain, lig_resno)
                if pose_idx == 0:
                    continue
                res = pose.residue(pose_idx)
                if atoms == "ALL_ATOMS" or (isinstance(atoms, list) and "ALL_ATOMS" in atoms):
                    atom_names = [res.atom_name(i).strip() for i in range(1, res.natoms() + 1)]
                elif atoms == "ALL_HEAVY" or (isinstance(atoms, list) and "ALL_HEAVY" in atoms):
                    atom_names = [
                        res.atom_name(i).strip()
                        for i in range(1, res.natoms() + 1)
                        if not res.atom_is_hydrogen(i)
                    ]
                else:
                    atom_names = atoms if isinstance(atoms, list) else [atoms]
                for atom_name in atom_names:
                    expanded.add((lig_chain, lig_resno, atom_name))
                continue
            try:
                if ":" in res_key:
                    chain, resno = res_key.split(":")
                    resno = int(resno)
                else:
                    chain = res_key[0]
                    resno = int(res_key[1:])
            except Exception:
                continue

            if atoms == "ALL_HEAVY" or (isinstance(atoms, list) and "ALL_HEAVY" in atoms):
                # Will be populated dynamically
                continue
            if atoms == "ALL_ATOMS" or (isinstance(atoms, list) and "ALL_ATOMS" in atoms):
                # Will be populated dynamically
                continue

            for atom in (atoms if isinstance(atoms, list) else [atoms]):
                self._explicit_constrained_atoms_set.add((chain, resno, atom))

        # Convert positions to sets for quick lookup
        self.catres_set: Set[Tuple[str, int]] = set(self.catres_positions)
        self.motif_set: Set[Tuple[str, int]] = set(self.motif_positions)

        # PyRosetta objects (lazy loaded)
        self._pyrosetta_available = None
        self._pose = None
        self._sfxn = None
        self._constrained_atoms_set_cache: Optional[Set[Tuple[str, int, str]]] = None
        self._constrained_atoms_pose_id: Optional[int] = None

    def _get_constrained_atoms_set(self, pose) -> Set[Tuple[str, int, str]]:
        """Expand ALL_ATOMS / ALL_HEAVY constraints based on a pose."""
        pose_id = id(pose)
        if self._constrained_atoms_pose_id == pose_id and self._constrained_atoms_set_cache is not None:
            return self._constrained_atoms_set_cache

        expanded: Set[Tuple[str, int, str]] = set(self._explicit_constrained_atoms_set)

        for res_key, atoms in self.constrained_atoms.items():
            if str(res_key).lower() == "ligand":
                continue
            try:
                if ":" in res_key:
                    chain, resno = res_key.split(":")
                    resno = int(resno)
                else:
                    chain = res_key[0]
                    resno = int(res_key[1:])
            except Exception:
                continue

            pose_idx = pose.pdb_info().pdb2pose(chain, resno)
            if pose_idx == 0:
                continue

            res = pose.residue(pose_idx)

            if atoms == "ALL_ATOMS" or (isinstance(atoms, list) and "ALL_ATOMS" in atoms):
                atom_names = [res.atom_name(i).strip() for i in range(1, res.natoms() + 1)]
            elif atoms == "ALL_HEAVY" or (isinstance(atoms, list) and "ALL_HEAVY" in atoms):
                atom_names = [
                    res.atom_name(i).strip()
                    for i in range(1, res.natoms() + 1)
                    if not res.atom_is_hydrogen(i)
                ]
            else:
                atom_names = atoms if isinstance(atoms, list) else [atoms]

            for atom_name in atom_names:
                expanded.add((chain, resno, atom_name))

        self._constrained_atoms_set_cache = expanded
        self._constrained_atoms_pose_id = pose_id
        return expanded

    @property
    def pyrosetta_available(self) -> bool:
        """Check if PyRosetta is available."""
        if self._pyrosetta_available is None:
            self._pyrosetta_available = init_pyrosetta_if_needed(self.params_files)
        return self._pyrosetta_available

    def _container_fallback_allowed(self) -> bool:
        """Return True if container fallback for metrics is permitted."""
        return (
            self.use_container_fallback
            and not self._disable_container_fallback
            and bool(self.pyrosetta_image)
        )

    def _get_container_metrics(self, mode: str = "all") -> Dict:
        """Run metrics inside PyRosetta container and return results."""
        if self._container_metrics_cache is not None:
            if self._container_metrics_mode == mode or self._container_metrics_mode == "comprehensive":
                return self._container_metrics_cache

        if not self._container_fallback_allowed():
            return {"error": "Container fallback disabled or pyrosetta_image not set"}

        if self.pyrosetta_image and not os.path.exists(self.pyrosetta_image):
            return {"error": f"PyRosetta image not found: {self.pyrosetta_image}"}

        script_path = Path(__file__).with_name("rosetta_metrics.py")
        if not script_path.exists():
            return {"error": f"Container metrics script not found: {script_path}"}

        # Write constraints and catres positions to files in a writable scratch dir
        base = Path(self.designed_pdb)
        scratch_dir = base.parent
        if not os.access(scratch_dir, os.W_OK):
            scratch_dir = Path(tempfile.mkdtemp(prefix="fastmpnn_metrics_"))

        def _write_inputs(target_dir: Path):
            cst = target_dir / f"{base.stem}.metrics.cst.json"
            catres = target_dir / f"{base.stem}.metrics.catres.json"
            motif = target_dir / f"{base.stem}.metrics.motif.json"
            out = target_dir / f"{base.stem}.metrics.container.json"
            with open(cst, "w") as f:
                json.dump(self.constrained_atoms, f)
            with open(catres, "w") as f:
                json.dump(self.catres_positions, f)
            with open(motif, "w") as f:
                json.dump(self.motif_positions, f)
            return cst, catres, motif, out

        try:
            cst_json, catres_json, motif_json, out_json = _write_inputs(scratch_dir)
        except Exception as e:
            LOGGER.warning(f"Failed to write metrics inputs in {scratch_dir}: {e}")
            scratch_dir = Path(tempfile.mkdtemp(prefix="fastmpnn_metrics_"))
            try:
                cst_json, catres_json, motif_json, out_json = _write_inputs(scratch_dir)
            except Exception as e2:
                return {"error": f"Failed to write metrics inputs: {e2}"}

        ligand_info_arg = None
        if self.ligand_info:
            lig_chain, lig_resname, lig_resno = self.ligand_info
            ligand_info_arg = f"{lig_chain},{lig_resname},{lig_resno}"

        cmd = [
            "apptainer", "exec", self.pyrosetta_image,
            "python", str(script_path),
            "--designed_pdb", self.designed_pdb,
            "--step02_pdb", self.step02_pdb,
            "--output_json", str(out_json),
            "--mode", mode,
            "--bond_length_tolerance", str(self.bond_length_tolerance),
            "--bond_angle_tolerance", str(self.bond_angle_tolerance),
            "--catres_bond_tolerance", str(self.catres_bond_tolerance),
            "--catres_angle_tolerance", str(self.catres_angle_tolerance),
            "--constraints_json", str(cst_json),
            "--catres_positions_json", str(catres_json),
            "--motif_positions_json", str(motif_json),
        ]

        if self.step01_pdb:
            cmd.extend(["--step01_pdb", self.step01_pdb])

        if self.params_files:
            cmd.extend(["--params"] + self.params_files)

        if ligand_info_arg:
            cmd.extend(["--ligand_info", ligand_info_arg])

        LOGGER.info(f"Running metrics in container (mode={mode})")
        LOGGER.debug(f"Container metrics cmd: {' '.join(cmd)}")

        env = os.environ.copy()
        env["FASTMPNN_METRICS_IN_CONTAINER"] = "1"

        timeout = self.container_timeout or 1200
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Container metrics timed out after {timeout}s"}

        if result.returncode != 0:
            err = result.stderr[-500:] if result.stderr else "No stderr"
            return {"error": f"Container metrics failed: {err}"}

        if not out_json.exists():
            return {"error": "Container metrics produced no output JSON"}

        try:
            with open(out_json, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            return {"error": f"Failed to read container metrics JSON: {e}"}

        self._container_metrics_cache = metrics
        self._container_metrics_mode = mode
        return metrics

    def _load_pose(self, pdb_path: str):
        """Load a pose from PDB."""
        import pyrosetta as pyr
        return pyr.pose_from_pdb(pdb_path)

    def _get_scorefunction(self, name: str = "ref2015_cart"):
        """Get or create scorefunction."""
        if self._sfxn is None or getattr(self, "_sfxn_name", None) != name:
            from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
            self._sfxn = ScoreFunctionFactory.create_score_function(name)
            self._sfxn_name = name
        return self._sfxn

    def calculate_rosetta_score(self, scorefunction: str = "ref2015_cart") -> Dict:
        """Calculate Rosetta total score for the designed structure."""
        if not self.pyrosetta_available:
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="score")
                if "error" not in container_metrics:
                    return container_metrics.get("rosetta_score", {"error": "rosetta_score missing in container metrics"})
            return {"error": "PyRosetta not available"}

        try:
            import pyrosetta as pyr
            pose = self._load_pose(self.designed_pdb)
            sfxn = self._get_scorefunction(scorefunction)
            total_score = float(sfxn(pose))
            return {
                "scorefunction": scorefunction,
                "total_score": round(total_score, 4),
            }
        except Exception as e:
            return {"error": str(e)}

    def calculate_sequence_metrics(self) -> Dict:
        """Calculate sequence-based metrics.

        Returns:
            Dict with sequence_identity, mutations, num_mutations
        """
        from ..module_utils.pdb_utils import read_pdb_atoms
        from ..module_utils.sequence_utils import (
            get_sequence_from_atoms,
            get_sequence_with_positions,
            calculate_sequence_identity,
            calculate_sequence_identity_from_maps,
            get_mutations_list,
            get_mutations_list_from_maps,
        )

        _, designed_atoms = read_pdb_atoms(self.designed_pdb)
        _, step02_atoms = read_pdb_atoms(self.step02_pdb)

        designed_seq = get_sequence_from_atoms(designed_atoms)
        step02_seq = get_sequence_from_atoms(step02_atoms)

        designed_map = get_sequence_with_positions(designed_atoms)
        step02_map = get_sequence_with_positions(step02_atoms)

        # Use provided step02_sequence if available
        if self.step02_sequence:
            step02_seq = self.step02_sequence

        if designed_map and step02_map:
            identity = calculate_sequence_identity_from_maps(step02_map, designed_map)
            mutations = get_mutations_list_from_maps(step02_map, designed_map)
        else:
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
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="all")
                if "error" not in container_metrics:
                    return container_metrics.get("rmsd", {"error": "RMSD missing in container metrics"})
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

    def calculate_tm_score(self) -> Dict:
        """Calculate TM-score vs step01 and step02 (CA-only, Kabsch-aligned)."""
        results = {
            "vs_step02": None,
            "vs_step01": None,
        }

        # PyRosetta path (if available) - fall back to internal implementation
        # We keep internal as the default for consistency and robustness
        try:
            results["vs_step02"] = self._calculate_tm_score_internal(self.step02_pdb, self.designed_pdb)
            if self.step01_pdb and os.path.exists(self.step01_pdb):
                results["vs_step01"] = self._calculate_tm_score_internal(self.step01_pdb, self.designed_pdb)
        except Exception as e:
            results["error"] = str(e)

        return results

    def calculate_lddt(self) -> Dict:
        """Calculate CA-only lDDT vs step01 and step02."""
        results = {
            "vs_step02": None,
            "vs_step01": None,
        }

        try:
            results["vs_step02"] = self._calculate_lddt_internal(self.step02_pdb, self.designed_pdb)
            if self.step01_pdb and os.path.exists(self.step01_pdb):
                results["vs_step01"] = self._calculate_lddt_internal(self.step01_pdb, self.designed_pdb)
        except Exception as e:
            results["error"] = str(e)

        return results

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
        constrained_set = self._get_constrained_atoms_set(pose_after)

        for res_key, atoms in self.constrained_atoms.items():
            if str(res_key).lower() == "ligand":
                continue
            try:
                if ":" in res_key:
                    chain, resno = res_key.split(":")
                    resno = int(resno)
                else:
                    chain = res_key[0]
                    resno = int(res_key[1:])
            except Exception:
                continue

            try:
                pose_idx_before = pose_before.pdb_info().pdb2pose(chain, resno)
                pose_idx_after = pose_after.pdb_info().pdb2pose(chain, resno)

                if pose_idx_before == 0 or pose_idx_after == 0:
                    continue

                res_before = pose_before.residue(pose_idx_before)
                res_after = pose_after.residue(pose_idx_after)

                # Determine atom names
                if atoms == "ALL_ATOMS" or (isinstance(atoms, list) and "ALL_ATOMS" in atoms):
                    atom_names = [
                        res_before.atom_name(i).strip()
                        for i in range(1, res_before.natoms() + 1)
                    ]
                elif atoms == "ALL_HEAVY" or (isinstance(atoms, list) and "ALL_HEAVY" in atoms):
                    atom_names = [
                        res_before.atom_name(i).strip()
                        for i in range(1, res_before.natoms() + 1)
                        if not res_before.atom_is_hydrogen(i)
                    ]
                else:
                    atom_names = atoms if isinstance(atoms, list) else [atoms]

                # If atom list is empty but we have expanded constraints, fall back to expanded set
                if not atom_names and constrained_set:
                    atom_names = [
                        a for (c, r, a) in constrained_set
                        if c == chain and r == resno
                    ]

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

    def _get_ca_coords_map(self, pdb_path: str) -> Dict[Tuple[str, int], np.ndarray]:
        """Get CA coordinates keyed by (chain, resno) from a PDB."""
        from ..module_utils.pdb_utils import read_pdb_atoms
        _, atoms = read_pdb_atoms(pdb_path)
        coords = {}
        for atom in atoms:
            if atom["record_type"] != "ATOM":
                continue
            if atom["atom_name"].strip() != "CA":
                continue
            key = (atom["chain"], atom["resno"])
            coords[key] = np.array([atom["x"], atom["y"], atom["z"]], dtype=float)
        return coords

    def _is_hydrogen_atom(self, atom: Dict) -> bool:
        """Return True if atom appears to be hydrogen."""
        elem = atom.get("element", "").strip().upper()
        if elem == "H":
            return True
        name = atom.get("atom_name", "").strip().upper()
        return name.startswith("H")

    def _get_ligand_heavy_coords_map(self, pdb_path: str) -> Dict[str, np.ndarray]:
        """Get ligand heavy-atom coordinates keyed by atom name."""
        if not self.ligand_info:
            return {}
        from ..module_utils.pdb_utils import read_pdb_atoms
        lig_chain, lig_resname, lig_resno = self.ligand_info
        _, atoms = read_pdb_atoms(pdb_path)

        def _collect(require_resname: bool) -> Dict[str, np.ndarray]:
            coords = {}
            for atom in atoms:
                if atom["chain"] != lig_chain or atom["resno"] != lig_resno:
                    continue
                if require_resname and atom["resname"] != lig_resname:
                    continue
                if self._is_hydrogen_atom(atom):
                    continue
                coords[atom["atom_name"].strip()] = np.array([atom["x"], atom["y"], atom["z"]], dtype=float)
            return coords

        coords = _collect(require_resname=True)
        if not coords:
            coords = _collect(require_resname=False)
        return coords

    def _vector1_to_list(self, vec) -> List[int]:
        """Convert PyRosetta vector1 types to a Python list (1-based or 0-based)."""
        if vec is None:
            return []
        if isinstance(vec, list):
            return [int(x) for x in vec]
        size = None
        if hasattr(vec, "size"):
            try:
                size = int(vec.size())
            except Exception:
                size = None
        if size is None:
            try:
                size = len(vec)
            except Exception:
                size = None

        if size is not None:
            if size <= 0:
                return []
            # Try 1-based indexing
            try:
                items = [vec[i] for i in range(1, size + 1)]
                vals = [int(x) for x in items if int(x) > 0]
                if vals:
                    return vals
            except Exception:
                pass
            # Try 0-based indexing
            try:
                items = [vec[i] for i in range(0, size)]
                vals = [int(x) for x in items if int(x) > 0]
                if vals:
                    return vals
            except Exception:
                pass

        try:
            return [int(x) for x in vec if int(x) > 0]
        except Exception:
            return []

    def _get_bonded_neighbors(self, res, atom_idx: int) -> List[int]:
        """Return bonded neighbor atom indices for a residue atom."""
        # Try residue API
        fn = getattr(res, "bonded_neighbor", None)
        if fn is not None:
            try:
                neighbors = self._vector1_to_list(fn(atom_idx))
                if neighbors:
                    return neighbors
            except TypeError:
                # Some builds expose bonded_neighbor(atom_idx, idx)
                try:
                    n = int(res.n_bonded_neighbor(atom_idx))
                    neighbors = []
                    for j in range(1, n + 1):
                        neighbors.append(int(fn(atom_idx, j)))
                    return [x for x in neighbors if x > 0]
                except Exception:
                    pass
            except Exception:
                pass

        # Try residue type API
        try:
            rtype = res.type()
        except Exception:
            rtype = None
        if rtype is not None:
            fn = getattr(rtype, "bonded_neighbor", None)
            if fn is not None:
                try:
                    neighbors = self._vector1_to_list(fn(atom_idx))
                    if neighbors:
                        return neighbors
                except Exception:
                    pass

        return []

    def _get_matched_ca_coords(
        self,
        ref_pdb: str,
        model_pdb: str,
    ) -> Tuple[List[Tuple[str, int]], np.ndarray, np.ndarray]:
        """Get matched CA coordinates for common residues between two PDBs."""
        ref_map = self._get_ca_coords_map(ref_pdb)
        model_map = self._get_ca_coords_map(model_pdb)
        keys = sorted(set(ref_map.keys()) & set(model_map.keys()))
        if not keys:
            return [], np.zeros((0, 3)), np.zeros((0, 3))
        ref_coords = np.stack([ref_map[k] for k in keys], axis=0)
        model_coords = np.stack([model_map[k] for k in keys], axis=0)
        return keys, ref_coords, model_coords

    def _kabsch(self, ref_coords: np.ndarray, model_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optimal rotation/translation (Kabsch) to align model onto ref."""
        ref_centroid = ref_coords.mean(axis=0)
        model_centroid = model_coords.mean(axis=0)
        ref_centered = ref_coords - ref_centroid
        model_centered = model_coords - model_centroid

        h = model_centered.T @ ref_centered
        u, s, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T
        t = ref_centroid - r @ model_centroid
        return r, t

    def _apply_transform(self, coords: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply rotation and translation to coordinates."""
        return (r @ coords.T).T + t

    def _calculate_tm_score_internal(self, ref_pdb: str, model_pdb: str) -> Dict:
        """Calculate TM-score using CA-only coordinates with Kabsch alignment."""
        keys, ref_coords, model_coords = self._get_matched_ca_coords(ref_pdb, model_pdb)
        n = len(keys)
        if n < 3:
            return {"error": "Not enough matched residues for TM-score", "num_matched": n}

        r, t = self._kabsch(ref_coords, model_coords)
        model_aligned = self._apply_transform(model_coords, r, t)
        diffs = ref_coords - model_aligned
        dists = np.sqrt((diffs ** 2).sum(axis=1))

        if n <= 15:
            d0 = 0.5
        else:
            d0 = 1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8
            d0 = max(d0, 0.5)

        tm = float(np.mean(1.0 / (1.0 + (dists / d0) ** 2)))
        rmsd = float(np.sqrt(np.mean(dists ** 2)))

        return {
            "tm_score": round(tm, 4),
            "rmsd_ca_aligned": round(rmsd, 4),
            "num_matched": n,
            "d0": round(float(d0), 4),
            "alignment": "kabsch_ca",
        }

    def _calculate_lddt_internal(self, ref_pdb: str, model_pdb: str) -> Dict:
        """Calculate CA-only lDDT vs reference (no alignment needed)."""
        keys, ref_coords, model_coords = self._get_matched_ca_coords(ref_pdb, model_pdb)
        n = len(keys)
        if n < 3:
            return {"error": "Not enough matched residues for lDDT", "num_matched": n}

        cutoff = 15.0
        thresholds = [0.5, 1.0, 2.0, 4.0]
        per_residue = []

        ref_dmat = np.linalg.norm(ref_coords[:, None, :] - ref_coords[None, :, :], axis=2)
        model_dmat = np.linalg.norm(model_coords[:, None, :] - model_coords[None, :, :], axis=2)

        for i in range(n):
            # consider neighbors within cutoff (exclude self)
            mask = (ref_dmat[i] <= cutoff) & (np.arange(n) != i)
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            ref_d = ref_dmat[i, idx]
            model_d = model_dmat[i, idx]
            diffs = np.abs(model_d - ref_d)

            frac = []
            for thr in thresholds:
                frac.append(float(np.mean(diffs < thr)))
            per_residue.append(float(np.mean(frac)))

        if not per_residue:
            return {"error": "No residue pairs within lDDT cutoff", "num_matched": n}

        return {
            "lddt_ca": round(float(np.mean(per_residue)), 4),
            "num_matched": n,
            "cutoff": cutoff,
            "thresholds": thresholds,
        }

    def calculate_bond_geometry(self) -> Dict:
        """Calculate bond length/angle geometry metrics.

        Returns:
            Dict with bond_length_geometry and bond_angle_geometry
        """
        if not self.pyrosetta_available:
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="all")
                if "error" not in container_metrics:
                    return container_metrics.get("bond_geometry", {"error": "Bond geometry missing in container metrics"})
            return {"error": "PyRosetta not available"}

        pose = self._load_pose(self.designed_pdb)

        constrained_set = self._get_constrained_atoms_set(pose)

        # Calculate bond length deviations
        bond_metrics = self._calculate_bond_metrics(pose, constrained_set)

        # Calculate bond angle deviations
        angle_metrics = self._calculate_angle_metrics(pose, constrained_set)

        payload = {
            "bond_length_geometry": bond_metrics,
            "bond_angle_geometry": angle_metrics,
        }

        if (
            bond_metrics.get("all", {}).get("count", 0) == 0
            and angle_metrics.get("all", {}).get("count", 0) == 0
        ):
            payload["warning"] = "No bond/angle entries were evaluated; check PyRosetta neighbor iteration."

        return payload

    def _calculate_bond_metrics(self, pose, constrained_set: Set[Tuple[str, int, str]]) -> Dict:
        """Calculate bond length deviations from ideal using tripeptide reference.

        Creates an ideal A-X-A tripeptide for each residue type to get ideal
        bond lengths, which is more robust than using res.type().bond_length().
        """
        import pyrosetta as pyr

        all_deviations = []
        unconstrained_deviations = []

        # Cache ideal reference poses by residue 1-letter code
        ref_cache = {}

        for i in range(1, pose.size() + 1):
            res = pose.residue(i)
            if res.is_ligand():
                continue

            chain = pose.pdb_info().chain(i)
            resno = pose.pdb_info().number(i)

            # Get or create ideal reference tripeptide
            res_name1 = res.name1()
            if res_name1 not in ref_cache:
                try:
                    ref_pose = pyr.pose_from_sequence("A" + res_name1 + "A")
                    ref_cache[res_name1] = ref_pose.residue(2)
                except Exception:
                    continue
            ref_res = ref_cache.get(res_name1)
            if ref_res is None:
                continue

            # Check bonds (heavy atoms only)
            for atom1_idx in range(1, res.natoms() + 1):
                if res.atom_type(atom1_idx).element() == "H":
                    continue
                atom1_name = res.atom_name(atom1_idx).strip()

                if not ref_res.has(atom1_name):
                    continue

                atom1_constrained = (chain, resno, atom1_name) in constrained_set

                # Get bonded neighbors
                try:
                    bonded = res.bonded_neighbor(atom1_idx)
                    neighbors = [int(bonded[j]) for j in range(1, len(bonded) + 1)]
                except Exception:
                    continue

                for atom2_idx in neighbors:
                    if atom2_idx <= atom1_idx:  # Avoid double counting
                        continue
                    if res.atom_type(atom2_idx).element() == "H":
                        continue
                    atom2_name = res.atom_name(atom2_idx).strip()

                    if not ref_res.has(atom2_name):
                        continue

                    atom2_constrained = (chain, resno, atom2_name) in constrained_set
                    both_constrained = atom1_constrained and atom2_constrained

                    # Calculate actual vs ideal bond length
                    actual = (res.xyz(atom1_name) - res.xyz(atom2_name)).norm()
                    ideal = (ref_res.xyz(atom1_name) - ref_res.xyz(atom2_name)).norm()
                    deviation = abs(actual - ideal)

                    all_deviations.append(deviation)
                    if not both_constrained:
                        unconstrained_deviations.append(deviation)

        def aggregate(devs):
            if not devs:
                return {"mean": 0.0, "max": 0.0, "std": 0.0, "count": 0}
            return {
                "mean": round(float(np.mean(devs)), 3),
                "max": round(float(np.max(devs)), 3),
                "std": round(float(np.std(devs)), 3),
                "count": len(devs),
            }

        return {
            "all": aggregate(all_deviations),
            "unconstrained_only": aggregate(unconstrained_deviations),
        }

    def _calculate_angle_metrics(self, pose, constrained_set: Set[Tuple[str, int, str]]) -> Dict:
        """Calculate bond angle deviations from ideal using tripeptide reference.

        For each central atom with 2+ bonded neighbors, calculate the angle
        and compare to the ideal angle from a reference A-X-A tripeptide.
        """
        import pyrosetta as pyr
        import math

        all_deviations = []
        unconstrained_deviations = []

        # Cache ideal reference poses by residue 1-letter code
        ref_cache = {}

        for i in range(1, pose.size() + 1):
            res = pose.residue(i)
            if res.is_ligand():
                continue

            chain = pose.pdb_info().chain(i)
            resno = pose.pdb_info().number(i)

            # Get or create ideal reference tripeptide
            res_name1 = res.name1()
            if res_name1 not in ref_cache:
                try:
                    ref_pose = pyr.pose_from_sequence("A" + res_name1 + "A")
                    ref_cache[res_name1] = ref_pose.residue(2)
                except Exception:
                    continue
            ref_res = ref_cache.get(res_name1)
            if ref_res is None:
                continue

            # For each potential central atom
            for atom2_idx in range(1, res.natoms() + 1):
                if res.atom_type(atom2_idx).element() == "H":
                    continue
                atom2_name = res.atom_name(atom2_idx).strip()

                if not ref_res.has(atom2_name):
                    continue

                atom2_constrained = (chain, resno, atom2_name) in constrained_set

                # Get bonded neighbors
                try:
                    bonded = res.bonded_neighbor(atom2_idx)
                    neighbors = [int(bonded[j]) for j in range(1, len(bonded) + 1)]
                except Exception:
                    continue

                if len(neighbors) < 2:
                    continue

                # Check all pairs of neighbors
                for idx1 in range(len(neighbors)):
                    for idx2 in range(idx1 + 1, len(neighbors)):
                        atom1_idx = neighbors[idx1]
                        atom3_idx = neighbors[idx2]

                        if res.atom_type(atom1_idx).element() == "H":
                            continue
                        if res.atom_type(atom3_idx).element() == "H":
                            continue

                        atom1_name = res.atom_name(atom1_idx).strip()
                        atom3_name = res.atom_name(atom3_idx).strip()

                        if not ref_res.has(atom1_name) or not ref_res.has(atom3_name):
                            continue

                        atom1_constrained = (chain, resno, atom1_name) in constrained_set
                        atom3_constrained = (chain, resno, atom3_name) in constrained_set
                        all_constrained = atom1_constrained and atom2_constrained and atom3_constrained

                        # Calculate actual angle
                        xyz1 = res.xyz(atom1_name)
                        xyz2 = res.xyz(atom2_name)
                        xyz3 = res.xyz(atom3_name)

                        v1 = np.array([xyz1.x - xyz2.x, xyz1.y - xyz2.y, xyz1.z - xyz2.z])
                        v2 = np.array([xyz3.x - xyz2.x, xyz3.y - xyz2.y, xyz3.z - xyz2.z])

                        dot = np.dot(v1, v2)
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        if norm1 < 1e-6 or norm2 < 1e-6:
                            continue

                        cos_actual = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
                        actual_deg = math.degrees(math.acos(cos_actual))

                        # Calculate ideal angle from reference
                        ref_xyz1 = ref_res.xyz(atom1_name)
                        ref_xyz2 = ref_res.xyz(atom2_name)
                        ref_xyz3 = ref_res.xyz(atom3_name)

                        ref_v1 = np.array([ref_xyz1.x - ref_xyz2.x, ref_xyz1.y - ref_xyz2.y, ref_xyz1.z - ref_xyz2.z])
                        ref_v2 = np.array([ref_xyz3.x - ref_xyz2.x, ref_xyz3.y - ref_xyz2.y, ref_xyz3.z - ref_xyz2.z])

                        ref_dot = np.dot(ref_v1, ref_v2)
                        ref_norm1 = np.linalg.norm(ref_v1)
                        ref_norm2 = np.linalg.norm(ref_v2)
                        if ref_norm1 < 1e-6 or ref_norm2 < 1e-6:
                            continue

                        cos_ideal = np.clip(ref_dot / (ref_norm1 * ref_norm2), -1.0, 1.0)
                        ideal_deg = math.degrees(math.acos(cos_ideal))

                        deviation = abs(actual_deg - ideal_deg)
                        all_deviations.append(deviation)

                        if not all_constrained:
                            unconstrained_deviations.append(deviation)

        def aggregate(devs):
            if not devs:
                return {"mean": 0.0, "max": 0.0, "std": 0.0, "count": 0}
            return {
                "mean": round(float(np.mean(devs)), 1),
                "max": round(float(np.max(devs)), 1),
                "std": round(float(np.std(devs)), 1),
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
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="comprehensive")
                if "error" not in container_metrics:
                    return container_metrics.get("clash_detection", {"error": "Clash detection missing in container metrics"})
            return {"error": "PyRosetta not available"}

        try:
            import pyrosetta as pyr
            from pyrosetta.rosetta.core.scoring import ScoreType

            pose = self._load_pose(self.designed_pdb)
            constrained_set = self._get_constrained_atoms_set(pose)
            constrained_set = self._get_constrained_atoms_set(pose)
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
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="comprehensive")
                if "error" not in container_metrics:
                    return container_metrics.get("catres_bond_geometry", {"error": "Catres bond geometry missing in container metrics"})
            return {"error": "PyRosetta not available"}

        try:
            import pyrosetta as pyr
            import math

            pose = self._load_pose(self.designed_pdb)
            constrained_set = self._get_constrained_atoms_set(pose)

            bond_deviations = []
            angle_deviations = []
            bond_failures = []
            angle_failures = []

            # Cache ideal reference tripeptide poses by 1-letter code (same as main bond geometry)
            ref_cache = {}

            for chain, resno in self.catres_positions:
                try:
                    pose_idx = pose.pdb_info().pdb2pose(chain, resno)
                    if pose_idx == 0:
                        continue

                    res = pose.residue(pose_idx)
                    if res.is_ligand():
                        continue

                    # Get or create ideal reference tripeptide for this residue type
                    res_name1 = res.name1()
                    if res_name1 not in ref_cache:
                        try:
                            ref_pose = pyr.pose_from_sequence("A" + res_name1 + "A")
                            ref_cache[res_name1] = ref_pose.residue(2)
                        except Exception:
                            continue
                    ref_res = ref_cache.get(res_name1)
                    if ref_res is None:
                        continue

                    # Check bonds using tripeptide reference
                    for atom1_idx in range(1, res.natoms() + 1):
                        if res.atom_type(atom1_idx).element() == "H":
                            continue
                        atom1_name = res.atom_name(atom1_idx).strip()
                        if not ref_res.has(atom1_name):
                            continue

                        atom1_constrained = (chain, resno, atom1_name) in constrained_set

                        try:
                            bonded = res.bonded_neighbor(atom1_idx)
                            neighbors = [int(bonded[j]) for j in range(1, len(bonded) + 1)]
                        except Exception:
                            continue

                        for atom2_idx in neighbors:
                            if atom2_idx <= atom1_idx:
                                continue
                            if res.atom_type(atom2_idx).element() == "H":
                                continue
                            atom2_name = res.atom_name(atom2_idx).strip()
                            if not ref_res.has(atom2_name):
                                continue

                            atom2_constrained = (chain, resno, atom2_name) in constrained_set
                            if atom1_constrained and atom2_constrained:
                                continue  # Skip fully constrained bonds

                            # Calculate actual vs ideal bond length using reference
                            actual = (res.xyz(atom1_name) - res.xyz(atom2_name)).norm()
                            ideal = (ref_res.xyz(atom1_name) - ref_res.xyz(atom2_name)).norm()
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

                    # Check angles using tripeptide reference
                    for atom2_idx in range(1, res.natoms() + 1):
                        if res.atom_type(atom2_idx).element() == "H":
                            continue
                        atom2_name = res.atom_name(atom2_idx).strip()
                        if not ref_res.has(atom2_name):
                            continue

                        atom2_constrained = (chain, resno, atom2_name) in constrained_set

                        try:
                            bonded = res.bonded_neighbor(atom2_idx)
                            neighbors = [int(bonded[j]) for j in range(1, len(bonded) + 1)]
                        except Exception:
                            continue

                        if len(neighbors) < 2:
                            continue

                        for idx1 in range(len(neighbors)):
                            for idx2 in range(idx1 + 1, len(neighbors)):
                                atom1_idx = neighbors[idx1]
                                atom3_idx = neighbors[idx2]

                                if res.atom_type(atom1_idx).element() == "H":
                                    continue
                                if res.atom_type(atom3_idx).element() == "H":
                                    continue

                                atom1_name = res.atom_name(atom1_idx).strip()
                                atom3_name = res.atom_name(atom3_idx).strip()

                                if not ref_res.has(atom1_name) or not ref_res.has(atom3_name):
                                    continue

                                atom1_constrained = (chain, resno, atom1_name) in constrained_set
                                atom3_constrained = (chain, resno, atom3_name) in constrained_set
                                if atom1_constrained and atom2_constrained and atom3_constrained:
                                    continue  # Skip fully constrained angles

                                # Calculate actual angle
                                xyz1 = res.xyz(atom1_name)
                                xyz2 = res.xyz(atom2_name)
                                xyz3 = res.xyz(atom3_name)
                                v1 = np.array([xyz1.x - xyz2.x, xyz1.y - xyz2.y, xyz1.z - xyz2.z])
                                v2 = np.array([xyz3.x - xyz2.x, xyz3.y - xyz2.y, xyz3.z - xyz2.z])
                                norm1 = np.linalg.norm(v1)
                                norm2 = np.linalg.norm(v2)
                                if norm1 < 1e-6 or norm2 < 1e-6:
                                    continue
                                cos_actual = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                                actual_deg = math.degrees(math.acos(cos_actual))

                                # Calculate ideal angle from reference
                                ref_xyz1 = ref_res.xyz(atom1_name)
                                ref_xyz2 = ref_res.xyz(atom2_name)
                                ref_xyz3 = ref_res.xyz(atom3_name)
                                ref_v1 = np.array([ref_xyz1.x - ref_xyz2.x, ref_xyz1.y - ref_xyz2.y, ref_xyz1.z - ref_xyz2.z])
                                ref_v2 = np.array([ref_xyz3.x - ref_xyz2.x, ref_xyz3.y - ref_xyz2.y, ref_xyz3.z - ref_xyz2.z])
                                ref_norm1 = np.linalg.norm(ref_v1)
                                ref_norm2 = np.linalg.norm(ref_v2)
                                if ref_norm1 < 1e-6 or ref_norm2 < 1e-6:
                                    continue
                                cos_ideal = np.clip(np.dot(ref_v1, ref_v2) / (ref_norm1 * ref_norm2), -1.0, 1.0)
                                ideal_deg = math.degrees(math.acos(cos_ideal))

                                deviation = abs(actual_deg - ideal_deg)

                                angle_info = {
                                    "chain": chain,
                                    "resno": resno,
                                    "resname": res.name3(),
                                    "angle": f"{atom1_name}-{atom2_name}-{atom3_name}",
                                    "deviation": round(float(deviation), 1)
                                }

                                angle_deviations.append(deviation)

                                if deviation > self.catres_angle_tolerance:
                                    angle_failures.append(angle_info)

                except Exception as e:
                    LOGGER.warning(f"Error processing catres {chain}{resno}: {e}")

            # Sort failures by severity
            bond_failures.sort(key=lambda x: x["deviation"], reverse=True)
            angle_failures.sort(key=lambda x: x["deviation"], reverse=True)

            # Track per-residue pass/fail status
            failing_residues_bonds = set()
            failing_residues_angles = set()
            for f in bond_failures:
                failing_residues_bonds.add((f["chain"], f["resno"]))
            for f in angle_failures:
                failing_residues_angles.add((f["chain"], f["resno"]))

            # Residues that fail either bond or angle checks
            failing_residues_any = failing_residues_bonds | failing_residues_angles

            num_catres = len(self.catres_positions)
            num_passing = num_catres - len(failing_residues_any)

            payload = {
                "summary": {
                    "num_catres": num_catres,
                    "num_passing": num_passing,
                    "num_failing": len(failing_residues_any),
                    "fraction_passing": round(num_passing / num_catres, 3) if num_catres > 0 else 1.0,
                    "failing_residues": [f"{c}:{r}" for c, r in sorted(failing_residues_any)],
                },
                "bond_geometry": {
                    "mean_deviation": round(float(np.mean(bond_deviations)), 4) if bond_deviations else 0.0,
                    "max_deviation": round(float(np.max(bond_deviations)), 4) if bond_deviations else 0.0,
                    "num_bonds": len(bond_deviations),
                    "num_failures": len(bond_failures),
                    "tolerance": self.catres_bond_tolerance,
                    "worst_offenders": bond_failures[:10]
                },
                "angle_geometry": {
                    "mean_deviation": round(float(np.mean(angle_deviations)), 4) if angle_deviations else 0.0,
                    "max_deviation": round(float(np.max(angle_deviations)), 4) if angle_deviations else 0.0,
                    "num_angles": len(angle_deviations),
                    "num_failures": len(angle_failures),
                    "tolerance": self.catres_angle_tolerance,
                    "worst_offenders": angle_failures[:10]
                }
            }

            if len(bond_deviations) == 0 and len(angle_deviations) == 0:
                payload["warning"] = "No catres bond/angle entries evaluated."

            return payload

        except Exception as e:
            LOGGER.warning(f"Error calculating catres bond geometry: {e}")
            return {"error": str(e)}

    def calculate_dunbrack_quality(self) -> Dict:
        """Calculate Dunbrack rotamer quality for catalytic residues.

        Returns:
            Dict with per-catres Dunbrack energies and overall stats
        """
        if not self.pyrosetta_available:
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="comprehensive")
                if "error" not in container_metrics:
                    return container_metrics.get("dunbrack_quality", {"error": "Dunbrack quality missing in container metrics"})
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
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="comprehensive")
                if "error" not in container_metrics:
                    return container_metrics.get("secondary_structure", {"error": "Secondary structure missing in container metrics"})
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
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="comprehensive")
                if "error" not in container_metrics:
                    return container_metrics.get("sasa", {"error": "SASA missing in container metrics"})
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
        if not self.ligand_info:
            return {"error": "No ligand info"}
        if not self.pyrosetta_available:
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="comprehensive")
                if "error" not in container_metrics:
                    return container_metrics.get("ligand_interface", {"error": "Ligand interface missing in container metrics"})
            return {"error": "PyRosetta not available"}

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
        if not self.step01_pdb or not self.ligand_info:
            return {"error": "No step01 PDB or ligand info"}

        try:
            step01_lig = self._get_ligand_heavy_coords_map(self.step01_pdb)
            designed_lig = self._get_ligand_heavy_coords_map(self.designed_pdb)
            common = sorted(set(step01_lig.keys()) & set(designed_lig.keys()))
            if len(common) < 3:
                return {
                    "error": "Not enough ligand atoms for alignment",
                    "num_ligand_atoms_aligned": len(common),
                }

            step01_coords = np.stack([step01_lig[name] for name in common], axis=0)
            designed_coords = np.stack([designed_lig[name] for name in common], axis=0)
            r, t = self._kabsch(step01_coords, designed_coords)

            keys, ref_ca, model_ca = self._get_matched_ca_coords(self.step01_pdb, self.designed_pdb)
            if len(keys) < 3:
                return {"error": "Not enough matched CA residues for RMSD", "num_matched": len(keys)}

            model_aligned = self._apply_transform(model_ca, r, t)
            diffs = ref_ca - model_aligned
            ca_rmsd = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))

            return {
                "ca_rmsd_after_ligand_alignment": round(ca_rmsd, 4),
                "num_ligand_atoms_aligned": len(common),
                "num_matched_ca": len(keys),
                "alignment": "ligand_kabsch",
            }

        except Exception as e:
            LOGGER.warning(f"Error calculating CA RMSD after ligand alignment: {e}")
            if self._container_fallback_allowed():
                container_metrics = self._get_container_metrics(mode="comprehensive")
                if "error" not in container_metrics:
                    return container_metrics.get("ca_rmsd_ligand_aligned", {"error": "CA RMSD missing in container metrics"})
            return {"error": str(e)}

    def _check_mutations_at_positions(self, positions: List[Tuple[str, int]], label: str) -> Dict:
        """Generic mutation check for a set of positions vs step01."""
        if not self.step01_pdb:
            return {"error": "No step01 PDB available for comparison"}

        try:
            from ..module_utils.pdb_utils import read_pdb_atoms

            _, step01_atoms = read_pdb_atoms(self.step01_pdb)
            _, designed_atoms = read_pdb_atoms(self.designed_pdb)

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

            mutations = []
            for chain, resno in positions:
                key = (chain, resno)
                step01_resname = step01_resnames.get(key)
                designed_resname = designed_resnames.get(key)

                if step01_resname and designed_resname:
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
                    f"{label}_preserved": False,
                    "mutations_detected": mutations,
                    "error": f"{label} residues were mutated: {len(mutations)} mutations found"
                }

            return {
                f"{label}_preserved": True,
                "mutations_detected": [],
                f"num_{label}_checked": len(positions)
            }

        except Exception as e:
            LOGGER.warning(f"Error checking {label} mutations: {e}")
            return {"error": str(e)}

    def check_catres_mutations(self) -> Dict:
        """Verify that NO catalytic residues (catres_subset) were mutated."""
        return self._check_mutations_at_positions(self.catres_positions, "catres")

    def check_motif_mutations(self) -> Dict:
        """Verify that NO REMARK 666 motif residues (catres + conserved) were mutated."""
        return self._check_mutations_at_positions(self.motif_positions, "motif")

    def check_geometry_converged(self) -> Dict:
        """Check if geometry has converged to acceptable tolerances.

        Returns:
            Dict with bond_length_converged, bond_angle_converged, overall_converged
        """
        geometry = self.calculate_bond_geometry()

        if "error" in geometry:
            return {"error": geometry["error"]}

        bond_count = geometry["bond_length_geometry"]["all"]["count"]
        angle_count = geometry["bond_angle_geometry"]["all"]["count"]
        if bond_count == 0 and angle_count == 0:
            return {"error": "No bond/angle entries evaluated; cannot assess convergence"}

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

        # Container fallback if host PyRosetta is unavailable
        if not self.pyrosetta_available and self._container_fallback_allowed():
            LOGGER.info("PyRosetta unavailable on host; falling back to container metrics (all)")
            container_metrics = self._get_container_metrics(mode="all")
            if "error" not in container_metrics:
                return container_metrics
            LOGGER.warning(f"Container metrics failed, falling back to partial metrics: {container_metrics.get('error')}")

        metrics = {
            "metadata": {},
            "sequence_metrics": {},
            "rmsd": {},
            "tm_score": {},
            "lddt": {},
            "bond_geometry": {},
            "convergence": {},
            "constraint_verification": {},
            "motif_mutation_check": {},
            "catres_bond_geometry": {},  # Always include detailed violations
        }

        # Metadata
        metrics["metadata"]["designed_pdb"] = os.path.abspath(self.designed_pdb)
        metrics["metadata"]["step02_pdb"] = os.path.abspath(self.step02_pdb) if self.step02_pdb else None
        metrics["metadata"]["step01_pdb"] = os.path.abspath(self.step01_pdb) if self.step01_pdb else None
        metrics["metadata"]["bond_length_tolerance"] = self.bond_length_tolerance
        metrics["metadata"]["bond_angle_tolerance"] = self.bond_angle_tolerance
        metrics["metadata"]["num_catres"] = len(self.catres_positions)
        metrics["metadata"]["num_motif"] = len(self.motif_positions)

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

        # TM-score
        try:
            metrics["tm_score"] = self.calculate_tm_score()
        except Exception as e:
            LOGGER.warning(f"Error calculating TM-score: {e}")
            metrics["tm_score"] = {"error": str(e)}

        # lDDT
        try:
            metrics["lddt"] = self.calculate_lddt()
        except Exception as e:
            LOGGER.warning(f"Error calculating lDDT: {e}")
            metrics["lddt"] = {"error": str(e)}

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

        # Motif mutation check (all REMARK 666 positions)
        try:
            metrics["motif_mutation_check"] = self.check_motif_mutations()
        except Exception as e:
            LOGGER.warning(f"Error checking motif mutations: {e}")
            metrics["motif_mutation_check"] = {"error": str(e)}

        # Catres-specific bond geometry with detailed violations
        # Always included (not just in debug mode) for diagnostics
        try:
            metrics["catres_bond_geometry"] = self.calculate_catres_bond_geometry()
        except Exception as e:
            LOGGER.warning(f"Error calculating catres bond geometry: {e}")
            metrics["catres_bond_geometry"] = {"error": str(e)}

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

        # Container fallback if host PyRosetta is unavailable
        if not self.pyrosetta_available and self._container_fallback_allowed():
            LOGGER.info("PyRosetta unavailable on host; falling back to container metrics (comprehensive)")
            container_metrics = self._get_container_metrics(mode="comprehensive")
            if "error" not in container_metrics:
                return container_metrics
            LOGGER.warning(f"Container metrics failed, falling back to partial metrics: {container_metrics.get('error')}")

        metrics = {
            "metadata": {},
            "sequence_metrics": {},
            "rmsd": {},
            "tm_score": {},
            "lddt": {},
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
            "motif_mutation_check": {},
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
        metrics["metadata"]["num_motif"] = len(self.motif_positions)
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

        # TM-score
        try:
            metrics["tm_score"] = self.calculate_tm_score()
        except Exception as e:
            LOGGER.warning(f"Error calculating TM-score: {e}")
            metrics["tm_score"] = {"error": str(e)}

        # lDDT
        try:
            metrics["lddt"] = self.calculate_lddt()
        except Exception as e:
            LOGGER.warning(f"Error calculating lDDT: {e}")
            metrics["lddt"] = {"error": str(e)}

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

        # Motif mutation check (all REMARK 666 positions)
        try:
            metrics["motif_mutation_check"] = self.check_motif_mutations()
        except Exception as e:
            LOGGER.warning(f"Error checking motif mutations: {e}")
            metrics["motif_mutation_check"] = {"error": str(e)}

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
