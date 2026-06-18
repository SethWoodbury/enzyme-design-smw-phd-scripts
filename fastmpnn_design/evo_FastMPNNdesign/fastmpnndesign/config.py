"""
Configuration dataclasses for fastmpnndesign.

Defines all configuration structures used throughout the pipeline.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from fastmpnndesign.constants import (
    DEFAULT_MPNN_RUNNER, DEFAULT_MODEL_TYPE, DEFAULT_ENHANCE_MODEL,
    DEFAULT_TEMPERATURE, DEFAULT_BATCHES, DEFAULT_BATCH_SIZE, DEFAULT_OMIT_AA,
    DEFAULT_SC_DENOISING_STEPS, DEFAULT_APPTAINER_IMAGE,
    DEFAULT_ROSETTA_PATH, DEFAULT_PYROSETTA_PATH, DEFAULT_SCOREFUNCTION,
    DEFAULT_PYROSETTA_IMAGE,
    PRIMARY_CONTACT_CUTOFF, SECONDARY_CONTACT_CUTOFF, METAL_CONTACT_CUTOFF,
    COORD_CST_WEIGHT, COORD_CST_STDEV, MOBILE_RADIUS, CART_BONDED_WEIGHT,
    FASTRELAX_CYCLES, LIGAND_CST_STDEV, ALLOW_CATRES_BB,
    USE_MULTISTAGE_RELAX, INITIAL_COORD_CST_WEIGHT, FINAL_COORD_CST_WEIGHT,
    INITIAL_FA_REP_SCALE, N_RELAX_STAGES,
    DEFAULT_N_CYCLES, DEFAULT_N_CANDIDATES, DEFAULT_N_KEEP,
    DEFAULT_N_FINAL, DEFAULT_SLURM_TIME, DEFAULT_SLURM_CPUS, DEFAULT_SLURM_MEM
)


@dataclass
class CatalyticResidue:
    """Represents a catalytic residue parsed from REMARK 666."""
    catres_index: int  # 1-indexed order from REMARK 666
    chain: str
    resnum: int
    resname: str
    icode: str = ""  # Insertion code
    cst_block: Optional[int] = None
    cst_var: Optional[int] = None
    raw_line: str = ""

    @property
    def rosetta_resid(self) -> str:
        """Return residue ID in Rosetta format (e.g., '150A')."""
        icode_str = self.icode if self.icode and self.icode.strip() else ""
        return f"{self.resnum}{icode_str}{self.chain}"

    @property
    def pdb_resid(self) -> str:
        """Return residue ID in PDB format (e.g., 'A150')."""
        icode_str = self.icode if self.icode and self.icode.strip() else ""
        return f"{self.chain}{self.resnum}{icode_str}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Contact:
    """Represents a contact between ligand/metal and protein."""
    ligand_chain: str
    ligand_resnum: int
    ligand_resname: str
    ligand_atom: str
    protein_chain: str
    protein_resnum: int
    protein_resname: str
    protein_atom: str
    distance: float
    contact_type: str  # 'metal', 'primary', 'secondary'
    priority: int
    is_heteroatom_contact: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CatresCatresContact:
    """Represents an interaction between two catalytic residues."""
    # First residue (res1)
    chain1: str
    resnum1: int
    resname1: str
    atom1: str
    # Second residue (res2)
    chain2: str
    resnum2: int
    resname2: str
    atom2: str
    # Interaction details
    distance: float
    interaction_type: str  # 'hbond', 'salt_bridge', 'pi_stack'
    priority: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def res1_id(self) -> str:
        """Return residue 1 ID in PDB format (e.g., 'A150')."""
        return f"{self.chain1}{self.resnum1}"

    @property
    def res2_id(self) -> str:
        """Return residue 2 ID in PDB format (e.g., 'A152')."""
        return f"{self.chain2}{self.resnum2}"


@dataclass
class MPNNConfig:
    """Configuration for LigandMPNN execution."""
    mpnn_runner: Path = field(default_factory=lambda: Path(DEFAULT_MPNN_RUNNER))
    model_type: str = DEFAULT_MODEL_TYPE
    enhance_model: Optional[str] = DEFAULT_ENHANCE_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    number_of_batches: int = DEFAULT_BATCHES
    batch_size: int = DEFAULT_BATCH_SIZE
    pack_side_chains: bool = True
    sc_num_denoising_steps: int = DEFAULT_SC_DENOISING_STEPS
    omit_AA: str = DEFAULT_OMIT_AA
    use_apptainer: bool = False
    apptainer_image: Path = field(default_factory=lambda: Path(DEFAULT_APPTAINER_IMAGE))
    ligand_mpnn_use_side_chain_context: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['mpnn_runner'] = str(self.mpnn_runner)
        d['apptainer_image'] = str(self.apptainer_image)
        return d


@dataclass
class ConstraintConfig:
    """Configuration for constraint generation."""
    primary_contact_cutoff: float = PRIMARY_CONTACT_CUTOFF
    secondary_contact_cutoff: float = SECONDARY_CONTACT_CUTOFF
    metal_cutoff: float = METAL_CONTACT_CUTOFF
    coord_cst_weight: float = COORD_CST_WEIGHT
    coord_cst_stdev: float = COORD_CST_STDEV
    ref_pdb: Optional[Path] = None
    cst_file: Optional[Path] = None  # Legacy constraint file

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['ref_pdb'] = str(self.ref_pdb) if self.ref_pdb else None
        d['cst_file'] = str(self.cst_file) if self.cst_file else None
        return d


@dataclass
class RelaxConfig:
    """Configuration for PyRosetta FastRelax.

    Features:
    - Completely frozen ligand residues (no movement at all)
    - Increased backbone mobility for protein near active site
    - Heavy geometry penalty via cart_bonded weight
    - Multi-stage relaxation with ramped constraints
    - Per-cycle progress logging
    """
    rosetta_path: Path = field(default_factory=lambda: Path(DEFAULT_ROSETTA_PATH))
    pyrosetta_path: Path = field(default_factory=lambda: Path(DEFAULT_PYROSETTA_PATH))
    scorefunction: str = DEFAULT_SCOREFUNCTION
    use_pyrosetta: bool = True
    fastrelax_cycles: int = FASTRELAX_CYCLES
    mobile_radius: float = MOBILE_RADIUS
    cart_bonded_weight: float = CART_BONDED_WEIGHT
    coord_cst_weight: float = COORD_CST_WEIGHT  # Coordinate constraint weight for scorefunction (legacy)
    ligand_cst_stdev: float = LIGAND_CST_STDEV  # Constraint stdev for ligand freezing
    allow_catres_bb: bool = ALLOW_CATRES_BB  # Allow backbone movement for catres
    weight_flexibility: bool = False  # Use B-factor/pLDDT for flexibility weighting
    use_pyrosetta_image: bool = True  # Use pyrosetta.sif container for relax
    pyrosetta_image: Path = field(default_factory=lambda: Path(DEFAULT_PYROSETTA_IMAGE))
    # Multi-stage relaxation parameters
    use_multistage_relax: bool = USE_MULTISTAGE_RELAX  # Use multi-stage protocol
    initial_coord_cst_weight: float = INITIAL_COORD_CST_WEIGHT  # Stage 1 constraint weight
    final_coord_cst_weight: float = FINAL_COORD_CST_WEIGHT  # Final stage constraint weight
    initial_fa_rep_scale: float = INITIAL_FA_REP_SCALE  # Stage 1 fa_rep scale
    n_relax_stages: int = N_RELAX_STAGES  # Number of relaxation stages

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['rosetta_path'] = str(self.rosetta_path)
        d['pyrosetta_path'] = str(self.pyrosetta_path)
        d['pyrosetta_image'] = str(self.pyrosetta_image)
        return d


@dataclass
class CatresConfig:
    """Configuration for catalytic residue handling."""
    catres_subset: Optional[List[int]] = None  # Indices for tight constraints (None = ALL)
    redesign_non_subset_catres: bool = False  # Allow redesigning non-subset catres

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineConfig:
    """Configuration for iterative design pipeline."""
    n_cycles: int = DEFAULT_N_CYCLES
    n_candidates: int = DEFAULT_N_CANDIDATES
    n_keep: int = DEFAULT_N_KEEP
    n_final: int = DEFAULT_N_FINAL
    design_shell_radius: float = 12.0  # Only redesign residues within this radius of ligand
    fix_outside_shell: bool = True  # Fix residues outside the design shell

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SlurmConfig:
    """Configuration for SLURM job submission."""
    partition: Optional[str] = None
    time: str = DEFAULT_SLURM_TIME
    cpus: int = DEFAULT_SLURM_CPUS
    mem: str = DEFAULT_SLURM_MEM

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    """Master configuration for a complete run."""
    # Required inputs
    pdb: Path = field(default_factory=lambda: Path("."))
    params: List[Path] = field(default_factory=list)

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./fastmpnn_output"))
    prefix: str = "design"

    # Sub-configurations
    mpnn: MPNNConfig = field(default_factory=MPNNConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    catres: CatresConfig = field(default_factory=CatresConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # Execution options
    single_thread: bool = False
    dry_run: bool = False
    verbose: bool = True
    log_file: Optional[Path] = None
    generate_sbatch: bool = False
    cleanup: bool = False  # Delete intermediate directories after successful completion

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pdb': str(self.pdb),
            'params': [str(p) for p in self.params],
            'output_dir': str(self.output_dir),
            'prefix': self.prefix,
            'mpnn': self.mpnn.to_dict(),
            'constraints': self.constraints.to_dict(),
            'relax': self.relax.to_dict(),
            'catres': self.catres.to_dict(),
            'pipeline': self.pipeline.to_dict(),
            'slurm': self.slurm.to_dict(),
            'single_thread': self.single_thread,
            'dry_run': self.dry_run,
            'verbose': self.verbose,
            'log_file': str(self.log_file) if self.log_file else None,
            'generate_sbatch': self.generate_sbatch,
            'cleanup': self.cleanup
        }

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RunConfig':
        """Create RunConfig from dictionary."""
        return cls(
            pdb=Path(d['pdb']),
            params=[Path(p) for p in d.get('params', [])],
            output_dir=Path(d.get('output_dir', './fastmpnn_output')),
            prefix=d.get('prefix', 'design'),
            mpnn=MPNNConfig(
                mpnn_runner=Path(d['mpnn'].get('mpnn_runner', DEFAULT_MPNN_RUNNER)),
                model_type=d['mpnn'].get('model_type', DEFAULT_MODEL_TYPE),
                enhance_model=d['mpnn'].get('enhance_model'),
                temperature=d['mpnn'].get('temperature', DEFAULT_TEMPERATURE),
                number_of_batches=d['mpnn'].get('number_of_batches', DEFAULT_BATCHES),
                batch_size=d['mpnn'].get('batch_size', DEFAULT_BATCH_SIZE),
                pack_side_chains=d['mpnn'].get('pack_side_chains', True),
                sc_num_denoising_steps=d['mpnn'].get('sc_num_denoising_steps', DEFAULT_SC_DENOISING_STEPS),
                omit_AA=d['mpnn'].get('omit_AA', DEFAULT_OMIT_AA),
                use_apptainer=d['mpnn'].get('use_apptainer', False),
                apptainer_image=Path(d['mpnn'].get('apptainer_image', DEFAULT_APPTAINER_IMAGE)),
                ligand_mpnn_use_side_chain_context=d['mpnn'].get('ligand_mpnn_use_side_chain_context', True)
            ),
            constraints=ConstraintConfig(
                primary_contact_cutoff=d['constraints'].get('primary_contact_cutoff', PRIMARY_CONTACT_CUTOFF),
                secondary_contact_cutoff=d['constraints'].get('secondary_contact_cutoff', SECONDARY_CONTACT_CUTOFF),
                metal_cutoff=d['constraints'].get('metal_cutoff', METAL_CONTACT_CUTOFF),
                coord_cst_weight=d['constraints'].get('coord_cst_weight', COORD_CST_WEIGHT),
                coord_cst_stdev=d['constraints'].get('coord_cst_stdev', COORD_CST_STDEV),
                ref_pdb=Path(d['constraints']['ref_pdb']) if d['constraints'].get('ref_pdb') else None,
                cst_file=Path(d['constraints']['cst_file']) if d['constraints'].get('cst_file') else None
            ),
            relax=RelaxConfig(
                rosetta_path=Path(d['relax'].get('rosetta_path', DEFAULT_ROSETTA_PATH)),
                pyrosetta_path=Path(d['relax'].get('pyrosetta_path', DEFAULT_PYROSETTA_PATH)),
                scorefunction=d['relax'].get('scorefunction', DEFAULT_SCOREFUNCTION),
                use_pyrosetta=d['relax'].get('use_pyrosetta', True),
                fastrelax_cycles=d['relax'].get('fastrelax_cycles', FASTRELAX_CYCLES),
                mobile_radius=d['relax'].get('mobile_radius', MOBILE_RADIUS),
                cart_bonded_weight=d['relax'].get('cart_bonded_weight', CART_BONDED_WEIGHT),
                coord_cst_weight=d['relax'].get('coord_cst_weight', COORD_CST_WEIGHT),
                ligand_cst_stdev=d['relax'].get('ligand_cst_stdev', LIGAND_CST_STDEV),
                allow_catres_bb=d['relax'].get('allow_catres_bb', ALLOW_CATRES_BB),
                weight_flexibility=d['relax'].get('weight_flexibility', False),
                use_pyrosetta_image=d['relax'].get('use_pyrosetta_image', True),
                pyrosetta_image=Path(d['relax'].get('pyrosetta_image', DEFAULT_PYROSETTA_IMAGE)),
                use_multistage_relax=d['relax'].get('use_multistage_relax', USE_MULTISTAGE_RELAX),
                initial_coord_cst_weight=d['relax'].get('initial_coord_cst_weight', INITIAL_COORD_CST_WEIGHT),
                final_coord_cst_weight=d['relax'].get('final_coord_cst_weight', FINAL_COORD_CST_WEIGHT),
                initial_fa_rep_scale=d['relax'].get('initial_fa_rep_scale', INITIAL_FA_REP_SCALE),
                n_relax_stages=d['relax'].get('n_relax_stages', N_RELAX_STAGES)
            ),
            catres=CatresConfig(
                catres_subset=d['catres'].get('catres_subset'),
                redesign_non_subset_catres=d['catres'].get('redesign_non_subset_catres', False)
            ),
            pipeline=PipelineConfig(
                n_cycles=d['pipeline'].get('n_cycles', DEFAULT_N_CYCLES),
                n_candidates=d['pipeline'].get('n_candidates', DEFAULT_N_CANDIDATES),
                n_keep=d['pipeline'].get('n_keep', DEFAULT_N_KEEP),
                n_final=d['pipeline'].get('n_final', DEFAULT_N_FINAL),
                design_shell_radius=d['pipeline'].get('design_shell_radius', 12.0),
                fix_outside_shell=d['pipeline'].get('fix_outside_shell', True)
            ),
            slurm=SlurmConfig(
                partition=d['slurm'].get('partition'),
                time=d['slurm'].get('time', DEFAULT_SLURM_TIME),
                cpus=d['slurm'].get('cpus', DEFAULT_SLURM_CPUS),
                mem=d['slurm'].get('mem', DEFAULT_SLURM_MEM)
            ),
            single_thread=d.get('single_thread', False),
            dry_run=d.get('dry_run', False),
            verbose=d.get('verbose', True),
            log_file=Path(d['log_file']) if d.get('log_file') else None,
            generate_sbatch=d.get('generate_sbatch', False),
            cleanup=d.get('cleanup', False)
        )

    @classmethod
    def load(cls, path: Path) -> 'RunConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)


@dataclass
class CycleResult:
    """Result from one design cycle."""
    cycle_num: int
    input_pdb: Path
    mpnn_output_dir: Path
    relax_output_dir: Path
    selected_pdbs: List[Path]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cycle_num': self.cycle_num,
            'input_pdb': str(self.input_pdb),
            'mpnn_output_dir': str(self.mpnn_output_dir),
            'relax_output_dir': str(self.relax_output_dir),
            'selected_pdbs': [str(p) for p in self.selected_pdbs],
            'metrics': self.metrics
        }


@dataclass
class DesignCandidate:
    """Represents a single design candidate with associated metrics."""
    pdb_path: Path
    sequence: str
    mpnn_score: float
    rosetta_score: Optional[float] = None
    mean_displacement: Optional[float] = None
    max_displacement: Optional[float] = None
    pct_within_tolerance: Optional[float] = None
    cart_bonded_score: Optional[float] = None
    cycle: int = 0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pdb_path': str(self.pdb_path),
            'sequence': self.sequence,
            'mpnn_score': self.mpnn_score,
            'rosetta_score': self.rosetta_score,
            'mean_displacement': self.mean_displacement,
            'max_displacement': self.max_displacement,
            'pct_within_tolerance': self.pct_within_tolerance,
            'cart_bonded_score': self.cart_bonded_score,
            'cycle': self.cycle,
            'rank': self.rank
        }
