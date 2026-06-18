"""
Configuration dataclasses for enhanced_fastmpnndesign.

Provides type-safe configuration structures for all pipeline components.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from constants import (
    DEFAULT_MPNN_RUNNER, DEFAULT_MODEL_TYPE, DEFAULT_ENHANCE_MODEL,
    DEFAULT_TEMPERATURE, DEFAULT_BATCHES, DEFAULT_BATCH_SIZE, DEFAULT_OMIT_AA,
    DEFAULT_SC_DENOISING_STEPS, DEFAULT_APPTAINER_IMAGE,
    DEFAULT_SCOREFUNCTION, DEFAULT_DALPHABALL, DEFAULT_PARAMS_PATH,
    DEFAULT_BIAS_VALUE, DEFAULT_BIAS_AAS, DEFAULT_HBOND_ACCEPT_PROBABILITY,
    CART_BONDED_WEIGHT, PRO_CLOSE_WEIGHT, USE_CARTESIAN_PRERELAX,
    DEFAULT_LAYER_DIST_BB, DEFAULT_LAYER_DIST_SC, DEFAULT_LAYER_CUTS,
    SECOND_LAYER_TEMPERATURES
)


# =============================================================================
# Catalytic Residue Data Structure
# =============================================================================

@dataclass
class CatalyticResidue:
    """
    Represents a catalytic residue parsed from REMARK 666.

    Attributes:
        catres_index: 1-indexed order from REMARK 666 (the constraint block number)
        chain: Chain identifier
        resnum: Residue number in PDB numbering
        resname: 3-letter residue name
        icode: Insertion code (if any)
        cst_block: Constraint block number
        cst_var: Constraint variant number
        raw_line: Original REMARK 666 line for reference
        seqpos: PyRosetta sequence position (set at runtime)
    """
    catres_index: int
    chain: str
    resnum: int
    resname: str
    icode: str = ""
    cst_block: Optional[int] = None
    cst_var: Optional[int] = None
    raw_line: str = ""
    _seqpos: Optional[int] = field(default=None, repr=False)

    # Template info (from MATCH TEMPLATE portion)
    template_chain: Optional[str] = None
    template_resname: Optional[str] = None
    template_resnum: Optional[int] = None

    @property
    def pdb_resid(self) -> str:
        """Return residue ID in format 'A150' or 'A150A' if icode present."""
        return f"{self.chain}{self.resnum}{self.icode}"

    @property
    def seqpos(self) -> Optional[int]:
        """Rosetta sequence position (set at runtime after pose loading)."""
        return self._seqpos

    @seqpos.setter
    def seqpos(self, value: int):
        self._seqpos = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'catres_index': self.catres_index,
            'chain': self.chain,
            'resnum': self.resnum,
            'resname': self.resname,
            'icode': self.icode,
            'cst_block': self.cst_block,
            'cst_var': self.cst_var,
            'pdb_resid': self.pdb_resid,
            'seqpos': self._seqpos,
        }


# =============================================================================
# MPNN Configuration
# =============================================================================

@dataclass
class MPNNConfig:
    """
    Configuration for LigandMPNN/EnhancedMPNN execution.

    Exposes all key LigandMPNN args as specified in requirements.
    """
    # Runner path
    mpnn_runner: Path = field(default_factory=lambda: Path(DEFAULT_MPNN_RUNNER))

    # Model settings
    model_type: str = DEFAULT_MODEL_TYPE
    enhance_model: Optional[str] = DEFAULT_ENHANCE_MODEL
    use_enhanced_mpnn: bool = True

    # Sampling parameters
    temperature: float = DEFAULT_TEMPERATURE
    number_of_batches: int = DEFAULT_BATCHES
    batch_size: int = DEFAULT_BATCH_SIZE

    # Side chain settings
    pack_side_chains: bool = True
    sc_num_denoising_steps: int = DEFAULT_SC_DENOISING_STEPS
    ligand_mpnn_use_side_chain_context: bool = True

    # Amino acid restrictions
    omit_AA: str = DEFAULT_OMIT_AA
    repack_everything: bool = False

    # Execution mode
    use_apptainer: bool = False
    apptainer_image: Path = field(default_factory=lambda: Path(DEFAULT_APPTAINER_IMAGE))

    # Output suffix
    packed_suffix: str = "_packed_"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'mpnn_runner': str(self.mpnn_runner),
            'model_type': self.model_type,
            'enhance_model': self.enhance_model,
            'use_enhanced_mpnn': self.use_enhanced_mpnn,
            'temperature': self.temperature,
            'number_of_batches': self.number_of_batches,
            'batch_size': self.batch_size,
            'pack_side_chains': self.pack_side_chains,
            'sc_num_denoising_steps': self.sc_num_denoising_steps,
            'ligand_mpnn_use_side_chain_context': self.ligand_mpnn_use_side_chain_context,
            'omit_AA': self.omit_AA,
            'repack_everything': self.repack_everything,
            'use_apptainer': self.use_apptainer,
            'apptainer_image': str(self.apptainer_image),
            'packed_suffix': self.packed_suffix,
        }


# =============================================================================
# Catalytic Residue Configuration
# =============================================================================

@dataclass
class CatresConfig:
    """
    Configuration for catalytic residue handling.

    Controls which catres get tight geometry constraints and which can be redesigned.
    """
    # Subset of catres indices (by REMARK 666 index) for tight geometry constraints
    # Default None means ALL catres get tight constraints
    catres_subset: Optional[List[int]] = None

    # If True, catres NOT in subset may be redesigned
    # IMPORTANT: catres are fixed by default; this is the only way to redesign them
    redesign_non_subset_catres: bool = False

    # Whether to fix all catres by default (standard behavior)
    fix_all_catres: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'catres_subset': self.catres_subset,
            'redesign_non_subset_catres': self.redesign_non_subset_catres,
            'fix_all_catres': self.fix_all_catres,
        }


# =============================================================================
# Bias Configuration
# =============================================================================

@dataclass
class BiasConfig:
    """Configuration for MPNN position-specific amino acid bias."""
    # Ligand atoms for bias calculation
    bias_atoms: Optional[List[str]] = None

    # Bias value (negative = favor, positive = disfavor)
    position_bias: float = DEFAULT_BIAS_VALUE

    # Amino acids to bias (1-letter codes)
    bias_AAs: str = DEFAULT_BIAS_AAS

    # Distance cuts for bias calculation
    bias_cuts: Tuple[float, ...] = field(default_factory=lambda: (5.0, 7.0, 9.0, 11.0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'bias_atoms': self.bias_atoms,
            'position_bias': self.position_bias,
            'bias_AAs': self.bias_AAs,
            'bias_cuts': list(self.bias_cuts),
        }


# =============================================================================
# Relaxation Configuration
# =============================================================================

@dataclass
class RelaxConfig:
    """Configuration for pre-relaxation with CartesianFastRelax."""
    # Use Cartesian pre-relaxation
    cartesian: bool = USE_CARTESIAN_PRERELAX

    # Scorefunction weights for Cartesian relax
    cart_bonded_weight: float = CART_BONDED_WEIGHT
    pro_close_weight: float = PRO_CLOSE_WEIGHT

    # Use crude (fast) FastRelax
    crude: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'cartesian': self.cartesian,
            'cart_bonded_weight': self.cart_bonded_weight,
            'pro_close_weight': self.pro_close_weight,
            'crude': self.crude,
        }


# =============================================================================
# PyRosetta Configuration
# =============================================================================

@dataclass
class RosettaConfig:
    """Configuration for PyRosetta initialization."""
    # Scorefunction name
    scorefunction: str = DEFAULT_SCOREFUNCTION

    # DAlphaBall path for hole detection
    dalphaball_path: Path = field(default_factory=lambda: Path(DEFAULT_DALPHABALL))

    # Ligand/NCAA params files
    params: List[Path] = field(default_factory=list)

    # Threading
    multithreading: bool = True

    # Preserve header (REMARK lines) in output PDBs
    preserve_header: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'scorefunction': self.scorefunction,
            'dalphaball_path': str(self.dalphaball_path),
            'params': [str(p) for p in self.params],
            'multithreading': self.multithreading,
            'preserve_header': self.preserve_header,
        }


# =============================================================================
# Protocol Configuration
# =============================================================================

@dataclass
class ProtocolConfig:
    """Configuration for the design protocol."""
    # Protocol file path (overrides protocol_text)
    protocol_file: Optional[Path] = None

    # Inline protocol text
    protocol_text: Optional[str] = None

    # Constraint file for enzyme design
    cstfile: Optional[Path] = None

    # HBond keeper parameters
    hbond_accept_probability: float = DEFAULT_HBOND_ACCEPT_PROBABILITY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'protocol_file': str(self.protocol_file) if self.protocol_file else None,
            'protocol_text': self.protocol_text,
            'cstfile': str(self.cstfile) if self.cstfile else None,
            'hbond_accept_probability': self.hbond_accept_probability,
        }


# =============================================================================
# Scoring Configuration
# =============================================================================

@dataclass
class ScoringConfig:
    """Configuration for design scoring and filtering."""
    # Custom scoring script path
    scoring_script: Optional[Path] = None

    # Apply filter to outputs
    apply_filter: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'scoring_script': str(self.scoring_script) if self.scoring_script else None,
            'apply_filter': self.apply_filter,
        }


# =============================================================================
# Layer Detection Configuration
# =============================================================================

@dataclass
class LayerConfig:
    """Configuration for design layer detection."""
    # 2nd layer fixed position detection distances
    dist_bb: float = DEFAULT_LAYER_DIST_BB
    dist_sc: float = DEFAULT_LAYER_DIST_SC

    # Layer cuts for pocket detection
    layer_cuts: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_LAYER_CUTS)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'dist_bb': self.dist_bb,
            'dist_sc': self.dist_sc,
            'layer_cuts': list(self.layer_cuts),
        }


# =============================================================================
# Second Layer MPNN Configuration
# =============================================================================

@dataclass
class SecondLayerMPNNConfig:
    """Configuration for 2nd layer MPNN refinement."""
    # Enable 2nd layer MPNN
    enabled: bool = False

    # Temperatures for sampling
    temperatures: Tuple[float, ...] = field(default_factory=lambda: SECOND_LAYER_TEMPERATURES)

    # Batch settings
    number_of_batches: int = 1
    batch_size: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'enabled': self.enabled,
            'temperatures': list(self.temperatures),
            'number_of_batches': self.number_of_batches,
            'batch_size': self.batch_size,
        }


# =============================================================================
# Master Run Configuration
# =============================================================================

@dataclass
class RunConfig:
    """
    Master configuration for a complete design run.

    Aggregates all sub-configurations and provides serialization.
    """
    # Required inputs
    pdb: Path = field(default_factory=lambda: Path("."))

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("."))
    prefix: str = ""
    suffix: str = ""
    nstruct: int = 1

    # Design positions (if None, auto-detect)
    design_pos: Optional[List[int]] = None
    keep_pos: Optional[List[int]] = None
    detect_pocket: bool = False

    # Sub-configurations
    mpnn: MPNNConfig = field(default_factory=MPNNConfig)
    catres: CatresConfig = field(default_factory=CatresConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    rosetta: RosettaConfig = field(default_factory=RosettaConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    layer: LayerConfig = field(default_factory=LayerConfig)
    second_layer_mpnn: SecondLayerMPNNConfig = field(default_factory=SecondLayerMPNNConfig)

    # Execution flags
    verbose: bool = True
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pdb': str(self.pdb),
            'output_dir': str(self.output_dir),
            'prefix': self.prefix,
            'suffix': self.suffix,
            'nstruct': self.nstruct,
            'design_pos': self.design_pos,
            'keep_pos': self.keep_pos,
            'detect_pocket': self.detect_pocket,
            'mpnn': self.mpnn.to_dict(),
            'catres': self.catres.to_dict(),
            'bias': self.bias.to_dict(),
            'relax': self.relax.to_dict(),
            'rosetta': self.rosetta.to_dict(),
            'protocol': self.protocol.to_dict(),
            'scoring': self.scoring.to_dict(),
            'layer': self.layer.to_dict(),
            'second_layer_mpnn': self.second_layer_mpnn.to_dict(),
            'verbose': self.verbose,
            'dry_run': self.dry_run,
        }

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'RunConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RunConfig':
        """Create RunConfig from dictionary."""
        # Parse sub-configs
        mpnn_dict = d.pop('mpnn', {})
        catres_dict = d.pop('catres', {})
        bias_dict = d.pop('bias', {})
        relax_dict = d.pop('relax', {})
        rosetta_dict = d.pop('rosetta', {})
        protocol_dict = d.pop('protocol', {})
        scoring_dict = d.pop('scoring', {})
        layer_dict = d.pop('layer', {})
        second_layer_dict = d.pop('second_layer_mpnn', {})

        # Convert paths
        if 'pdb' in d:
            d['pdb'] = Path(d['pdb'])
        if 'output_dir' in d:
            d['output_dir'] = Path(d['output_dir'])

        # Build config
        config = cls(**d)

        # Set sub-configs
        if mpnn_dict:
            if 'mpnn_runner' in mpnn_dict:
                mpnn_dict['mpnn_runner'] = Path(mpnn_dict['mpnn_runner'])
            if 'apptainer_image' in mpnn_dict:
                mpnn_dict['apptainer_image'] = Path(mpnn_dict['apptainer_image'])
            config.mpnn = MPNNConfig(**mpnn_dict)

        if catres_dict:
            config.catres = CatresConfig(**catres_dict)

        if bias_dict:
            if 'bias_cuts' in bias_dict:
                bias_dict['bias_cuts'] = tuple(bias_dict['bias_cuts'])
            config.bias = BiasConfig(**bias_dict)

        if relax_dict:
            config.relax = RelaxConfig(**relax_dict)

        if rosetta_dict:
            if 'dalphaball_path' in rosetta_dict:
                rosetta_dict['dalphaball_path'] = Path(rosetta_dict['dalphaball_path'])
            if 'params' in rosetta_dict:
                rosetta_dict['params'] = [Path(p) for p in rosetta_dict['params']]
            config.rosetta = RosettaConfig(**rosetta_dict)

        if protocol_dict:
            if 'protocol_file' in protocol_dict and protocol_dict['protocol_file']:
                protocol_dict['protocol_file'] = Path(protocol_dict['protocol_file'])
            if 'cstfile' in protocol_dict and protocol_dict['cstfile']:
                protocol_dict['cstfile'] = Path(protocol_dict['cstfile'])
            config.protocol = ProtocolConfig(**protocol_dict)

        if scoring_dict:
            if 'scoring_script' in scoring_dict and scoring_dict['scoring_script']:
                scoring_dict['scoring_script'] = Path(scoring_dict['scoring_script'])
            config.scoring = ScoringConfig(**scoring_dict)

        if layer_dict:
            if 'layer_cuts' in layer_dict:
                layer_dict['layer_cuts'] = tuple(layer_dict['layer_cuts'])
            config.layer = LayerConfig(**layer_dict)

        if second_layer_dict:
            if 'temperatures' in second_layer_dict:
                second_layer_dict['temperatures'] = tuple(second_layer_dict['temperatures'])
            config.second_layer_mpnn = SecondLayerMPNNConfig(**second_layer_dict)

        return config
