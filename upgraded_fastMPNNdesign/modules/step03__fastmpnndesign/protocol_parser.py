"""Protocol definition and parsing for FastMPNN design.

This module provides:
- StepType enum for different protocol step types
- Step dataclasses for configuring each step type
- ProtocolParser for parsing custom protocol strings (text format)
- ProtocolFileParser for loading protocols from JSON or text files

Protocol file formats:
    JSON format (protocol.json):
        {
          "steps": [
            {"type": "mpnn", "temperature": 0.1, "num_designs": 1, "spheres": ["primary"]},
            {"type": "cart_relax", "repeats": 2, "stages": 3},
            {"type": "mpnn", "temperature": 0.1, "num_designs": 4},
            {"type": "torsional_relax", "repeats": 1, "stages": 3}
          ]
        }

    Simple text format (protocol.txt):
        mpnn:T0.1:N1:spheres=primary
        cart_relax:R2S3
        mpnn:T0.1:N4
        torsional_relax:R1S3

Built-in protocol files live in the protocols/ directory.
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple

LOGGER = logging.getLogger(__name__)


class StepType(Enum):
    """Types of protocol steps."""
    # MPNN design steps - new naming (scope-based)
    MPNN = "mpnn"                                 # Standard MPNN design (default: core)
    MPNN_MULTI = "mpnn_multi"                     # Parallel/pool multiple MPNN strategies
    DESIGN_CORE = "design_core"                   # Design DESIGN_CORE only (0-6Å)
    DESIGN_CORE_SHELL = "design_core_shell"       # Design DESIGN_CORE + DESIGN_SHELL (0-8Å)
    DESIGN_SHELL_ONLY = "design_shell_only"       # Design DESIGN_SHELL only, freeze core (6-8Å)
    DESIGN_SHELL_FLEX = "design_shell_flex"       # Design shell + flex (6-12Å)
    DESIGN_FLEX_ONLY = "design_flex_only"         # Design flex zone only (8-12Å)
    DESIGN_CORE_SHELL_FLEX = "design_core_shell_flex"  # Design core + shell + flex (0-12Å)
    DESIGN_DISTANT_ONLY = "design_distant_only"   # Design distant/frozen only (>12Å)
    DESIGN_SHELL_FLEX_DISTANT = "design_shell_flex_distant"  # Shell + flex + distant (>6Å)
    DESIGN_GLOBAL = "design_global"               # Global redesign (all non-fixed residues)

    # Legacy MPNN step names (aliases for backwards compatibility)
    MPNN_PRIMARY = "mpnn_primary"                 # Alias for DESIGN_CORE
    MPNN_SECONDARY = "mpnn_secondary"             # Alias for DESIGN_CORE_SHELL
    MPNN_2ND_SHELL = "mpnn_2nd_shell"             # Alias for DESIGN_SHELL_ONLY
    MPNN_ALL = "mpnn_all"                         # Alias for DESIGN_GLOBAL

    # Rosetta relaxation steps (time-consuming)
    CART_RELAX = "cart_relax"               # Cartesian FastRelax (for bond geometry)

    # Rosetta relaxation steps (fast)
    TORSIONAL_RELAX = "torsional_relax"     # Torsional FastRelax
    MINIMIZE = "minimize"                    # Minimization only
    REPACK = "repack"                        # Sidechain repacking

    # Selection/filtering steps
    SELECT_BEST = "select_best"             # Select top N structures
    CLUSTER = "cluster"                      # Cluster structures (sequence/structure)
    KEEP_CLUSTER_BEST = "keep_cluster_best"  # Keep best per cluster

    # Scorefunction manipulation
    SCALE_SCORETERM = "scale_scoreterm"     # Scale scorefunction term(s)

    # Dynamic configuration
    SET_OPTIONS = "set_options"             # Update runtime defaults
    SET_LAYER_CUTS = "set_layer_cuts"       # Update sphere distance cutoffs

    # Interaction-based fixing
    KEEP_INTERACTIONS = "keep_interactions" # Add fixed residues based on interactions

    # Plugin task operation
    TASK_OPERATION = "task_operation"       # Custom task operation plugin

    # Runtime guard
    TIME_CHECK = "time_check"               # Conditional time-based branch

    # Final diversification
    FINAL_DIVERSIFY = "final_diversify"     # Multi-temp MPNN with target count


@dataclass
class MPNNStep:
    """Configuration for an MPNN design step."""
    step_type: StepType = StepType.MPNN
    temperature: Optional[float] = 0.1
    num_designs: Optional[int] = 8
    batch_size: Optional[int] = 1
    design_scope: str = "core"  # New: core, core_shell, shell_only, global
    design_spheres: List[str] = field(default_factory=lambda: ["core"])  # Legacy support
    use_sc_context: bool = True
    pack_side_chains: bool = True
    sc_denoising_steps: int = 3
    repack_everything: Optional[bool] = None
    omit_aa: Optional[str] = None
    enhance: Optional[str] = None
    bias_aa: Optional[Dict[str, float]] = None
    bias_aa_per_residue: Optional[Dict[str, Dict[str, float]]] = None
    num_designs_explicit: bool = True

    # New: scope-based factory methods
    @classmethod
    def core(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Create MPNN step for core sphere only (0-6Å)."""
        return cls(
            step_type=StepType.DESIGN_CORE,
            temperature=temperature,
            num_designs=num_designs,
            design_scope="core",
            design_spheres=["core"],
            **kwargs
        )

    @classmethod
    def core_shell(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Create MPNN step for core + shell spheres (0-8Å)."""
        return cls(
            step_type=StepType.DESIGN_CORE_SHELL,
            temperature=temperature,
            num_designs=num_designs,
            design_scope="core_shell",
            design_spheres=["core", "shell"],
            **kwargs
        )

    @classmethod
    def shell_only(cls, temperature: float = 0.1, num_designs: int = 4, **kwargs) -> "MPNNStep":
        """Create MPNN step for shell only, freeze core (6-8Å)."""
        return cls(
            step_type=StepType.DESIGN_SHELL_ONLY,
            temperature=temperature,
            num_designs=num_designs,
            design_scope="shell_only",
            design_spheres=["shell"],
            **kwargs
        )

    @classmethod
    def shell_flex(cls, temperature: float = 0.1, num_designs: int = 4, **kwargs) -> "MPNNStep":
        """Create MPNN step for shell + flex zones (6-12Å)."""
        return cls(
            step_type=StepType.DESIGN_SHELL_FLEX,
            temperature=temperature,
            num_designs=num_designs,
            design_scope="shell_flex",
            design_spheres=["shell", "flex"],
            **kwargs
        )

    @classmethod
    def flex_only(cls, temperature: float = 0.1, num_designs: int = 4, **kwargs) -> "MPNNStep":
        """Create MPNN step for flex zone only (8-12Å)."""
        return cls(
            step_type=StepType.DESIGN_FLEX_ONLY,
            temperature=temperature,
            num_designs=num_designs,
            design_scope="flex_only",
            design_spheres=["flex"],
            **kwargs
        )

    @classmethod
    def core_shell_flex(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Create MPNN step for core + shell + flex zones (0-12Å)."""
        return cls(
            step_type=StepType.DESIGN_CORE_SHELL_FLEX,
            temperature=temperature,
            num_designs=num_designs,
            design_scope="core_shell_flex",
            design_spheres=["core", "shell", "flex"],
            **kwargs
        )

    @classmethod
    def distant_only(cls, temperature: float = 0.1, num_designs: int = 4, **kwargs) -> "MPNNStep":
        """Create MPNN step for distant/frozen zone only (>12Å)."""
        return cls(
            step_type=StepType.DESIGN_DISTANT_ONLY,
            temperature=temperature,
            num_designs=num_designs,
            design_scope="distant_only",
            design_spheres=["distant"],
            **kwargs
        )

    @classmethod
    def shell_flex_distant(cls, temperature: float = 0.1, num_designs: int = 4, **kwargs) -> "MPNNStep":
        """Create MPNN step for shell + flex + distant zones (>6Å, everything except core)."""
        return cls(
            step_type=StepType.DESIGN_SHELL_FLEX_DISTANT,
            temperature=temperature,
            num_designs=num_designs,
            design_scope="shell_flex_distant",
            design_spheres=["shell", "flex", "distant"],
            **kwargs
        )

    @classmethod
    def global_design(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Create MPNN step for global redesign (all non-fixed residues except catres)."""
        return cls(
            step_type=StepType.DESIGN_GLOBAL,
            temperature=temperature,
            num_designs=num_designs,
            design_scope="global",
            design_spheres=["global"],
            **kwargs
        )


@dataclass
class MPNNMultiStep:
    """Run multiple MPNN strategies in parallel and pool outputs."""
    step_type: StepType = StepType.MPNN_MULTI
    strategies: List[MPNNStep] = field(default_factory=list)
    dedupe_pool: bool = True
    parallel: bool = False
    max_workers: Optional[int] = None
    min_workers: Optional[int] = None
    use_mpnn_server: Optional[bool] = None
    comment: Optional[str] = None

    # Legacy factory methods (for backwards compatibility)
    @classmethod
    def primary(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Legacy: Create MPNN step for primary sphere only. Use core() instead."""
        return cls.core(temperature=temperature, num_designs=num_designs, **kwargs)

    @classmethod
    def secondary(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Legacy: Create MPNN step for primary + secondary. Use core_shell() instead."""
        return cls.core_shell(temperature=temperature, num_designs=num_designs, **kwargs)

    @classmethod
    def second_shell(cls, temperature: float = 0.1, num_designs: int = 4, **kwargs) -> "MPNNStep":
        """Legacy: Create MPNN step for 2nd shell. Use shell_only() instead."""
        return cls.shell_only(temperature=temperature, num_designs=num_designs, **kwargs)

    @classmethod
    def all_spheres(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Legacy: Create MPNN step for global redesign. Use global_design() instead."""
        return cls.global_design(temperature=temperature, num_designs=num_designs, **kwargs)


@dataclass
class CartRelaxStep:
    """Configuration for Cartesian relaxation step."""
    step_type: StepType = StepType.CART_RELAX
    repeats: int = 2
    stages: int = 3
    cart_bonded_weight: float = 2.0
    coord_cst_weight: float = 750.0
    coord_cst_stdev: float = 0.01
    global_coord_cst_weight: float = 0.0
    global_coord_cst_stdev: float = 0.5
    enable_bond_geometry_min: bool = True
    scorefunction: str = "ref2015_cart"
    fa_rep_weight: Optional[float] = None
    relax_rounds: int = 5
    relax_inner_cycles: Optional[int] = None
    nstruct: int = 1
    until_converged: bool = False  # Run until geometry converges
    bond_length_tolerance: float = 0.05
    bond_angle_tolerance: float = 10.0
    score_term_weights: Optional[Dict[str, float]] = None


@dataclass
class TorsionalRelaxStep:
    """Configuration for torsional relaxation step."""
    step_type: StepType = StepType.TORSIONAL_RELAX
    repeats: int = 2
    stages: int = 3
    coord_cst_weight: float = 750.0
    coord_cst_stdev: float = 0.01
    global_coord_cst_weight: float = 0.0
    global_coord_cst_stdev: float = 0.5
    scorefunction: str = "beta_jan25"
    fa_rep_weight: Optional[float] = None
    relax_rounds: int = 5
    relax_inner_cycles: Optional[int] = None
    nstruct: int = 1
    score_term_weights: Optional[Dict[str, float]] = None


@dataclass
class MinimizeStep:
    """Configuration for minimization step."""
    step_type: StepType = StepType.MINIMIZE
    tolerance: float = 0.01
    max_iter: int = 200
    coord_cst_weight: float = 750.0
    coord_cst_stdev: float = 0.01
    global_coord_cst_weight: float = 0.0
    global_coord_cst_stdev: float = 0.5
    scorefunction: str = "beta_jan25"
    fa_rep_weight: Optional[float] = None
    cartesian: bool = False
    cart_bonded_weight: float = 2.0
    min_backbone_rmsd_cutoff: Optional[float] = None
    score_term_weights: Optional[Dict[str, float]] = None
    minimize_scope: Optional[str] = None  # core, core_shell, core_shell_flex, global


@dataclass
class RepackStep:
    """Configuration for repacking step."""
    step_type: StepType = StepType.REPACK
    repack_scope: str = "core_shell_flex"  # core, core_shell, core_shell_flex, global
    repack_shell: Optional[float] = None  # Legacy: Angstroms around design region
    coord_cst_weight: float = 750.0
    coord_cst_stdev: float = 0.01
    global_coord_cst_weight: float = 0.0
    global_coord_cst_stdev: float = 0.5
    scorefunction: str = "beta_jan25"
    fa_rep_weight: Optional[float] = None
    score_term_weights: Optional[Dict[str, float]] = None


@dataclass
class SetLayerCutsStep:
    """Configuration for updating sphere distance cutoffs mid-protocol.

    Allows dynamic adjustment of design/flex boundaries during protocol execution.
    """
    step_type: StepType = StepType.SET_LAYER_CUTS
    core_cutoff: Optional[float] = None      # DESIGN_CORE boundary (default 6.0Å)
    shell_cutoff: Optional[float] = None     # DESIGN_SHELL boundary (default 8.0Å)
    flex_cutoff: Optional[float] = None      # FLEX boundary (default 12.0Å)


@dataclass
class SelectBestStep:
    """Configuration for structure selection step."""
    step_type: StepType = StepType.SELECT_BEST
    n: int = 1
    metric: str = "geometry"  # "geometry", "score", "smart", "sequence_similarity_high/low", "ca_rmsd"
    bond_length_tolerance: Optional[float] = None
    bond_angle_tolerance: Optional[float] = None
    geom_similarity_epsilon: float = 0.005
    rmsd_similarity_epsilon: float = 0.05
    sequence_ref: str = "parent"  # parent|step02|step01
    scorefunction: Optional[str] = None


@dataclass
class ClusterStep:
    """Configuration for clustering step."""
    step_type: StepType = StepType.CLUSTER
    method: str = "sequence"  # "sequence" or "structure"
    n_clusters: Optional[int] = None
    threshold: Optional[float] = None  # identity (sequence) or RMSD (structure)
    sequence_ref: str = "parent"  # for sequence clustering reference


@dataclass
class KeepClusterBestStep:
    """Configuration for keeping best per cluster."""
    step_type: StepType = StepType.KEEP_CLUSTER_BEST
    n: int = 1
    metric: str = "geometry"
    scorefunction: Optional[str] = None


@dataclass
class ScaleScoreTermStep:
    """Scale scorefunction term(s) for subsequent Rosetta steps."""
    step_type: StepType = StepType.SCALE_SCORETERM
    terms: Dict[str, Any] = field(default_factory=dict)
    scope: str = "global"  # "global" or "next"


@dataclass
class SetOptionsStep:
    """Update runtime defaults for subsequent steps."""
    step_type: StepType = StepType.SET_OPTIONS
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeepInteractionsStep:
    """Add fixed residues based on interaction analysis."""
    step_type: StepType = StepType.KEEP_INTERACTIONS
    target: str = "catres"  # catres|motif|ligand|catres_or_motif
    interaction_types: List[str] = field(default_factory=lambda: ["hbond"])
    probability: float = 0.5
    mutator_atoms: str = "either"  # sidechain|backbone|either
    target_atoms: str = "either"   # sidechain|backbone|either
    include_ligand_interactions: bool = False
    include_catres_interactions: bool = True
    strong_interaction_types: Optional[List[str]] = None
    hbond_accept_probability: Optional[float] = None


@dataclass
class TaskOperationStep:
    """Custom task operation hook."""
    step_type: StepType = StepType.TASK_OPERATION
    module: str = ""
    function: str = "compute"
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeCheckStep:
    """Conditional time-based branch."""
    step_type: StepType = StepType.TIME_CHECK
    max_elapsed: Optional[float] = None          # seconds
    min_remaining: Optional[float] = None        # seconds
    max_runtime_fraction: Optional[float] = None # 0-1
    then_steps: List[Any] = field(default_factory=list)
    mode: str = "replace_remaining"              # replace_remaining|continue
    target_total_designs: Optional[int] = None   # optional quota helper


@dataclass
class FinalDiversifyStep:
    """Configuration for final diversification step.

    Runs MPNN at multiple temperatures, deduplicates across temps,
    then clusters/expands to meet target_count (default: --num_final_designs).
    """
    step_type: StepType = StepType.FINAL_DIVERSIFY

    # Temperatures for MPNN sampling
    temperatures: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

    # Target number of outputs (None = use --num_final_designs)
    target_count: Optional[int] = None

    # Designs per temperature per input structure
    designs_per_temp: int = 3

    # Max iterations if expansion needed
    max_iterations: int = 3

    # Design scope (inherited from shell_only by default)
    design_scope: str = "shell_only"
    design_spheres: List[str] = field(default_factory=lambda: ["shell"])

    # MPNN options
    use_sc_context: bool = True
    pack_side_chains: bool = True
    sc_denoising_steps: int = 3
    omit_aa: str = "CM"
    enhance: Optional[str] = "plddt_3_20240930-f9c9ea0f"
    batch_size: int = 1

    # Dynamic scaling threshold (0.0-1.0)
    # If max_possible < overshoot_threshold × target, double batch_size & designs_per_temp
    overshoot_threshold: float = 0.75

    # Fallback: if max_iterations exhausted and still short, try one more round with shell+flex
    fallback_include_flex: bool = True


# Type alias for any protocol step
ProtocolStep = Union[
    MPNNStep,
    MPNNMultiStep,
    CartRelaxStep,
    TorsionalRelaxStep,
    MinimizeStep,
    RepackStep,
    SelectBestStep,
    ClusterStep,
    KeepClusterBestStep,
    ScaleScoreTermStep,
    SetOptionsStep,
    KeepInteractionsStep,
    TaskOperationStep,
    TimeCheckStep,
    FinalDiversifyStep,
]


# =============================================================================
# Protocol Parser
# =============================================================================

class ProtocolParser:
    """Parse protocol definition strings into step objects.

    Protocol format examples:
        Custom format:
            "mpnn:T0.1:N4 -> cart_relax:R2S3 -> mpnn:T0.1:N2 -> torsional_relax:R1S3"

        Step syntax:
            mpnn:T<temp>:N<num>[:spheres=primary,secondary]
            cart_relax:R<repeats>S<stages>[:until_converged]
            torsional_relax:R<repeats>S<stages>
            minimize:T<tolerance>
            repack
            select_best:N<num>[:metric=geometry]
    """

    @classmethod
    def parse(cls, protocol_str: str) -> List[ProtocolStep]:
        """Parse protocol string into list of steps."""
        protocol_str = protocol_str.strip()

        # Parse custom protocol string
        LOGGER.info(f"Parsing custom protocol: {protocol_str}")
        steps = []
        for step_str in protocol_str.split("->"):
            step_str = step_str.strip()
            if not step_str:
                continue
            step = cls._parse_step(step_str)
            if step:
                steps.append(step)

        return steps

    @classmethod
    def _parse_step(cls, step_str: str) -> Optional[ProtocolStep]:
        """Parse a single step definition."""
        parts = step_str.split(":")
        step_type_str = parts[0].strip().lower()
        params = parts[1:] if len(parts) > 1 else []

        # Design steps - new naming (scope-based)
        if step_type_str in ("design_core", "mpnn", "mpnn_primary"):
            return cls._parse_mpnn_step(params, spheres=["core"], step_type=StepType.DESIGN_CORE)
        elif step_type_str in ("design_core_shell", "mpnn_secondary"):
            return cls._parse_mpnn_step(params, spheres=["core", "shell"], step_type=StepType.DESIGN_CORE_SHELL)
        elif step_type_str in ("design_shell_only", "mpnn_2nd_shell"):
            return cls._parse_mpnn_step(params, spheres=["shell"], step_type=StepType.DESIGN_SHELL_ONLY)
        elif step_type_str in ("design_shell_flex", "shell_flex"):
            return cls._parse_mpnn_step(params, spheres=["shell", "flex"], step_type=StepType.DESIGN_SHELL_FLEX)
        elif step_type_str in ("design_flex_only", "flex_only"):
            return cls._parse_mpnn_step(params, spheres=["flex"], step_type=StepType.DESIGN_FLEX_ONLY)
        elif step_type_str in ("design_core_shell_flex", "core_shell_flex"):
            return cls._parse_mpnn_step(params, spheres=["core", "shell", "flex"], step_type=StepType.DESIGN_CORE_SHELL_FLEX)
        elif step_type_str in ("design_distant_only", "distant_only", "frozen_only"):
            return cls._parse_mpnn_step(params, spheres=["distant"], step_type=StepType.DESIGN_DISTANT_ONLY)
        elif step_type_str in ("design_shell_flex_distant", "shell_flex_distant"):
            return cls._parse_mpnn_step(params, spheres=["shell", "flex", "distant"], step_type=StepType.DESIGN_SHELL_FLEX_DISTANT)
        elif step_type_str in ("design_global", "mpnn_all"):
            return cls._parse_mpnn_step(params, spheres=["global"], step_type=StepType.DESIGN_GLOBAL)
        # Relaxation steps
        elif step_type_str == "cart_relax":
            return cls._parse_cart_relax_step(params)
        elif step_type_str in ("torsional_relax", "tors_relax"):
            return cls._parse_torsional_relax_step(params)
        elif step_type_str in ("minimize", "min"):
            return cls._parse_minimize_step(params)
        elif step_type_str in ("repack", "rosetta_repack_only"):
            return cls._parse_repack_step(params)
        # Selection steps
        elif step_type_str in ("select_best", "select"):
            return cls._parse_select_best_step(params)
        elif step_type_str in ("cluster",):
            return cls._parse_cluster_step(params)
        elif step_type_str in ("keep_cluster_best", "keepclusterbest"):
            return cls._parse_keep_cluster_best_step(params)
        # Configuration steps
        elif step_type_str in ("scale", "scale_scoreterm"):
            return cls._parse_scale_step(params)
        elif step_type_str in ("set", "set_options"):
            return cls._parse_set_step(params)
        elif step_type_str in ("set_layer_cuts", "layer_cuts"):
            return cls._parse_set_layer_cuts_step(params)
        # Interaction steps
        elif step_type_str in ("keep_interactions", "keep_interaction"):
            return cls._parse_keep_interactions_step(params)
        elif step_type_str in ("keep_hbonds", "keep_hbond"):
            return cls._parse_keep_interactions_step(params, default_types=["hbond"])
        # Plugin steps
        elif step_type_str in ("task_operation", "taskop"):
            return cls._parse_task_operation_step(params)
        # Comments (skip)
        elif step_type_str == "comment":
            return None
        else:
            LOGGER.warning(f"Unknown step type: {step_type_str}")
            return None

    @classmethod
    def _split_kv_params(cls, params: List[str]) -> Tuple[Dict[str, str], List[str]]:
        """Split params into key=value dict and positional tokens."""
        kv: Dict[str, str] = {}
        positional: List[str] = []
        for p in params:
            p = p.strip()
            if not p:
                continue
            if "=" in p:
                key, val = p.split("=", 1)
                kv[key.strip().lower()] = val.strip()
            else:
                positional.append(p)
        return kv, positional

    @classmethod
    def _parse_mpnn_step(
        cls,
        params: List[str],
        spheres: List[str] = None,
        step_type: StepType = StepType.MPNN,
    ) -> MPNNStep:
        """Parse MPNN step parameters."""
        temperature: Optional[float] = None
        num_designs: Optional[int] = None
        batch_size: Optional[int] = None
        design_spheres = spheres or ["core"]
        design_scope: Optional[str] = None
        omit_aa: Optional[str] = None
        enhance: Optional[str] = None
        use_sc_context: Optional[bool] = None
        pack_side_chains: Optional[bool] = None
        sc_denoising_steps: Optional[int] = None
        repack_everything: Optional[bool] = None
        num_designs_explicit = False
        kv, positional = cls._split_kv_params(params)

        for p in positional:
            p = p.strip()
            if p.startswith("T"):
                temperature = float(p[1:])
            elif p.startswith("N"):
                num_designs = int(p[1:])
                num_designs_explicit = True
            elif p.startswith("B"):
                batch_size = int(p[1:])
            elif p.startswith("spheres="):
                design_spheres = [s.strip() for s in p[8:].split(",") if s.strip()]
            elif p.startswith("scope="):
                design_scope = p[6:].strip()

        if "temperature" in kv and temperature is None:
            temperature = float(kv["temperature"])
        if "num_designs" in kv and num_designs is None:
            num_designs = int(kv["num_designs"])
            num_designs_explicit = True
        if "batch_size" in kv and batch_size is None:
            batch_size = int(kv["batch_size"])
        if "omit_aa" in kv:
            omit_aa = kv["omit_aa"]
        if "enhance" in kv or "enhance_model" in kv:
            enhance = kv.get("enhance", kv.get("enhance_model"))
        if "use_sc_context" in kv:
            use_sc_context = kv["use_sc_context"].lower() in ("1", "true", "yes", "y")
        if "pack_side_chains" in kv:
            pack_side_chains = kv["pack_side_chains"].lower() in ("1", "true", "yes", "y")
        if "sc_denoising_steps" in kv:
            sc_denoising_steps = int(kv["sc_denoising_steps"])
        if "repack_everything" in kv:
            repack_everything = kv["repack_everything"].lower() in ("1", "true", "yes", "y")
        if "spheres" in kv:
            design_spheres = [s.strip() for s in kv["spheres"].split(",") if s.strip()]
        if "scope" in kv or "design_scope" in kv:
            design_scope = kv.get("scope", kv.get("design_scope"))

        # Normalize sphere names (legacy -> new)
        sphere_map = {
            "primary": "core",
            "secondary": "shell",
            "repack_primary": "flex",
            "repack_secondary": "flex",
        }
        design_spheres = [sphere_map.get(s.lower(), s.lower()) for s in design_spheres if s.strip()]

        # Infer design_scope from spheres if not explicitly set
        if design_scope is None:
            if design_spheres == ["core"]:
                design_scope = "core"
            elif set(design_spheres) == {"core", "shell"}:
                design_scope = "core_shell"
            elif design_spheres == ["shell"]:
                design_scope = "shell_only"
            elif "global" in design_spheres:
                design_scope = "global"
            else:
                design_scope = "core"  # Default

        return MPNNStep(
            step_type=step_type,
            temperature=temperature,
            num_designs=num_designs,
            batch_size=batch_size,
            design_spheres=design_spheres,
            design_scope=design_scope,
            omit_aa=omit_aa,
            num_designs_explicit=num_designs_explicit,
            use_sc_context=use_sc_context if use_sc_context is not None else True,
            pack_side_chains=pack_side_chains if pack_side_chains is not None else True,
            sc_denoising_steps=sc_denoising_steps if sc_denoising_steps is not None else 3,
            repack_everything=repack_everything,
            enhance=enhance,
        )

    @classmethod
    def _parse_cart_relax_step(cls, params: List[str]) -> CartRelaxStep:
        """Parse Cartesian relax step parameters.

        Supported parameters:
            R<repeats>S<stages> - e.g. R3S4 for 3 repeats, 4 stages
            until_converged - run until geometry converges
            cb=<float> - cart_bonded_weight (default 2.0, try 3.0-5.0 for stricter geometry)
            fr=<float> - fa_rep_weight (default ~0.55, try 0.1-0.3 to allow clashes)
            sf=<name> - scorefunction name (default ref2015_cart)
            rounds=<int> - relax_rounds (default 5)
            bl_tol=<float> - bond_length_tolerance for convergence (default 0.05A)
            ba_tol=<float> - bond_angle_tolerance for convergence (default 10.0 deg)
        """
        repeats = 2
        stages = 3
        until_converged = False
        cart_bonded_weight = 2.0
        fa_rep_weight = None
        scorefunction = "ref2015_cart"
        relax_rounds = 5
        bond_length_tolerance = 0.05
        bond_angle_tolerance = 10.0

        for p in params:
            p = p.strip()
            # R<repeats>S<stages> format
            match = re.match(r"R(\d+)S(\d+)", p, re.IGNORECASE)
            if match:
                repeats = int(match.group(1))
                stages = int(match.group(2))
            elif p.lower() == "until_converged":
                until_converged = True
            elif p.startswith("cb="):
                cart_bonded_weight = float(p[3:])
            elif p.startswith("fr="):
                fa_rep_weight = float(p[3:])
            elif p.startswith("sf="):
                scorefunction = p[3:]
            elif p.startswith("rounds="):
                relax_rounds = int(p[7:])
            elif p.startswith("bl_tol="):
                bond_length_tolerance = float(p[7:])
            elif p.startswith("ba_tol="):
                bond_angle_tolerance = float(p[7:])

        return CartRelaxStep(
            repeats=repeats,
            stages=stages,
            until_converged=until_converged,
            cart_bonded_weight=cart_bonded_weight,
            fa_rep_weight=fa_rep_weight,
            scorefunction=scorefunction,
            relax_rounds=relax_rounds,
            bond_length_tolerance=bond_length_tolerance,
            bond_angle_tolerance=bond_angle_tolerance,
        )

    @classmethod
    def _parse_torsional_relax_step(cls, params: List[str]) -> TorsionalRelaxStep:
        """Parse torsional relax step parameters.

        Supported parameters:
            R<repeats>S<stages> - e.g. R2S3 for 2 repeats, 3 stages
            fr=<float> - fa_rep_weight (default ~0.55)
            sf=<name> - scorefunction name (default beta_jan25)
            rounds=<int> - relax_rounds (default 5)
        """
        repeats = 2
        stages = 3
        fa_rep_weight = None
        scorefunction = "beta_jan25"
        relax_rounds = 5

        for p in params:
            p = p.strip()
            match = re.match(r"R(\d+)S(\d+)", p, re.IGNORECASE)
            if match:
                repeats = int(match.group(1))
                stages = int(match.group(2))
            elif p.startswith("fr="):
                fa_rep_weight = float(p[3:])
            elif p.startswith("sf="):
                scorefunction = p[3:]
            elif p.startswith("rounds="):
                relax_rounds = int(p[7:])

        return TorsionalRelaxStep(
            repeats=repeats,
            stages=stages,
            fa_rep_weight=fa_rep_weight,
            scorefunction=scorefunction,
            relax_rounds=relax_rounds,
        )

    @classmethod
    def _parse_minimize_step(cls, params: List[str]) -> MinimizeStep:
        """Parse minimize step parameters."""
        tolerance = 0.01
        max_iter = 200
        cartesian = False
        scorefunction = "beta_jan25"
        fa_rep_weight = None
        coord_cst_weight = 750.0
        coord_cst_stdev = 0.01
        global_coord_cst_weight = 0.0
        global_coord_cst_stdev = 0.5
        cart_bonded_weight = 2.0
        min_backbone_rmsd_cutoff = None
        minimize_scope = None
        kv, positional = cls._split_kv_params(params)

        for p in positional:
            p = p.strip()
            if p.startswith("T"):
                tolerance = float(p[1:])
            elif p.startswith("I"):
                max_iter = int(p[1:])
            elif p.lower() in ("cart", "cartesian"):
                cartesian = True

        if "tolerance" in kv:
            tolerance = float(kv["tolerance"])
        if "max_iter" in kv:
            max_iter = int(kv["max_iter"])
        if "cartesian" in kv:
            cartesian = kv["cartesian"].lower() in ("1", "true", "yes", "y")
        if "scorefunction" in kv or "sf" in kv:
            scorefunction = kv.get("scorefunction", kv.get("sf"))
        if "fa_rep" in kv:
            fa_rep_weight = float(kv["fa_rep"])
        if "coord_cst_weight" in kv:
            coord_cst_weight = float(kv["coord_cst_weight"])
        if "coord_cst_stdev" in kv:
            coord_cst_stdev = float(kv["coord_cst_stdev"])
        if "global_coord_cst_weight" in kv:
            global_coord_cst_weight = float(kv["global_coord_cst_weight"])
        if "global_coord_cst_stdev" in kv:
            global_coord_cst_stdev = float(kv["global_coord_cst_stdev"])
        if "cart_bonded_weight" in kv:
            cart_bonded_weight = float(kv["cart_bonded_weight"])
        if "min_backbone_rmsd_cutoff" in kv:
            min_backbone_rmsd_cutoff = float(kv["min_backbone_rmsd_cutoff"])
        if "minimize_scope" in kv or "scope" in kv:
            minimize_scope = kv.get("minimize_scope", kv.get("scope"))

        return MinimizeStep(
            tolerance=tolerance,
            max_iter=max_iter,
            cartesian=cartesian,
            scorefunction=scorefunction,
            fa_rep_weight=fa_rep_weight,
            coord_cst_weight=coord_cst_weight,
            coord_cst_stdev=coord_cst_stdev,
            global_coord_cst_weight=global_coord_cst_weight,
            global_coord_cst_stdev=global_coord_cst_stdev,
            cart_bonded_weight=cart_bonded_weight,
            min_backbone_rmsd_cutoff=min_backbone_rmsd_cutoff,
            minimize_scope=minimize_scope,
        )

    @classmethod
    def _parse_repack_step(cls, params: List[str]) -> RepackStep:
        """Parse repack step parameters.

        Supports repack_scope: core, core_shell, core_shell_flex (default), global
        """
        repack_scope = "core_shell_flex"  # New default
        repack_shell = None  # Legacy parameter
        scorefunction = "beta_jan25"
        fa_rep_weight = None
        coord_cst_weight = 750.0
        coord_cst_stdev = 0.01
        global_coord_cst_weight = 0.0
        global_coord_cst_stdev = 0.5
        kv, positional = cls._split_kv_params(params)

        for p in positional:
            p = p.strip()
            if p.startswith("shell="):
                repack_shell = float(p[6:])
            elif p.startswith("R"):
                repack_shell = float(p[1:])
            elif p in ("core", "core_shell", "core_shell_flex", "global"):
                repack_scope = p

        # New scope parameter
        if "repack_scope" in kv or "scope" in kv:
            repack_scope = kv.get("repack_scope", kv.get("scope", repack_scope))
        # Legacy shell parameter
        if "repack_shell" in kv:
            repack_shell = float(kv["repack_shell"])
        if "scorefunction" in kv or "sf" in kv:
            scorefunction = kv.get("scorefunction", kv.get("sf"))
        if "fa_rep" in kv:
            fa_rep_weight = float(kv["fa_rep"])
        if "coord_cst_weight" in kv:
            coord_cst_weight = float(kv["coord_cst_weight"])
        if "coord_cst_stdev" in kv:
            coord_cst_stdev = float(kv["coord_cst_stdev"])
        if "global_coord_cst_weight" in kv:
            global_coord_cst_weight = float(kv["global_coord_cst_weight"])
        if "global_coord_cst_stdev" in kv:
            global_coord_cst_stdev = float(kv["global_coord_cst_stdev"])

        return RepackStep(
            repack_scope=repack_scope,
            repack_shell=repack_shell,
            scorefunction=scorefunction,
            fa_rep_weight=fa_rep_weight,
            coord_cst_weight=coord_cst_weight,
            coord_cst_stdev=coord_cst_stdev,
            global_coord_cst_weight=global_coord_cst_weight,
            global_coord_cst_stdev=global_coord_cst_stdev,
        )

    @classmethod
    def _parse_select_best_step(cls, params: List[str]) -> SelectBestStep:
        """Parse select best step parameters."""
        n = 1
        metric = "geometry"
        bond_length_tolerance = None
        bond_angle_tolerance = None
        geom_similarity_epsilon = 0.005
        rmsd_similarity_epsilon = 0.05
        sequence_ref = "parent"
        scorefunction = None
        kv, positional = cls._split_kv_params(params)

        for p in positional:
            p = p.strip()
            if p.startswith("N"):
                n = int(p[1:])
            elif p.startswith("metric="):
                metric = p[7:]
        if "n" in kv:
            n = int(kv["n"])
        if "metric" in kv:
            metric = kv["metric"]
        if "bond_length_tolerance" in kv:
            bond_length_tolerance = float(kv["bond_length_tolerance"])
        if "bond_angle_tolerance" in kv:
            bond_angle_tolerance = float(kv["bond_angle_tolerance"])
        if "geom_similarity_epsilon" in kv:
            geom_similarity_epsilon = float(kv["geom_similarity_epsilon"])
        if "rmsd_similarity_epsilon" in kv:
            rmsd_similarity_epsilon = float(kv["rmsd_similarity_epsilon"])
        if "sequence_ref" in kv:
            sequence_ref = kv["sequence_ref"]
        if "scorefunction" in kv or "sf" in kv:
            scorefunction = kv.get("scorefunction", kv.get("sf"))

        return SelectBestStep(
            n=n,
            metric=metric,
            bond_length_tolerance=bond_length_tolerance,
            bond_angle_tolerance=bond_angle_tolerance,
            geom_similarity_epsilon=geom_similarity_epsilon,
            rmsd_similarity_epsilon=rmsd_similarity_epsilon,
            sequence_ref=sequence_ref,
            scorefunction=scorefunction,
        )

    @classmethod
    def _parse_scale_step(cls, params: List[str]) -> ScaleScoreTermStep:
        kv, positional = cls._split_kv_params(params)
        terms: Dict[str, Any] = {}
        scope = kv.get("scope", "global")
        reset_tokens = {"reset", "default", "none", "null"}
        if positional:
            for i in range(0, len(positional), 2):
                term = positional[i]
                if i + 1 < len(positional):
                    raw = positional[i + 1]
                    if isinstance(raw, str) and raw.strip().lower() in reset_tokens:
                        terms[term] = "reset"
                        continue
                    try:
                        terms[term] = float(raw)
                    except ValueError:
                        continue
        for k, v in kv.items():
            if k == "scope":
                continue
            if isinstance(v, str) and v.strip().lower() in reset_tokens:
                terms[k] = "reset"
                continue
            try:
                terms[k] = float(v)
            except ValueError:
                continue
        return ScaleScoreTermStep(terms=terms, scope=scope)

    @classmethod
    def _parse_set_step(cls, params: List[str]) -> SetOptionsStep:
        kv, positional = cls._split_kv_params(params)
        options: Dict[str, Any] = {}
        if positional:
            for i in range(0, len(positional), 2):
                key = positional[i]
                if i + 1 < len(positional):
                    options[key] = positional[i + 1]
        for k, v in kv.items():
            options[k] = v
        return SetOptionsStep(options=options)

    @classmethod
    def _parse_set_layer_cuts_step(cls, params: List[str]) -> SetLayerCutsStep:
        """Parse set_layer_cuts step parameters.

        Syntax:
            set_layer_cuts:core=6.0:shell=8.0:flex=12.0
            set_layer_cuts:6.0:8.0:12.0
        """
        kv, positional = cls._split_kv_params(params)

        core_cutoff = None
        shell_cutoff = None
        flex_cutoff = None

        # Handle positional arguments (in order: core, shell, flex)
        if len(positional) >= 1:
            core_cutoff = float(positional[0])
        if len(positional) >= 2:
            shell_cutoff = float(positional[1])
        if len(positional) >= 3:
            flex_cutoff = float(positional[2])

        # Key-value overrides
        if "core" in kv or "core_cutoff" in kv:
            core_cutoff = float(kv.get("core", kv.get("core_cutoff")))
        if "shell" in kv or "shell_cutoff" in kv:
            shell_cutoff = float(kv.get("shell", kv.get("shell_cutoff")))
        if "flex" in kv or "flex_cutoff" in kv:
            flex_cutoff = float(kv.get("flex", kv.get("flex_cutoff")))

        return SetLayerCutsStep(
            core_cutoff=core_cutoff,
            shell_cutoff=shell_cutoff,
            flex_cutoff=flex_cutoff,
        )

    @classmethod
    def _parse_keep_interactions_step(
        cls,
        params: List[str],
        default_types: Optional[List[str]] = None,
    ) -> KeepInteractionsStep:
        kv, positional = cls._split_kv_params(params)
        target = kv.get("target", "catres")
        types = default_types or ["hbond"]
        if "types" in kv:
            types = [t.strip() for t in kv["types"].split(",") if t.strip()]
        probability = float(kv.get("prob", kv.get("probability", 0.5)))
        mutator_atoms = kv.get("mutator", kv.get("mutator_atoms", "either"))
        target_atoms = kv.get("target_atoms", kv.get("target_atoms_scope", "either"))
        include_ligand = kv.get("include_ligand", None)
        include_catres = kv.get("include_catres", None)
        strong_types = None
        if "strong_types" in kv:
            strong_types = [t.strip() for t in kv["strong_types"].split(",") if t.strip()]
        hbond_accept_probability = None
        if "hbond_accept_probability" in kv:
            hbond_accept_probability = float(kv["hbond_accept_probability"])

        if include_ligand is not None:
            include_ligand = include_ligand.lower() in ("1", "true", "yes", "y")
        if include_catres is not None:
            include_catres = include_catres.lower() in ("1", "true", "yes", "y")

        return KeepInteractionsStep(
            target=target,
            interaction_types=types,
            probability=probability,
            mutator_atoms=mutator_atoms,
            target_atoms=target_atoms,
            include_ligand_interactions=bool(include_ligand) if include_ligand is not None else False,
            include_catres_interactions=bool(include_catres) if include_catres is not None else True,
            strong_interaction_types=strong_types,
            hbond_accept_probability=hbond_accept_probability,
        )

    @classmethod
    def _parse_cluster_step(cls, params: List[str]) -> ClusterStep:
        kv, positional = cls._split_kv_params(params)
        method = "sequence"
        n_clusters = None
        threshold = None
        if positional:
            method = positional[0].lower()
            for p in positional[1:]:
                if p.startswith("N"):
                    n_clusters = int(p[1:])
                elif p.startswith("T"):
                    threshold = float(p[1:])
        if "method" in kv:
            method = kv["method"]
        if "n" in kv or "n_clusters" in kv:
            n_clusters = int(kv.get("n_clusters", kv.get("n")))
        if "threshold" in kv:
            threshold = float(kv["threshold"])
        return ClusterStep(method=method, n_clusters=n_clusters, threshold=threshold)

    @classmethod
    def _parse_keep_cluster_best_step(cls, params: List[str]) -> KeepClusterBestStep:
        n = 1
        metric = "geometry"
        kv, positional = cls._split_kv_params(params)
        for p in positional:
            p = p.strip()
            if p.startswith("N"):
                n = int(p[1:])
        if "n" in kv:
            n = int(kv["n"])
        if "metric" in kv:
            metric = kv["metric"]
        return KeepClusterBestStep(n=n, metric=metric)

    @classmethod
    def _parse_task_operation_step(cls, params: List[str]) -> TaskOperationStep:
        kv, positional = cls._split_kv_params(params)
        module = kv.get("module", "")
        function = kv.get("function", "compute")
        args: Dict[str, Any] = {}
        # Support positional: module function
        if positional:
            module = positional[0]
            if len(positional) > 1:
                function = positional[1]
        # Support args as key=value pairs prefixed with arg:
        for k, v in kv.items():
            if k in ("module", "function"):
                continue
            args[k] = v
        return TaskOperationStep(module=module, function=function, args=args)

    @classmethod
    def describe_protocol(cls, steps: List[ProtocolStep]) -> str:
        """Generate human-readable description of a protocol."""
        lines = []
        for i, step in enumerate(steps, 1):
            if isinstance(step, MPNNStep):
                spheres = ",".join(step.design_spheres)
                lines.append(f"{i}. MPNN design (T={step.temperature}, N={step.num_designs}, spheres={spheres})")
            elif isinstance(step, MPNNMultiStep):
                strategy_desc = []
                for s in step.strategies:
                    spheres = ",".join(s.design_spheres)
                    strategy_desc.append(f"{s.step_type.value}(T={s.temperature},N={s.num_designs},spheres={spheres})")
                lines.append(f"{i}. MPNN multi ({len(step.strategies)} strategies): " + "; ".join(strategy_desc))
            elif isinstance(step, CartRelaxStep):
                conv = " until converged" if step.until_converged else ""
                lines.append(f"{i}. Cartesian relax (R{step.repeats}S{step.stages}{conv})")
            elif isinstance(step, TorsionalRelaxStep):
                lines.append(f"{i}. Torsional relax (R{step.repeats}S{step.stages})")
            elif isinstance(step, MinimizeStep):
                lines.append(f"{i}. Minimize (tol={step.tolerance})")
            elif isinstance(step, RepackStep):
                scope = f" scope={step.repack_scope}" if step.repack_scope else ""
                shell = f" shell={step.repack_shell}Å" if step.repack_shell else ""
                lines.append(f"{i}. Repack{scope}{shell}")
            elif isinstance(step, SelectBestStep):
                lines.append(f"{i}. Select best {step.n} by {step.metric}")
            elif isinstance(step, ClusterStep):
                detail = f"method={step.method}"
                if step.n_clusters:
                    detail += f", k={step.n_clusters}"
                if step.threshold is not None:
                    detail += f", threshold={step.threshold}"
                lines.append(f"{i}. Cluster ({detail})")
            elif isinstance(step, KeepClusterBestStep):
                lines.append(f"{i}. Keep best {step.n} per cluster by {step.metric}")
            elif isinstance(step, ScaleScoreTermStep):
                lines.append(f"{i}. Scale scoreterms {step.terms} (scope={step.scope})")
            elif isinstance(step, SetOptionsStep):
                lines.append(f"{i}. Set options {step.options}")
            elif isinstance(step, KeepInteractionsStep):
                lines.append(f"{i}. Keep interactions target={step.target} types={step.interaction_types} p={step.probability}")
            elif isinstance(step, TaskOperationStep):
                lines.append(f"{i}. Task operation {step.module}:{step.function}")
            elif isinstance(step, SetLayerCutsStep):
                cuts = []
                if step.core_cutoff is not None:
                    cuts.append(f"core={step.core_cutoff}Å")
                if step.shell_cutoff is not None:
                    cuts.append(f"shell={step.shell_cutoff}Å")
                if step.flex_cutoff is not None:
                    cuts.append(f"flex={step.flex_cutoff}Å")
                lines.append(f"{i}. Set layer cuts ({', '.join(cuts) if cuts else 'no changes'})")
            else:
                lines.append(f"{i}. Unknown step: {step}")

        return "\n".join(lines)


# =============================================================================
# Protocol File Parser
# =============================================================================

class ProtocolValidationError(Exception):
    """Raised when protocol file validation fails."""
    pass


class ProtocolFileParser:
    """Parse protocol definitions from JSON or text files.

    Supports two formats:

    JSON format (protocol.json):
        {
          "steps": [
            {"type": "mpnn", "temperature": 0.1, "num_designs": 1, "spheres": ["primary"]},
            {"type": "cart_relax", "repeats": 2, "stages": 3},
            {"type": "mpnn", "temperature": 0.1, "num_designs": 4},
            {"type": "torsional_relax", "repeats": 1, "stages": 3}
          ]
        }

    Simple text format (protocol.txt):
        mpnn:T0.1:N1:spheres=primary
        cart_relax:R2S3
        mpnn:T0.1:N4
        torsional_relax:R1S3

    Text format uses the same syntax as custom protocol strings but with newlines
    instead of '->' separators.
    """

    # Valid step types for validation
    VALID_STEP_TYPES = {
        # New design step names
        "design_core", "design_core_shell", "design_shell_only",
        "design_shell_flex", "design_flex_only", "design_core_shell_flex",
        "design_distant_only", "design_shell_flex_distant", "design_global",
        # Legacy MPNN step names
        "mpnn", "mpnn_primary", "mpnn_secondary", "mpnn_2nd_shell", "mpnn_all",
        "mpnn_multi",
        # Relaxation
        "cart_relax", "torsional_relax", "tors_relax",
        # Other steps
        "minimize", "min", "repack", "select_best", "select",
        "scale", "scale_scoreterm", "set", "set_options",
        "set_layer_cuts",  # Dynamic layer cutoffs
        "keep_interactions", "keep_hbonds",
        "cluster", "keep_cluster_best",
        "task_operation",
        "time_check",
        "final_diversify",  # Multi-temp MPNN diversification
        "comment",  # Documentation comments (ignored during execution)
    }

    # JSON parameter validation schemas
    STEP_SCHEMAS = {
        "mpnn": {
            "required": [],
            "optional": {
                "temperature": (float, int),
                "num_designs": int,
                "batch_size": int,
                "spheres": list,
                "use_sc_context": bool,
                "pack_side_chains": bool,
                "sc_denoising_steps": int,
                "omit_aa": str,
                "enhance": str,
                "repack_everything": bool,
            },
            "defaults": {
                "temperature": 0.1,
                "num_designs": 8,
                "batch_size": 1,
                "spheres": ["primary"],
            }
        },
        "mpnn_multi": {
            "required": ["strategies"],
            "optional": {
                "strategies": list,
                "defaults": dict,
                "dedupe_pool": bool,
                "parallel": bool,
                "max_workers": int,
                "min_workers": int,
                "use_mpnn_server": bool,
            },
            "defaults": {
                "dedupe_pool": True,
            }
        },
        "cart_relax": {
            "required": [],
            "optional": {
                "repeats": int,
                "stages": int,
                "cart_bonded_weight": (float, int),
                "coord_cst_weight": (float, int),
                "coord_cst_stdev": (float, int),
                "global_coord_cst_weight": (float, int),
                "global_coord_cst_stdev": (float, int),
                "scorefunction": str,
                "fa_rep_weight": (float, int, type(None)),
                "relax_rounds": int,
                "relax_inner_cycles": (int, type(None)),
                "nstruct": int,
                "until_converged": bool,
                "bond_length_tolerance": (float, int),
                "bond_angle_tolerance": (float, int),
            },
            "defaults": {
                "repeats": 2,
                "stages": 3,
                "global_coord_cst_weight": 0.0,
                "global_coord_cst_stdev": 0.5,
                "relax_rounds": 5,
                "nstruct": 1,
            }
        },
        "torsional_relax": {
            "required": [],
            "optional": {
                "repeats": int,
                "stages": int,
                "coord_cst_weight": (float, int),
                "coord_cst_stdev": (float, int),
                "global_coord_cst_weight": (float, int),
                "global_coord_cst_stdev": (float, int),
                "scorefunction": str,
                "fa_rep_weight": (float, int, type(None)),
                "relax_rounds": int,
                "relax_inner_cycles": (int, type(None)),
                "nstruct": int,
            },
            "defaults": {
                "repeats": 2,
                "stages": 3,
                "global_coord_cst_weight": 0.0,
                "global_coord_cst_stdev": 0.5,
                "relax_rounds": 5,
                "nstruct": 1,
            }
        },
        "minimize": {
            "required": [],
            "optional": {
                "tolerance": (float, int),
                "max_iter": int,
                "coord_cst_weight": (float, int),
                "coord_cst_stdev": (float, int),
                "global_coord_cst_weight": (float, int),
                "global_coord_cst_stdev": (float, int),
                "scorefunction": str,
                "fa_rep_weight": (float, int, type(None)),
                "cartesian": bool,
                "cart_bonded_weight": (float, int),
                "min_backbone_rmsd_cutoff": (float, int, type(None)),
                "minimize_scope": str,
                "scope": str,
            },
            "defaults": {
                "tolerance": 0.01,
                "max_iter": 200,
            }
        },
        "repack": {
            "required": [],
            "optional": {
                "repack_scope": str,  # core, core_shell, core_shell_flex, global
                "repack_shell": (float, int, type(None)),  # Legacy
                "coord_cst_weight": (float, int),
                "coord_cst_stdev": (float, int),
                "global_coord_cst_weight": (float, int),
                "global_coord_cst_stdev": (float, int),
                "scorefunction": str,
                "fa_rep_weight": (float, int, type(None)),
            },
            "defaults": {
                "repack_scope": "core_shell_flex",
            }
        },
        "select_best": {
            "required": [],
            "optional": {
                "n": int,
                "metric": str,
                "bond_length_tolerance": (float, int, type(None)),
                "bond_angle_tolerance": (float, int, type(None)),
                "geom_similarity_epsilon": (float, int),
                "rmsd_similarity_epsilon": (float, int),
                "sequence_ref": str,
                "scorefunction": (str, type(None)),
            },
            "defaults": {
                "n": 1,
                "metric": "geometry",
            }
        },
        "scale": {
            "required": [],
            "optional": {
                "terms": dict,
                "scope": str,
            },
            "defaults": {}
        },
        "set": {
            "required": [],
            "optional": {
                "options": dict,
            },
            "defaults": {}
        },
        "keep_interactions": {
            "required": [],
            "optional": {
                "target": str,
                "interaction_types": list,
                "probability": (float, int),
                "mutator_atoms": str,
                "target_atoms": str,
                "include_ligand_interactions": bool,
                "include_catres_interactions": bool,
                "strong_interaction_types": (list, type(None)),
                "hbond_accept_probability": (float, int, type(None)),
            },
            "defaults": {}
        },
        "cluster": {
            "required": [],
            "optional": {
                "method": str,
                "n_clusters": (int, type(None)),
                "threshold": (float, int, type(None)),
            },
            "defaults": {}
        },
        "keep_cluster_best": {
            "required": [],
            "optional": {
                "n": int,
                "metric": str,
                "scorefunction": (str, type(None)),
            },
            "defaults": {
                "n": 1,
                "metric": "geometry",
            }
        },
        "task_operation": {
            "required": ["module"],
            "optional": {
                "function": str,
                "args": dict,
            },
            "defaults": {
                "function": "compute",
                "args": {},
            }
        },
        "time_check": {
            "required": [],
            "optional": {
                "max_elapsed": (float, int),
                "min_remaining": (float, int),
                "max_runtime_fraction": (float, int),
                "then": list,
                "mode": str,
                "target_total_designs": (int, float),
            },
            "defaults": {}
        },
        # New design step schemas (same as mpnn but with implied scope)
        "design_core": {
            "required": [],
            "optional": {
                "temperature": (float, int),
                "num_designs": int,
                "batch_size": int,
                "use_sc_context": bool,
                "pack_side_chains": bool,
                "sc_denoising_steps": int,
                "omit_aa": str,
                "enhance": str,
                "repack_everything": bool,
            },
            "defaults": {
                "temperature": 0.1,
                "num_designs": 8,
                "batch_size": 1,
            }
        },
        "design_core_shell": {
            "required": [],
            "optional": {
                "temperature": (float, int),
                "num_designs": int,
                "batch_size": int,
                "use_sc_context": bool,
                "pack_side_chains": bool,
                "sc_denoising_steps": int,
                "omit_aa": str,
                "enhance": str,
                "repack_everything": bool,
            },
            "defaults": {
                "temperature": 0.1,
                "num_designs": 8,
                "batch_size": 1,
            }
        },
        "design_shell_only": {
            "required": [],
            "optional": {
                "temperature": (float, int),
                "num_designs": int,
                "batch_size": int,
                "use_sc_context": bool,
                "pack_side_chains": bool,
                "sc_denoising_steps": int,
                "omit_aa": str,
                "enhance": str,
                "repack_everything": bool,
            },
            "defaults": {
                "temperature": 0.1,
                "num_designs": 4,
                "batch_size": 1,
            }
        },
        "design_global": {
            "required": [],
            "optional": {
                "temperature": (float, int),
                "num_designs": int,
                "batch_size": int,
                "use_sc_context": bool,
                "pack_side_chains": bool,
                "sc_denoising_steps": int,
                "omit_aa": str,
                "enhance": str,
                "repack_everything": bool,
            },
            "defaults": {
                "temperature": 0.3,
                "num_designs": 32,
                "batch_size": 1,
            }
        },
        "set_layer_cuts": {
            "required": [],
            "optional": {
                "core_cutoff": (float, int),
                "shell_cutoff": (float, int),
                "flex_cutoff": (float, int),
            },
            "defaults": {}
        },
        "comment": {
            "required": [],
            "optional": {
                "text": str,
            },
            "defaults": {}
        },
        "final_diversify": {
            "required": [],
            "optional": {
                "temperatures": list,
                "target_count": int,
                "designs_per_temp": int,
                "max_iterations": int,
                "design_scope": str,
                "use_sc_context": bool,
                "pack_side_chains": bool,
                "sc_denoising_steps": int,
                "omit_aa": str,
                "enhance": str,
                "batch_size": int,
                "overshoot_threshold": (float, int),
                "fallback_include_flex": bool,
            },
            "defaults": {
                "temperatures": [0.1, 0.2, 0.3],
                "designs_per_temp": 3,
                "max_iterations": 3,
                "design_scope": "shell_only",
                "overshoot_threshold": 0.75,
                "fallback_include_flex": True,
            }
        },
    }

    @classmethod
    def load_from_file(cls, file_path: str) -> List[ProtocolStep]:
        """Load and parse a protocol from a file.

        Args:
            file_path: Path to protocol file (.json or .txt)

        Returns:
            List of protocol steps

        Raises:
            FileNotFoundError: If file doesn't exist
            ProtocolValidationError: If protocol format is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Protocol file not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        with open(file_path, "r") as f:
            content = f.read()

        if file_ext == ".json":
            return cls._parse_json_protocol(content, file_path)
        elif file_ext in (".txt", ".protocol"):
            return cls._parse_text_protocol(content, file_path)
        else:
            # Try to detect format
            content_stripped = content.strip()
            if content_stripped.startswith("{"):
                return cls._parse_json_protocol(content, file_path)
            else:
                return cls._parse_text_protocol(content, file_path)

    @classmethod
    def _parse_json_protocol(cls, content: str, file_path: str) -> List[ProtocolStep]:
        """Parse JSON format protocol file.

        Args:
            content: JSON string content
            file_path: Original file path (for error messages)

        Returns:
            List of protocol steps
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ProtocolValidationError(
                f"Invalid JSON in protocol file {file_path}: {e}"
            )

        # Validate top-level structure
        if not isinstance(data, dict):
            raise ProtocolValidationError(
                f"Protocol file {file_path} must contain a JSON object"
            )

        if "steps" not in data:
            raise ProtocolValidationError(
                f"Protocol file {file_path} must contain a 'steps' array"
            )

        if not isinstance(data["steps"], list):
            raise ProtocolValidationError(
                f"'steps' in {file_path} must be an array"
            )

        if len(data["steps"]) == 0:
            raise ProtocolValidationError(
                f"Protocol file {file_path} must contain at least one step"
            )

        # Parse each step
        steps = []
        for i, step_data in enumerate(data["steps"]):
            try:
                step = cls._parse_json_step(step_data, i)
                if step is not None:  # Filter out comments
                    steps.append(step)
            except ProtocolValidationError as e:
                raise ProtocolValidationError(
                    f"Error in step {i + 1} of {file_path}: {e}"
                )

        LOGGER.info(f"Loaded protocol from {file_path}: {len(steps)} executable steps")
        return steps

    @classmethod
    def _parse_json_step(cls, step_data: Dict[str, Any], step_index: int) -> ProtocolStep:
        """Parse a single step from JSON format.

        Args:
            step_data: Dictionary containing step configuration
            step_index: Index for error messages

        Returns:
            Protocol step object
        """
        if not isinstance(step_data, dict):
            raise ProtocolValidationError(
                f"Step must be an object, got {type(step_data).__name__}"
            )

        if "type" not in step_data:
            raise ProtocolValidationError("Step must have a 'type' field")

        step_type = step_data["type"].lower()

        # Normalize step type aliases
        type_aliases = {
            "tors_relax": "torsional_relax",
            "min": "minimize",
            "select": "select_best",
            "scale_scoreterm": "scale",
            "set_options": "set",
            "keep_hbonds": "keep_interactions",
            "rosetta_repack_only": "repack",
            "mpnn_group": "mpnn_multi",
        }
        step_type = type_aliases.get(step_type, step_type)

        # Validate step type
        if step_type not in cls.VALID_STEP_TYPES and step_type not in type_aliases.values():
            raise ProtocolValidationError(
                f"Unknown step type '{step_type}'. Valid types: {', '.join(sorted(cls.VALID_STEP_TYPES))}"
            )

        # All design step types use the "mpnn" schema for validation
        design_step_types = (
            "design_core", "design_core_shell", "design_shell_only",
            "design_shell_flex", "design_flex_only", "design_core_shell_flex",
            "design_distant_only", "design_shell_flex_distant", "design_global",
            "mpnn_primary", "mpnn_secondary", "mpnn_2nd_shell", "mpnn_all",
        )

        # Multi-strategy MPNN step (pool multiple strategies)
        if step_type == "mpnn_multi" or (step_type in design_step_types or step_type == "mpnn") and "strategies" in step_data:
            return cls._create_mpnn_multi_step_from_json(step_data, step_index)

        # Get schema for validation
        schema_type = step_type
        if step_type in design_step_types:
            schema_type = "mpnn"

        schema = cls.STEP_SCHEMAS.get(schema_type)
        if schema:
            cls._validate_step_params(step_data, schema, step_type)

        # Create step object based on type
        # New design step types (preferred)
        if step_type == "design_core":
            return cls._create_design_step_from_json(step_data, "core")
        elif step_type == "design_core_shell":
            return cls._create_design_step_from_json(step_data, "core_shell")
        elif step_type == "design_shell_only":
            return cls._create_design_step_from_json(step_data, "shell_only")
        elif step_type == "design_shell_flex":
            return cls._create_design_step_from_json(step_data, "shell_flex")
        elif step_type == "design_flex_only":
            return cls._create_design_step_from_json(step_data, "flex_only")
        elif step_type == "design_core_shell_flex":
            return cls._create_design_step_from_json(step_data, "core_shell_flex")
        elif step_type == "design_distant_only":
            return cls._create_design_step_from_json(step_data, "distant_only")
        elif step_type == "design_shell_flex_distant":
            return cls._create_design_step_from_json(step_data, "shell_flex_distant")
        elif step_type == "design_global":
            return cls._create_design_step_from_json(step_data, "global")
        # Legacy MPNN step types
        elif step_type in ("mpnn", "mpnn_primary", "mpnn_secondary", "mpnn_2nd_shell", "mpnn_all"):
            return cls._create_mpnn_step_from_json(step_data, step_type)
        elif step_type == "cart_relax":
            return cls._create_cart_relax_step_from_json(step_data)
        elif step_type == "torsional_relax":
            return cls._create_torsional_relax_step_from_json(step_data)
        elif step_type == "minimize":
            return cls._create_minimize_step_from_json(step_data)
        elif step_type == "repack":
            return cls._create_repack_step_from_json(step_data)
        elif step_type == "select_best":
            return cls._create_select_best_step_from_json(step_data)
        elif step_type in ("scale", "scale_scoreterm"):
            return cls._create_scale_step_from_json(step_data)
        elif step_type in ("set", "set_options"):
            return cls._create_set_step_from_json(step_data)
        elif step_type in ("keep_interactions", "keep_hbonds"):
            return cls._create_keep_interactions_step_from_json(step_data)
        elif step_type == "cluster":
            return cls._create_cluster_step_from_json(step_data)
        elif step_type == "keep_cluster_best":
            return cls._create_keep_cluster_best_step_from_json(step_data)
        elif step_type == "task_operation":
            return cls._create_task_operation_step_from_json(step_data)
        elif step_type == "set_layer_cuts":
            return cls._create_set_layer_cuts_step_from_json(step_data)
        elif step_type == "time_check":
            return cls._create_time_check_step_from_json(step_data)
        elif step_type == "final_diversify":
            return cls._create_final_diversify_step_from_json(step_data)
        elif step_type == "comment":
            # Comments are ignored during execution, return None to be filtered out
            return None
        else:
            raise ProtocolValidationError(f"Unhandled step type: {step_type}")

    @classmethod
    def _validate_step_params(
        cls,
        step_data: Dict[str, Any],
        schema: Dict[str, Any],
        step_type: str
    ) -> None:
        """Validate step parameters against schema.

        Args:
            step_data: Step configuration dictionary
            schema: Validation schema
            step_type: Step type name for error messages
        """
        # Check required parameters
        for param in schema.get("required", []):
            if param not in step_data:
                raise ProtocolValidationError(
                    f"Missing required parameter '{param}' for {step_type}"
                )

        # Check parameter types
        optional = schema.get("optional", {})
        for param, value in step_data.items():
            if param == "type":
                continue

            # Allow "comment" field for all step types (documentation only)
            if param == "comment":
                continue

            if param not in optional and param not in schema.get("required", []):
                LOGGER.warning(f"Unknown parameter '{param}' for {step_type}")
                continue

            expected_type = optional.get(param)
            if expected_type:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        raise ProtocolValidationError(
                            f"Parameter '{param}' must be one of {expected_type}, got {type(value).__name__}"
                        )
                elif not isinstance(value, expected_type):
                    raise ProtocolValidationError(
                        f"Parameter '{param}' must be {expected_type.__name__}, got {type(value).__name__}"
                    )

    @classmethod
    def _create_mpnn_step_from_json(cls, step_data: Dict[str, Any], step_type: str) -> MPNNStep:
        """Create MPNN step from JSON data."""
        # Determine spheres based on step type
        if step_type == "mpnn_primary":
            default_spheres = ["primary"]
            actual_step_type = StepType.MPNN_PRIMARY
        elif step_type == "mpnn_secondary":
            default_spheres = ["primary", "secondary"]
            actual_step_type = StepType.MPNN_SECONDARY
        elif step_type == "mpnn_2nd_shell":
            default_spheres = ["secondary"]
            actual_step_type = StepType.MPNN_2ND_SHELL
        elif step_type == "mpnn_all":
            default_spheres = ["global"]
            actual_step_type = StepType.MPNN_ALL
        else:
            default_spheres = ["primary"]
            actual_step_type = StepType.MPNN

        spheres = step_data.get("spheres", default_spheres)
        if isinstance(spheres, list):
            spheres = [str(s).strip().lower() for s in spheres if str(s).strip()]

        # Validate spheres
        valid_spheres = {
            "primary", "secondary",
            "repack", "repack_primary", "repack_secondary",
            "distant", "all", "global",
        }
        for s in spheres:
            if s not in valid_spheres:
                raise ProtocolValidationError(
                    f"Invalid sphere '{s}'. Valid spheres: {', '.join(valid_spheres)}"
                )

        temperature = float(step_data["temperature"]) if "temperature" in step_data else None
        num_designs = int(step_data["num_designs"]) if "num_designs" in step_data else None
        batch_size = int(step_data["batch_size"]) if "batch_size" in step_data else None
        num_designs_explicit = "num_designs" in step_data

        return MPNNStep(
            step_type=actual_step_type,
            temperature=temperature,
            num_designs=num_designs,
            batch_size=batch_size,
            design_spheres=spheres,
            use_sc_context=step_data.get("use_sc_context", True),
            pack_side_chains=step_data.get("pack_side_chains", True),
            sc_denoising_steps=int(step_data.get("sc_denoising_steps", 3)),
            omit_aa=step_data.get("omit_aa"),
            enhance=step_data.get("enhance"),
            repack_everything=step_data.get("repack_everything"),
            num_designs_explicit=num_designs_explicit,
        )

    @classmethod
    def _create_mpnn_multi_step_from_json(
        cls, step_data: Dict[str, Any], step_index: int
    ) -> MPNNMultiStep:
        """Create a multi-strategy MPNN step from JSON data."""
        strategies = step_data.get("strategies")
        if not isinstance(strategies, list) or not strategies:
            raise ProtocolValidationError("mpnn_multi requires a non-empty 'strategies' list")

        defaults = step_data.get("defaults", {})
        if defaults and not isinstance(defaults, dict):
            raise ProtocolValidationError("mpnn_multi 'defaults' must be a dict if provided")

        parsed_strategies: List[MPNNStep] = []
        for i, strategy in enumerate(strategies, 1):
            if not isinstance(strategy, dict):
                raise ProtocolValidationError(f"mpnn_multi strategy {i} must be an object")
            merged = dict(defaults)
            merged.update(strategy)
            merged.pop("strategies", None)
            if "type" not in merged:
                merged["type"] = "mpnn"
            try:
                parsed = cls._parse_json_step(merged, step_index)
            except ProtocolValidationError as e:
                raise ProtocolValidationError(f"mpnn_multi strategy {i}: {e}")
            if not isinstance(parsed, MPNNStep):
                raise ProtocolValidationError(
                    f"mpnn_multi strategy {i} must be an MPNN design step, got {type(parsed).__name__}"
                )
            parsed_strategies.append(parsed)

        return MPNNMultiStep(
            strategies=parsed_strategies,
            dedupe_pool=bool(step_data.get("dedupe_pool", True)),
            parallel=bool(step_data.get("parallel", False)),
            max_workers=step_data.get("max_workers"),
            min_workers=step_data.get("min_workers"),
            use_mpnn_server=step_data.get("use_mpnn_server"),
            comment=step_data.get("comment"),
        )

    @classmethod
    def _create_design_step_from_json(cls, step_data: Dict[str, Any], scope: str) -> MPNNStep:
        """Create MPNN design step from JSON data with explicit design scope.

        Args:
            step_data: JSON step configuration
            scope: Design scope (core, core_shell, shell_only, global)

        Returns:
            MPNNStep configured for the specified scope
        """
        # Map scope to step type and spheres
        scope_config = {
            "core": (StepType.DESIGN_CORE, ["core"]),
            "core_shell": (StepType.DESIGN_CORE_SHELL, ["core", "shell"]),
            "shell_only": (StepType.DESIGN_SHELL_ONLY, ["shell"]),
            "shell_flex": (StepType.DESIGN_SHELL_FLEX, ["shell", "flex"]),
            "flex_only": (StepType.DESIGN_FLEX_ONLY, ["flex"]),
            "core_shell_flex": (StepType.DESIGN_CORE_SHELL_FLEX, ["core", "shell", "flex"]),
            "distant_only": (StepType.DESIGN_DISTANT_ONLY, ["distant"]),
            "shell_flex_distant": (StepType.DESIGN_SHELL_FLEX_DISTANT, ["shell", "flex", "distant"]),
            "global": (StepType.DESIGN_GLOBAL, ["global"]),
        }

        if scope not in scope_config:
            raise ProtocolValidationError(f"Invalid design scope: {scope}")

        step_type, spheres = scope_config[scope]

        temperature = float(step_data["temperature"]) if "temperature" in step_data else None
        num_designs = int(step_data["num_designs"]) if "num_designs" in step_data else None
        batch_size = int(step_data["batch_size"]) if "batch_size" in step_data else None
        num_designs_explicit = "num_designs" in step_data

        return MPNNStep(
            step_type=step_type,
            temperature=temperature,
            num_designs=num_designs,
            batch_size=batch_size,
            design_scope=scope,
            design_spheres=spheres,
            use_sc_context=step_data.get("use_sc_context", True),
            pack_side_chains=step_data.get("pack_side_chains", True),
            sc_denoising_steps=int(step_data.get("sc_denoising_steps", 3)),
            omit_aa=step_data.get("omit_aa"),
            enhance=step_data.get("enhance"),
            repack_everything=step_data.get("repack_everything"),
            num_designs_explicit=num_designs_explicit,
        )

    @classmethod
    def _create_set_layer_cuts_step_from_json(cls, step_data: Dict[str, Any]) -> SetLayerCutsStep:
        """Create SetLayerCuts step from JSON data."""
        return SetLayerCutsStep(
            core_cutoff=float(step_data["core_cutoff"]) if "core_cutoff" in step_data else None,
            shell_cutoff=float(step_data["shell_cutoff"]) if "shell_cutoff" in step_data else None,
            flex_cutoff=float(step_data["flex_cutoff"]) if "flex_cutoff" in step_data else None,
        )

    @classmethod
    def _create_cart_relax_step_from_json(cls, step_data: Dict[str, Any]) -> CartRelaxStep:
        """Create CartRelax step from JSON data."""
        fa_rep_weight = step_data.get("fa_rep_weight")
        if fa_rep_weight is not None:
            fa_rep_weight = float(fa_rep_weight)

        relax_inner_cycles = step_data.get("relax_inner_cycles")
        if relax_inner_cycles is not None:
            relax_inner_cycles = int(relax_inner_cycles)

        return CartRelaxStep(
            repeats=int(step_data.get("repeats", 2)),
            stages=int(step_data.get("stages", 3)),
            cart_bonded_weight=float(step_data.get("cart_bonded_weight", 2.0)),
            coord_cst_weight=float(step_data.get("coord_cst_weight", 750.0)),
            coord_cst_stdev=float(step_data.get("coord_cst_stdev", 0.01)),
            global_coord_cst_weight=float(step_data.get("global_coord_cst_weight", 0.0)),
            global_coord_cst_stdev=float(step_data.get("global_coord_cst_stdev", 0.5)),
            enable_bond_geometry_min=step_data.get("enable_bond_geometry_min", True),
            scorefunction=step_data.get("scorefunction", "ref2015_cart"),
            fa_rep_weight=fa_rep_weight,
            relax_rounds=int(step_data.get("relax_rounds", 5)),
            relax_inner_cycles=relax_inner_cycles,
            nstruct=int(step_data.get("nstruct", 1)),
            until_converged=step_data.get("until_converged", False),
            bond_length_tolerance=float(step_data.get("bond_length_tolerance", 0.05)),
            bond_angle_tolerance=float(step_data.get("bond_angle_tolerance", 10.0)),
        )

    @classmethod
    def _create_torsional_relax_step_from_json(cls, step_data: Dict[str, Any]) -> TorsionalRelaxStep:
        """Create TorsionalRelax step from JSON data."""
        fa_rep_weight = step_data.get("fa_rep_weight")
        if fa_rep_weight is not None:
            fa_rep_weight = float(fa_rep_weight)

        relax_inner_cycles = step_data.get("relax_inner_cycles")
        if relax_inner_cycles is not None:
            relax_inner_cycles = int(relax_inner_cycles)

        return TorsionalRelaxStep(
            repeats=int(step_data.get("repeats", 2)),
            stages=int(step_data.get("stages", 3)),
            coord_cst_weight=float(step_data.get("coord_cst_weight", 750.0)),
            coord_cst_stdev=float(step_data.get("coord_cst_stdev", 0.01)),
            global_coord_cst_weight=float(step_data.get("global_coord_cst_weight", 0.0)),
            global_coord_cst_stdev=float(step_data.get("global_coord_cst_stdev", 0.5)),
            scorefunction=step_data.get("scorefunction", "beta_jan25"),
            fa_rep_weight=fa_rep_weight,
            relax_rounds=int(step_data.get("relax_rounds", 5)),
            relax_inner_cycles=relax_inner_cycles,
            nstruct=int(step_data.get("nstruct", 1)),
        )

    @classmethod
    def _create_minimize_step_from_json(cls, step_data: Dict[str, Any]) -> MinimizeStep:
        """Create Minimize step from JSON data."""
        return MinimizeStep(
            tolerance=float(step_data.get("tolerance", 0.01)),
            max_iter=int(step_data.get("max_iter", 200)),
            coord_cst_weight=float(step_data.get("coord_cst_weight", 750.0)),
            coord_cst_stdev=float(step_data.get("coord_cst_stdev", 0.01)),
            global_coord_cst_weight=float(step_data.get("global_coord_cst_weight", 0.0)),
            global_coord_cst_stdev=float(step_data.get("global_coord_cst_stdev", 0.5)),
            scorefunction=step_data.get("scorefunction", "beta_jan25"),
            fa_rep_weight=step_data.get("fa_rep_weight"),
            cartesian=bool(step_data.get("cartesian", False)),
            cart_bonded_weight=float(step_data.get("cart_bonded_weight", 2.0)),
            min_backbone_rmsd_cutoff=step_data.get("min_backbone_rmsd_cutoff"),
            minimize_scope=step_data.get("minimize_scope", step_data.get("scope")),
        )

    @classmethod
    def _create_repack_step_from_json(cls, step_data: Dict[str, Any]) -> RepackStep:
        """Create Repack step from JSON data."""
        repack_shell = step_data.get("repack_shell")
        if repack_shell is not None:
            repack_shell = float(repack_shell)

        # Validate repack_scope
        repack_scope = step_data.get("repack_scope", "core_shell_flex")
        valid_scopes = {"core", "core_shell", "core_shell_flex", "global"}
        if repack_scope not in valid_scopes:
            raise ProtocolValidationError(
                f"Invalid repack_scope '{repack_scope}'. Valid scopes: {', '.join(valid_scopes)}"
            )

        return RepackStep(
            repack_scope=repack_scope,
            repack_shell=repack_shell,
            coord_cst_weight=float(step_data.get("coord_cst_weight", 750.0)),
            coord_cst_stdev=float(step_data.get("coord_cst_stdev", 0.01)),
            global_coord_cst_weight=float(step_data.get("global_coord_cst_weight", 0.0)),
            global_coord_cst_stdev=float(step_data.get("global_coord_cst_stdev", 0.5)),
            scorefunction=step_data.get("scorefunction", "beta_jan25"),
            fa_rep_weight=step_data.get("fa_rep_weight"),
        )

    @classmethod
    def _create_select_best_step_from_json(cls, step_data: Dict[str, Any]) -> SelectBestStep:
        """Create SelectBest step from JSON data."""
        metric = step_data.get("metric", "geometry")
        valid_metrics = {"geometry", "score", "constraint", "sequence_diversity", "smart", "rosetta_score"}
        if metric not in valid_metrics:
            raise ProtocolValidationError(
                f"Invalid metric '{metric}'. Valid metrics: {', '.join(valid_metrics)}"
            )

        return SelectBestStep(
            n=int(step_data.get("n", 1)),
            metric=metric,
            bond_length_tolerance=step_data.get("bond_length_tolerance"),
            bond_angle_tolerance=step_data.get("bond_angle_tolerance"),
            geom_similarity_epsilon=float(step_data.get("geom_similarity_epsilon", 0.005)),
            rmsd_similarity_epsilon=float(step_data.get("rmsd_similarity_epsilon", 0.05)),
            sequence_ref=str(step_data.get("sequence_ref", "parent")),
            scorefunction=step_data.get("scorefunction"),
        )

    @classmethod
    def _create_scale_step_from_json(cls, step_data: Dict[str, Any]) -> ScaleScoreTermStep:
        terms = step_data.get("terms", {})
        scope = step_data.get("scope", "global")
        if not isinstance(terms, dict):
            raise ProtocolValidationError("scale step requires 'terms' dict")
        # Ensure float values
        parsed_terms = {}
        for k, v in terms.items():
            if isinstance(v, str) and v.strip().lower() in ("reset", "default", "none", "null"):
                parsed_terms[k] = "reset"
                continue
            try:
                parsed_terms[k] = float(v)
            except Exception:
                continue
        return ScaleScoreTermStep(terms=parsed_terms, scope=scope)

    @classmethod
    def _create_set_step_from_json(cls, step_data: Dict[str, Any]) -> SetOptionsStep:
        options = step_data.get("options", {})
        if not isinstance(options, dict):
            raise ProtocolValidationError("set step requires 'options' dict")
        return SetOptionsStep(options=options)

    @classmethod
    def _create_keep_interactions_step_from_json(cls, step_data: Dict[str, Any]) -> KeepInteractionsStep:
        interaction_types = step_data.get("interaction_types")
        if interaction_types is None:
            interaction_types = step_data.get("types", ["hbond"])
        return KeepInteractionsStep(
            target=step_data.get("target", "catres"),
            interaction_types=interaction_types,
            probability=float(step_data.get("probability", 0.5)),
            mutator_atoms=step_data.get("mutator_atoms", "either"),
            target_atoms=step_data.get("target_atoms", "either"),
            include_ligand_interactions=bool(step_data.get("include_ligand_interactions", False)),
            include_catres_interactions=bool(step_data.get("include_catres_interactions", True)),
            strong_interaction_types=step_data.get("strong_interaction_types"),
            hbond_accept_probability=step_data.get("hbond_accept_probability"),
        )

    @classmethod
    def _create_cluster_step_from_json(cls, step_data: Dict[str, Any]) -> ClusterStep:
        return ClusterStep(
            method=step_data.get("method", "sequence"),
            n_clusters=step_data.get("n_clusters"),
            threshold=step_data.get("threshold"),
        )

    @classmethod
    def _create_keep_cluster_best_step_from_json(cls, step_data: Dict[str, Any]) -> KeepClusterBestStep:
        return KeepClusterBestStep(
            n=int(step_data.get("n", 1)),
            metric=step_data.get("metric", "geometry"),
            scorefunction=step_data.get("scorefunction"),
        )

    @classmethod
    def _create_task_operation_step_from_json(cls, step_data: Dict[str, Any]) -> TaskOperationStep:
        module = step_data.get("module", "")
        if not module:
            raise ProtocolValidationError("task_operation step requires 'module'")
        return TaskOperationStep(
            module=module,
            function=step_data.get("function", "compute"),
            args=step_data.get("args", {}) or {},
        )

    @classmethod
    def _create_time_check_step_from_json(cls, step_data: Dict[str, Any]) -> TimeCheckStep:
        then_steps_raw = step_data.get("then", [])
        if not isinstance(then_steps_raw, list):
            raise ProtocolValidationError("time_check requires a 'then' list of steps")

        then_steps: List[ProtocolStep] = []
        for sub in then_steps_raw:
            if not isinstance(sub, dict):
                raise ProtocolValidationError("time_check 'then' entries must be step objects")
            step = cls._parse_json_step(sub, -1)
            if step is not None:
                then_steps.append(step)

        return TimeCheckStep(
            max_elapsed=step_data.get("max_elapsed"),
            min_remaining=step_data.get("min_remaining"),
            max_runtime_fraction=step_data.get("max_runtime_fraction"),
            then_steps=then_steps,
            mode=step_data.get("mode", "replace_remaining"),
            target_total_designs=step_data.get("target_total_designs"),
        )

    @classmethod
    def _create_final_diversify_step_from_json(cls, step_data: Dict[str, Any]) -> FinalDiversifyStep:
        """Create FinalDiversifyStep from JSON dict."""
        temps = step_data.get("temperatures", [0.1, 0.2, 0.3])
        if isinstance(temps, str):
            temps = [float(t.strip()) for t in temps.split(",")]

        scope = step_data.get("design_scope", "shell_only")
        scope_to_spheres = {
            "core": ["core"],
            "core_shell": ["core", "shell"],
            "shell_only": ["shell"],
            "shell_flex": ["shell", "flex"],
            "flex_only": ["flex"],
            "core_shell_flex": ["core", "shell", "flex"],
            "distant_only": ["distant"],
            "shell_flex_distant": ["shell", "flex", "distant"],
            "global": ["global"],  # Special: all non-fixed residues except catres
        }
        spheres = scope_to_spheres.get(scope, ["shell"])

        return FinalDiversifyStep(
            temperatures=[float(t) for t in temps],
            target_count=step_data.get("target_count"),
            designs_per_temp=int(step_data.get("designs_per_temp", 3)),
            max_iterations=int(step_data.get("max_iterations", 3)),
            design_scope=scope,
            design_spheres=spheres,
            use_sc_context=bool(step_data.get("use_sc_context", True)),
            pack_side_chains=bool(step_data.get("pack_side_chains", True)),
            sc_denoising_steps=int(step_data.get("sc_denoising_steps", 3)),
            omit_aa=step_data.get("omit_aa", "CM"),
            enhance=step_data.get("enhance", "plddt_3_20240930-f9c9ea0f"),
            batch_size=int(step_data.get("batch_size", 1)),
            overshoot_threshold=float(step_data.get("overshoot_threshold", 0.75)),
            fallback_include_flex=bool(step_data.get("fallback_include_flex", True)),
        )

    @classmethod
    def _parse_text_protocol(cls, content: str, file_path: str) -> List[ProtocolStep]:
        """Parse text format protocol file.

        Text format uses the same syntax as custom protocol strings, with one step
        per line. Also supports legacy-style whitespace-separated commands.

        Args:
            content: Text file content
            file_path: Original file path (for error messages)

        Returns:
            List of protocol steps
        """
        lines = []
        for line in content.strip().split("\n"):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            lines.append(line)

        if not lines:
            raise ProtocolValidationError(
                f"Protocol file {file_path} contains no valid steps"
            )

        steps: List[ProtocolStep] = []
        for line in lines:
            # Inline protocol lines with -> are handled by ProtocolParser
            if "->" in line:
                steps.extend(ProtocolParser.parse(line))
                continue

            # Try native parser first
            step = ProtocolParser._parse_step(line)
            if step is None:
                step = cls._parse_legacy_line(line)
            if step:
                steps.append(step)

        if not steps:
            raise ProtocolValidationError(
                f"Failed to parse any steps from {file_path}"
            )

        LOGGER.info(f"Loaded protocol from {file_path}: {len(steps)} steps")
        return steps

    @classmethod
    def _parse_legacy_line(cls, line: str) -> Optional[ProtocolStep]:
        """Parse legacy whitespace-separated protocol lines."""
        tokens = [t for t in line.strip().split() if t]
        if not tokens:
            return None
        cmd = tokens[0].lower()
        args = tokens[1:]

        # scale:term weight
        if cmd.startswith("scale:"):
            term = cmd.split(":", 1)[1]
            params = [term] + args
            return ProtocolParser._parse_scale_step(params)
        if cmd == "scale":
            return ProtocolParser._parse_scale_step(args)

        # mpnn temperature N
        if cmd.startswith("mpnn"):
            params: List[str] = []
            if len(args) >= 1 and re.match(r"^[0-9.]+$", args[0]):
                params.append(f"T{args[0]}")
            if len(args) >= 2 and args[1].isdigit():
                params.append(f"N{args[1]}")
            if len(args) > 2:
                params.extend(args[2:])
            spheres = ["primary"]
            if cmd == "mpnn_secondary":
                spheres = ["primary", "secondary"]
            elif cmd == "mpnn_2nd_shell":
                spheres = ["secondary"]
            elif cmd == "mpnn_all":
                spheres = ["global"]
            return ProtocolParser._parse_mpnn_step(params, spheres=spheres)

        # minimize
        if cmd in ("min", "minimize"):
            params: List[str] = []
            if args:
                if re.match(r"^[0-9.]+$", args[0]):
                    params.append(f"T{args[0]}")
                    params.extend(args[1:])
                else:
                    params.extend(args)
            return ProtocolParser._parse_minimize_step(params)

        # repack
        if cmd in ("repack", "rosetta_repack_only"):
            return ProtocolParser._parse_repack_step(args)

        # select_best / keep_best
        if cmd in ("select_best", "select", "keep_best"):
            params: List[str] = []
            if args:
                if args[0].isdigit():
                    params.append(f"N{args[0]}")
                    params.extend(args[1:])
                else:
                    params.extend(args)
            return ProtocolParser._parse_select_best_step(params)

        # cluster
        if cmd == "cluster":
            return ProtocolParser._parse_cluster_step(args)
        if cmd == "keep_cluster_best":
            return ProtocolParser._parse_keep_cluster_best_step(args)

        # keep_interactions
        if cmd in ("keep_interactions", "keep_hbonds"):
            return ProtocolParser._parse_keep_interactions_step(
                args, default_types=["hbond"] if cmd == "keep_hbonds" else None
            )

        # task_operation
        if cmd in ("task_operation", "taskop"):
            return ProtocolParser._parse_task_operation_step(args)

        # cart/torsional relax
        if cmd == "cart_relax":
            return ProtocolParser._parse_cart_relax_step(args)
        if cmd in ("torsional_relax", "tors_relax"):
            return ProtocolParser._parse_torsional_relax_step(args)

        return None

    @classmethod
    def validate_file(cls, file_path: str) -> Dict[str, Any]:
        """Validate a protocol file without fully parsing it.

        Args:
            file_path: Path to protocol file

        Returns:
            Dictionary with validation results:
                - valid: bool
                - format: 'json' or 'text'
                - num_steps: int (if valid)
                - errors: list of error messages (if invalid)
                - warnings: list of warning messages
        """
        result = {
            "valid": False,
            "format": None,
            "num_steps": 0,
            "errors": [],
            "warnings": [],
        }

        if not os.path.exists(file_path):
            result["errors"].append(f"File not found: {file_path}")
            return result

        file_ext = os.path.splitext(file_path)[1].lower()
        result["format"] = "json" if file_ext == ".json" else "text"

        try:
            steps = cls.load_from_file(file_path)
            result["valid"] = True
            result["num_steps"] = len(steps)
        except ProtocolValidationError as e:
            result["errors"].append(str(e))
        except Exception as e:
            result["errors"].append(f"Unexpected error: {e}")

        return result

    @classmethod
    def get_example_json(cls) -> str:
        """Return example JSON protocol for documentation."""
        example = {
            "steps": [
                {"type": "mpnn_multi", "strategies": [
                    {"type": "design_core_shell", "temperature": 0.1, "num_designs": 1},
                    {"type": "design_shell_flex", "temperature": 0.2, "num_designs": 1},
                ]},
                {"type": "cart_relax", "repeats": 2, "stages": 3},
                {"type": "mpnn", "temperature": 0.1, "num_designs": 4},
                {"type": "torsional_relax", "repeats": 1, "stages": 3}
            ]
        }
        return json.dumps(example, indent=2)

    @classmethod
    def get_example_text(cls) -> str:
        """Return example text protocol for documentation."""
        return """# Example protocol file
# Lines starting with # are comments

mpnn:T0.1:N1:spheres=primary
cart_relax:R2S3
mpnn:T0.1:N4
torsional_relax:R1S3"""
