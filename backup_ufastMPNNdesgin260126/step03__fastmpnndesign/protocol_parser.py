"""Protocol definition and parsing for FastMPNN design.

This module provides:
- StepType enum for different protocol step types
- Step dataclasses for configuring each step type
- ProtocolParser for parsing preset names and custom protocol strings
- ProtocolFileParser for loading protocols from JSON or text files
- PRESETS dictionary with predefined protocols

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
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

LOGGER = logging.getLogger(__name__)


class StepType(Enum):
    """Types of protocol steps."""
    # MPNN design steps
    MPNN = "mpnn"                           # Standard MPNN design
    MPNN_PRIMARY = "mpnn_primary"           # Design primary sphere only
    MPNN_SECONDARY = "mpnn_secondary"       # Design primary + secondary
    MPNN_2ND_SHELL = "mpnn_2nd_shell"       # Fix inner, design outer (H-bond keepers)
    MPNN_ALL = "mpnn_all"                   # Design all spheres including repack

    # Rosetta relaxation steps (time-consuming)
    CART_RELAX = "cart_relax"               # Cartesian FastRelax (for bond geometry)

    # Rosetta relaxation steps (fast)
    TORSIONAL_RELAX = "torsional_relax"     # Torsional FastRelax
    MINIMIZE = "minimize"                    # Minimization only
    REPACK = "repack"                        # Sidechain repacking

    # Selection/filtering steps
    SELECT_BEST = "select_best"             # Select top N structures


@dataclass
class MPNNStep:
    """Configuration for an MPNN design step."""
    step_type: StepType = StepType.MPNN
    temperature: float = 0.1
    num_designs: int = 8
    batch_size: int = 1
    design_spheres: List[str] = field(default_factory=lambda: ["primary"])
    use_sc_context: bool = True
    pack_side_chains: bool = True
    sc_denoising_steps: int = 3

    @classmethod
    def primary(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Create MPNN step for primary sphere only."""
        return cls(
            step_type=StepType.MPNN_PRIMARY,
            temperature=temperature,
            num_designs=num_designs,
            design_spheres=["primary"],
            **kwargs
        )

    @classmethod
    def secondary(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Create MPNN step for primary + secondary spheres."""
        return cls(
            step_type=StepType.MPNN_SECONDARY,
            temperature=temperature,
            num_designs=num_designs,
            design_spheres=["primary", "secondary"],
            **kwargs
        )

    @classmethod
    def second_shell(cls, temperature: float = 0.1, num_designs: int = 4, **kwargs) -> "MPNNStep":
        """Create MPNN step for 2nd shell (fix inner, design outer)."""
        return cls(
            step_type=StepType.MPNN_2ND_SHELL,
            temperature=temperature,
            num_designs=num_designs,
            design_spheres=["secondary"],
            **kwargs
        )

    @classmethod
    def all_spheres(cls, temperature: float = 0.1, num_designs: int = 8, **kwargs) -> "MPNNStep":
        """Create MPNN step for all spheres including repack region."""
        return cls(
            step_type=StepType.MPNN_ALL,
            temperature=temperature,
            num_designs=num_designs,
            design_spheres=["primary", "secondary", "repack"],
            **kwargs
        )


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


@dataclass
class MinimizeStep:
    """Configuration for minimization step."""
    step_type: StepType = StepType.MINIMIZE
    tolerance: float = 0.01
    max_iter: int = 200
    coord_cst_weight: float = 750.0
    scorefunction: str = "beta_jan25"


@dataclass
class RepackStep:
    """Configuration for repacking step."""
    step_type: StepType = StepType.REPACK
    repack_shell: Optional[float] = None  # Angstroms around design region
    coord_cst_weight: float = 750.0
    scorefunction: str = "beta_jan25"


@dataclass
class SelectBestStep:
    """Configuration for structure selection step."""
    step_type: StepType = StepType.SELECT_BEST
    n: int = 1
    metric: str = "geometry"  # "geometry", "score", "constraint", "sequence_diversity"


# Type alias for any protocol step
ProtocolStep = Union[MPNNStep, CartRelaxStep, TorsionalRelaxStep, MinimizeStep, RepackStep, SelectBestStep]


# =============================================================================
# Preset Protocols
# =============================================================================

PRESETS = {
    "fast": [
        # Phase 1: Light geometry fix
        MPNNStep.primary(temperature=0.1, num_designs=2),
        CartRelaxStep(repeats=1, stages=2),
        # Phase 2: Design + torsional optimize
        MPNNStep.primary(temperature=0.1, num_designs=4),
        TorsionalRelaxStep(repeats=1, stages=3),
    ],

    "balanced": [
        # Phase 1: Geometry optimization (cart relax until converged)
        MPNNStep.primary(temperature=0.2, num_designs=2),
        CartRelaxStep(repeats=2, stages=3, until_converged=True),
        # Select best 1 structure for further optimization
        SelectBestStep(n=1, metric="geometry"),
        # Phase 2: Iterative design
        MPNNStep.primary(temperature=0.1, num_designs=8),
        TorsionalRelaxStep(repeats=2, stages=3),
        MPNNStep.primary(temperature=0.1, num_designs=4),
        TorsionalRelaxStep(repeats=1, stages=3),
    ],

    "thorough": [
        # Phase 1: Extensive geometry optimization
        MPNNStep.primary(temperature=0.3, num_designs=2),
        CartRelaxStep(repeats=3, stages=4, until_converged=True),
        SelectBestStep(n=1, metric="geometry"),
        # Phase 2: Multi-round design
        MPNNStep.primary(temperature=0.2, num_designs=8),
        TorsionalRelaxStep(repeats=2, stages=3),
        MPNNStep.primary(temperature=0.1, num_designs=8),
        TorsionalRelaxStep(repeats=2, stages=3),
        # Phase 3: Secondary sphere
        MPNNStep.second_shell(temperature=0.1, num_designs=4),
        TorsionalRelaxStep(repeats=1, stages=3),
    ],

    "aggressive": [
        # More designs, less strict optimization
        MPNNStep.secondary(temperature=0.3, num_designs=8),
        CartRelaxStep(repeats=1, stages=2),
        MPNNStep.secondary(temperature=0.2, num_designs=8),
        TorsionalRelaxStep(repeats=1, stages=2),
        MPNNStep.primary(temperature=0.1, num_designs=8),
        TorsionalRelaxStep(repeats=1, stages=2),
    ],

    "geometry_only": [
        # Focus on geometry optimization (no sequence design)
        CartRelaxStep(repeats=3, stages=5, until_converged=True),
    ],

    "design_only": [
        # Single round design (fast testing)
        MPNNStep.primary(temperature=0.1, num_designs=4),
        TorsionalRelaxStep(repeats=1, stages=3),
    ],

    # =========================================================================
    # BREADTH: High diversity exploration mode
    # =========================================================================
    # Goal: Explore diverse sequence and structure space
    # Strategy: Higher MPNN temperatures (0.3 -> 0.25 -> 0.2) with multiple
    #   rounds and increasing design counts. Light relaxation between rounds
    #   to maintain diversity and explore broad sequence/structure space.
    # Use when: You want to sample widely different sequences and discover
    #   novel solutions far from the input structure.
    # Total designs: 4 + 8 + 12 = 24 structures
    "breadth": [
        # Phase 1: High diversity sampling
        MPNNStep.primary(temperature=0.3, num_designs=4),
        TorsionalRelaxStep(repeats=1, stages=2),
        # Phase 2: Medium diversity expansion
        MPNNStep.primary(temperature=0.25, num_designs=8),
        TorsionalRelaxStep(repeats=1, stages=2),
        # Phase 3: Broader scope with more designs
        MPNNStep.secondary(temperature=0.2, num_designs=12),
        TorsionalRelaxStep(repeats=1, stages=2),
    ],

    # =========================================================================
    # DEPTH: Conservative refinement mode
    # =========================================================================
    # Goal: Polish sequences close to input structure
    # Strategy: Very low MPNN temperatures (0.05 -> 0.05 -> 0.1) with
    #   multiple rounds of the same design count. Thorough relaxation with
    #   more repeats/stages for careful optimization. Prioritizes quality
    #   over diversity.
    # Use when: You have a good starting structure and want to make small,
    #   conservative improvements while staying close to the original.
    # Total designs: 4 + 4 + 4 = 12 structures
    "depth": [
        # Phase 1: Low diversity, thorough geometry
        MPNNStep.primary(temperature=0.05, num_designs=4),
        CartRelaxStep(repeats=2, stages=3),
        # Phase 2: Repeat low diversity with more relaxation
        MPNNStep.primary(temperature=0.05, num_designs=4),
        TorsionalRelaxStep(repeats=3, stages=3),
        # Phase 3: Very conservative final round
        MPNNStep.primary(temperature=0.1, num_designs=4),
        TorsionalRelaxStep(repeats=2, stages=3),
    ],

    # =========================================================================
    # ITERATIVE_REFINE: Multi-stage progressive refinement
    # =========================================================================
    # Goal: Progressive refinement from moderate to low temperature
    # Strategy: Start with moderate diversity (T=0.2), progressively reduce
    #   temperature (0.2 -> 0.15 -> 0.1 -> 0.1) while increasing design count
    #   and relaxation thoroughness. Ends with selection of best 5 by score.
    # Use when: You want a systematic approach that balances exploration and
    #   exploitation, gradually converging to high-quality designs.
    # Total designs: 1 + 1 + 10 + 10 = 22 structures, then select best 5
    "iterative_refine": [
        # Phase 1: Moderate temperature
        MPNNStep.primary(temperature=0.2, num_designs=1),
        TorsionalRelaxStep(repeats=1, stages=2),
        # Phase 2: Lower temperature
        MPNNStep.primary(temperature=0.15, num_designs=1),
        TorsionalRelaxStep(repeats=1, stages=2),
        # Phase 3: Low temperature with more designs
        MPNNStep.primary(temperature=0.1, num_designs=10),
        TorsionalRelaxStep(repeats=2, stages=3),
        # Phase 4: Final low temperature round
        MPNNStep.primary(temperature=0.1, num_designs=10),
        TorsionalRelaxStep(repeats=1, stages=3),
        # Phase 5: Select best 5
        SelectBestStep(n=5, metric="score"),
    ],

    # =========================================================================
    # PROGRESSIVE: Progressive design scope expansion
    # =========================================================================
    # Goal: Start narrow, progressively expand design scope outward
    # Strategy: Begin with primary sphere only at moderate temperature (T=0.2),
    #   then reduce temperature (T=0.1) and increase designs, finally expand
    #   to secondary sphere. Builds outward from core interface.
    # Use when: You want to design the critical core first, then expand to
    #   surrounding residues in a controlled manner.
    # Total designs: 2 + 5 + 10 = 17 structures
    "progressive": [
        # Phase 1: Primary sphere only, moderate temperature
        MPNNStep.primary(temperature=0.2, num_designs=2),
        TorsionalRelaxStep(repeats=1, stages=2),
        # Phase 2: Primary sphere, lower temperature
        MPNNStep.primary(temperature=0.1, num_designs=5),
        TorsionalRelaxStep(repeats=1, stages=3),
        # Phase 3: Expand to secondary sphere
        MPNNStep.secondary(temperature=0.1, num_designs=10),
        TorsionalRelaxStep(repeats=2, stages=3),
    ],

    # =========================================================================
    # GEOMETRY_FIRST: Geometry optimization before sequence design
    # =========================================================================
    # Goal: Fix geometry issues first, then design sequences on clean structure
    # Strategy: Thorough Cartesian relaxation (R3S3 until converged) to fix
    #   bond geometry and bad contacts, select best geometry, then run MPNN
    #   design with many sequences (N=20) on the optimized backbone.
    # Use when: Input structure has geometry problems (bad bond lengths/angles)
    #   that should be resolved before sequence design.
    # Total designs: 20 structures on geometry-optimized backbone
    "geometry_first": [
        # Phase 1: Extensive geometry optimization
        CartRelaxStep(repeats=3, stages=3, until_converged=True),
        SelectBestStep(n=1, metric="geometry"),
        # Phase 2: Design on optimized geometry
        MPNNStep.primary(temperature=0.1, num_designs=20),
        TorsionalRelaxStep(repeats=2, stages=3),
    ],

    # =========================================================================
    # DESIGN_SECONDARY_SHELL: Focus on outer shell design
    # =========================================================================
    # Goal: Design the secondary shell while keeping primary sphere stable
    # Strategy: Light primary sphere design first (T=0.1, N=2) to establish
    #   core, then focus on secondary shell (2nd shell mode) with moderate
    #   temperature (T=0.15, N=10) followed by refinement (T=0.1, N=5).
    # Use when: Core interface is good but you want to optimize the outer
    #   supporting shell of residues (H-bond network, packing).
    # Total designs: 2 + 10 + 5 = 17 structures
    "design_secondary_shell": [
        # Phase 1: Light primary sphere design
        MPNNStep.primary(temperature=0.1, num_designs=2),
        TorsionalRelaxStep(repeats=1, stages=2),
        # Phase 2: Secondary shell medium diversity
        MPNNStep.second_shell(temperature=0.15, num_designs=10),
        TorsionalRelaxStep(repeats=2, stages=3),
        # Phase 3: Secondary shell low diversity refinement
        MPNNStep.second_shell(temperature=0.1, num_designs=5),
        TorsionalRelaxStep(repeats=1, stages=3),
    ],
}


# =============================================================================
# Protocol Parser
# =============================================================================

class ProtocolParser:
    """Parse protocol definition strings into step objects.

    Protocol format examples:
        Preset names:
            "balanced"
            "fast"
            "thorough"

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

        # Check for preset (case-insensitive)
        preset_key = protocol_str.lower()
        if preset_key in PRESETS:
            LOGGER.info(f"Using preset protocol: {preset_key}")
            return PRESETS[preset_key]

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

        if step_type_str in ("mpnn", "mpnn_primary"):
            return cls._parse_mpnn_step(params, spheres=["primary"])
        elif step_type_str == "mpnn_secondary":
            return cls._parse_mpnn_step(params, spheres=["primary", "secondary"])
        elif step_type_str == "mpnn_2nd_shell":
            return cls._parse_mpnn_step(params, spheres=["secondary"], step_type=StepType.MPNN_2ND_SHELL)
        elif step_type_str == "mpnn_all":
            return cls._parse_mpnn_step(params, spheres=["primary", "secondary", "repack"], step_type=StepType.MPNN_ALL)
        elif step_type_str == "cart_relax":
            return cls._parse_cart_relax_step(params)
        elif step_type_str in ("torsional_relax", "tors_relax"):
            return cls._parse_torsional_relax_step(params)
        elif step_type_str in ("minimize", "min"):
            return cls._parse_minimize_step(params)
        elif step_type_str == "repack":
            return cls._parse_repack_step(params)
        elif step_type_str in ("select_best", "select"):
            return cls._parse_select_best_step(params)
        else:
            LOGGER.warning(f"Unknown step type: {step_type_str}")
            return None

    @classmethod
    def _parse_mpnn_step(
        cls,
        params: List[str],
        spheres: List[str] = None,
        step_type: StepType = StepType.MPNN,
    ) -> MPNNStep:
        """Parse MPNN step parameters."""
        temperature = 0.1
        num_designs = 8
        batch_size = 1
        design_spheres = spheres or ["primary"]

        for p in params:
            p = p.strip()
            if p.startswith("T"):
                temperature = float(p[1:])
            elif p.startswith("N"):
                num_designs = int(p[1:])
            elif p.startswith("B"):
                batch_size = int(p[1:])
            elif p.startswith("spheres="):
                design_spheres = p[8:].split(",")

        return MPNNStep(
            step_type=step_type,
            temperature=temperature,
            num_designs=num_designs,
            batch_size=batch_size,
            design_spheres=design_spheres,
        )

    @classmethod
    def _parse_cart_relax_step(cls, params: List[str]) -> CartRelaxStep:
        """Parse Cartesian relax step parameters."""
        repeats = 2
        stages = 3
        until_converged = False
        cart_bonded_weight = 2.0

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

        return CartRelaxStep(
            repeats=repeats,
            stages=stages,
            until_converged=until_converged,
            cart_bonded_weight=cart_bonded_weight,
        )

    @classmethod
    def _parse_torsional_relax_step(cls, params: List[str]) -> TorsionalRelaxStep:
        """Parse torsional relax step parameters."""
        repeats = 2
        stages = 3

        for p in params:
            p = p.strip()
            match = re.match(r"R(\d+)S(\d+)", p, re.IGNORECASE)
            if match:
                repeats = int(match.group(1))
                stages = int(match.group(2))

        return TorsionalRelaxStep(repeats=repeats, stages=stages)

    @classmethod
    def _parse_minimize_step(cls, params: List[str]) -> MinimizeStep:
        """Parse minimize step parameters."""
        tolerance = 0.01
        max_iter = 200

        for p in params:
            p = p.strip()
            if p.startswith("T"):
                tolerance = float(p[1:])
            elif p.startswith("I"):
                max_iter = int(p[1:])

        return MinimizeStep(tolerance=tolerance, max_iter=max_iter)

    @classmethod
    def _parse_repack_step(cls, params: List[str]) -> RepackStep:
        """Parse repack step parameters."""
        repack_shell = None

        for p in params:
            p = p.strip()
            if p.startswith("shell="):
                repack_shell = float(p[6:])

        return RepackStep(repack_shell=repack_shell)

    @classmethod
    def _parse_select_best_step(cls, params: List[str]) -> SelectBestStep:
        """Parse select best step parameters."""
        n = 1
        metric = "geometry"

        for p in params:
            p = p.strip()
            if p.startswith("N"):
                n = int(p[1:])
            elif p.startswith("metric="):
                metric = p[7:]

        return SelectBestStep(n=n, metric=metric)

    @classmethod
    def get_presets(cls) -> List[str]:
        """Get list of available preset names."""
        return list(PRESETS.keys())

    @classmethod
    def describe_protocol(cls, steps: List[ProtocolStep]) -> str:
        """Generate human-readable description of a protocol."""
        lines = []
        for i, step in enumerate(steps, 1):
            if isinstance(step, MPNNStep):
                spheres = ",".join(step.design_spheres)
                lines.append(f"{i}. MPNN design (T={step.temperature}, N={step.num_designs}, spheres={spheres})")
            elif isinstance(step, CartRelaxStep):
                conv = " until converged" if step.until_converged else ""
                lines.append(f"{i}. Cartesian relax (R{step.repeats}S{step.stages}{conv})")
            elif isinstance(step, TorsionalRelaxStep):
                lines.append(f"{i}. Torsional relax (R{step.repeats}S{step.stages})")
            elif isinstance(step, MinimizeStep):
                lines.append(f"{i}. Minimize (tol={step.tolerance})")
            elif isinstance(step, RepackStep):
                shell = f" shell={step.repack_shell}A" if step.repack_shell else ""
                lines.append(f"{i}. Repack{shell}")
            elif isinstance(step, SelectBestStep):
                lines.append(f"{i}. Select best {step.n} by {step.metric}")
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
        "mpnn", "mpnn_primary", "mpnn_secondary", "mpnn_2nd_shell", "mpnn_all",
        "cart_relax", "torsional_relax", "tors_relax",
        "minimize", "min", "repack", "select_best", "select"
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
            },
            "defaults": {
                "temperature": 0.1,
                "num_designs": 8,
                "batch_size": 1,
                "spheres": ["primary"],
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
            },
            "defaults": {
                "tolerance": 0.01,
                "max_iter": 200,
            }
        },
        "repack": {
            "required": [],
            "optional": {
                "repack_shell": (float, int, type(None)),
                "coord_cst_weight": (float, int),
            },
            "defaults": {}
        },
        "select_best": {
            "required": [],
            "optional": {
                "n": int,
                "metric": str,
            },
            "defaults": {
                "n": 1,
                "metric": "geometry",
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
                steps.append(step)
            except ProtocolValidationError as e:
                raise ProtocolValidationError(
                    f"Error in step {i + 1} of {file_path}: {e}"
                )

        LOGGER.info(f"Loaded protocol from {file_path}: {len(steps)} steps")
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
        }
        step_type = type_aliases.get(step_type, step_type)

        # Validate step type
        if step_type not in cls.VALID_STEP_TYPES and step_type not in type_aliases.values():
            raise ProtocolValidationError(
                f"Unknown step type '{step_type}'. Valid types: {', '.join(sorted(cls.VALID_STEP_TYPES))}"
            )

        # Get schema for validation
        schema_type = step_type
        if step_type in ("mpnn_primary", "mpnn_secondary", "mpnn_2nd_shell", "mpnn_all"):
            schema_type = "mpnn"

        schema = cls.STEP_SCHEMAS.get(schema_type)
        if schema:
            cls._validate_step_params(step_data, schema, step_type)

        # Create step object based on type
        if step_type in ("mpnn", "mpnn_primary", "mpnn_secondary", "mpnn_2nd_shell", "mpnn_all"):
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
            default_spheres = ["primary", "secondary", "repack"]
            actual_step_type = StepType.MPNN_ALL
        else:
            default_spheres = ["primary"]
            actual_step_type = StepType.MPNN

        spheres = step_data.get("spheres", default_spheres)

        # Validate spheres
        valid_spheres = {"primary", "secondary", "repack"}
        for s in spheres:
            if s not in valid_spheres:
                raise ProtocolValidationError(
                    f"Invalid sphere '{s}'. Valid spheres: {', '.join(valid_spheres)}"
                )

        return MPNNStep(
            step_type=actual_step_type,
            temperature=float(step_data.get("temperature", 0.1)),
            num_designs=int(step_data.get("num_designs", 8)),
            batch_size=int(step_data.get("batch_size", 1)),
            design_spheres=spheres,
            use_sc_context=step_data.get("use_sc_context", True),
            pack_side_chains=step_data.get("pack_side_chains", True),
            sc_denoising_steps=int(step_data.get("sc_denoising_steps", 3)),
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
        )

    @classmethod
    def _create_repack_step_from_json(cls, step_data: Dict[str, Any]) -> RepackStep:
        """Create Repack step from JSON data."""
        repack_shell = step_data.get("repack_shell")
        if repack_shell is not None:
            repack_shell = float(repack_shell)

        return RepackStep(
            repack_shell=repack_shell,
            coord_cst_weight=float(step_data.get("coord_cst_weight", 750.0)),
        )

    @classmethod
    def _create_select_best_step_from_json(cls, step_data: Dict[str, Any]) -> SelectBestStep:
        """Create SelectBest step from JSON data."""
        metric = step_data.get("metric", "geometry")
        valid_metrics = {"geometry", "score", "constraint", "sequence_diversity"}
        if metric not in valid_metrics:
            raise ProtocolValidationError(
                f"Invalid metric '{metric}'. Valid metrics: {', '.join(valid_metrics)}"
            )

        return SelectBestStep(
            n=int(step_data.get("n", 1)),
            metric=metric,
        )

    @classmethod
    def _parse_text_protocol(cls, content: str, file_path: str) -> List[ProtocolStep]:
        """Parse text format protocol file.

        Text format uses the same syntax as custom protocol strings, with one step
        per line (lines can also be separated by '->').

        Args:
            content: Text file content
            file_path: Original file path (for error messages)

        Returns:
            List of protocol steps
        """
        # Replace newlines with '->' to create standard protocol string
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

        # Join with '->' separator
        protocol_str = " -> ".join(lines)

        # Validate step types before parsing
        for i, line in enumerate(lines):
            step_type = line.split(":")[0].strip().lower()
            if step_type not in cls.VALID_STEP_TYPES:
                raise ProtocolValidationError(
                    f"Unknown step type '{step_type}' on line {i + 1} of {file_path}. "
                    f"Valid types: {', '.join(sorted(cls.VALID_STEP_TYPES))}"
                )

        # Use existing ProtocolParser
        steps = ProtocolParser.parse(protocol_str)

        if not steps:
            raise ProtocolValidationError(
                f"Failed to parse any steps from {file_path}"
            )

        LOGGER.info(f"Loaded protocol from {file_path}: {len(steps)} steps")
        return steps

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
                {"type": "mpnn", "temperature": 0.1, "num_designs": 1, "spheres": ["primary"]},
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
