#!/usr/bin/env python3
"""
Unified CLI Runner for the Enzyme Design Pipeline.

Orchestrates three sequential steps:
  1. Catalytic residue alignment (align_catres.py)
  2. Constrained Cartesian relaxation (constrained_cart_relax.py)
  3. FastMPNN design with Rosetta refinement (fastmpnn_design.py)

Usage:
    python run_pipeline.py --input_pdb input.pdb --ref_pdb ref.pdb \\
                           --params ligand.params --output_dir results/

For full options: python run_pipeline.py --help
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_constants import (
    DEFAULT_CONTAINER_RUNTIME,
    DEFAULT_UNIVERSAL_CONTAINER,
    DEFAULT_PYROSETTA_CONTAINER,
    STEP1_SCRIPT,
    STEP2_SCRIPT,
    STEP3_SCRIPT,
    STEP1_DEFAULTS,
    STEP2_DEFAULTS,
    STEP3_DEFAULTS,
    STEP3_DEFAULT_PROTOCOL_FILE,
    STEP1_ARG_MAPPING,
    STEP2_ARG_MAPPING,
    STEP3_ARG_MAPPING,
    STEP1_OUTPUT_PATTERNS,
    STEP2_OUTPUT_PATTERNS,
    STEP3_OUTPUT_PATTERNS,
    STEP2_SCOREFUNCTION_CHOICES,
    STEP3_SCOREFUNCTION_CART_CHOICES,
    STEP3_SCOREFUNCTION_TORSIONAL_CHOICES,
    HEADER_WIDTH,
    SECTION_CHAR,
    SUBSECTION_CHAR,
    get_project_root,
    get_step_script_path,
    get_container_type,
    format_duration,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
LOGGER = logging.getLogger(__name__)

def parse_bool(value: str) -> bool:
    """Parse a string into a boolean."""
    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}
    if isinstance(value, bool):
        return value
    if value is None:
        raise argparse.ArgumentTypeError("Boolean value required")
    val = str(value).strip().lower()
    if val in truthy:
        return True
    if val in falsy:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: '{value}'")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StepOutputs:
    """Container for outputs from a pipeline step."""
    step: int
    output_dir: Path
    files: Dict[str, Path] = field(default_factory=dict)
    duration: float = 0.0
    success: bool = False


class PipelineError(Exception):
    """Raised when a pipeline step fails."""
    def __init__(self, step: int, message: str, returncode: int = 1):
        self.step = step
        self.message = message
        self.returncode = returncode
        super().__init__(f"Step {step} failed: {message}")


# =============================================================================
# Pipeline Runner
# =============================================================================

class PipelineRunner:
    """Orchestrates the enzyme design pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.original_basename = Path(args.input_pdb).stem
        self.internal_basename = self._resolve_internal_basename()
        self.output_dir = Path(args.output_dir).resolve()
        self.project_root = get_project_root()
        self.work_dir: Optional[Path] = None
        self.run_id = self._make_run_id()
        self.output_tag = self._resolve_output_tag()
        self.append_output_tag = bool(getattr(args, "append_output_tag", False))
        self.timings: Dict[str, float] = {}
        self.files_created: List[Path] = []

    def _make_run_id(self) -> str:
        """Create a unique run ID for work directory isolation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nonce = uuid.uuid4().hex[:6]
        return f"{timestamp}_{os.getpid()}_{nonce}"

    def _resolve_internal_basename(self) -> str:
        """Determine internal basename for intermediate files."""
        if not getattr(self.args, "short_internal_basename", False):
            return self.original_basename
        seed = str(Path(self.args.input_pdb).resolve())
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
        return f"p{digest}"

    def _auto_output_tag(self) -> Optional[str]:
        """Derive an output tag from SLURM env vars when present."""
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        proc_id = os.environ.get("SLURM_PROCID") or os.environ.get("SLURM_LOCALID")
        if job_id and array_id:
            return f"job{job_id}_task{array_id}"
        if job_id and proc_id:
            return f"job{job_id}_proc{proc_id}"
        if job_id:
            return f"job{job_id}"
        return None

    def _resolve_output_tag(self) -> Optional[str]:
        """Resolve output tag from CLI or SLURM env."""
        tag = (self.args.output_tag or "").strip()
        if tag:
            return tag
        return self._auto_output_tag()

    def _final_basename(self) -> str:
        """Basename for final output files (optionally includes output_tag)."""
        if self.append_output_tag and self.output_tag:
            return f"{self.original_basename}_{self.output_tag}"
        return self.original_basename

    def _auto_mpnn_server_port(self) -> Optional[int]:
        """Derive a deterministic MPNN server port for SLURM array isolation."""
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        proc_id = os.environ.get("SLURM_PROCID") or os.environ.get("SLURM_LOCALID")
        if not job_id and not array_id and not proc_id:
            return None
        seed = f"{job_id or 'job'}-{array_id or proc_id or '0'}"
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        return 20000 + (int(digest, 16) % 40000)

    def _create_work_dir(self) -> Path:
        """Create unique working directory for intermediates."""
        work_dir = self.output_dir / f".pipeline_work_{self.internal_basename}_{self.run_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _get_container_path(self, container_type: str) -> str:
        """Get container path by type."""
        if container_type == "universal":
            return self.args.universal_container
        elif container_type == "pyrosetta":
            return self.args.pyrosetta_container
        else:
            raise ValueError(f"Unknown container type: {container_type}")

    def _get_nested_container_binds(self) -> List[str]:
        """Get bind mounts needed for nested container execution.

        Returns bind arguments that allow running apptainer inside the container.
        This is needed for step 3, which runs MPNN in universal.sif and
        Rosetta in pyrosetta.sif as subprocesses.
        """
        bind_args = []

        # Find apptainer binary
        runtime = self.args.container_runtime
        runtime_path = shutil.which(runtime)
        if not runtime_path:
            LOGGER.warning(f"Could not find {runtime} binary for nested container support")
            return bind_args

        # Bind the container runtime binary
        bind_args.extend(["--bind", f"{runtime_path}:{runtime_path}"])

        # Bind apptainer/singularity helper libraries
        # These are typically in /usr/libexec/apptainer or /usr/libexec/singularity
        libexec_paths = [
            f"/usr/libexec/{runtime}",
            "/usr/libexec/apptainer",
            "/usr/libexec/singularity",
        ]
        for libexec_path in libexec_paths:
            if Path(libexec_path).exists():
                bind_args.extend(["--bind", f"{libexec_path}:{libexec_path}"])
                break

        # Bind the container images themselves so they're accessible inside
        for container_path in [self.args.universal_container, self.args.pyrosetta_container]:
            resolved = str(Path(container_path).resolve())
            bind_args.extend(["--bind", f"{resolved}:{resolved}"])

        # Also bind the container versions directory if it's a symlink
        for container_path in [self.args.universal_container, self.args.pyrosetta_container]:
            p = Path(container_path)
            if p.is_symlink():
                real_path = str(p.resolve())
                # Bind the directory containing the real file
                real_dir = str(Path(real_path).parent)
                bind_args.extend(["--bind", f"{real_dir}:{real_dir}"])

        return bind_args

    def _build_container_command(
        self,
        container_type: str,
        script_path: Path,
        script_args: List[str],
        module_name: Optional[str] = None,
        enable_nested_containers: bool = False,
        extra_bind_dirs: Optional[List[Path]] = None,
    ) -> List[str]:
        """Build the container execution command.

        Args:
            container_type: Type of container ("universal" or "pyrosetta")
            script_path: Path to the script (used if module_name not provided)
            script_args: Arguments to pass to the script
            module_name: If provided, run as `python -m module_name` instead of script
            enable_nested_containers: If True, bind apptainer and container images
                                      to allow nested container execution (for step 3)
        """
        container_path = self._get_container_path(container_type)

        # Collect all directories that need to be bound
        bind_dirs = set()
        bind_dirs.add(str(self.project_root))
        bind_dirs.add(str(self.output_dir))
        if self.work_dir:
            bind_dirs.add(str(self.work_dir))
        if extra_bind_dirs:
            for bind_dir in extra_bind_dirs:
                bind_dirs.add(str(Path(bind_dir).resolve()))

        # Add input file directories
        bind_dirs.add(str(Path(self.args.input_pdb).parent.resolve()))
        bind_dirs.add(str(Path(self.args.ref_pdb).parent.resolve()))
        for params_file in self.args.params:
            bind_dirs.add(str(Path(params_file).parent.resolve()))

        # Build bind mount arguments
        bind_args = []
        for bind_dir in bind_dirs:
            bind_args.extend(["--bind", f"{bind_dir}:{bind_dir}"])

        # Add nested container support binds if requested (for step 3)
        if enable_nested_containers:
            bind_args.extend(self._get_nested_container_binds())

        # Set PYTHONPATH to project root for module imports
        # Use bash wrapper to append to existing PYTHONPATH instead of replacing it
        script_args_str = " ".join(f'"{arg}"' for arg in script_args)

        if module_name:
            # Run as module for scripts that use relative imports
            inner_cmd = f'export PYTHONPATH="${{PYTHONPATH:+$PYTHONPATH:}}{self.project_root}" && python -m {module_name} {script_args_str}'
        else:
            # Run as script directly
            inner_cmd = f'python "{script_path}" {script_args_str}'

        cmd = [self.args.container_runtime, "exec"]
        if getattr(self.args, "container_nv", False):
            cmd.append("--nv")
        cmd.extend([
            *bind_args,
            container_path,
            "/bin/bash",
            "-c",
            inner_cmd,
        ])

        return cmd

    def _run_command(self, cmd: List[str], step: int) -> subprocess.CompletedProcess:
        """Execute a command and handle output."""
        if self.args.dry_run:
            print(f"  [DRY RUN] Would execute:")
            print(f"    {' '.join(cmd)}")
            # Return a mock result for dry run
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if self.args.verbose or self.args.debug:
            LOGGER.debug(f"  Executing: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            LOGGER.error(f"  Command failed with return code {result.returncode}")
            if result.stderr:
                LOGGER.error(f"  STDERR:\n{result.stderr}")
            raise PipelineError(step, f"Command exited with code {result.returncode}", result.returncode)

        return result

    def _find_output_files(self, output_dir: Path, patterns: Dict[str, str]) -> Dict[str, Path]:
        """Find output files matching patterns in a directory."""
        found = {}
        for key, pattern in patterns.items():
            matches = list(output_dir.glob(pattern))
            if matches:
                # Sort by modification time, take most recent
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                found[key] = matches[0]
        return found

    def _mock_output_files(self, output_dir: Path, patterns: Dict[str, str]) -> Dict[str, Path]:
        """Generate mock output file paths for dry run mode."""
        mock_files = {}
        for key, pattern in patterns.items():
            # Convert glob pattern to a concrete filename
            mock_name = pattern.replace("*", self.internal_basename)
            mock_files[key] = output_dir / mock_name
        return mock_files

    def _print_header(self):
        """Print the pipeline header."""
        print(SECTION_CHAR * HEADER_WIDTH)
        print(" Enzyme Design Pipeline")
        print(SECTION_CHAR * HEADER_WIDTH)
        print(f" Input PDB:    {self.args.input_pdb}")
        print(f" Reference:    {self.args.ref_pdb}")
        print(f" Params:       {', '.join(self.args.params)}")
        print(f" Step2 preset: {self.args.step2_preset or 'none'}")
        print(f" Output:       {self.args.output_dir}")
        if self.internal_basename != self.original_basename:
            print(f" Internal ID:  {self.internal_basename}")
        if self.output_tag:
            print(f" Output Tag:   {self.output_tag}")
        if self.work_dir:
            print(f" Work dir:     {self.work_dir}")
        metrics_subdir = (self.args.metrics_subdir or "").strip()
        if self.args.no_metrics_subdir or not metrics_subdir:
            print(" Metrics dir:  output_dir (root)")
        else:
            print(f" Metrics dir:  {self.output_dir / metrics_subdir}")
        print(f" Run ID:       {self.run_id}")
        if getattr(self.args, "container_nv", False):
            print(" GPU:          Enabled (--nv)")
        if self.args.dry_run:
            print(f" Mode:         DRY RUN (no execution)")
        print(SECTION_CHAR * HEADER_WIDTH)
        print()

    def _print_step_header(self, step: int, title: str):
        """Print a step header."""
        print(f"[Step {step}/3] {title}")
        print(SUBSECTION_CHAR * HEADER_WIDTH)

    def _print_step_complete(self, step: int, duration: float, outputs: List[str]):
        """Print step completion message."""
        print(f"  Completed in {format_duration(duration)}")
        if outputs:
            print(f"  Outputs: {', '.join(outputs)}")
        print()

    def _print_summary(self, total_duration: float):
        """Print the final summary."""
        print(SECTION_CHAR * HEADER_WIDTH)
        if self.args.dry_run:
            print(" Dry Run Complete!")
        else:
            print(" Pipeline Complete!")
        print(SECTION_CHAR * HEADER_WIDTH)
        print(f" Total time:     {format_duration(total_duration)}")

        if self.args.dry_run:
            # Show what would be created
            num_designs = self.args.step3_num_final_designs
            if num_designs is None:
                num_designs = STEP3_DEFAULTS["num_final_designs"]
            if num_designs is None:
                print(" Would create:   designs based on step3 protocol target_count")
            else:
                output_bn = self._final_basename()
                print(f" Would create:   {num_designs + 1} files")
                print(f"   - {self.output_dir}/{output_bn}_design_00.pdb ... {output_bn}_design_{num_designs-1:02d}.pdb")
                print(f"   - {self.output_dir}/{output_bn}_design_metrics.json")
        else:
            print(f" Files created:  {len(self.files_created)}")
            # Show first few files
            max_show = 5
            for i, f in enumerate(self.files_created[:max_show]):
                print(f"   - {f}")
            if len(self.files_created) > max_show:
                print(f"   - ... ({len(self.files_created) - max_show} more)")

        if self.args.keep_intermediates and self.work_dir and not self.args.dry_run:
            print(f" Intermediates:  Retained at {self.work_dir}")
        elif not self.args.dry_run:
            print(f" Intermediates:  Cleaned up (use --keep_intermediates to retain)")
        if not self.args.dry_run:
            metrics_subdir = (self.args.metrics_subdir or "").strip()
            if self.args.no_metrics_subdir or not metrics_subdir:
                metrics_dir = self.output_dir
            else:
                metrics_dir = self.output_dir / metrics_subdir
            print(f" Metrics:     {metrics_dir}")
        print(SECTION_CHAR * HEADER_WIDTH)

    # -------------------------------------------------------------------------
    # Step 1: Catalytic Residue Alignment
    # -------------------------------------------------------------------------

    def _build_step1_args(self) -> List[str]:
        """Build command-line arguments for step 1."""
        step1_dir = self.work_dir / "step01"
        if not self.args.dry_run:
            step1_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "--input_pdb", str(Path(self.args.input_pdb).resolve()),
            "--ref_pdb", str(Path(self.args.ref_pdb).resolve()),
            "--outdir", str(step1_dir),
        ]

        if self.args.catres_subset:
            args.extend(["--catres_subset", self.args.catres_subset])
        if self.internal_basename != self.original_basename:
            args.extend(["--outfile_bn", self.internal_basename])

        # Map pipeline args to step args
        for pipeline_arg, step_arg in STEP1_ARG_MAPPING.items():
            value = getattr(self.args, pipeline_arg, None)
            if value:
                args.append(f"--{step_arg}")

        if self.args.verbose:
            args.append("--verbose")

        return args

    def run_step1(self) -> StepOutputs:
        """Run step 1: Catalytic Residue Alignment."""
        self._print_step_header(1, "Catalytic Residue Alignment")
        step1_dir = self.work_dir / "step01" if self.work_dir else None

        if self.args.skip_step1:
            # Use existing outputs
            step1_dir = Path(self.args.step1_output_dir).resolve()
            print(f"  Skipping step 1, using outputs from: {step1_dir}")
            outputs = StepOutputs(
                step=1,
                output_dir=step1_dir,
                files=self._find_output_files(step1_dir, STEP1_OUTPUT_PATTERNS),
                success=True,
            )
            print()
            return outputs

        start_time = time.time()
        script_path = get_step_script_path(1)
        script_args = self._build_step1_args()

        print(f"  Running: {script_path.name}")
        if step1_dir:
            print(f"  Step dir:   {step1_dir}")

        cmd = self._build_container_command("universal", script_path, script_args)
        self._run_command(cmd, step=1)

        duration = time.time() - start_time
        self.timings["step1"] = duration

        step1_dir = self.work_dir / "step01"

        # In dry run mode, mock the output files
        if self.args.dry_run:
            files = self._mock_output_files(step1_dir, STEP1_OUTPUT_PATTERNS)
        else:
            files = self._find_output_files(step1_dir, STEP1_OUTPUT_PATTERNS)

        outputs = StepOutputs(
            step=1,
            output_dir=step1_dir,
            files=files,
            duration=duration,
            success=True,
        )

        output_names = [f.name for f in outputs.files.values()]
        self._print_step_complete(1, duration, output_names)

        return outputs

    # -------------------------------------------------------------------------
    # Step 2: Constrained Cartesian Relaxation
    # -------------------------------------------------------------------------

    def _build_step2_args(self, step1_outputs: StepOutputs) -> Tuple[List[str], List[Path]]:
        """Build command-line arguments for step 2."""
        step2_dir = self.work_dir / "step02"
        if not self.args.dry_run:
            step2_dir.mkdir(parents=True, exist_ok=True)

        # Get the constraints JSON from step 1
        constraints_json = step1_outputs.files.get("constraints_json")
        if not constraints_json and not self.args.dry_run:
            raise PipelineError(2, "Step 1 constraints JSON not found")
        if not constraints_json:
            # Mock path for dry run
            constraints_json = step1_outputs.output_dir / f"{self.internal_basename}_recommended_atom_cst.json"
        extra_bind_dirs = [constraints_json.parent]

        args = [
            "--step01_json", str(constraints_json),
            "--params", *[str(Path(p).resolve()) for p in self.args.params],
            "--output_dir", str(step2_dir),
        ]

        if self.args.catres_subset:
            args.extend(["--catres_subset", self.args.catres_subset])

        # Collect step2-specific args
        step2_args = {}
        for pipeline_arg, step_arg in STEP2_ARG_MAPPING.items():
            value = getattr(self.args, pipeline_arg, None)
            if value is not None:
                step2_args[step_arg] = value
        # Hard override for bond geometry minimization
        if self.args.step2_enable_bond_geometry_min is not None:
            if self.args.step2_enable_bond_geometry_min:
                step2_args["enable_bond_geometry_min"] = True
            else:
                step2_args["disable_bond_geometry_min"] = True

        # Hard override for auto-expand behavior
        if self.args.step2_auto_expand_mobile is not None:
            if self.args.step2_auto_expand_mobile:
                step2_args["auto_expand_mobile"] = True
            else:
                step2_args["no_auto_expand_mobile"] = True

        # Hard override for catres convergence requirement
        if self.args.step2_require_catres_converged is not None:
            if self.args.step2_require_catres_converged:
                step2_args["require_catres_converged"] = True
            else:
                step2_args["no_require_catres_converged"] = True

        # Convert to CLI args
        for key, value in step2_args.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.extend([f"--{key}", str(value)])

        if self.args.debug:
            args.append("--debug")

        return args, extra_bind_dirs

    def run_step2(self, step1_outputs: StepOutputs) -> StepOutputs:
        """Run step 2: Constrained Cartesian Relaxation."""
        self._print_step_header(2, "Constrained Cartesian Relaxation")
        step2_dir = self.work_dir / "step02" if self.work_dir else None

        if self.args.skip_step2:
            # Use existing outputs
            step2_dir = Path(self.args.step2_output_dir).resolve()
            print(f"  Skipping step 2, using outputs from: {step2_dir}")
            outputs = StepOutputs(
                step=2,
                output_dir=step2_dir,
                files=self._find_output_files(step2_dir, STEP2_OUTPUT_PATTERNS),
                success=True,
            )
            print()
            return outputs

        start_time = time.time()
        script_path = get_step_script_path(2)
        script_args, extra_bind_dirs = self._build_step2_args(step1_outputs)

        print(f"  Running: {script_path.name}")
        if step2_dir:
            print(f"  Step dir:   {step2_dir}")

        cmd = self._build_container_command(
            "pyrosetta",
            script_path,
            script_args,
            extra_bind_dirs=extra_bind_dirs,
        )
        self._run_command(cmd, step=2)

        duration = time.time() - start_time
        self.timings["step2"] = duration

        step2_dir = self.work_dir / "step02"

        # In dry run mode, mock the output files
        if self.args.dry_run:
            files = self._mock_output_files(step2_dir, STEP2_OUTPUT_PATTERNS)
        else:
            files = self._find_output_files(step2_dir, STEP2_OUTPUT_PATTERNS)

        outputs = StepOutputs(
            step=2,
            output_dir=step2_dir,
            files=files,
            duration=duration,
            success=True,
        )

        output_names = [f.name for f in outputs.files.values()]
        self._print_step_complete(2, duration, output_names)

        return outputs

    # -------------------------------------------------------------------------
    # Step 3: FastMPNN Design
    # -------------------------------------------------------------------------

    def _build_step3_args(self, step2_outputs: StepOutputs) -> Tuple[List[str], List[Path]]:
        """Build command-line arguments for step 3."""
        step3_dir = self.work_dir / "step03"
        if not self.args.dry_run:
            step3_dir.mkdir(parents=True, exist_ok=True)

        # Get the metrics JSON from step 2
        metrics_json = step2_outputs.files.get("metrics_json")
        if not metrics_json and not self.args.dry_run:
            raise PipelineError(3, "Step 2 metrics JSON not found")
        if not metrics_json:
            # Mock path for dry run
            metrics_json = step2_outputs.output_dir / f"{self.internal_basename}_aligned_relaxed_metrics.json"
        extra_bind_dirs = [metrics_json.parent]

        args = [
            "--step02_json", str(metrics_json),
            "--params", *[str(Path(p).resolve()) for p in self.args.params],
            "--output_dir", str(step3_dir),
        ]

        if self.args.catres_subset:
            args.extend(["--catres_subset", self.args.catres_subset])

        # Collect step3-specific args
        step3_args = {}
        for pipeline_arg, step_arg in STEP3_ARG_MAPPING.items():
            value = getattr(self.args, pipeline_arg, None)
            if value is not None:
                step3_args[step_arg] = value

        # Propagate global keep_intermediates to step3 unless explicitly set
        if self.args.keep_intermediates and "keep_intermediates" not in step3_args:
            step3_args["keep_intermediates"] = True

        # If protocol looks like a file path, treat it as protocol_file
        if "protocol" in step3_args and step3_args["protocol"]:
            protocol_value = str(step3_args["protocol"])
            protocol_path_hint = Path(protocol_value)
            if (
                protocol_path_hint.suffix in {".json", ".txt"} or
                protocol_path_hint.is_absolute() or
                "/" in protocol_value or "\\" in protocol_value
            ):
                step3_args.pop("protocol", None)
                step3_args["protocol_file"] = protocol_value

        # Default to step03 protocol JSON unless user overrides it
        if (
            "protocol" not in step3_args and
            "protocol_file" not in step3_args
        ):
            step3_args["protocol_file"] = STEP3_DEFAULT_PROTOCOL_FILE

        # Ensure step3 uses the same PyRosetta image as the pipeline container config
        if "pyrosetta_image" not in step3_args:
            step3_args["pyrosetta_image"] = self.args.pyrosetta_container
        # Ensure MPNN uses the pipeline universal container unless overridden
        if "mpnn_container_image" not in step3_args:
            step3_args["mpnn_container_image"] = self.args.universal_container
        if step3_args.get("mpnn_container_image"):
            extra_bind_dirs.append(Path(step3_args["mpnn_container_image"]).resolve().parent)

        # Auto-assign MPNN server port for SLURM arrays unless user sets it
        if (
            self.args.step3_mpnn_server_port is None and
            not self.args.step3_no_mpnn_server
        ):
            auto_port = self._auto_mpnn_server_port()
            if auto_port:
                step3_args["mpnn-server-port"] = auto_port

        # Resolve protocol file path to absolute (project-relative if needed) and validate
        if "protocol_file" in step3_args and step3_args["protocol_file"]:
            protocol_path = Path(step3_args["protocol_file"])
            if not protocol_path.is_absolute():
                # First try relative to CWD, then project root, then step03 protocols/ dir
                candidate = protocol_path.resolve()
                if not candidate.exists():
                    candidate = (self.project_root / protocol_path).resolve()
                if not candidate.exists():
                    protocols_dir = self.project_root / "modules/step03__fastmpnndesign/protocols"
                    candidate = (protocols_dir / protocol_path.name).resolve()
                protocol_path = candidate
            step3_args["protocol_file"] = str(protocol_path)
            if not protocol_path.exists():
                raise PipelineError(3, f"Protocol file not found: {protocol_path}")
            extra_bind_dirs.append(protocol_path.parent)

        # Convert to CLI args
        for key, value in step3_args.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            elif isinstance(value, (list, tuple)):
                # Handle list arguments like layer_cuts
                args.append(f"--{key}")
                args.extend([str(v) for v in value])
            else:
                args.extend([f"--{key}", str(value)])

        if self.args.verbose:
            args.append("--verbose")
        if self.args.debug:
            args.append("--debug")

        return args, extra_bind_dirs

    def run_step3(self, step2_outputs: StepOutputs) -> StepOutputs:
        """Run step 3: FastMPNN Design with Rosetta Refinement."""
        self._print_step_header(3, "FastMPNN Design + Rosetta Refinement")
        step3_dir = self.work_dir / "step03" if self.work_dir else None

        if self.args.skip_step3:
            print("  Skipping step 3 as requested")
            print()
            return StepOutputs(step=3, output_dir=Path(), success=True)

        start_time = time.time()
        script_path = get_step_script_path(3)
        script_args, extra_bind_dirs = self._build_step3_args(step2_outputs)

        # Check if containers are the same (unified container mode)
        # When using the same container for everything, step 3 should run
        # PyRosetta in-process instead of spawning subprocess containers
        containers_differ = (
            Path(self.args.universal_container).resolve() !=
            Path(self.args.pyrosetta_container).resolve()
        )

        if containers_differ:
            print(f"  Running: {script_path.name}")
            print("  Note: Using different containers for MPNN and Rosetta (nested container mode)")
        else:
            print(f"  Running: {script_path.name} (unified container, in-process Rosetta)")
            # When using unified container, run Rosetta in-process to avoid
            # nested container issues (can't call apptainer from inside container)
            script_args.append("--rosetta_in_process")
        if step3_dir:
            print(f"  Step dir:   {step3_dir}")

        cmd = self._build_container_command(
            "pyrosetta", script_path, script_args,
            module_name="modules.step03__fastmpnndesign.fastmpnn_design",
            enable_nested_containers=containers_differ,
            extra_bind_dirs=extra_bind_dirs,
        )
        self._run_command(cmd, step=3)

        duration = time.time() - start_time
        self.timings["step3"] = duration

        step3_dir = self.work_dir / "step03"

        # In dry run mode, mock the output files
        if self.args.dry_run:
            files = self._mock_output_files(step3_dir, STEP3_OUTPUT_PATTERNS)
            num_designs = self.args.step3_num_final_designs
            if num_designs is None:
                num_designs = STEP3_DEFAULTS["num_final_designs"]
        else:
            files = self._find_output_files(step3_dir, STEP3_OUTPUT_PATTERNS)
            num_designs = len(list(step3_dir.glob("design_*.pdb")))
            # Sanity check against results JSON if present
            results_json = step3_dir / "fastmpnn_design_results.json"
            if results_json.exists():
                try:
                    with open(results_json, "r") as f:
                        results = json.load(f)
                    reported = len(results.get("output_designs", []))
                    if reported != num_designs:
                        print(f"  WARNING: design_*.pdb count ({num_designs}) != results.json output_designs ({reported})")
                    if self.args.step3_num_final_designs is not None and reported < self.args.step3_num_final_designs:
                        print(
                            f"  WARNING: output_designs ({reported}) < requested "
                            f"--step3_num_final_designs {self.args.step3_num_final_designs} (dedupe/constraints may apply)"
                        )
                except Exception:
                    print("  WARNING: Failed to parse step3 results JSON for sanity checks")

        outputs = StepOutputs(
            step=3,
            output_dir=step3_dir,
            files=files,
            duration=duration,
            success=True,
        )

        print(f"  Completed in {format_duration(duration)}")
        if num_designs is None:
            print("  Outputs: protocol target_count designs (dry run)")
        else:
            print(f"  Outputs: {num_designs} designs generated")
        print()

        return outputs

    # -------------------------------------------------------------------------
    # Final Output Handling
    # -------------------------------------------------------------------------

    def _copy_final_outputs(self, step3_outputs: StepOutputs):
        """Copy final designs to output_dir with proper naming."""
        if self.args.skip_step3 or self.args.dry_run:
            return

        step3_dir = step3_outputs.output_dir
        output_bn = self._final_basename()

        # Copy design PDBs with basename prefix
        design_pdbs = sorted(step3_dir.glob("design_*.pdb"))
        if not design_pdbs:
            print("  WARNING: No design_*.pdb files found to copy")
        for pdb in design_pdbs:
            # Extract design number from original name (design_00.pdb -> 00)
            num = pdb.stem.replace("design_", "")
            new_name = f"{output_bn}_design_{num}.pdb"
            dest = self.output_dir / new_name
            shutil.copy2(pdb, dest)
            self.files_created.append(dest)

        # Copy and rename the results JSON
        results_json = step3_dir / "fastmpnn_design_results.json"
        if results_json.exists():
            metrics_subdir = (self.args.metrics_subdir or "").strip()
            if self.args.no_metrics_subdir or not metrics_subdir:
                metrics_dir = self.output_dir
            else:
                metrics_dir = self.output_dir / metrics_subdir
                metrics_dir.mkdir(parents=True, exist_ok=True)
            dest = metrics_dir / f"{output_bn}_design_metrics.json"

            try:
                with open(results_json, "r") as f:
                    results = json.load(f)
            except Exception:
                results = None

            if results:
                results = self._sanitize_metrics_results(results, output_bn)
                with open(dest, "w") as f:
                    json.dump(results, f, indent=2)
            else:
                shutil.copy2(results_json, dest)

            self.files_created.append(dest)
            if results:
                reported = len(results.get("output_designs", []))
                if reported != len(design_pdbs):
                    print(
                        f"  WARNING: results.json output_designs ({reported}) != "
                        f"design_*.pdb count ({len(design_pdbs)})"
                    )

    def _sanitize_metrics_results(self, results: Dict[str, Any], output_bn: str) -> Dict[str, Any]:
        """Normalize metrics JSON paths and metadata for final outputs."""
        output_bn = output_bn or self._final_basename()
        output_dir = self.output_dir
        keep_intermediates = bool(self.args.keep_intermediates)
        scrub_history = not getattr(self.args, "no_metrics_history_scrub", False)

        # Insert original inputs for traceability
        metadata = results.get("metadata", {})
        metadata["input_pdb"] = str(self.args.input_pdb)
        metadata["ref_pdb"] = str(self.args.ref_pdb)
        if self.output_tag:
            metadata["output_tag"] = self.output_tag
            metadata["append_output_tag"] = bool(self.append_output_tag)
        # Capture SLURM identifiers (metadata only; never in filenames)
        metadata["slurm_job_id"] = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
        metadata["slurm_array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID")
        metadata["slurm_procid"] = os.environ.get("SLURM_PROCID") or os.environ.get("SLURM_LOCALID")
        results["metadata"] = metadata

        # Remove step01/step02 paths when intermediates are not kept
        if not keep_intermediates:
            if "step01_pdb" in metadata:
                metadata["step01_pdb"] = None
            if "step02_pdb" in metadata:
                metadata["step02_pdb"] = None

        # Normalize output_designs
        designs = results.get("output_designs", [])
        for design in designs:
            rank = design.get("rank")
            if rank is None:
                continue
            final_pdb = output_dir / f"{output_bn}_design_{int(rank):02d}.pdb"
            design["pdb_path"] = str(final_pdb)
            design["pdb"] = str(final_pdb)

            metrics = design.get("metrics") or {}
            metadat = metrics.get("metadata") or {}
            metadat["input_pdb"] = str(self.args.input_pdb)
            metadat["ref_pdb"] = str(self.args.ref_pdb)
            if not keep_intermediates:
                metadat["step01_pdb"] = None
                metadat["step02_pdb"] = None
            else:
                # Ensure consistent paths when kept
                if "step01_pdb" in metadat:
                    metadat["step01_pdb"] = metadat.get("step01_pdb")
                if "step02_pdb" in metadat:
                    metadat["step02_pdb"] = metadat.get("step02_pdb")
            metadat["designed_pdb"] = str(final_pdb)
            metrics["metadata"] = metadat
            design["metrics"] = metrics

        results["output_designs"] = designs

        # Pull constant metrics fields up into top-level metadata to reduce repetition
        constants = {}

        def _collect_uniform(values):
            vals = [v for v in values if v is not None]
            if not vals:
                return None
            first = vals[0]
            if all(v == first for v in vals):
                return first
            return None

        meta_constant_keys = [
            "bond_length_tolerance",
            "bond_angle_tolerance",
            "catres_bond_tolerance",
            "catres_angle_tolerance",
            "num_catres",
            "num_motif",
        ]

        # Collect constants from output designs
        meta_values = {k: [] for k in meta_constant_keys}
        lddt_thresholds_vals = []
        lddt_cutoff_vals = []
        for design in designs:
            metrics = design.get("metrics") or {}
            metadat = metrics.get("metadata") or {}
            for k in meta_constant_keys:
                meta_values[k].append(metadat.get(k))
            lddt = metrics.get("lddt") or {}
            for key in ("vs_step02", "vs_step01"):
                entry = lddt.get(key) or {}
                if "thresholds" in entry:
                    lddt_thresholds_vals.append(entry.get("thresholds"))
                if "cutoff" in entry:
                    lddt_cutoff_vals.append(entry.get("cutoff"))

        for k, vals in meta_values.items():
            val = _collect_uniform(vals)
            if val is not None:
                constants[k] = val

        lddt_thresholds = _collect_uniform(lddt_thresholds_vals)
        if lddt_thresholds is not None:
            constants["lddt_thresholds"] = lddt_thresholds
        lddt_cutoff = _collect_uniform(lddt_cutoff_vals)
        if lddt_cutoff is not None:
            constants["lddt_cutoff"] = lddt_cutoff

        if constants:
            metadata.setdefault("metrics_constants", {})
            metadata["metrics_constants"].update(constants)

        def _strip_constants_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
            metadat = metrics.get("metadata") or {}
            for k in meta_constant_keys:
                if k in constants:
                    metadat.pop(k, None)
            metrics["metadata"] = metadat

            lddt = metrics.get("lddt") or {}
            for key in ("vs_step02", "vs_step01"):
                entry = lddt.get(key)
                if not entry:
                    continue
                if "lddt_thresholds" in constants:
                    entry.pop("thresholds", None)
                if "lddt_cutoff" in constants:
                    entry.pop("cutoff", None)
                lddt[key] = entry
            metrics["lddt"] = lddt
            return metrics

        if constants:
            for design in designs:
                metrics = design.get("metrics") or {}
                design["metrics"] = _strip_constants_from_metrics(metrics)

        # Scrub intermediate pdb paths when intermediates are not kept
        history = results.get("metrics_history", [])
        for entry in history:
            for structure in entry.get("structures", []):
                if not keep_intermediates and scrub_history and "pdb" in structure:
                    structure["pdb"] = None
                metrics = structure.get("metrics") or {}
                metadat = metrics.get("metadata") or {}
                metadat["input_pdb"] = str(self.args.input_pdb)
                metadat["ref_pdb"] = str(self.args.ref_pdb)
                if not keep_intermediates:
                    metadat["step01_pdb"] = None
                    metadat["step02_pdb"] = None
                metrics["metadata"] = metadat
                if constants:
                    metrics = _strip_constants_from_metrics(metrics)
                structure["metrics"] = metrics
        results["metrics_history"] = history

        return results

    def _cleanup(self):
        """Remove work_dir unless --keep_intermediates."""
        if self.args.dry_run:
            return

        if not self.args.keep_intermediates and self.work_dir and self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary dictionary of the pipeline run."""
        return {
            "basename": self.original_basename,
            "internal_basename": self.internal_basename,
            "input_pdb": str(self.args.input_pdb),
            "ref_pdb": str(self.args.ref_pdb),
            "params": self.args.params,
            "output_dir": str(self.output_dir),
            "step2_preset": self.args.step2_preset,
            "timings": self.timings,
            "files_created": [str(f) for f in self.files_created],
            "dry_run": self.args.dry_run,
        }

    # -------------------------------------------------------------------------
    # Main Run Method
    # -------------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline."""
        start_time = time.time()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create work directory (unless dry run or skipping all steps)
        if not self.args.dry_run:
            self.work_dir = self._create_work_dir()
        else:
            # For dry run, create a mock work dir path
            self.work_dir = self.output_dir / f".pipeline_work_{self.internal_basename}_{self.run_id}"

        self._print_header()

        try:
            # Run steps
            step1_outputs = self.run_step1()
            step2_outputs = self.run_step2(step1_outputs)
            step3_outputs = self.run_step3(step2_outputs)

            # Copy final outputs
            self._copy_final_outputs(step3_outputs)

            # Cleanup
            self._cleanup()

            # Print summary
            total_duration = time.time() - start_time
            self.timings["total"] = total_duration
            self._print_summary(total_duration)

            return self._generate_summary()

        except PipelineError as e:
            LOGGER.error(f"\nPipeline failed at step {e.step}: {e.message}")
            if self.work_dir and self.work_dir.exists():
                LOGGER.error(f"Intermediate files preserved at: {self.work_dir}")
            raise


# =============================================================================
# Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified CLI Runner for the Enzyme Design Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with defaults
  %(prog)s --input_pdb input.pdb --ref_pdb ref.pdb --params ligand.params --output_dir results/

  # Fast step2 preset
  %(prog)s --input_pdb input.pdb --ref_pdb ref.pdb --params ligand.params --output_dir results/ --step2_preset fast

  # Dry run to see commands
  %(prog)s --input_pdb input.pdb --ref_pdb ref.pdb --params ligand.params --output_dir results/ --dry_run

  # Resume from step 2 with existing step1 outputs
  %(prog)s --input_pdb input.pdb --ref_pdb ref.pdb --params ligand.params --output_dir results/ \\
           --skip_step1 --step1_output_dir previous_run/step01/

For more information, see README.md
        """
    )

    # -------------------------------------------------------------------------
    # Global Required Arguments
    # -------------------------------------------------------------------------
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("--input_pdb", required=True,
                          help="Input PDB file (structure prediction with ligand aligned)")
    required.add_argument("--ref_pdb", required=True,
                          help="Reference PDB file (theozyme)")
    required.add_argument("--params", nargs="+", required=True,
                          help="Ligand .params file(s)")
    required.add_argument("--output_dir", required=True,
                          help="Output directory for final results")

    # -------------------------------------------------------------------------
    # Global Optional Arguments
    # -------------------------------------------------------------------------
    global_opts = parser.add_argument_group("Global Options")
    global_opts.add_argument("--catres_subset", default=None,
                             help="Comma-separated REMARK 666 block indices for catalytic motif")
    global_opts.add_argument("--verbose", action="store_true",
                             help="Enable verbose logging")
    global_opts.add_argument("--debug", action="store_true",
                             help="Enable debug logging")
    global_opts.add_argument("--quiet", action="store_true",
                             help="Minimal output")
    global_opts.add_argument("--dry_run", action="store_true",
                             help="Print commands without executing")
    global_opts.add_argument("--keep_intermediates", action="store_true",
                             help="Keep work directory after completion")
    global_opts.add_argument("--output_tag", default=None,
                             help="Optional tag captured in metadata (and optionally filenames). "
                                  "If not set, SLURM job/task IDs are used when available (metadata only)")
    global_opts.add_argument("--append_output_tag", action="store_true",
                             help="Append output_tag to final output filenames (default: metadata only)")
    global_opts.add_argument("--short_internal_basename", action="store_true",
                             help="Use short hash-based basenames for intermediates (final outputs unchanged)")
    global_opts.add_argument("--metrics_subdir", default="scores_and_metrics",
                             help="Subdirectory for final metrics JSON (default: scores_and_metrics). "
                                  "Use --no_metrics_subdir to disable")
    global_opts.add_argument("--no_metrics_subdir", action="store_true",
                             help="Place final metrics JSON in output_dir root instead of a subdirectory")
    global_opts.add_argument("--no_metrics_history_scrub", action="store_true",
                             help="Do not scrub intermediate metrics_history PDB paths in the final metrics JSON")

    # -------------------------------------------------------------------------
    # Execution Control
    # -------------------------------------------------------------------------
    exec_ctrl = parser.add_argument_group("Execution Control")
    exec_ctrl.add_argument("--skip_step1", action="store_true",
                           help="Skip step 1 (use existing outputs)")
    exec_ctrl.add_argument("--skip_step2", action="store_true",
                           help="Skip step 2 (use existing outputs)")
    exec_ctrl.add_argument("--skip_step3", action="store_true",
                           help="Skip step 3")
    exec_ctrl.add_argument("--step1_output_dir", default=None,
                           help="Path to existing step1 outputs (required if --skip_step1)")
    exec_ctrl.add_argument("--step2_output_dir", default=None,
                           help="Path to existing step2 outputs (required if --skip_step2)")

    # -------------------------------------------------------------------------
    # Container Configuration
    # -------------------------------------------------------------------------
    container = parser.add_argument_group("Container Configuration")
    container.add_argument("--container_runtime", default=DEFAULT_CONTAINER_RUNTIME,
                           choices=["apptainer", "singularity"],
                           help=f"Container runtime (default: {DEFAULT_CONTAINER_RUNTIME})")
    container.add_argument("--universal_container", default=DEFAULT_UNIVERSAL_CONTAINER,
                           help=f"Path to universal.sif (default: {DEFAULT_UNIVERSAL_CONTAINER})")
    container.add_argument("--pyrosetta_container", default=DEFAULT_PYROSETTA_CONTAINER,
                           help=f"Path to pyrosetta.sif (default: {DEFAULT_PYROSETTA_CONTAINER})")
    container.add_argument("--container_nv", "--nv", dest="container_nv", action="store_true",
                           help="Enable GPU passthrough for apptainer/singularity (--nv)")

    # Protocol choices (for help text only; allow custom names without strict choices)
    protocol_dir = get_project_root() / "modules/step03__fastmpnndesign/protocols"
    protocol_choices = sorted(p.stem for p in protocol_dir.glob("*.json")) if protocol_dir.exists() else []
    protocol_choices_str = ", ".join(protocol_choices) if protocol_choices else "default"

    # -------------------------------------------------------------------------
    # Step 1 Arguments
    # -------------------------------------------------------------------------
    step1 = parser.add_argument_group("Step 1: Catalytic Residue Alignment")
    step1.add_argument("--step1_strict_backbone_importance", action="store_true",
                       help="Backbone-to-backbone H-bonds alone don't make backbone_important=True")
    step1.add_argument("--step1_exclude_bb_only_hbond_constraints", action="store_true",
                       help="Don't include backbone atoms in constraints when only important for BB-BB H-bonds")
    step1.add_argument("--step1_flex_res_move_all_sc", action="store_true",
                       help="For ARG/LYS: move entire sidechain instead of just tip atoms")
    step1.add_argument("--step1_flex_res_constrain_all_sc", action="store_true",
                       help="For ARG/LYS: constrain entire sidechain (implies flex_res_move_all_sc)")

    # -------------------------------------------------------------------------
    # Step 2 Arguments
    # -------------------------------------------------------------------------
    step2 = parser.add_argument_group("Step 2: Constrained Cartesian Relaxation")
    step2.add_argument("--step2_preset", choices=["fast", "balanced", "thorough", "aggressive"],
                       help="Override step2 preset")
    step2.add_argument("--step2_coord_cst_weight", type=float,
                       help=f"Coordinate constraint weight (default: {STEP2_DEFAULTS['coord_cst_weight']})")
    step2.add_argument("--step2_coord_cst_stdev", type=float,
                       help=f"Constraint stdev in Angstroms (default: {STEP2_DEFAULTS['coord_cst_stdev']})")
    step2.add_argument("--step2_cart_bonded_weight", type=float,
                       help=f"Cart_bonded term weight (default: {STEP2_DEFAULTS['cart_bonded_weight']})")
    step2.add_argument("--step2_mobile_radius", type=float,
                       help=f"Mobile region radius in Angstroms (default: {STEP2_DEFAULTS['mobile_radius']})")
    step2.add_argument("--step2_fastrelax_repeats", type=int,
                       help=f"FastRelax repeats (default: {STEP2_DEFAULTS['fastrelax_repeats']})")
    step2.add_argument("--step2_fastrelax_ramp_stages", type=int,
                       help=f"FastRelax ramp stages (default: {STEP2_DEFAULTS['fastrelax_ramp_stages']})")
    step2.add_argument("--step2_bond_length_tolerance", type=float,
                       help=f"Bond length tolerance in Angstroms (default: {STEP2_DEFAULTS['bond_length_tolerance']})")
    step2.add_argument("--step2_bond_angle_tolerance", type=float,
                       help=f"Bond angle tolerance in degrees (default: {STEP2_DEFAULTS['bond_angle_tolerance']})")
    step2.add_argument("--step2_sequence_neighbor_buffer", type=int,
                       help=f"Include residues +/- N from catres (default: {STEP2_DEFAULTS['sequence_neighbor_buffer']})")
    step2.add_argument("--step2_max_adaptive_rounds", type=int,
                       help=f"Max adaptive rounds (default: {STEP2_DEFAULTS['max_adaptive_rounds']})")
    step2.add_argument("--step2_scorefunction", choices=STEP2_SCOREFUNCTION_CHOICES,
                       help=f"Scorefunction (default: {STEP2_DEFAULTS['scorefunction']})")
    step2.add_argument("--step2_max_runtime", type=int,
                       help=f"Max runtime in seconds (default: {STEP2_DEFAULTS['max_runtime']})")
    step2.add_argument("--step2_cart_bonded_scale_factor", type=float,
                       help=f"Cart_bonded scale factor when not converging (default: {STEP2_DEFAULTS['cart_bonded_scale_factor']})")
    step2.add_argument("--step2_cart_bonded_max", type=float,
                       help=f"Max cart_bonded weight cap (default: {STEP2_DEFAULTS['cart_bonded_max']})")
    step2.add_argument("--step2_fa_rep_scale", type=float,
                       help=f"Scale factor for fa_rep term (default: {STEP2_DEFAULTS['fa_rep_scale']})")
    step2.add_argument("--step2_fa_atr_scale", type=float,
                       help=f"Scale factor for fa_atr term (default: {STEP2_DEFAULTS['fa_atr_scale']})")
    step2.add_argument("--step2_fa_elec_scale", type=float,
                       help=f"Scale factor for fa_elec term (default: {STEP2_DEFAULTS['fa_elec_scale']})")
    step2.add_argument("--step2_ramp_fa_rep", action="store_true",
                       help="Ramp fa_rep across adaptive rounds (default: False)")
    step2.add_argument("--step2_fa_rep_min_scale", type=float,
                       help=f"Starting fa_rep scale when ramping (default: {STEP2_DEFAULTS['fa_rep_min_scale']})")
    step2.add_argument("--step2_auto_expand_mobile", type=parse_bool, default=None,
                       help="Enable/disable automatic mobile region expansion (true/false, hard override)")
    step2.add_argument("--step2_expansion_radius", type=float,
                       help=f"Radius for mobile expansion in Angstroms (default: {STEP2_DEFAULTS['expansion_radius']})")
    step2.add_argument("--step2_max_expansions", type=int,
                       help=f"Maximum mobile expansions (default: {STEP2_DEFAULTS['max_expansions']})")
    step2.add_argument("--step2_catres_bond_tolerance", type=float,
                       help=f"Catres-specific bond tolerance (default: {STEP2_DEFAULTS['catres_bond_tolerance']})")
    step2.add_argument("--step2_catres_angle_tolerance", type=float,
                       help=f"Catres-specific angle tolerance (default: {STEP2_DEFAULTS['catres_angle_tolerance']})")
    step2.add_argument("--step2_require_catres_converged", type=parse_bool, default=None,
                       help="Require catres geometry convergence (true/false, hard override)")
    step2.add_argument("--step2_enable_bond_geometry_min", type=parse_bool, default=None,
                       help="Enable/disable bond geometry minimization (true/false, hard override)")

    # -------------------------------------------------------------------------
    # Step 3 Arguments
    # -------------------------------------------------------------------------
    step3 = parser.add_argument_group("Step 3: FastMPNN Design")

    # Protocol settings
    step3.add_argument("--step3_protocol", default=None,
                       help=f"Protocol name (JSON basename in step03 protocols/). "
                            f"Available: {protocol_choices_str}")
    step3.add_argument("--step3_protocol_file", default=None,
                       help="Path to protocol file (.json or .txt). "
                            f"Overrides --step3_protocol. Default: {STEP3_DEFAULT_PROTOCOL_FILE} "
                            "if neither protocol nor protocol file is provided")

    # Design settings
    step3.add_argument("--step3_design_secondary_sphere", action="store_true",
                       help="Include secondary sphere in design")
    step3.add_argument("--step3_design_gly_pro", action="store_true",
                       help="Allow GLY/PRO redesign")
    step3.add_argument("--step3_layer_cuts", type=float, nargs=3, metavar=("CORE", "SHELL", "FLEX"),
                       help="Layer cutoffs in Angstroms (default: 6.0 8.0 12.0)")

    # MPNN settings
    step3.add_argument("--step3_mpnn_spheres", default=None,
                       help="Override MPNN design spheres (e.g., 'core,shell')")
    step3.add_argument("--step3_mpnn_temperature", type=float,
                       help=f"MPNN sampling temperature (default: {STEP3_DEFAULTS['mpnn_temperature']})")
    step3.add_argument("--step3_mpnn_num_designs", type=int,
                       help=f"Number of MPNN designs per round (default: {STEP3_DEFAULTS['mpnn_num_designs']})")
    step3.add_argument("--step3_mpnn_num_designs_after_first", type=int,
                       help="Designs per subsequent MPNN rounds (default: same as mpnn_num_designs)")
    step3.add_argument("--step3_mpnn_batch_size", type=int,
                       help=f"MPNN batch size (default: {STEP3_DEFAULTS['mpnn_batch_size']})")
    step3.add_argument("--step3_mpnn_omit_aa", default=None,
                       help=f"Amino acids to exclude from design (default: {STEP3_DEFAULTS['mpnn_omit_aa']})")
    step3.add_argument("--step3_mpnn_use_gpu", action="store_true",
                       help="Force MPNN to use GPU (enables --nv for MPNN subprocess/server)")
    step3.add_argument("--step3_mpnn_no_gpu", action="store_true",
                       help="Force MPNN to run on CPU")
    step3.add_argument("--step3_mpnn_container_image", default=None,
                       help="Apptainer image for MPNN (defaults to --universal_container)")

    # Constraint settings
    step3.add_argument("--step3_coord_cst_weight", type=float,
                       help=f"Coordinate constraint weight (default: {STEP3_DEFAULTS['coord_cst_weight']})")
    step3.add_argument("--step3_coord_cst_stdev", type=float,
                       help=f"Constraint stdev (default: {STEP3_DEFAULTS['coord_cst_stdev']})")
    step3.add_argument("--step3_global_coord_cst_weight", type=float,
                       help=f"Global constraint weight for all atoms (default: {STEP3_DEFAULTS['global_coord_cst_weight']})")
    step3.add_argument("--step3_global_coord_cst_stdev", type=float,
                       help=f"Global constraint stdev (default: {STEP3_DEFAULTS['global_coord_cst_stdev']})")

    # Scorefunction settings
    step3.add_argument("--step3_scorefunction_cart", choices=STEP3_SCOREFUNCTION_CART_CHOICES,
                       help=f"Cartesian scorefunction (default: {STEP3_DEFAULTS['scorefunction_cart']})")
    step3.add_argument("--step3_scorefunction_torsional", choices=STEP3_SCOREFUNCTION_TORSIONAL_CHOICES,
                       help=f"Torsional scorefunction (default: {STEP3_DEFAULTS['scorefunction_torsional']})")
    step3.add_argument("--step3_fa_rep_weight", type=float,
                       help="Override fa_rep term weight")
    step3.add_argument("--step3_cart_bonded_weight", type=float,
                       help=f"Cart_bonded term weight (default: {STEP3_DEFAULTS['cart_bonded_weight']})")

    # Relaxation settings
    step3.add_argument("--step3_relax_rounds", type=int,
                       help=f"FastRelax repeats per relax step (default: {STEP3_DEFAULTS['relax_rounds']})")
    step3.add_argument("--step3_relax_inner_cycles", type=int,
                       help="FastRelax inner cycles")
    step3.add_argument("--step3_bond_length_tolerance", type=float,
                       help="Bond length deviation tolerance (Angstroms)")
    step3.add_argument("--step3_bond_angle_tolerance", type=float,
                       help="Bond angle deviation tolerance (degrees)")

    # Interaction conservation is controlled via protocol keep_interactions steps

    # Backbone H-bond constraints
    step3.add_argument("--step3_include_bb_hbond_constraints", action="store_true",
                       help="Include backbone atoms in constraints for BB-BB H-bonds")

    # Workflow settings
    step3.add_argument("--step3_skip_initial_cart_relax", action="store_true",
                       help="Skip initial cartesian relaxation")

    # Output settings
    step3.add_argument("--step3_num_final_designs", type=int,
                       help="Number of final designs to output (overrides protocol target_count)")
    step3.add_argument("--step3_max_runtime", type=int,
                       help=f"Max runtime in seconds (default: {STEP3_DEFAULTS['max_runtime']})")
    step3.add_argument("--step3_rosetta_timeout", type=int,
                       help=f"Timeout for individual Rosetta calls (default: {STEP3_DEFAULTS['rosetta_timeout']})")
    step3.add_argument("--step3_cart_relax_max_rounds", type=int,
                       help=f"Max adaptive Cartesian relax rounds (default: {STEP3_DEFAULTS['cart_relax_max_rounds']})")
    step3.add_argument("--step3_keep_intermediates", action="store_true",
                       help="Keep intermediate files from step3")
    # MPNN server controls
    step3.add_argument("--step3_no_mpnn_server", action="store_true",
                       help="Disable MPNN server (use subprocess per call)")
    step3.add_argument("--step3_mpnn_server_host", default=None,
                       help="MPNN server hostname (default: localhost)")
    step3.add_argument("--step3_mpnn_server_port", type=int, default=None,
                       help="MPNN server port (default: 5000; auto-assigned for SLURM arrays if unset)")
    step3.add_argument("--step3_no_auto_start_mpnn_server", action="store_true",
                       help="Do not auto-start MPNN server on first call")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    # Check required files exist
    if not Path(args.input_pdb).exists():
        sys.exit(f"Error: Input PDB not found: {args.input_pdb}")
    if not Path(args.ref_pdb).exists():
        sys.exit(f"Error: Reference PDB not found: {args.ref_pdb}")
    for params_file in args.params:
        if not Path(params_file).exists():
            sys.exit(f"Error: Params file not found: {params_file}")

    # Check skip logic consistency
    if args.skip_step1 and not args.step1_output_dir:
        sys.exit("Error: --skip_step1 requires --step1_output_dir")
    if args.skip_step2 and not args.step2_output_dir:
        sys.exit("Error: --skip_step2 requires --step2_output_dir")

    # Check skip directories exist
    if args.step1_output_dir and not Path(args.step1_output_dir).exists():
        sys.exit(f"Error: Step1 output directory not found: {args.step1_output_dir}")
    if args.step2_output_dir and not Path(args.step2_output_dir).exists():
        sys.exit(f"Error: Step2 output directory not found: {args.step2_output_dir}")

    # Check containers exist (unless dry run)
    if not args.dry_run:
        if not Path(args.universal_container).exists():
            sys.exit(f"Error: Universal container not found: {args.universal_container}")
        if not Path(args.pyrosetta_container).exists():
            sys.exit(f"Error: PyRosetta container not found: {args.pyrosetta_container}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Validate arguments
    validate_args(args)

    # Run pipeline
    runner = PipelineRunner(args)
    try:
        summary = runner.run()
        return 0
    except PipelineError as e:
        return e.returncode
    except KeyboardInterrupt:
        LOGGER.error("\nPipeline interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
