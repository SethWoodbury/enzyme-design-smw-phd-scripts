#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Stage Tracker for Enzyme Design

Provides structured logging and metric tracking throughout the design pipeline.
This enables:
- Stage-by-stage metric logging
- Catalytic residue RMSD tracking
- Constraint score history
- Checkpoint snapshots (PDB + metrics)
- Summary reports for debugging

Usage:
    tracker = PipelineTracker(output_dir="/path/to/output", verbose=True)

    tracker.begin_stage("initialization", "Loading input and setting up constraints")
    tracker.log_metric("n_catalytic_residues", 10)
    tracker.end_stage()

    tracker.begin_stage("mpnn_design", "MPNN design at temperature 0.1")
    tracker.log_catres_rmsd(rmsd_result, label="post_mpnn")
    tracker.checkpoint("after_mpnn", pose=pose)
    tracker.end_stage()

    print(tracker.summary())
    tracker.save_report()
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import pyrosetta
    import pyrosetta.distributed.io
    HAS_PYROSETTA = True
except ImportError:
    HAS_PYROSETTA = False


class PipelineTracker:
    """
    Track metrics and state throughout the enzyme design pipeline.

    Provides:
    - Stage-by-stage metric logging
    - Catalytic residue RMSD tracking
    - Constraint score history
    - Checkpoint snapshots
    - Summary reports
    """

    def __init__(self, output_dir: str = None, verbose: bool = True, name: str = None):
        """
        Initialize the pipeline tracker.

        Arguments:
            output_dir: Directory for saving reports and checkpoints (default: cwd)
            verbose: Print progress to stdout
            name: Optional name for this tracking session
        """
        self.output_dir = output_dir or os.getcwd()
        self.verbose = verbose
        self.name = name or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.stages: List[Dict] = []
        self.current_stage: Optional[Dict] = None
        self.start_time = time.time()
        self.checkpoints: Dict[str, Any] = {}
        self.global_metrics: Dict[str, Any] = {}

    def begin_stage(self, stage_name: str, description: str = ""):
        """
        Begin tracking a new pipeline stage.

        Arguments:
            stage_name: Short identifier for the stage (e.g., "mpnn_design")
            description: Human-readable description
        """
        # End previous stage if not ended
        if self.current_stage is not None:
            self.end_stage()

        self.current_stage = {
            'name': stage_name,
            'description': description,
            'start_time': time.time(),
            'metrics': {},
            'warnings': [],
            'errors': []
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  STAGE: {stage_name}")
            if description:
                print(f"  {description}")
            print(f"{'='*60}")

    def end_stage(self, success: bool = True):
        """
        End the current stage and record duration.

        Arguments:
            success: Whether the stage completed successfully
        """
        if self.current_stage is None:
            return

        self.current_stage['end_time'] = time.time()
        self.current_stage['duration'] = self.current_stage['end_time'] - self.current_stage['start_time']
        self.current_stage['success'] = success
        self.stages.append(self.current_stage)

        if self.verbose:
            status = "COMPLETED" if success else "FAILED"
            print(f"  [{status}] Duration: {self.current_stage['duration']:.2f}s")

        self.current_stage = None

    def log_metric(self, name: str, value: Any, unit: str = ""):
        """
        Log a metric for the current stage.

        Arguments:
            name: Metric name
            value: Metric value (should be JSON-serializable)
            unit: Optional unit string (e.g., "A", "REU", "s")
        """
        if self.current_stage is None:
            self.begin_stage("unnamed")

        self.current_stage['metrics'][name] = {
            'value': value,
            'unit': unit,
            'timestamp': time.time() - self.start_time
        }

        if self.verbose:
            if unit:
                print(f"    {name}: {value} {unit}")
            else:
                print(f"    {name}: {value}")

    def log_catres_rmsd(self, rmsd_result: Dict, label: str = ""):
        """
        Log catalytic residue RMSD from calculate_catalytic_residue_rmsd().

        Arguments:
            rmsd_result: Result dict from calculate_catalytic_residue_rmsd()
            label: Optional label to distinguish multiple RMSD measurements
        """
        key = f"catres_rmsd{'_' + label if label else ''}"
        self.log_metric(f"{key}_overall", rmsd_result.get('overall_rmsd', 0.0), "A")

        # Log per-residue RMSD
        for resno, info in rmsd_result.get('per_residue', {}).items():
            name3 = info.get('name3', 'UNK')
            rmsd_val = info.get('rmsd', 0.0)
            self.current_stage['metrics'][f"{key}_{name3}{resno}"] = {
                'value': rmsd_val,
                'unit': 'A',
                'timestamp': time.time() - self.start_time
            }

    def log_constraint_scores(self, cat_scores: Dict, label: str = ""):
        """
        Log constraint scores from evaluate_catalytic_constraint_scores().

        Arguments:
            cat_scores: Result dict from evaluate_catalytic_constraint_scores()
            label: Optional label to distinguish measurements
        """
        key = f"cst_score{'_' + label if label else ''}"

        total_cst = 0.0
        for resno, scores in cat_scores.items():
            name3 = scores.get('name3', 'UNK')
            total = scores.get('total', 0.0)
            total_cst += total
            self.current_stage['metrics'][f"{key}_{name3}{resno}_total"] = {
                'value': total,
                'unit': 'REU',
                'timestamp': time.time() - self.start_time
            }

        self.log_metric(f"{key}_total", total_cst, "REU")

    def log_warning(self, message: str):
        """
        Log a warning for the current stage.

        Arguments:
            message: Warning message
        """
        if self.current_stage is None:
            self.begin_stage("unnamed")

        self.current_stage['warnings'].append({
            'message': message,
            'timestamp': time.time() - self.start_time
        })

        if self.verbose:
            print(f"    WARNING: {message}")

    def log_error(self, message: str):
        """
        Log an error for the current stage.

        Arguments:
            message: Error message
        """
        if self.current_stage is None:
            self.begin_stage("unnamed")

        self.current_stage['errors'].append({
            'message': message,
            'timestamp': time.time() - self.start_time
        })

        if self.verbose:
            print(f"    ERROR: {message}")

    def checkpoint(self, name: str, pose=None, metrics: Dict = None):
        """
        Save a checkpoint (pose snapshot + metrics).

        Arguments:
            name: Checkpoint name (e.g., "after_mpnn", "post_minimization")
            pose: Optional PyRosetta Pose to save
            metrics: Optional additional metrics dict
        """
        checkpoint_data = {
            'name': name,
            'timestamp': time.time() - self.start_time,
            'metrics': metrics or {}
        }

        if pose is not None and HAS_PYROSETTA:
            # Save PDB string
            checkpoint_data['pdb_str'] = pyrosetta.distributed.io.to_pdbstring(pose)
            checkpoint_data['sequence'] = pose.sequence()
            checkpoint_data['n_residues'] = pose.size()

        self.checkpoints[name] = checkpoint_data

        if self.verbose:
            print(f"    [CHECKPOINT] Saved: {name}")

    def get_checkpoint(self, name: str) -> Optional[Dict]:
        """
        Retrieve a saved checkpoint.

        Arguments:
            name: Checkpoint name

        Returns:
            Checkpoint data dict or None if not found
        """
        return self.checkpoints.get(name)

    def get_checkpoint_pose(self, name: str):
        """
        Load a pose from a saved checkpoint.

        Arguments:
            name: Checkpoint name

        Returns:
            PyRosetta Pose or None if checkpoint doesn't exist or has no pose
        """
        if not HAS_PYROSETTA:
            raise ImportError("PyRosetta is required to load checkpoint poses")

        checkpoint = self.checkpoints.get(name)
        if checkpoint is None or 'pdb_str' not in checkpoint:
            return None

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(checkpoint['pdb_str'])
            temp_path = f.name

        try:
            pose = pyrosetta.pose_from_file(temp_path)
            return pose
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def set_global_metric(self, name: str, value: Any):
        """
        Set a global metric that persists across stages.

        Arguments:
            name: Metric name
            value: Metric value
        """
        self.global_metrics[name] = value

    def summary(self) -> str:
        """
        Generate a summary report of all stages.

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 70,
            f"PIPELINE SUMMARY: {self.name}",
            "=" * 70,
            f"Total duration: {time.time() - self.start_time:.2f}s",
            f"Stages completed: {len(self.stages)}",
            f"Checkpoints saved: {len(self.checkpoints)}",
            ""
        ]

        # Global metrics
        if self.global_metrics:
            lines.append("Global Metrics:")
            for name, value in self.global_metrics.items():
                lines.append(f"  {name}: {value}")
            lines.append("")

        # Per-stage summary
        for stage in self.stages:
            status = "OK" if stage['success'] else "FAIL"
            lines.append(f"[{status}] {stage['name']} ({stage['duration']:.2f}s)")

            # Key metrics (RMSD and constraint scores)
            rmsd_metrics = {}
            cst_metrics = {}
            other_metrics = {}

            for metric_name, metric_data in stage['metrics'].items():
                val = metric_data['value']
                if 'rmsd' in metric_name.lower() and 'overall' in metric_name.lower():
                    rmsd_metrics[metric_name] = val
                elif 'cst_score' in metric_name.lower() and metric_name.endswith('_total'):
                    cst_metrics[metric_name] = val
                elif not any(x in metric_name for x in ['rmsd_', 'cst_score_']):
                    other_metrics[metric_name] = metric_data

            # Print key metrics
            for name, val in rmsd_metrics.items():
                flag = " !!!" if isinstance(val, (int, float)) and val > 1.5 else ""
                lines.append(f"      {name}: {val:.3f} A{flag}")
            for name, val in cst_metrics.items():
                lines.append(f"      {name}: {val:.3f} REU")

            # Warnings and errors
            if stage['warnings']:
                lines.append(f"      Warnings: {len(stage['warnings'])}")
            if stage['errors']:
                lines.append(f"      Errors: {len(stage['errors'])}")
                for err in stage['errors'][:3]:  # Show first 3 errors
                    lines.append(f"        - {err['message'][:60]}...")

        lines.append("=" * 70)
        return "\n".join(lines)

    def save_report(self, filename: str = None, include_pdb: bool = False):
        """
        Save detailed JSON report.

        Arguments:
            filename: Output filename (default: pipeline_report.json in output_dir)
            include_pdb: Whether to include PDB strings in checkpoint data
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f"{self.name}_report.json")

        # Prepare checkpoint data (optionally excluding PDB strings)
        checkpoint_data = {}
        for name, cp in self.checkpoints.items():
            cp_copy = {k: v for k, v in cp.items()}
            if not include_pdb and 'pdb_str' in cp_copy:
                cp_copy['pdb_str'] = f"<{len(cp['pdb_str'])} chars>"
            checkpoint_data[name] = cp_copy

        report = {
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'total_duration': time.time() - self.start_time,
            'global_metrics': self.global_metrics,
            'stages': self.stages,
            'checkpoints': checkpoint_data,
            'summary': {
                'n_stages': len(self.stages),
                'n_checkpoints': len(self.checkpoints),
                'all_success': all(s['success'] for s in self.stages) if self.stages else True,
                'total_warnings': sum(len(s['warnings']) for s in self.stages),
                'total_errors': sum(len(s['errors']) for s in self.stages)
            }
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        if self.verbose:
            print(f"Report saved: {filename}")

        return filename

    def save_checkpoint_pdbs(self, output_dir: str = None):
        """
        Save all checkpoint PDBs to separate files.

        Arguments:
            output_dir: Directory for PDB files (default: self.output_dir/checkpoints/)

        Returns:
            list: Paths to saved PDB files
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "checkpoints")

        os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        for name, cp in self.checkpoints.items():
            if 'pdb_str' in cp:
                pdb_path = os.path.join(output_dir, f"{name}.pdb")
                with open(pdb_path, 'w') as f:
                    f.write(cp['pdb_str'])
                saved_files.append(pdb_path)

                if self.verbose:
                    print(f"  Saved checkpoint PDB: {pdb_path}")

        return saved_files


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tracker_for_design(pdb_name: str, output_dir: str, verbose: bool = True) -> PipelineTracker:
    """
    Create a tracker configured for enzyme design pipeline.

    Arguments:
        pdb_name: Name of the input PDB (used in tracker name)
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Configured PipelineTracker
    """
    tracker_name = f"design_{os.path.splitext(os.path.basename(pdb_name))[0]}"
    return PipelineTracker(output_dir=output_dir, verbose=verbose, name=tracker_name)
