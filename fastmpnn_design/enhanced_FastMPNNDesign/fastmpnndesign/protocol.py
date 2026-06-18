"""
Protocol parsing and execution for enhanced_fastmpnndesign.

Handles parsing of the multi-step design protocol from text files or strings.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Union, Optional, Any
from pathlib import Path
from enum import Enum, auto

# Add package directory to path for standalone execution
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from constants import DEFAULT_PROTOCOL
from logging_config import get_logger

logger = get_logger("protocol")


class ProtocolStepType(Enum):
    """Types of protocol steps."""
    SCALE = auto()           # scale:weight_name value
    MPNN = auto()            # mpnn temperature n_sequences
    REPACK = auto()          # repack
    MIN = auto()             # min tolerance
    KEEP_BEST = auto()       # keep_best n
    TASK_OPERATION = auto()  # task_operation name


@dataclass
class ProtocolStep:
    """A parsed protocol step."""
    step_type: ProtocolStepType
    args: List[Union[str, float, int]] = field(default_factory=list)
    raw_line: str = ""

    def __str__(self) -> str:
        return f"{self.step_type.name}: {self.args}"


def parse_protocol_line(line: str) -> Optional[ProtocolStep]:
    """
    Parse a single protocol line.

    Supported commands:
    - scale:<weight_name> <value>
    - mpnn <temperature> <n_sequences>
    - repack
    - min <tolerance>
    - keep_best <n>
    - task_operation <name>

    Args:
        line: Protocol line to parse

    Returns:
        ProtocolStep or None if line is empty/comment
    """
    line = line.strip()

    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None

    parts = line.split()
    if not parts:
        return None

    cmd = parts[0].lower()

    # scale:weight_name value
    if cmd.startswith('scale:'):
        weight_name = cmd.split(':')[1]
        value = float(parts[1]) if len(parts) > 1 else 1.0
        return ProtocolStep(
            step_type=ProtocolStepType.SCALE,
            args=[weight_name, value],
            raw_line=line
        )

    # mpnn temperature n_sequences
    elif cmd == 'mpnn':
        temp = float(parts[1]) if len(parts) > 1 else 0.1
        n_seq = int(parts[2]) if len(parts) > 2 else 1
        return ProtocolStep(
            step_type=ProtocolStepType.MPNN,
            args=[temp, n_seq],
            raw_line=line
        )

    # repack
    elif cmd == 'repack':
        return ProtocolStep(
            step_type=ProtocolStepType.REPACK,
            args=[],
            raw_line=line
        )

    # min tolerance
    elif cmd == 'min':
        tolerance = float(parts[1]) if len(parts) > 1 else 0.01
        return ProtocolStep(
            step_type=ProtocolStepType.MIN,
            args=[tolerance],
            raw_line=line
        )

    # keep_best n
    elif cmd == 'keep_best':
        n = int(parts[1]) if len(parts) > 1 else 5
        return ProtocolStep(
            step_type=ProtocolStepType.KEEP_BEST,
            args=[n],
            raw_line=line
        )

    # task_operation name
    elif cmd == 'task_operation':
        name = parts[1] if len(parts) > 1 else ""
        return ProtocolStep(
            step_type=ProtocolStepType.TASK_OPERATION,
            args=[name],
            raw_line=line
        )

    else:
        logger.warning(f"Unknown protocol command: {cmd}")
        return None


def parse_protocol(
    protocol_input: Union[str, Path, List[str], None]
) -> List[ProtocolStep]:
    """
    Parse a protocol definition into executable steps.

    Handles:
    - File path to protocol file
    - Inline protocol string
    - List of protocol lines
    - None (returns default protocol)

    Args:
        protocol_input: Protocol source

    Returns:
        List of ProtocolStep objects
    """
    if protocol_input is None:
        logger.info("Using default protocol")
        return parse_protocol(DEFAULT_PROTOCOL)

    lines = []

    if isinstance(protocol_input, Path):
        if protocol_input.exists():
            logger.info(f"Reading protocol from {protocol_input}")
            with open(protocol_input, 'r') as f:
                lines = f.readlines()
        else:
            raise FileNotFoundError(f"Protocol file not found: {protocol_input}")

    elif isinstance(protocol_input, str):
        lines = protocol_input.strip().split('\n')

    elif isinstance(protocol_input, list):
        lines = protocol_input

    # Parse each line
    steps = []
    for line in lines:
        step = parse_protocol_line(line)
        if step:
            steps.append(step)

    logger.info(f"Parsed {len(steps)} protocol steps")
    return steps


def get_default_protocol() -> List[ProtocolStep]:
    """
    Return the default protocol matching original script lines 310-341.

    Returns:
        List of ProtocolStep objects
    """
    return parse_protocol(DEFAULT_PROTOCOL)


def protocol_to_string(steps: List[ProtocolStep]) -> str:
    """
    Convert protocol steps back to string format.

    Args:
        steps: List of ProtocolStep objects

    Returns:
        Protocol as string
    """
    lines = []
    for step in steps:
        if step.step_type == ProtocolStepType.SCALE:
            lines.append(f"scale:{step.args[0]} {step.args[1]}")
        elif step.step_type == ProtocolStepType.MPNN:
            lines.append(f"mpnn {step.args[0]} {step.args[1]}")
        elif step.step_type == ProtocolStepType.REPACK:
            lines.append("repack")
        elif step.step_type == ProtocolStepType.MIN:
            lines.append(f"min {step.args[0]}")
        elif step.step_type == ProtocolStepType.KEEP_BEST:
            lines.append(f"keep_best {step.args[0]}")
        elif step.step_type == ProtocolStepType.TASK_OPERATION:
            lines.append(f"task_operation {step.args[0]}")

    return '\n'.join(lines)


def protocol_summary(steps: List[ProtocolStep]) -> str:
    """
    Generate a summary of the protocol.

    Args:
        steps: List of ProtocolStep objects

    Returns:
        Summary string
    """
    mpnn_count = sum(1 for s in steps if s.step_type == ProtocolStepType.MPNN)
    min_count = sum(1 for s in steps if s.step_type == ProtocolStepType.MIN)
    keep_best = [s.args[0] for s in steps if s.step_type == ProtocolStepType.KEEP_BEST]

    return (
        f"Protocol: {len(steps)} steps, "
        f"{mpnn_count} MPNN rounds, "
        f"{min_count} minimizations, "
        f"keep_best: {keep_best}"
    )
