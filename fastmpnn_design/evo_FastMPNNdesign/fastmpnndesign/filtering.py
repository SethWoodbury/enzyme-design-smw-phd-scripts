"""
Candidate filtering and ranking for design selection.

Provides ranking strategies and filtering functions for selecting
best design candidates based on geometry quality, scores, and diversity.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum

from fastmpnndesign.metrics import CandidateMetrics, GeometryMetrics
from fastmpnndesign.logging_config import get_logger

logger = get_logger("filtering")


class RankingStrategy(Enum):
    """Available ranking strategies for candidate selection."""
    GEOMETRY_QUALITY = "geometry_quality"
    ROSETTA_SCORE = "rosetta_score"
    MPNN_SCORE = "mpnn_score"
    COMBINED = "combined"


@dataclass
class FilterCriteria:
    """Criteria for filtering candidates."""
    max_mean_displacement: Optional[float] = 0.5  # Angstroms
    max_max_displacement: Optional[float] = 1.0   # Angstroms
    min_pct_within_tolerance: Optional[float] = 50.0  # Percentage
    max_cart_bonded_score: Optional[float] = 5.0
    max_total_score: Optional[float] = None


def filter_by_geometry(
    candidates: List[CandidateMetrics],
    criteria: FilterCriteria
) -> List[CandidateMetrics]:
    """
    Filter candidates by geometry quality criteria.

    Args:
        candidates: List of candidates with computed metrics.
        criteria: FilterCriteria object.

    Returns:
        Filtered list of candidates.
    """
    filtered = []

    for c in candidates:
        geom = c.geometry

        # Check mean displacement
        if criteria.max_mean_displacement is not None:
            if geom.mean_displacement > criteria.max_mean_displacement:
                logger.debug(
                    f"Filtered {c.pdb_path.name}: mean_disp={geom.mean_displacement:.3f} "
                    f"> {criteria.max_mean_displacement}"
                )
                continue

        # Check max displacement
        if criteria.max_max_displacement is not None:
            if geom.max_displacement > criteria.max_max_displacement:
                logger.debug(
                    f"Filtered {c.pdb_path.name}: max_disp={geom.max_displacement:.3f} "
                    f"> {criteria.max_max_displacement}"
                )
                continue

        # Check percentage within tolerance
        if criteria.min_pct_within_tolerance is not None:
            if geom.pct_within_0_5A < criteria.min_pct_within_tolerance:
                logger.debug(
                    f"Filtered {c.pdb_path.name}: pct_within={geom.pct_within_0_5A:.1f}% "
                    f"< {criteria.min_pct_within_tolerance}%"
                )
                continue

        filtered.append(c)

    logger.info(
        f"Geometry filter: {len(candidates)} -> {len(filtered)} candidates "
        f"({len(candidates) - len(filtered)} filtered)"
    )
    return filtered


def filter_by_scoring(
    candidates: List[CandidateMetrics],
    criteria: FilterCriteria
) -> List[CandidateMetrics]:
    """
    Filter candidates by Rosetta scoring criteria.

    Args:
        candidates: List of candidates with computed metrics.
        criteria: FilterCriteria object.

    Returns:
        Filtered list of candidates.
    """
    filtered = []

    for c in candidates:
        scoring = c.scoring

        # Check cart_bonded score
        if criteria.max_cart_bonded_score is not None:
            if scoring.cart_bonded_score > criteria.max_cart_bonded_score:
                logger.debug(
                    f"Filtered {c.pdb_path.name}: cart_bonded={scoring.cart_bonded_score:.2f} "
                    f"> {criteria.max_cart_bonded_score}"
                )
                continue

        # Check total score
        if criteria.max_total_score is not None:
            if scoring.total_score > criteria.max_total_score:
                logger.debug(
                    f"Filtered {c.pdb_path.name}: total_score={scoring.total_score:.2f} "
                    f"> {criteria.max_total_score}"
                )
                continue

        filtered.append(c)

    logger.info(
        f"Scoring filter: {len(candidates)} -> {len(filtered)} candidates"
    )
    return filtered


def rank_by_geometry(candidates: List[CandidateMetrics]) -> List[CandidateMetrics]:
    """
    Rank candidates by geometry quality (lower displacement = better).

    Primary sort: mean displacement (ascending)
    Secondary sort: max displacement (ascending)
    Tertiary sort: percentage within 0.1A tolerance (descending)
    """
    def sort_key(c: CandidateMetrics) -> Tuple[float, float, float]:
        return (
            c.geometry.mean_displacement,
            c.geometry.max_displacement,
            -c.geometry.pct_within_0_1A
        )

    ranked = sorted(candidates, key=sort_key)

    # Assign ranks
    for i, c in enumerate(ranked):
        c.rank = i + 1

    return ranked


def rank_by_rosetta_score(candidates: List[CandidateMetrics]) -> List[CandidateMetrics]:
    """
    Rank candidates by Rosetta total score (lower = better).
    """
    ranked = sorted(candidates, key=lambda c: c.scoring.total_score)

    for i, c in enumerate(ranked):
        c.rank = i + 1

    return ranked


def rank_by_mpnn_score(candidates: List[CandidateMetrics]) -> List[CandidateMetrics]:
    """
    Rank candidates by MPNN score (lower = better, more negative = better).
    """
    ranked = sorted(
        candidates,
        key=lambda c: c.mpnn_score if c.mpnn_score is not None else float('inf')
    )

    for i, c in enumerate(ranked):
        c.rank = i + 1

    return ranked


def rank_by_combined(
    candidates: List[CandidateMetrics],
    geometry_weight: float = 0.5,
    score_weight: float = 0.3,
    mpnn_weight: float = 0.2
) -> List[CandidateMetrics]:
    """
    Rank candidates by combined weighted score.

    Normalizes each metric to 0-1 range and combines with weights.
    Lower combined score = better.
    """
    if not candidates:
        return []

    # Get ranges for normalization
    geom_disps = [c.geometry.mean_displacement for c in candidates]
    total_scores = [c.scoring.total_score for c in candidates]
    mpnn_scores = [c.mpnn_score for c in candidates if c.mpnn_score is not None]

    geom_min, geom_max = min(geom_disps), max(geom_disps)
    score_min, score_max = min(total_scores), max(total_scores)
    mpnn_min, mpnn_max = (min(mpnn_scores), max(mpnn_scores)) if mpnn_scores else (0, 1)

    def normalize(val: float, vmin: float, vmax: float) -> float:
        if vmax - vmin < 1e-6:
            return 0.5
        return (val - vmin) / (vmax - vmin)

    def combined_score(c: CandidateMetrics) -> float:
        geom_norm = normalize(c.geometry.mean_displacement, geom_min, geom_max)
        score_norm = normalize(c.scoring.total_score, score_min, score_max)

        mpnn_norm = 0.5  # Default if no MPNN score
        if c.mpnn_score is not None:
            mpnn_norm = normalize(c.mpnn_score, mpnn_min, mpnn_max)

        return (
            geometry_weight * geom_norm +
            score_weight * score_norm +
            mpnn_weight * mpnn_norm
        )

    ranked = sorted(candidates, key=combined_score)

    for i, c in enumerate(ranked):
        c.rank = i + 1

    return ranked


def select_best(
    candidates: List[CandidateMetrics],
    n_keep: int,
    strategy: RankingStrategy = RankingStrategy.GEOMETRY_QUALITY,
    filter_criteria: Optional[FilterCriteria] = None
) -> List[CandidateMetrics]:
    """
    Select best N candidates using specified strategy.

    Args:
        candidates: List of candidates to select from.
        n_keep: Number of candidates to keep.
        strategy: Ranking strategy to use.
        filter_criteria: Optional filtering criteria to apply first.

    Returns:
        List of best N candidates.
    """
    if not candidates:
        return []

    # Apply filtering if criteria provided
    if filter_criteria:
        candidates = filter_by_geometry(candidates, filter_criteria)
        candidates = filter_by_scoring(candidates, filter_criteria)

    if not candidates:
        logger.warning("No candidates remaining after filtering")
        return []

    # Rank based on strategy
    if strategy == RankingStrategy.GEOMETRY_QUALITY:
        ranked = rank_by_geometry(candidates)
    elif strategy == RankingStrategy.ROSETTA_SCORE:
        ranked = rank_by_rosetta_score(candidates)
    elif strategy == RankingStrategy.MPNN_SCORE:
        ranked = rank_by_mpnn_score(candidates)
    elif strategy == RankingStrategy.COMBINED:
        ranked = rank_by_combined(candidates)
    else:
        ranked = rank_by_geometry(candidates)

    # Select top N
    selected = ranked[:n_keep]

    logger.info(
        f"Selected {len(selected)}/{len(ranked)} candidates using {strategy.value}"
    )
    for c in selected:
        logger.debug(
            f"  Rank {c.rank}: {c.pdb_path.name} "
            f"(mean_disp={c.geometry.mean_displacement:.3f}, "
            f"score={c.scoring.total_score:.1f})"
        )

    return selected


def compute_sequence_diversity(
    candidates: List[CandidateMetrics],
    active_site_only: bool = True
) -> Dict[str, Any]:
    """
    Compute sequence diversity metrics across candidates.

    Args:
        candidates: List of candidates.
        active_site_only: If True, compute diversity only for active site.

    Returns:
        Dictionary with diversity metrics.
    """
    if not candidates:
        return {'n_unique': 0, 'diversity': 0.0}

    sequences = []
    for c in candidates:
        if active_site_only and c.sequence.active_site_sequence:
            sequences.append(c.sequence.active_site_sequence)
        else:
            sequences.append(c.sequence.sequence)

    # Count unique sequences
    unique_seqs = set(sequences)
    n_unique = len(unique_seqs)

    # Compute pairwise diversity (Hamming distance)
    total_dist = 0
    n_pairs = 0
    for i, s1 in enumerate(sequences):
        for s2 in sequences[i+1:]:
            if len(s1) == len(s2):
                dist = sum(1 for a, b in zip(s1, s2) if a != b)
                total_dist += dist
                n_pairs += 1

    avg_diversity = total_dist / n_pairs if n_pairs > 0 else 0.0

    return {
        'n_unique': n_unique,
        'n_total': len(sequences),
        'avg_pairwise_distance': avg_diversity,
        'diversity_ratio': n_unique / len(sequences) if sequences else 0.0
    }


def diversify_selection(
    candidates: List[CandidateMetrics],
    n_select: int,
    min_hamming_distance: int = 2
) -> List[CandidateMetrics]:
    """
    Select diverse candidates ensuring minimum sequence distance.

    Uses greedy selection: pick best, then pick best that is sufficiently
    different from already selected.

    Args:
        candidates: Pre-ranked list of candidates.
        n_select: Number to select.
        min_hamming_distance: Minimum Hamming distance between selected sequences.

    Returns:
        List of diverse selected candidates.
    """
    if not candidates:
        return []

    selected = [candidates[0]]

    for c in candidates[1:]:
        if len(selected) >= n_select:
            break

        # Check distance to all selected
        seq = c.sequence.active_site_sequence or c.sequence.sequence
        is_diverse = True

        for sel in selected:
            sel_seq = sel.sequence.active_site_sequence or sel.sequence.sequence
            if len(seq) == len(sel_seq):
                dist = sum(1 for a, b in zip(seq, sel_seq) if a != b)
                if dist < min_hamming_distance:
                    is_diverse = False
                    break

        if is_diverse:
            selected.append(c)

    logger.info(
        f"Diversity selection: {len(selected)}/{n_select} candidates "
        f"(min_dist={min_hamming_distance})"
    )

    return selected
