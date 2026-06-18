"""Geometry utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class Transform:
    """A 3D rigid body transformation (rotation + translation)."""
    rotation: np.ndarray
    translation: np.ndarray


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> Transform:
    """Compute the optimal rotation and translation to align mobile onto target.

    Uses the Kabsch algorithm (singular value decomposition) to find the
    rotation matrix that minimizes the RMSD between the point sets.

    Args:
        mobile: Nx3 array of coordinates to be transformed
        target: Nx3 array of target coordinates

    Returns:
        Transform containing the rotation matrix and translation vector
    """
    LOGGER.debug("Kabsch alignment: %d points", len(mobile))

    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    LOGGER.debug("Mobile centroid: (%.4f, %.4f, %.4f)", *mobile_center)
    LOGGER.debug("Target centroid: (%.4f, %.4f, %.4f)", *target_center)

    covariance = mobile_centered.T @ target_centered
    v, s, w_t = np.linalg.svd(covariance)

    LOGGER.debug("SVD singular values: %.4f, %.4f, %.4f", *s)

    d = np.sign(np.linalg.det(v @ w_t))
    d_matrix = np.diag([1.0, 1.0, d])
    rotation = v @ d_matrix @ w_t
    translation = target_center - rotation @ mobile_center

    LOGGER.debug("Determinant sign (reflection check): %.1f", d)

    return Transform(rotation=rotation, translation=translation)


def rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """Calculate the root mean square deviation between two point sets.

    Args:
        mobile: Nx3 array of coordinates
        target: Nx3 array of target coordinates

    Returns:
        RMSD in the same units as the input coordinates (typically Angstroms)
    """
    diff = mobile - target
    result = float(np.sqrt((diff * diff).sum() / len(diff)))
    LOGGER.debug("RMSD calculated: %.4f Angstroms", result)
    return result
