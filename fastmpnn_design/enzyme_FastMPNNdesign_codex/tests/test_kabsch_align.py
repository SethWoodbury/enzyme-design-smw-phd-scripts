import numpy as np

from utils.geometry import kabsch_align, rmsd


def test_kabsch_identity():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transform = kabsch_align(points, points)
    aligned = (transform.rotation @ points.T).T + transform.translation
    assert rmsd(aligned, points) < 1e-6
