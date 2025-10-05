import numpy as np


def _as_boxv(box):
    """Return a float[3] array for the box lengths."""
    return np.array([box] * 3, float) if np.isscalar(box) else np.asarray(box, float)


def minimum_image(rij, box):
    """
    Apply the minimum-image convention to a displacement vector rij.
    If box is None, return rij unchanged.
    """
    if box is None:
        return rij
    b = _as_boxv(box)
    return rij - b * np.round(rij / b)


def wrap_positions(x, box):
    """
    Wrap absolute positions x into the periodic box [0, L) in each dimension.
    If box is None, return x unchanged.
    """
    if box is None:
        return x
    b = _as_boxv(box)
    return x - b * np.floor(x / b)
