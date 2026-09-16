"""
kernels.py
----------
Functions that build the small filter "kernels" (also called masks or
point-spread functions) used to blur an image.

A blur kernel is just a little grid of weights. To blur a pixel we take a
weighted average of that pixel and its neighbours. Two classic choices:

  * Box (average) kernel : every weight is equal -> plain averaging.
  * Gaussian kernel      : weights follow a bell curve -> smoother, more
                           natural looking blur with fewer artifacts.

Every kernel we return is NORMALISED (its weights sum to 1.0). That keeps the
overall brightness of the image unchanged after blurring.

Author: Siri Bikkasani
Course:  CSc 8830 Computer Vision - Module 3
"""

from __future__ import annotations
import numpy as np


def box_kernel(size: int) -> np.ndarray:
    """
    Build a size x size box (averaging) kernel.

    Example (size=3):
        1/9 * [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("kernel size must be a positive ODD integer (e.g. 3, 5, 7)")
    k = np.ones((size, size), dtype=np.float64)
    return k / k.sum()          # normalise so weights add up to 1


def gaussian_kernel(size: int, sigma: float | None = None) -> np.ndarray:
    """
    Build a size x size Gaussian kernel.

    The 2-D Gaussian is:  G(x, y) = exp( -(x^2 + y^2) / (2 * sigma^2) )
    We evaluate it on a grid centred at (0, 0), then normalise so the
    weights sum to 1.

    Parameters
    ----------
    size : int
        Odd kernel width/height (e.g. 3, 5, 7, ...). Bigger = more blur.
    sigma : float or None
        Standard deviation of the bell curve. If None, a sensible default
        tied to the kernel size is used (the common rule sigma = size/6,
        which puts +/-3 sigma at the kernel edges).

    Returns
    -------
    np.ndarray
        A normalised (size x size) Gaussian kernel.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("kernel size must be a positive ODD integer (e.g. 3, 5, 7)")
    if sigma is None or sigma <= 0:
        sigma = size / 6.0                       # default spread

    half = size // 2
    # 1-D coordinate axis: e.g. size=5 -> [-2, -1, 0, 1, 2]
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)                  # 2-D coordinate grids
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()                  # normalise to sum = 1


def make_kernel(kind: str, size: int, sigma: float | None = None) -> np.ndarray:
    """
    Convenience wrapper used by the rest of the app.

    kind : "box" or "gaussian"
    """
    kind = kind.lower()
    if kind == "box":
        return box_kernel(size)
    if kind == "gaussian":
        return gaussian_kernel(size, sigma)
    raise ValueError(f"unknown kernel kind: {kind!r} (use 'box' or 'gaussian')")
