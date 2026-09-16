"""
utils.py
--------
Small helper functions for loading images, converting them to NumPy arrays,
and saving results. Kept dependency-light: only NumPy and Pillow (PIL).

Author: Siri Bikkasani
Course:  CSc 8830 Computer Vision - Module 3
"""

from __future__ import annotations
import numpy as np
from PIL import Image


def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    """
    Load an image from disk as a float NumPy array with values in [0, 1].

    Parameters
    ----------
    path : str
        Path to the image file (jpg, png, ...).
    grayscale : bool
        If True, convert to a single-channel gray image (H, W).
        If False, keep 3 color channels (H, W, 3).

    Returns
    -------
    np.ndarray
        float64 array, values scaled to the range 0.0 .. 1.0.
    """
    img = Image.open(path)
    img = img.convert("L") if grayscale else img.convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Convert a float image in [0, 1] (or any range) back to displayable uint8.
    Values are clipped to [0, 1] first so nothing overflows.
    """
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def save_image(arr: np.ndarray, path: str) -> None:
    """Save a float image in [0, 1] to disk as a normal 8-bit image."""
    Image.fromarray(to_uint8(arr)).save(path)


def apply_per_channel(func, image: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Apply a 2-D filtering function to an image that may be gray (H, W)
    or color (H, W, 3). For color images we simply run the filter once
    on each of the R, G, B channels and stack the results back together.

    This lets the same convolution code work for both gray and color images.
    """
    if image.ndim == 2:                       # grayscale -> filter directly
        return func(image, *args, **kwargs)
    # color -> filter each channel independently, then re-stack
    channels = [func(image[:, :, c], *args, **kwargs) for c in range(image.shape[2])]
    return np.stack(channels, axis=2)
