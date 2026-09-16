"""
spatial.py
----------
Image blurring in the SPATIAL domain, implemented from scratch (i.e. we do NOT
call cv2.filter2D / cv2.GaussianBlur). This is the "filtering approach" the
assignment asks us to implement.

Blurring = 2-D convolution of the image with a small blur kernel.
For each output pixel we slide the (flipped) kernel over the image, multiply
overlapping values, and sum them up.

Two implementations are provided:
  1. convolve2d_naive  - the plain textbook nested-loop version. Easy to read,
                         but slow. Good for understanding + small examples.
  2. convolve2d        - a fast, vectorised "shift-and-add" version that gives
                         the exact same result. Used by the app on real images.

Boundary handling ("padding"):
  * "zero"    : treat pixels outside the image as 0. This makes the operation
                a true LINEAR convolution, which is exactly what the FFT method
                computes -- so we use "zero" when proving the two methods match.
  * "reflect" : mirror the edge pixels. Looks nicer (no dark border) for a
                pure visual blur.

Author: Siri Bikkasani
Course:  CSc 8830 Computer Vision - Module 3
"""

from __future__ import annotations
import numpy as np


def pad_image(image: np.ndarray, pad_h: int, pad_w: int, mode: str = "zero") -> np.ndarray:
    """Add a border of width (pad_h, pad_w) around the image."""
    if mode == "zero":
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    if mode == "reflect":
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    raise ValueError("mode must be 'zero' or 'reflect'")


def convolve2d_naive(image: np.ndarray, kernel: np.ndarray, mode: str = "zero") -> np.ndarray:
    """
    The plain, easy-to-follow version of 2-D convolution (nested loops).

    This is O(H * W * k * k) and is slow on big images, but it makes the
    definition of convolution crystal clear. We flip the kernel so that this
    is true convolution (not correlation).
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    kflip = kernel[::-1, ::-1]                     # flip kernel -> convolution
    padded = pad_image(image, ph, pw, mode)
    H, W = image.shape
    out = np.zeros((H, W), dtype=np.float64)

    for y in range(H):                            # for every output row
        for x in range(W):                        # for every output column
            region = padded[y:y + kh, x:x + kw]   # kh x kw neighbourhood
            out[y, x] = np.sum(region * kflip)    # weighted sum
    return out


def convolve2d(image: np.ndarray, kernel: np.ndarray, mode: str = "zero") -> np.ndarray:
    """
    Fast, vectorised 2-D convolution -- identical result to convolve2d_naive.

    Idea ("shift and add"): instead of looping over every pixel, we loop over
    the (few) kernel positions. For each kernel weight we take the whole padded
    image, shift it, multiply by that single weight, and accumulate. A kxk
    kernel therefore needs only k*k array operations regardless of image size.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    kflip = kernel[::-1, ::-1]                     # flip -> true convolution
    padded = pad_image(image, ph, pw, mode)
    H, W = image.shape
    out = np.zeros((H, W), dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            # padded[i:i+H, j:j+W] is the image shifted by (i, j).
            out += kflip[i, j] * padded[i:i + H, j:j + W]
    return out


def blur_spatial(image: np.ndarray, kernel: np.ndarray, mode: str = "zero") -> np.ndarray:
    """
    Public entry point: blur a (grayscale) image by convolving it with the
    given blur kernel in the spatial domain.
    """
    return convolve2d(image, kernel, mode=mode)
