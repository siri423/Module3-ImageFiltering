"""
frequency.py
------------
Image blurring in the FREQUENCY (Fourier) domain.

The Convolution Theorem says:

        convolution in space  <===>  multiplication in frequency
        f * g                 <===>  F . G          (element-wise product)

So instead of sliding a kernel over the image (spatial.py), we can:
    1. take the Fourier transform of the image      -> F
    2. take the Fourier transform of the kernel      -> G
    3. multiply them element-wise                    -> F . G
    4. take the inverse Fourier transform            -> the blurred image

This module implements exactly that, and is written so its result matches
the from-scratch spatial convolution (spatial.blur_spatial with mode="zero")
to within tiny floating-point rounding error. That match is our experimental
proof of the theorem.

Important detail -- LINEAR vs CIRCULAR convolution:
The plain FFT of two arrays gives *circular* convolution (the kernel wraps
around the image edges). To reproduce ordinary (linear) zero-padded
convolution we must first zero-pad BOTH arrays to size (H+kh-1, W+kw-1).
Then the circular convolution equals the linear one, and we crop the centre.

Author: Siri Bikkasani
Course:  CSc 8830 Computer Vision - Module 3
"""

from __future__ import annotations
import numpy as np


def blur_frequency(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Blur a grayscale image by MULTIPLYING in the Fourier domain.

    Returns an (H, W) image that matches the zero-padded spatial convolution
    of `image` with `kernel`.
    """
    H, W = image.shape
    kh, kw = kernel.shape

    # Size needed for LINEAR convolution (avoids wrap-around / circular effects)
    full_h, full_w = H + kh - 1, W + kw - 1

    # Step 1 & 2: Fourier transforms, each zero-padded to the full size.
    F = np.fft.fft2(image, s=(full_h, full_w))     # transform of the image
    G = np.fft.fft2(kernel, s=(full_h, full_w))    # transform of the kernel

    # Step 3: multiplication in frequency == convolution in space
    product = F * G

    # Step 4: inverse transform, keep the real part (imaginary part is ~0)
    full = np.real(np.fft.ifft2(product))

    # Crop the centre region so the output is the same (H, W) as the spatial
    # method. For odd kernels the offset is exactly half the kernel size.
    ph, pw = kh // 2, kw // 2
    return full[ph:ph + H, pw:pw + W]


def magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Return a log-magnitude spectrum of an image, shifted so the zero frequency
    (DC term) is in the centre. Handy for VISUALISING what the transform looks
    like. Output is scaled to [0, 1] for display.
    """
    F = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F)                    # move low freqs to centre
    mag = np.log1p(np.abs(F_shift))                # log so we can see detail
    mag -= mag.min()
    if mag.max() > 0:
        mag /= mag.max()
    return mag


def kernel_frequency_response(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Visualise the kernel's frequency response (its transfer function) at the
    given image `shape`. A blur kernel acts as a LOW-PASS filter: this image is
    bright in the centre (low frequencies kept) and dark at the edges (high
    frequencies, i.e. fine detail, removed). Scaled to [0, 1] for display.
    """
    G = np.fft.fft2(kernel, s=shape)
    G_shift = np.fft.fftshift(G)
    mag = np.abs(G_shift)
    if mag.max() > 0:
        mag /= mag.max()
    return mag
