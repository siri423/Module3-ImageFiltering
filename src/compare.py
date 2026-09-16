"""
compare.py
----------
Runs the actual EXPERIMENT that validates the Convolution Theorem:

    blur the SAME image with the SAME kernel using
        (a) from-scratch spatial convolution   (spatial.py)
        (b) Fourier-domain multiplication       (frequency.py)
    then measure how close the two results are.

If the theorem holds, the two images should be identical up to tiny
floating-point rounding error (max difference ~1e-15). This module reports
that difference and can also be run from the command line to generate all the
figures used in the report.

Author: Siri Bikkasani
Course:  CSc 8830 Computer Vision - Module 3
"""

from __future__ import annotations
import time
import numpy as np

from .kernels import make_kernel
from .spatial import blur_spatial
from .frequency import blur_frequency
from .utils import apply_per_channel


def error_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Compare two images and return numbers describing how different they are.

    max_abs_diff : the single largest pixel difference (the strongest evidence;
                   a value near machine epsilon means "the same image").
    mse          : mean squared error, averaged over all pixels.
    psnr_db      : peak signal-to-noise ratio in decibels. Infinite (or very
                   large) when the images are effectively identical.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    mse = float(np.mean(diff**2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))   # peak = 1.0 for [0,1] images
    return {"max_abs_diff": max_abs, "mse": mse, "psnr_db": psnr}


def compare_methods(image: np.ndarray,
                    kind: str = "gaussian",
                    size: int = 15,
                    sigma: float | None = None) -> dict:
    """
    Blur `image` both ways and bundle up everything the app/report needs.

    Works for grayscale (H, W) and color (H, W, 3) images.

    Returns a dict with:
        kernel, spatial, frequency, diff (absolute), metrics,
        time_spatial, time_frequency
    """
    kernel = make_kernel(kind, size, sigma)

    t0 = time.perf_counter()
    spatial = apply_per_channel(blur_spatial, image, kernel, mode="zero")
    t1 = time.perf_counter()
    frequency = apply_per_channel(blur_frequency, image, kernel)
    t2 = time.perf_counter()

    metrics = error_metrics(spatial, frequency)
    return {
        "kernel": kernel,
        "spatial": spatial,
        "frequency": frequency,
        "diff": np.abs(spatial - frequency),
        "metrics": metrics,
        "time_spatial": t1 - t0,
        "time_frequency": t2 - t1,
    }


# --------------------------------------------------------------------------
# Command-line entry point: generates all report figures into ../outputs/
# --------------------------------------------------------------------------
def _main() -> None:
    import argparse
    import os
    import matplotlib
    matplotlib.use("Agg")                          # no display needed
    import matplotlib.pyplot as plt
    from .utils import load_image, save_image, to_uint8
    from .frequency import magnitude_spectrum

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="Spatial vs Frequency blur experiment")
    ap.add_argument("--image", default=os.path.join(here, "data", "sample.jpg"))
    ap.add_argument("--kind", default="gaussian", choices=["gaussian", "box"])
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(here, "outputs"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img = load_image(args.image, grayscale=True)
    res = compare_methods(img, args.kind, args.size, args.sigma)
    m = res["metrics"]

    print(f"Image           : {args.image}  {img.shape}")
    print(f"Kernel          : {args.kind} {args.size}x{args.size}")
    print(f"max |diff|      : {m['max_abs_diff']:.3e}")
    print(f"MSE             : {m['mse']:.3e}")
    print(f"PSNR (dB)       : {m['psnr_db']}")
    print(f"time spatial    : {res['time_spatial']*1000:.1f} ms")
    print(f"time frequency  : {res['time_frequency']*1000:.1f} ms")

    # Save the plain result images
    save_image(img, os.path.join(args.out, "01_original.png"))
    save_image(res["spatial"], os.path.join(args.out, "02_blur_spatial.png"))
    save_image(res["frequency"], os.path.join(args.out, "03_blur_frequency.png"))

    # Figure A: side-by-side proof (original, spatial, frequency, diff)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(res["spatial"], cmap="gray"); ax[1].set_title("Spatial convolution")
    ax[2].imshow(res["frequency"], cmap="gray"); ax[2].set_title("Frequency multiplication")
    d = ax[3].imshow(res["diff"], cmap="magma")
    ax[3].set_title(f"|difference|  (max={m['max_abs_diff']:.1e})")
    fig.colorbar(d, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"Spatial vs Frequency blur  -  {args.kind} {args.size}x{args.size}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_comparison.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # Figure B: the frequency picture (image spectrum, kernel response, result)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    ax[0].imshow(magnitude_spectrum(img), cmap="gray")
    ax[0].set_title("Image spectrum |F(u,v)|  (log)")
    kresp = np.fft.fftshift(np.abs(np.fft.fft2(res["kernel"], s=img.shape)))
    kresp = kresp / kresp.max()
    ax[1].imshow(kresp, cmap="gray"); ax[1].set_title("Kernel response |G(u,v)|  (low-pass)")
    ax[2].imshow(magnitude_spectrum(res["frequency"]), cmap="gray")
    ax[2].set_title("Blurred spectrum |F.G|  (log)")
    for a in ax:
        a.axis("off")
    fig.suptitle("Why blurring is low-pass filtering: high frequencies are removed",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_spectra.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures written to: {args.out}")


if __name__ == "__main__":
    _main()
