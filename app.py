"""
app.py  -  CSc 8830 Computer Vision, Module 3
=============================================
Streamlit WEB APPLICATION. Every part of the assignment is reachable from this
single page through the tabs on screen:

    Overview            - what the assignment is + link to the GitHub repo
    1. Blur an image    - implement image blurring using a filtering approach
    2. Spatial = Freq   - prove convolution in space == multiplication in freq
    3. Theory           - the Convolution Theorem, written out

Run it with:
    streamlit run app.py
then open the URL it prints (usually http://localhost:8501).

Author: Siri Bikkasani
"""

import io
import os
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# our own modules (the "filtering approach" lives here, no OpenCV blur used)
from src.kernels import make_kernel
from src.spatial import blur_spatial
from src.frequency import blur_frequency, magnitude_spectrum
from src.compare import compare_methods
from src.utils import load_image, to_uint8, apply_per_channel

# ---- change this to YOUR repo once you create it on GitHub -----------------
GITHUB_URL = "https://github.com/siri423/Module3-ImageFiltering"
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample.jpg")

st.set_page_config(page_title="CSc 8830 - Module 3 - Image Filtering",
                   layout="wide")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_upload(file_bytes: bytes, grayscale: bool) -> np.ndarray:
    """Turn uploaded file bytes into a float image in [0,1]."""
    from PIL import Image
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("L") if grayscale else img.convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def get_input_image(grayscale: bool, key: str) -> np.ndarray:
    """Sidebar-independent image chooser: upload OR the built-in sample."""
    up = st.file_uploader("Upload your own image (optional)",
                          type=["jpg", "jpeg", "png", "bmp"], key=key)
    if up is not None:
        return _read_upload(up.getvalue(), grayscale)
    return load_image(SAMPLE_PATH, grayscale=grayscale)


def kernel_controls(prefix: str):
    """Shared kernel type / size / sigma controls. Returns (kind, size, sigma)."""
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Kernel type", ["gaussian", "box"], key=prefix + "kind")
    size = c2.slider("Kernel size (odd)", 3, 41, 15, step=2, key=prefix + "size")
    sigma = None
    if kind == "gaussian":
        sigma = c3.slider("Gaussian sigma", 0.5, 15.0, float(size) / 6.0,
                          step=0.5, key=prefix + "sigma")
    else:
        c3.markdown("*(box kernel has no sigma)*")
    return kind, size, sigma


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("CSc 8830 - Computer Vision - Module 3")
st.subheader("Image Blurring by Filtering: Spatial Domain vs Fourier Domain")

tabs = st.tabs(["Overview",
                "1. Blur an image",
                "2. Spatial = Frequency (proof)",
                "3. Theory"])

# ==========================================================================
# TAB 0 - Overview
# ==========================================================================
with tabs[0]:
    st.markdown(f"""
### What this app does
This web application implements **image blurring using a filtering approach**
and then demonstrates a fundamental result of signal processing, the
**Convolution Theorem**:

> Convolving an image with a filter in the **spatial domain** produces exactly
> the same result as **multiplying** their Fourier transforms in the
> **frequency domain**.

**Assignment parts, and where to find them here:**

| Requirement | Tab |
|---|---|
| Implement image blurring (filtering approach) | **1. Blur an image** |
| Show spatial filtering == Fourier-domain equivalent | **2. Spatial = Frequency** |
| Show convolution in space == multiplication in frequency | **2. Spatial = Frequency** |
| Theory / derivation | **3. Theory** (full write-up is in the PDF report) |

**GitHub repository:** [{GITHUB_URL}]({GITHUB_URL})

**How it works, briefly:** the blur is done with a kernel we build ourselves
(box or Gaussian) and a 2-D convolution we wrote by hand in `src/spatial.py`
(no `cv2.GaussianBlur`). The Fourier version in `src/frequency.py` takes the
FFT of the image and the kernel, multiplies them, and inverse-transforms.
Tab 2 blurs the same image both ways and reports the difference, which is on
the order of 1e-15 (machine precision) - i.e. the two results are identical.
""")
    st.info("Tip for your screen recording: walk through the tabs left to "
            "right - upload an image and blur it, then open Tab 2 to show the "
            "two methods agree, then Tab 3 for the theory.")

# ==========================================================================
# TAB 1 - Blur an image (the filtering implementation)
# ==========================================================================
with tabs[1]:
    st.markdown("### Blur an image using a filtering approach")
    st.caption("The blur is a 2-D convolution of the image with a blur kernel, "
               "implemented from scratch in `src/spatial.py`.")

    color = st.checkbox("Process in color (RGB)", value=False, key="blur_color")
    img = get_input_image(grayscale=not color, key="blur_upload")
    kind, size, sigma = kernel_controls("blur_")
    kernel = make_kernel(kind, size, sigma)

    blurred = apply_per_channel(blur_spatial, img, kernel, mode="reflect")

    c1, c2 = st.columns(2)
    c1.markdown("**Original**")
    c1.image(to_uint8(img), use_container_width=True)
    c2.markdown(f"**Blurred** - {kind} {size}x{size}"
                + (f", sigma={sigma:.1f}" if sigma else ""))
    c2.image(to_uint8(blurred), use_container_width=True)

    # show the kernel as a heatmap so students can see the weights
    with st.expander("See the blur kernel (the filter weights)"):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(kernel, cmap="viridis")
        ax.set_title(f"{kind} kernel {size}x{size}  (weights sum to 1)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)
        st.write(f"Kernel sum = {kernel.sum():.6f} (normalised), "
                 f"min = {kernel.min():.5f}, max = {kernel.max():.5f}")

    # let the user download the blurred result
    from PIL import Image as _PILImage
    _buf = io.BytesIO()
    _PILImage.fromarray(to_uint8(blurred)).save(_buf, format="PNG")
    st.download_button("Download blurred image (PNG)",
                       data=_buf.getvalue(),
                       file_name=f"blurred_{kind}_{size}.png",
                       mime="image/png", key="dl_blurred")

# ==========================================================================
# TAB 2 - Spatial == Frequency (the proof / experiment)
# ==========================================================================
with tabs[2]:
    st.markdown("### Proof by experiment: spatial convolution == frequency multiplication")
    st.caption("Same image, same kernel, blurred two different ways. If the "
               "Convolution Theorem is true, the two results must be identical.")

    img2 = get_input_image(grayscale=True, key="proof_upload")
    kind2, size2, sigma2 = kernel_controls("proof_")

    res = compare_methods(img2, kind2, size2, sigma2)
    m = res["metrics"]

    # numeric evidence
    a, b, c = st.columns(3)
    a.metric("max |difference|", f"{m['max_abs_diff']:.2e}",
             help="Largest pixel disagreement. ~1e-15 = machine precision.")
    b.metric("MSE", f"{m['mse']:.2e}")
    c.metric("PSNR (dB)", "inf" if np.isinf(m["psnr_db"]) else f"{m['psnr_db']:.1f}")

    if m["max_abs_diff"] < 1e-9:
        st.success("The two images are identical to floating-point precision. "
                   "Convolution in space == multiplication in frequency. QED.")

    # side-by-side visual
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(img2, cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(res["spatial"], cmap="gray"); ax[1].set_title("Spatial convolution")
    ax[2].imshow(res["frequency"], cmap="gray"); ax[2].set_title("Frequency multiplication")
    d = ax[3].imshow(res["diff"], cmap="magma")
    ax[3].set_title(f"|difference| (max={m['max_abs_diff']:.1e})")
    fig.colorbar(d, ax=ax[3], fraction=0.046)
    for x in ax:
        x.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(f"Timing on this image: spatial {res['time_spatial']*1000:.0f} ms, "
               f"frequency {res['time_frequency']*1000:.0f} ms.")

    # frequency-domain picture
    with st.expander("Show the frequency-domain view (why blur = low-pass filter)"):
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
        ax[0].imshow(magnitude_spectrum(img2), cmap="gray")
        ax[0].set_title("Image spectrum |F| (log)")
        kresp = np.fft.fftshift(np.abs(np.fft.fft2(res["kernel"], s=img2.shape)))
        kresp = kresp / kresp.max()
        ax[1].imshow(kresp, cmap="gray")
        ax[1].set_title("Kernel response |G| (low-pass)")
        ax[2].imshow(magnitude_spectrum(res["frequency"]), cmap="gray")
        ax[2].set_title("Blurred spectrum |F.G| (log)")
        for x in ax:
            x.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.write("The kernel response is bright in the centre (low frequencies "
                 "pass) and dark at the edges (high frequencies removed). "
                 "Multiplying by it deletes fine detail - that IS blurring.")

# ==========================================================================
# TAB 3 - Theory
# ==========================================================================
with tabs[3]:
    st.markdown(r"""
### The Convolution Theorem

**Spatial convolution.** Blurring an image $f$ with a kernel $g$ is a 2-D
convolution:

$$(f * g)(x,y) = \sum_{s}\sum_{t} f(s,t)\, g(x-s,\, y-t).$$

**Fourier transform.** The 2-D Discrete Fourier Transform (DFT) of an
$M\times N$ image is

$$F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y)\,
e^{-j 2\pi (ux/M + vy/N)}.$$

**The theorem.**

$$\boxed{\; f * g \;\;\Longleftrightarrow\;\; F \cdot G \;}$$

Convolution in the spatial domain equals **element-wise multiplication** in the
frequency domain (and vice-versa). So blurring can be done either way:

$$f * g \;=\; \mathcal{F}^{-1}\{\, \mathcal{F}\{f\}\cdot \mathcal{F}\{g\}\,\}.$$

**Why a blur kernel blurs.** A box or Gaussian kernel is a **low-pass filter**:
its transform $G(u,v)$ is large near the origin (low frequencies) and small far
away (high frequencies). Multiplying $F\cdot G$ keeps the smooth, low-frequency
content and suppresses the fine, high-frequency detail - which is exactly what
"blur" means.

**One practical caveat - linear vs circular convolution.** The plain FFT gives
*circular* convolution (the kernel wraps around image edges). To reproduce
ordinary zero-padded (linear) convolution we zero-pad both arrays to size
$(M+k-1)\times(N+k-1)$ before transforming, then crop the centre. This is done
in `src/frequency.py`, which is why Tab 2 shows agreement to ~$10^{-15}$.

---
The **full typed derivation** (including a small worked-by-hand example) is in
the PDF report submitted with this assignment.
""")
