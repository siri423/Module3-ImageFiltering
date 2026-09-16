# CSc 8830 Computer Vision — Module 3
## Image Blurring by Filtering: Spatial Domain vs. Fourier Domain

A web application that blurs an image using a **filtering approach** (convolution
with a blur kernel, implemented from scratch) and then demonstrates the
**Convolution Theorem**: convolving in the spatial domain produces the *same*
result as multiplying the Fourier transforms in the frequency domain.

**Author:** Siri Bikkasani · Georgia State University

---

## What each assignment requirement maps to

| Requirement | Where it lives |
|---|---|
| Implement image blurring (filtering approach) | `src/spatial.py` + web-app tab **“1. Blur an image”** |
| Show spatial filtering == Fourier-domain equivalent | `src/frequency.py`, `src/compare.py` + tab **“2. Spatial = Frequency”** |
| Show convolution in space == multiplication in frequency | tab **“2. Spatial = Frequency”** + tab **“3. Theory”** |
| Theory / derivation (typed) | `Module3_Report.pdf` (Sections 3–4) + tab **“3. Theory”** |
| Working demo as a **web application** | `app.py` (Streamlit) |

---

## Folder structure
```
Module3_ImageFiltering/
├── app.py               # Streamlit web app — every part is accessible here
├── requirements.txt
├── README.md
├── Module3_Report.pdf   # typed report: theory, worked example, evidence
├── src/
│   ├── kernels.py       # build box & Gaussian blur kernels
│   ├── spatial.py       # 2-D convolution from scratch (the blur)
│   ├── frequency.py     # FFT-based filtering (multiply in frequency)
│   ├── compare.py       # spatial-vs-frequency experiment + figure generation
│   └── utils.py         # image load/save helpers
├── data/
│   └── sample.jpg       # built-in test image (upload your own in the app too)
└── outputs/             # result figures used by the report
```

## Setup
```bash
pip install -r requirements.txt
```

## Run the web application (this is what you screen-record)
```bash
streamlit run app.py
```
Then open the URL it prints (usually http://localhost:8501) and walk through the
tabs left to right:

1. **Overview** — what the app does + the GitHub link.
2. **1. Blur an image** — upload an image (or use the built-in sample), pick a
   kernel (box or Gaussian) and size, and see it blurred. This is the filtering
   implementation.
3. **2. Spatial = Frequency (proof)** — the same image is blurred both ways and
   the app reports the `max |difference|` (about `1e-15`, i.e. identical) plus a
   side-by-side view and the frequency-domain spectra.
4. **3. Theory** — the Convolution Theorem written out.

## Run from the command line (optional — reproduces the report figures)
```bash
# Gaussian 15x15 blur, prints error metrics, writes figures to outputs/
python -m src.compare --kind gaussian --size 15

# Box 9x9 blur
python -m src.compare --kind box --size 9
```
Example output:
```
max |diff|      : 2.665e-15      <- spatial and frequency results are identical
MSE             : 2.226e-31
PSNR (dB)       : 306.5
```

## How it works, in one paragraph
A blur kernel (box or Gaussian, built in `kernels.py`) is a small grid of weights
that sum to 1. Blurring convolves the image with that kernel. `spatial.py` does
this directly with a from-scratch, vectorised “shift-and-add” convolution (no
`cv2.GaussianBlur`). `frequency.py` does the same job in the Fourier domain: it
takes the FFT of the image and of the kernel, multiplies them element-wise, and
inverse-transforms — because convolution in space equals multiplication in
frequency. `compare.py` runs both on the same input and measures the difference,
which is on the order of machine precision (`~1e-15`), confirming the theorem.

## Dependencies
Python 3.9+, `streamlit`, `numpy`, `pillow`, `matplotlib` (see `requirements.txt`).
