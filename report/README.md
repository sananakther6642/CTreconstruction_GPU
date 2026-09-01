# CT Volume Reconstruction — Final Course Report

This directory contains the complete final course report for **Topic 1-2: CT Volume Reconstruction** for the GPULAB course at the University of Stuttgart (Institute of Computer Architecture and Computer Engineering, ITI).

## Authors
- **Sanan Akther** (Matriculation No. 3704154)
- **Divyansh Sharma** (Matriculation No. 3758980)
- **Joseph Maher Labib Estawro** (Matriculation No. 3699049)

## Contents
- `final_report.tex`: Complete LaTeX source code for the final report ($\ge 10$ pages).
  - Theoretical formulations (Cone-beam CT geometry, Forward Projection $\mathcal{F}$, Backprojection $\mathcal{B}$, MLEM, OSEM).
  - Software architecture evolution and CLI interface.
  - Correctness audits and boundary condition debugging.
  - CPU ray-tiling memory optimizations (`FP_TILE=32`).
  - OpenCL GPU kernel engineering and Phase B kernel fusions (`divide_preprocess_img`, `vol_update_img`).
  - Work-group tuning sweeps and negative results (register pressure, branch-free clamp regressions).
  - Algorithmic acceleration via Ordered Subsets EM (OSEM) with golden-ratio coprime ordering.
  - Full 100-epoch multi-platform benchmark tables (AMD Hawaii PRO & NVIDIA GeForce GTX 680) and float32 validation results.

## Compilation Instructions
To compile `final_report.tex` to PDF locally using `pdflatex`:
```bash
pdflatex final_report.tex
pdflatex final_report.tex
```

### Overleaf Compilation
1. Upload `final_report.tex` to Overleaf.
2. Select the **pdfLaTeX** compiler (default).
3. Click **Recompile**.
