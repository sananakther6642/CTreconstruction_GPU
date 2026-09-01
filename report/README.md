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

### Command-Line (Local or Lab Server)
Run the root Makefile target:
```bash
make report
```
Or manually run `pdflatex`:
```bash
cd report
pdflatex final_report.tex
pdflatex final_report.tex
```

### Overleaf Compilation
1. Create a new Blank Project in Overleaf.
2. Upload `final_report.tex` into the project root.
3. Select the standard **pdfLaTeX** compiler.
4. Click **Recompile**.
   *(The document is 100% self-contained with no external image/bibtex dependencies and generates the complete $\ge 10$-page report PDF).*

## Final Submission Checklist
- [x] Length: $\ge 10$ pages (with structured sections, equations, tables, benchmarks).
- [x] Theoretical foundations (Cone-beam geometry, Feldkamp weights, Poisson model, MLEM, OSEM).
- [x] Complete hardware results from both platforms (AMD Hawaii PRO & NVIDIA GeForce GTX 680).
- [x] Numerical correctness validation against CPU reference ($\text{MSE} \approx 10^{-10}\text{--}10^{-7}$).
- [x] Code architecture evolution, negative results, and work-group profiling analysis.
