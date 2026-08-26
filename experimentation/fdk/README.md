# fdk (unmerged, branch `fdk`)

## What it is

A separate reconstruction mode: single-pass analytical cone-beam FDK
(Feldkamp-Davis-Kress) filtered backprojection, as an alternative to the
project's main iterative MLEM approach. Implemented as `ct_fdk.c/h`
(CPU-side Cooley-Tukey radix-2 FFT ramp filter, reuses the existing GPU
`bp_opt` kernel for the actual backprojection — no separate FDK kernel
file needed).

Note: this predates and is unrelated to a classmate's separate FDK-based
course project seen alongside this repo during cleanup — that was an
independent implementation by another team, not a source for this one.

## Pipeline

```
1. Cosine weight each detector pixel: w(u,v) = SDD / sqrt(SDD^2 + u^2 + v^2)
2. Per-row 1D ramp filter (Ram-Lak) via zero-padded FFT: H(k) = |k| / (N_fft * du)
3. Single GPU backprojection using bp_opt kernel (same kernel MLEM uses)
4. Normalization: bp_opt applies pi/N_angles weighting
```

## Result (from the branch's own measurements)

FDK is **~240x faster than 100-epoch CPU MLEM** (single analytical pass vs
100 iterative passes) but substantially lower quality: **MSE ~5e-4 vs CPU**,
compared to MLEM's **~1.4e-7**. A real speed/quality trade-off, not a
strict improvement — useful as a fast initial estimate, not a replacement
for the iterative approach this project's grading centers on.

One real bug found and fixed during development: the ramp filter's scale
was originally under-scaled by `1/N_fft^2`; fixed to the correct
`px/(2*N_fft)` normalization (`0bbb3e6`).

## What's here

`src/ct_fdk.c`, `src/ct_fdk.h` — the FDK implementation. Reference copies;
to build/run, check out the `fdk` branch directly (this branch has no
shared git history with `features` — diverged early and was never
rebased — so these files can't be cleanly merged, only copied for
documentation).

## Disposition

Not merged. This project's grading and report center on the MLEM
implementation (CPU/gpu-buf/gpu-img/gpu-opt); FDK was an early separate
exploration, kept as a documented alternative-algorithm comparison point.
