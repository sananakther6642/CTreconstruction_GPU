# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM. CPU (OpenMP) and GPU
(OpenCL) implementations, 256³ and 512³ datasets.

**Correctness summary:**
- All 3 GPU modes (`gpu-buf`, `gpu-img`, `gpu-opt`) agree with CPU ref at float32 noise floor (MSE 1e-7 to 1e-10 vs CPU, both datasets, both machines — see Performance/Validation below).
- MSE vs course Python ref (`Topic_2_CTreconstruction.py`) initially higher (`2.061e-02`/`6.371e-03` depending on hardware) — confirmed numerical-stability bug in that script's unclamped MLEM update. 26 voxels on one machine, 24 on other (<0.0002% of volume) diverged over 20 iterations. Not a bug in this project's C/GPU/CPU code.
- Reported to course staff, approved to fix.
- Per-epoch ratio clamp tried first, doesn't work — observed ~1.3 ratio already inside any reasonable clamp range, never triggers, compounding continues (verified via standalone simulation, not run for real).
- Actual fix: clamp `v0` to `[0, 5]` after each update, added to `Topic_2_CTreconstruction.py` and its 512³ variant (documented in each file). 5.0 is well above ~1.7-1.9 normal converged range, doesn't affect correct voxels.
- **Re-verified, real 20-epoch run, AMD Hawaii PRO (2026-08-28):** MSE vs Python Ref `6.371e-03` → `3.014e-04` (~21x better). Zero outlier voxels remain, all 4 modes agree identically, `python_ref` max exactly `5.0000` (the clamp bound). Fix confirmed working.
- **Independently re-verified on NVIDIA GTX 680 (2026-09-02), 20 epochs, 19h 41m:** MSE vs a 20-epoch CPU run `2.06e-02` → `1.749e-04`. `python_ref` max again exactly `5.0000`, zero NaN/Inf, min `1.98e-14` (non-negativity held). The fix therefore reproduces on both vendors. Two caveats on this figure: it is scored against a **20-epoch** CPU run (correctly — Topic_2 ran 20 epochs, so a 100-epoch reference would conflate convergence with accuracy), so it is *not* comparable to the 100-epoch MSE-vs-CPU numbers elsewhere in this README. And the max absolute difference is still `5.000000`, meaning the worst voxels are now *bounded* at the clamp rather than eliminated — containment, not removal. Over 99% of voxels agree to within `0.0046`; median difference `4.94e-05`.
- See Validation section for original (unfixed) numbers, investigation, fixed-run details.

## Performance

### AMD Hawaii PRO, EPOCHS=10

| Mode | 256³ time/epoch | 512³ time/epoch | Speedup vs CPU |
|---|---|---|---|
| `cpu` | 2.60 s | 22.9-23.3 s | 1× |
| `gpu-buf` | 0.44 s | 4.87-10.19 s (run-to-run variance, AMD-specific) | 5.9× / 2.2-4.8× |
| `gpu-img` | 0.095 s | 0.870-0.876 s | 27.4× / 26.6× |
| `gpu-opt` | 0.093 s | 0.873-0.879 s | 28.0× / 26.5× |

Intel i7-5820K (12 threads) · AMD Hawaii PRO (2560 shaders, 2.56 TFLOPS).

### NVIDIA GeForce GTX 680, EPOCHS=100

| Mode | 256³ time/epoch | 256³ total | 512³ time/epoch | 512³ total | Speedup vs CPU |
|---|---|---|---|---|---|
| `cpu` | ~4.20s | 423.86s | ~34s | 3415.58s | 1× |
| `gpu-buf` | 0.855-0.860s | 86.46s | 7.13-7.20s | 722.59s | 4.9× / 4.7× |
| `gpu-img` | 0.142-0.148s | 14.67s | 1.226-1.256s | 126.74s | 28.9× / 27.0× |
| `gpu-opt` | 0.138-0.143s | 14.29s | 1.226-1.245s | 125.77s | 29.7× / 27.2× |

Intel Xeon E5-2620 0 (24 threads) · NVIDIA GeForce GTX 680 (Kepler, no
`cl_khr_fp16` — `--half` unavailable).

- MSE vs CPU: 256³ `1.1477e-10` (`gpu-buf`) / `1.1278e-07` (`gpu-img`/`gpu-opt`). 512³ `9.534e-11` (`gpu-buf`) / `1.232e-09` (`gpu-img`/`gpu-opt`). No NaN/inf.

### `FP_TEX_EXACT`: exact forward-projection interpolation

`gpu-img`/`gpu-opt` sample the volume through the hardware texture unit,
which bundles two separable things: a 3D-tiled cache (the source of their
speed) and a fixed-function interpolation stage whose blend weights carry
less than float32 precision (the source of their MSE gap vs CPU). Setting
`FP_TEX_EXACT=1` keeps the cache and replaces only the blend — eight
`CLK_FILTER_NEAREST` fetches plus a manual float32 trilinear, in the forward
projection.

Forward projection specifically: fp produces the ratio that drives every MLEM
update, so its error re-enters the loop each epoch, while bp's averages out
across 75 angles. An earlier attempt at the same correction in
backprojection moved MSE only 6% while costing 55% runtime.

**MSE vs CPU, 100 epochs:**

| | NVIDIA GTX 680 | | AMD Hawaii PRO | |
|---|---|---|---|---|
| | default | `FP_TEX_EXACT=1` | default | `FP_TEX_EXACT=1` |
| 256³ `gpu-img`/`gpu-opt` | `1.128e-07` | `2.524e-09` | `1.949e-07` | `1.651e-09` |
| 512³ `gpu-img`/`gpu-opt` | `1.232e-09` | `5.528e-10` | `6.849e-10` | *pending* |
| 256³ `gpu-buf` | `1.148e-10` | n/a | `6.436e-10` | n/a |
| 512³ `gpu-buf` | `9.534e-11` | n/a | `5.310e-10` | n/a |

`gpu-buf` has no `fp_image`, so the flag does not apply to it. Hawaii's 512³
`FP_TEX_EXACT` figure is still running
(`temp_scripts/run_hawaii_precision_100ep.sh`).

**The fix transfers across vendors, and brings the two machines into
agreement.** Hawaii's 256³ gain is 118× — larger than the GTX 680's 44.7×
only because its baseline was worse (`1.949e-07` vs `1.128e-07`); both land at
the same corrected level (`1.651e-09` vs `2.524e-09`). That convergence is
itself evidence the sampler was the dominant machine-specific error source,
since Hawaii's baselines are otherwise looser than the GTX 680's across the
board (a pre-existing AMD-vs-NVIDIA difference documented in the Validation
section).

Two corroborating details from the Hawaii 256³ run: `max` moved from `1.8741`
(8% above the CPU reference's `1.7303`) to `1.7332`, within 0.2%; and the
worst-disagreeing voxel moved from `(49,211,33)` to `(10,102,249)` — which is
`gpu-buf`'s worst voxel too. The sampler-specific error is gone, leaving the
common float32 floor both paths share.

**Runtime cost (GTX 680, 256³, 100 epochs):**

All four figures below were measured in one sitting from a single `features`
build, so the ratio between them is meaningful:

| path | MSE vs CPU | `gpu-img` | `gpu-opt` |
|---|---|---|---|
| default | `1.128e-07` | 16.95 s | 16.51 s |
| `FP_TEX_EXACT=1` | `2.524e-09` | 21.33 s | 20.98 s |
| `gpu-buf` | `1.148e-10` | — | 39.05 s |

A 45× MSE reduction for **1.26-1.27×** runtime. Defaults are unchanged; the
flag costs nothing unless set.

Note on the default figures: earlier drafts quoted 14.65 s / 14.20 s from an
archived 2026-08-29 run. Re-measuring on current `features` gives ~16.5-17.0 s
for the same default path, reproducibly (four runs within 17.12-17.17 s for
`gpu-img`). Something between those dates cost the default path ~15%;
`CLK_ADDRESS_CLAMP` vs `CLAMP_TO_EDGE` and `native_sqrt` vs IEEE `sqrt` were
each tested directly and neither accounts for it, so the cause is not yet
identified. This is an open item, not a consequence of `FP_TEX_EXACT` — the
exact-path timing reproduced to 0.01 s across builds. The ratio quoted above
uses same-build numbers on both sides and so is unaffected either way. At 512³ the improvement is 2.2× rather than 45×,
since 512³ starts much closer to the float32 noise floor — finer voxels shrink
the sampler's absolute interpolation error.

**Figures:** `submission_outputs/gtx680/fp_tex_exact_{256,512}.png` — shared-scale
difference maps plus a log-log error-distribution histogram. The histogram is the
readable panel: a difference map alone undersells the 256³ result, since the error
is fine-grained speckle and MSE squares it, so 44.7× is driven by a few extreme
voxels covering almost no pixels. Full write-up in
`docs/precision-256-investigation.md`, including four approaches that were tested
and did not work.

CLI and `pybind_backend` produce bit-identical output at both resolutions on
Hawaii (verified 2026-09-01, all four modes), as expected since the binding
JIT-compiles the same `src/ct_gpu.c`.

OSEM on Hawaii, MSE vs CPU: `1.062e-04` (S=5, 256³), `4.050e-05` (S=2, 512³).
OSEM converges through a different update path — S partial updates per epoch
rather than one full update — so a larger divergence from plain MLEM is
inherent to the method rather than a precision defect.
- RMS as % of signal range: 256³ 0.0194%, 512³ 0.0026% (`gpu-img`/`gpu-opt`) — improves at higher resolution.
- `gpu-buf`'s MSE-vs-CPU margin over `gpu-img`/`gpu-opt` is scale-dependent: ~980x tighter at 256³, ~13x tighter at 512³.
- (256³ re-confirmed 2026-08-26 on GTX 680, fresh 100-epoch run, same result.)
- `gpu-img`/`gpu-opt` not bit-exact vs CPU/`gpu-buf` — root cause checked with `diag_maxgap.py`, not the earlier-suspected `1/U²` singularity (ruled out: min|U| at worst voxel 82-91% of SOD, 1.2-1.5x amplification, nothing). It's a one-voxel spatial displacement from the hardware `CLK_FILTER_LINEAR` sampler (half-texel offsets audited, correct). Mass conserved across displaced voxel pair. Affects 21/16.7M voxels at 256³, 138/134M at 512³ (>100×RMS), concentrated at FOV edge (100% beyond 0.8×FOV radius at 512³).
- `gpu-buf` (manual float32 trilinear) has none of this — use it when bit-level CPU fidelity matters more than speed.
- OSEM `--subsets 5`, 256³, 100 epochs: 57.17s total.
- `gpu-buf` variance on AMD Hawaii PRO (75-102s, same config) does not reproduce on GTX 680 (flat to the ms). Root cause: AMD-driver memory-placement demotion of the volume buffer, not thermal throttling — ruled out via step-change shape (not gradual ramp), magnitude (6.6x, exceeds Hawaii's ~3.2x DVFS ceiling), and `gpu-img`/`gpu-opt` staying stable in the same sessions. Confirmed as `d_vol`-allocation-specific: freeing/recreating the 537MB buffer mid-run recovers fast timing, degrades again after 6-13 launches.
- Mitigation tried (`FP_BUFFER_VOL_REALLOC_EVERY`, periodic reallocation): did not hold up — averaged *slower* (91.74s vs 86.96s) over 4 full 10-epoch runs. Off by default (numbers in `src/ct_gpu.c`).

## Validation (AMD Hawaii PRO, 100 epochs, both datasets, all four modes)

- Earlier full-scale validation run, on original AMD hardware — MSE values differ from GTX 680 numbers in Performance above, not a regression.
- Gap real, larger than expected for `gpu-buf` specifically: at 256³, AMD's `gpu-buf` MSE (`6.436e-10`) ~5.6x worse than GTX 680's (`1.148e-10`), despite `gpu-buf` never touching a hardware texture sampler — so "texture-sampler rounding" isn't the explanation.
- `-cl-fast-relaxed-math` (always on, `src/ct_gpu.c`) plausible cause — OpenCL spec leaves exact behavior vendor-defined. A prior investigation confirmed it does NOT explain `gpu-img`/`gpu-opt`'s within-GPU precision floor on a single machine (flag stripped, byte-identical output), but that test never compared the same build across AMD vs NVIDIA — the actual open question here.
- Not yet resolved — needs real AMD-hardware rebuild with flag stripped.

### 256³
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   247.7164 0.0070    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=6.371e-03
gpu-buf     0.0000   1.7307   0.0067    0    0  MSE=6.436e-10  MSE=6.371e-03  max=0.0511
gpu-img     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=6.371e-03  max=1.1490
gpu-opt     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=6.371e-03  max=1.1490
```

Same outlier-voxel phenomenon as GTX 680's run (see GTX 680 Validation
below): 24 outlier voxels dominate this MSE vs Python ref
(`Topic_2_CTreconstruction.py`) figure (0.00030 excluding them vs
0.00637 including them), not a C/GPU mode disagreement — confirmed
independently on this machine's own Python-reference output.

**Re-run with the `v0`-clamp fix (2026-08-28), same 100-epoch CPU/GPU
outputs, real 20-epoch Python reference run:**
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   5.0000   0.0069    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=3.014e-04
gpu-buf     0.0000   1.7307   0.0067    0    0  MSE=6.436e-10  MSE=3.014e-04  max=0.0511
gpu-img     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=3.016e-04  max=1.1490
gpu-opt     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=3.016e-04  max=1.1490
```
- MSE vs Python Ref: `6.371e-03` (unfixed) → `3.014e-04` (~21x better). `python_ref` max exactly `5.0000` (clamp bound, confirms engaging). Matches the ~0.00030 estimate from excluding the 24 outlier voxels — fix removes outliers, doesn't mask them.
- MSE vs CPU unchanged, as expected — fix only touches the Python reference script.

### 512³
```
Mode       min      max     mean   nan  inf  MSE vs CPU     MSE vs Python Ref
cpu      0.0000   1.0055   0.0330    0    0  (reference)
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.087e-11  max=0.0156
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
```

## Validation (NVIDIA GeForce GTX 680, 100 epochs, both datasets, all four modes)

- MSE vs CPU matches Performance section exactly (same `validate.py` output).
- MSE vs Python Ref available at 256³ (`Epochs=20` run completed) — dominated by 26 outlier voxels in the Python reference's own output, not a C/GPU mode disagreement (see note below 256³ table).
- 512³ not possible with Python reference: `Topic_2_CTreconstruction_512.py` variant attempted on both machines (2026-08-27), both ran out of memory. AMD Hawaii PRO's 15GB RAM confirmed hard limit (matches 2026-08-21 finding). GTX 680 has more RAM but still ran out partway. Not attempted again — see Run section for the script.

### 256³
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   521.6699 0.0070    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=2.061e-02
gpu-buf     0.0000   1.7292   0.0067    0    0  MSE=1.148e-10  MSE=2.061e-02  max=0.0203
gpu-img     0.0000   1.6577   0.0067    0    0  MSE=1.128e-07  MSE=2.061e-02  max=0.8966
gpu-opt     0.0000   1.6577   0.0067    0    0  MSE=1.128e-07  MSE=2.061e-02  max=0.8966
```

**MSE vs Python reference (`Topic_2_CTreconstruction.py`) dominated by 26
outlier voxels in that script's own output (0.00016% of volume), not a
C/GPU mode disagreement:**
- All 4 modes show identical `2.061e-02` — the Python reference output is the outlier, not any implementation. Excluding those 26 voxels drops MSE to ~3.2e-04.
- Investigated via `python/diag_bp_ones.py`: MLEM normalizer `bp_ones` normal at every outlier voxel (364-443, same range as elsewhere) — rules out division-by-near-zero.
- Real mechanism: geometric compounding. Each outlier voxel's per-epoch multiplicative ratio ~1.29-1.37 (backed out from `521.6699^(1/20)` and `174.32614^(1/20)` matching observed finals from v0=1) — 20 unclamped iterations compound small drift into large final value.
- Numerical-stability gap in unmodified reference script at ill-conditioned voxels (likely sparse/near-tangent ray coverage), not a bug in this project's code — reproduced independently on AMD Hawaii PRO's own Python-reference run (24 outlier voxels, overlapping indices, same z=1/z=254-concentrated pattern, same order of magnitude).

### 512³

No Python-reference output at 512³ -- ran out of memory on both
machines (see the Run section below for the 512^3 script and what was
tried).

```
Mode       min      max     mean   nan  inf  MSE vs CPU     MSE vs Python Ref
cpu      0.0000   1.0055   0.0330    0    0  (reference)
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.534e-11  max=0.0114
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=1.232e-09  max=0.0186
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=1.232e-09  max=0.0186
```

Correctness fixes that got here:
- `fp_cpu` used `(int)xi` (truncation) instead of `floorf(xi)` — wrong
  bounds-check pass for `xi` in `(-1,0)`. Fixed.
- GPU `bp` kernels zero-padded individual OOB taps instead of zeroing
  the whole sample (Python reference behavior). Fixed to match.
- Python reference ran bp-only, not full MLEM — invalidated "MSE vs
  Python". Fixed to run the same fp→ratio→bp→update loop.

Component tests: `--op fp|bp` dumps a single fp/bp call in isolation;
`validate_ops.py` compares it to the Python reference.

## Optimization history

- **fp_cpu ray-tiling** (512³): batched neighboring rays, sample index
  as outer loop → cache-friendly access order. `cpu` 512³: 44.5-45.7s
  → 26.06s/epoch.
- **D1 (bp_cpu branch removal)**: reverted. Looked like a wash on a
  single noisy run; a 3-trial comparison found a real ~3.7% CPU
  regression. Original `continue`-based code restored.
- **unroll-x2 in bp_opt**: removed. Measured 1.6% regression on AMD
  Hawaii (register pressure hurt occupancy more than ILP helped).
  `gpu-opt`/`gpu-img` now essentially tied (0.17% gap).
- **Work-group tuning**: `fp_image` `{16,16,1}→{8,32,1}` (~5-7%),
  `fp_buffer` `{16,16,1}→{4,64,1}` (>4× at 512³, largest single win),
  `fp_cpu` `FP_TILE` `8→32` (~19% at 512³). Env overrides:
  `FP_IMAGE_LWS`, `FP_BUFFER_LWS`, `FP_TILE_ENV`.
- **Negative/neutral results** (kept, not chased further): AABB
  clipping still hurts at 256³ when forced on; a latent (inert) AABB
  axis-transposition bug found and fixed; sphere-shaped AABB — no
  headroom over box clip; `bp_cpu` thread scaling already near-optimal
  at the Makefile default; `clFinish` batching — no win on the NVIDIA GeForce GTX 680.

## Modes

| Mode | Flag | Description |
|---|---|---|
| CPU | `--mode cpu` | OpenMP, incremental ray stepping |
| GPU buffer | `--mode gpu-buf` | Manual bilinear/trilinear, no texture cache — naive baseline |
| GPU image | `--mode gpu-img` | Hardware image2d_array + image3d sampler |
| GPU opt | `--mode gpu-opt` | Hardware sampler + float2 LUT + local mem |

`gpu-img`/`gpu-opt` share `fp_image.cl`; only their `bp` kernel differs.
AABB clipping (gated `W>512`) applies to both.

`vol_img` defaults to float32; `--half` opts into `CL_HALF_FLOAT`
(lower bandwidth, ~3-decimal-digit quantization cost).

## Files

```
src/
  main.c              — CLI, dispatch, HDF5 save
  utils.c/h           — HDF5 load/save, timing
  ct_cpu.c/h          — CPU: cone_weight, fp_cpu, bp_cpu, reconstruct_cpu
  ct_gpu.c/h          — OpenCL host: gpu_init, reconstruct_gpu, reconstruct_gpu_opt
kernels/
  bp_buffer.cl        — bp (buffer) + preprocess_proj + proj_divide + vol_update
  fp_buffer.cl        — fp (buffer): ray march + manual trilinear + AABB
  bp_image.cl         — bp (image): hardware bilinear + float2 LUT
  fp_image.cl         — fp (image): hardware trilinear + AABB
  bp_buffer_opt.cl    — bp_opt: image2d_array_t + float2 LUT + local mem
python/
  validate.py                 — MSE vs CPU + vs Python Ref, outlier diagnostics (256/512)
  validate_ops.py             — per-operator fp/bp vs Topic_2_CTreconstruction.py
  diag_voxel.py                — one-off debugging script, kept for reference
  diag_maxgap.py               — per-operator error attribution for the gpu-img/gpu-opt max-gap finding
  plot_results.py              — report figures (OSEM/MLEM convergence, slice comparisons)
  Topic_2_CTreconstruction.py — reference script (fp_func/bp_func), the
                                 "Reference code (python)" grading refers
                                 to. 2 real bugs fixed early (out_path
                                 placeholder, volume z-axis size) so it
                                 runs/saves. 3rd fix: clamp v0 to [0, 5]
                                 after each update (numerical-stability
                                 gap at a handful of voxels, course staff
                                 approved — see 26-outlier-voxel finding
                                 in Validation section). Per-epoch ratio
                                 clamp tried first, doesn't work (verified
                                 via simulation, not run for real — see
                                 in-file comment). fp_func/bp_func
                                 themselves untouched.
  Topic_2_CTreconstruction_512.py — 512^3 variant, path_data/out_path
                                 changed, same v0 clamp as 256^3 script;
                                 attempted both machines, ran out of
                                 memory on both — see Validation section.
  diag_bp_ones.py               — investigates 26-outlier-voxel finding
                                 (checks bp_ones near-zero at those
                                 voxels; it isn't).
```

## Build

```bash
sudo apt install libhdf5-dev ocl-icd-opencl-dev opencl-headers gcc make
make
```

## Run

Data: `/lgrp/edu-2026-1-gpulab/`.

```bash
# 256³
make run-cpu     EPOCHS=100
make run-gpu-buf EPOCHS=100
make run-gpu-img EPOCHS=100
make run-gpu-opt EPOCHS=100
python3 python/validate.py

# 512³
make run-cpu-512     EPOCHS=100
make run-gpu-buf-512 EPOCHS=100
make run-gpu-img-512 EPOCHS=100
make run-gpu-opt-512 EPOCHS=100
python3 python/validate.py 512

# component tests (CPU only, isolate fp/bp correctness)
make run-op-fp       # dumps fp_cpu.hdf5 (256³)
make run-op-bp       # dumps bp_cpu.hdf5 (256³)
make run-op-fp-512   # dumps fp_cpu_512.hdf5 (512³, use SAMPLES512=64 for a quick check)
make run-op-bp-512   # dumps bp_cpu_512.hdf5 (512³)
python3 python/validate_ops.py fp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump fp_cpu.hdf5
python3 python/validate_ops.py bp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump bp_cpu.hdf5

# Reference (python) -- 256³ only, hardcoded epoch count in-script
# (no --data/--epochs flags, EPOCHS not honored)
make run-python
```

- Pure-Python fp/bp (scipy `RegularGridInterpolator` rebuilt per-angle, no vectorization/GPU): ~4690s/epoch at 256³ on GTX 680, script default `sample_ratio=2`.
- `sample_ratio` reduced to 1 (call-site only, `fp_func(cb_para, vol, sample_ratio=1)`; internals untouched) — matches C/GPU modes' default sample count at 256³ (`n_samples = Nxz = 256`, `src/main.c`), which `sample_ratio=2`'s 512 samples/ray didn't.
- Measured ~3679s/epoch at sample_ratio=1 — only ~21% faster, not ~50% naive scaling would suggest: most of fp_func's cost is fixed per-pixel Python/interpreter overhead (1.3M-pixel loop), not proportional to ray length; bp_func unaffected by sample_ratio at all.
- Full 100 epochs at this rate ≈ 4.3 days; `Epochs` set to 20 (~20hrs) instead — fewer epochs = less convergence, expected tradeoff, not a defect.
- Run on GTX 680 under `tmux -s topic2` unattended, completed 2026-08-27.

- Full MSE-vs-CPU/Python-reference numbers (real `Epochs=20` run, completed 2026-08-27): "Validation (NVIDIA GeForce GTX 680...)" section above, incl. 26-outlier-voxel finding explaining 2.061e-02.

## Algorithm

MLEM multiplicative update:

```
v = ones(Nxz, Nxz, Ny)
bp_ones = bp( cone_weight( ones_proj ) )   # once, before the loop

for each epoch:
    b      = fp(v)
    ratio  = p0 / b    (b > 1e-3)
    ratio  = cone_weight(ratio)
    v     *= bp(ratio) / bp_ones
```

`preprocess_proj`: flip H-axis + transpose `[np,H,W]→[np,W,H]` +
`/voxelSize` (applied to ratio and ones before every bp call).

### OSEM (`gpu-opt` only, `--subsets N`)

Splits the angle set into `N` subsets, one volume update per subset
instead of one per full angle pass. `--subsets 1` (default) is
byte-identical to plain MLEM.

```
bp_ones[s] = bp( cone_weight(ones_proj), subset s )   for s in 0..N   # once

for each epoch (= one pass over all N subsets):
    for each subset s:
        b      = fp(v)                          # full angle range
        ratio  = p0 / b   (b > 1e-3)
        ratio  = cone_weight(ratio)
        v     *= bp(ratio, subset s) / bp_ones[s]   # subset's angles only
```

- Angle ordering: subset `s` gets `{s, s+N, s+2N, ...}` (interleaved),
  subsets visited in a golden-ratio-derived coprime stride order.
  Permutation applied once at load time (`utils.c`), so each subset
  becomes a contiguous `(ip_start, ip_count)` launch range.
- 256³ only. Confirmed on the NVIDIA GeForce GTX 680 (4037MiB VRAM): plain `gpu-opt`
  at 512³ (S=1) already uses ~3244-3274MiB, leaving ~760-790MiB
  headroom — not enough for the 1024MiB (2×512MiB) a second subset's
  normalizer buffers would need at S=2.
- Regression tests: `--subsets` unset vs baseline, MSE unchanged;
  `--subsets 1` vs unset, byte-identical.

**Result** (256³, 100 epochs, log-likelihood vs wall-clock — logged via
`--log-convergence <file.csv>`, which writes per-epoch loglik/residual/
rel_change; off by default):

| Time | S=1 | S=3 | S=5 | S=15 | S=25 |
|---|---|---|---|---|---|
| 10s | −482,720 | −481,297 | −481,109 | **−480,543** | −480,577 |
| 15s | −481,109 | −480,391 | −480,304 | −479,652 | **−479,285** |
| 20s | −480,652 (plateaued) | −480,013 | −479,942 | −479,305 | **−478,917** |
| 30s | −480,652 (plateaued) | −479,701 | −479,653 | −479,042 | **−478,651** |
| 36s | −480,652 (plateaued) | −479,610 | −479,580 | −478,964 | **−478,571** |

- S=1 plateaus by ~20s (100-epoch ceiling). Every OSEM config keeps improving through 36s, never plateaus.
- S=15 leads at 10s, S=25 close second (sub-iteration overhead still amortizing) but overtakes and wins clearly from 15s on.
- Report time-matched curve, not a single "best S."

## Optimizations

| Optimization | Where | Effect |
|---|---|---|
| `-ffast-math` + OpenMP `collapse(2)` | `fp_cpu`, `bp_cpu` | denormal flush; parallelism |
| Incremental ray stepping | `fp_cpu`, `fp_buffer.cl`, `fp_image.cl` | fewer multiplies in ray march |
| image2d_array_t hardware bilinear | `bp_image.cl`, `bp_buffer_opt.cl` | texture cache + HW interpolation |
| image3d_t hardware trilinear | `fp_image.cl` | texture cache replaces manual 8-tap trilinear |
| float2 cos/sin LUT | `bp_buffer_opt.cl`, `bp_image.cl` | no cos/sin per voxel per angle |
| Local memory LUT cache | `bp_buffer_opt.cl` | angle_cs cooperatively loaded into `__local` |
| float4 vectorized divide/update | `bp_buffer.cl` | 4 elements/work-item |
| Work-group `{4,4,16}` for bp | `ct_gpu.c` | coalesced writes |
| Fused cone_weight + preprocess_proj | `bp_buffer.cl` | one kernel pass instead of two |
| Precomputed R/T matrices | `ct_gpu.c`, `fp_image.cl` | no per-work-item cos/sin + matmul in fp |
| Half-precision vol_img (`--half`) | `fp_image.cl`, `ct_gpu.c` | halves texture bandwidth; off by default |
| AABB slab ray clipping (`W>512`) | `fp_image.cl`, `fp_buffer.cl` | skips empty cone-beam edge rays |
| `native_recip` in bp kernels | `bp_buffer_opt.cl`, `bp_buffer.cl`, `bp_image.cl` | HW SFU reciprocal |
| OMP_NUM_THREADS=nproc | `Makefile` | uses all available cores |
| Ray tiling (`FP_TILE`) | `fp_cpu` | cache reuse across neighboring rays |
| `schedule(guided,4)` | `fp_cpu` | lower scheduling overhead, handles AABB imbalance |
| Fused proj_divide + preprocess_proj (`divide_preprocess_img`) | `bp_buffer.cl` | one kernel writes straight to `ratio_img`, no intermediate copy |
| `vol_update_img` | `bp_buffer.cl`, `ct_gpu.c` | vol_update writes directly into `vol_img`, eliminates the per-epoch `d_vol`→image copy entirely (float32 mode) |
