# CT Volume Reconstruction — GPU Lab Project

Cone-beam CT reconstruction using iterative MLEM on CPU (OpenMP) and GPU (OpenCL).
Supports 256³ and 512³ datasets. All GPU modes validated: no NaN/Inf, MSE vs CPU within float32 rounding bounds (see Validation below — this was previously a real bug, now fixed and re-measured).

## Performance Results

### 256³ dataset (512×512 detector, 75 angles, n\_samples=256)

Measured on `pool15-01`, **EPOCHS=10** (10-epoch, not 100 — see note),
current code (post `FP_TILE=32` and `fp_buffer` work-group changes):

| Mode | Time/epoch | Speedup vs CPU |
|------|-----------|----------------|
| `cpu` (12-thread OpenMP, `FP_TILE=32`) | 2.60 s | 1× |
| `gpu-buf` (`{4,64,1}`) | 0.44 s | **5.9×** |
| `gpu-img` | 0.095 s | **27.4×** |
| `gpu-opt` | **0.093 s** | **28.0×** |

### 512³ dataset (1120×1184 detector, 75 angles, n\_samples=512)

Measured on `pool15-01`, **EPOCHS=10**, current code. Two independent
10-epoch runs shown for `gpu-buf`, whose per-epoch time varies more than
the other modes' — see note below.

| Mode | Time/epoch | Speedup vs CPU |
|------|-----------|----------------|
| `cpu` (12-thread OpenMP, `FP_TILE=32`) | 22.9-23.3 s | 1× |
| `gpu-buf` (`{4,64,1}`) | 4.87-10.19 s (run-to-run) | **2.3-4.8×** |
| `gpu-img` | 0.870-0.876 s | **26.6×** |
| `gpu-opt` | 0.873-0.879 s | **26.5×** |

**Hardware:** Intel Core i7-5820K @ 3.30GHz (12 logical cores) · AMD Hawaii PRO (Radeon R9 290/390, 2560 shaders, 2.56 TFLOPS) · `pool15-01.cis.iti.uni-stuttgart.de`

### kale (NVIDIA GTX 680), 100-epoch confirmed numbers

The tables above are `pool15-01` (AMD Hawaii) at 10-epoch scale, from
earlier in the project. Work has since moved to `kale` (NVIDIA GTX 680,
Kepler architecture) as the standing development machine — a different
GPU vendor/architecture, so none of the AMD-tuned work-group values
transfer directly (re-tuned separately, see "gpu-buf variance confirmed
AMD-driver-specific" below). These are full 100-epoch runs, matching the
epoch count actually used for the report's convergence/quality claims,
not a 10-epoch spot-check.

**256³** (512×512 detector, 75 angles):

| Mode | Time/epoch (steady) | Total (100 epochs) | Speedup vs CPU |
|---|---|---|---|
| `cpu` (24-thread OpenMP) | ~4.20s | 423.86s | 1× |
| `gpu-buf` | 0.855-0.860s | 86.46s | **4.9×** |
| `gpu-img` | 0.142-0.148s | 14.67s | **28.9×** |
| `gpu-opt` | 0.138-0.143s | 14.29s | **29.7×** |

**512³** (1120×1184 detector, 75 angles):

| Mode | Time/epoch (steady) | Total (100 epochs) | Speedup vs CPU |
|---|---|---|---|
| `cpu` (24-thread OpenMP) | ~34s | 3415.58s | 1× |
| `gpu-buf` | 7.13-7.20s | 722.59s | **4.7×** |
| `gpu-img` | 1.226-1.256s | 126.74s | **27.0×** |
| `gpu-opt` | 1.226-1.245s | 125.77s | **27.2×** |

**Hardware:** Intel Xeon E5-2620 0 @ 2.00GHz (24 logical cores) · NVIDIA
GeForce GTX 680 (Kepler, no `cl_khr_fp16` — `--half` unavailable on this
card) · `kale.cis.iti.uni-stuttgart.de`

Validated at both scales (`validate.py` / `validate.py 512`): all four
modes agree, no NaN/inf. 256³: MSE vs CPU `1.148e-10` (`gpu-buf`),
`1.128e-07` (`gpu-img`/`gpu-opt`). 512³: MSE vs CPU `9.534e-11`
(`gpu-buf`), `1.232e-09` (`gpu-img`/`gpu-opt`).

OSEM (`--subsets 5`), 256³, 100 epochs: 57.17s total (~0.57s/epoch
steady state) — scales with subset count as expected (~5× a single
sub-iteration's angle-range cost, since one epoch is 5 sub-iterations).

> **`gpu-buf` on pool15-01 (AMD Hawaii) shows real run-to-run variance**
> (75-102s for the same 10-epoch config) that does **not** reproduce on
> kale/NVIDIA (flat to the millisecond across 40 repeats). Root-caused to
> AMD-driver-side memory-placement demotion of the large `d_vol`
> allocation under sustained access, not thermal throttling — confirmed
> via GPU-side event profiling and a dedicated repeat-slab diagnostic
> (`--diag repeat-slab`). No reliable mitigation found; report a range,
> not a single number, when citing `gpu-buf` on pool15-01 specifically.
> Full investigation: `sessions/2026-08-25-readme-cleanup-investigation-log.md`.

### Key optimizations that drove 512³ speedup

> ⚠ Predates the correctness fixes above; per-stage numbers not re-measured.

| Change | bp (ms) | fp (ms) | Total (s) |
|--------|---------|---------|-----------|
| Baseline | 718 | 409 | 1.149 |
| Work-group `{8,8,4}→{4,4,16}` (coalesced z-writes) | **290** | 409 | 0.724 |
| AABB ray clipping in fp\_image (gated W>512) | 290 | **342** | 0.659 |
| `native_recip` in bp\_opt | 290 | 342 | 0.657 |

Note: the `native_recip` row shows no measured bp/fp delta (290→290,
342→342) — its effect on the previous total (0.659→0.657s) is within
measurement noise, not a real driver of the speedup. Left in the table for
history; don't cite it as a proven win without re-measuring in isolation.

## Validation (100 epochs)

Full-scale run: both datasets, all four modes, matching epoch counts
throughout — the epoch-mismatch caveat from earlier 10-epoch checks no
longer applies anywhere below.

> ⚠ This table predates the `FP_TILE=8→32` and `fp_buffer` work-group
> `{16,16,1}→{4,64,1}` changes (see Performance Results above). Both were
> re-validated at 10 epochs after landing (MSE at the float32 noise floor,
> same as here) but not yet re-confirmed at the full 100-epoch scale this
> table uses. Re-run `python3 validate.py` / `validate.py 512` at
> `EPOCHS=100` to refresh this table.

### 256³
```
Mode       min      max     mean   nan  inf  MSE vs CPU         MSE vs Python
python   0.0002   0.1225   0.0144    0    0  (python ref)      -
cpu      0.0000   1.7303   0.0067    0    0  (reference)        MSE=1.215e-03
gpu-buf  0.0000   1.7307   0.0067    0    0  MSE=6.436e-10  max=0.0511  MSE=1.215e-03
gpu-img  0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  max=1.1490  MSE=1.215e-03
gpu-opt  0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  max=1.1490  MSE=1.215e-03
```

### 512³
```
Mode       min      max     mean   nan  inf  MSE vs CPU
cpu      0.0000   1.0055   0.0330    0    0  (reference)
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.087e-11  max=0.0156
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
```

All three GPU modes agree with CPU at essentially the float32 noise floor
at both scales — confirms the correctness fixes below hold up over full
100-epoch convergence, not just the shorter 10-epoch checks used while
debugging. `max` is naturally larger at 100 epochs than at 10 (compounding
iteration), but MSE stays tiny throughout.

**What was fixed to get here** (earlier this project, `max` was ~1.17-1.25
at only 10 epochs, attributed in this README to "float32 rounding" — that
framing was wrong):
1. `fp_cpu` used `int x0 = (int)xi` (truncates toward zero) instead of
   `floorf(xi)`. For `xi` in `(-1, 0)` this gives `x0=0` instead of `-1`,
   which wrongly passes the `(unsigned)x0 < Nxz-1` bounds check and samples
   `volume[0]` with a bogus weight for a ray that should have missed the
   volume entirely. Found via a component test comparing `fp_cpu(ones)`
   against the Python reference pixel-by-pixel — one pixel had C=1.107,
   Python=0.0. Fixed in `src/ct_cpu.c` (fp_cpu only — the two `(int)uf`/
   `(int)vf` casts in `bp_cpu` are safe, already guarded by an explicit
   `< 0` check before the cast).
2. GPU `bp` kernels (`bp_buffer.cl`, `bp_image.cl`, `bp_buffer_opt.cl`)
   zero-padded individual out-of-bounds interpolation taps; the Python
   reference (`scipy.interpolate.RegularGridInterpolator(fill_value=0)`)
   zeros the *entire* sample if any tap is out of bounds — verified
   empirically, not documented behavior. GPU kernels now match: an
   explicit whole-cell bounds check gates the texture read / tap lookup.
3. `run_python_reference.py` previously ran a single bp-only MLEM step (no
   `fp` at all) and called that "MSE vs Python" — comparing two different
   algorithms. It now runs the same full fp→ratio→bp→update loop as
   `reconstruct_cpu`/`reconstruct_gpu`, with a `jitter=False` mode so its
   fixed-step ray march matches the C/GPU kernels' (which have no jitter).

Component-test tools added for future debugging: `--op fp|bp` on
`ct_recon` dumps a single fp/bp call in isolation (bypasses accumulated
MLEM iteration error); `validate_ops.py` compares that dump against the
Python reference per-operator.

### CPU 512³ speedup: fp_cpu ray-tiling rewrite
`fp_cpu` was memory-latency-bound at 512³ (8-tap trilinear gather, each
spanning up to ~2MB). Fix: batch `FP_TILE=32` neighboring detector rows
together and iterate the sample index as the outer loop, so nearby rays'
volume reads cluster in time instead of scattering. Pure
memory-access-order change — same math, same weights, verified against
the Python reference. Result: `cpu` 512³ 26.06s/epoch (100 epochs), down
from a pre-tiling 44.5-45.7s/epoch. Full investigation:
`sessions/2026-08-25-readme-cleanup-investigation-log.md`.

### Phase D: bp_cpu branch removal, reverted after a real regression was found

D1 (replaced a `continue` guard in `bp_cpu`'s inner loop with a
branch-free clamp) initially looked like a wash, but a rigorous
3-trial-each comparison during pre-merge validation found a real ~3.7%
CPU-time regression — reverted. D2/D3 were investigated and correctly
not pursued (`bp_cpu` isn't the bottleneck; D3's target branch never
fires on this project's detector width). Full story, including how the
regression was found and isolated:
`sessions/2026-08-25-readme-cleanup-investigation-log.md`.

### gpu-opt vs gpu-img: unroll-x2 was measured harmful, removed

`bp_opt`'s unroll-x2 loop (meant to hide texture-fetch latency via ILP)
was actually a 1.6% regression on AMD Hawaii — the doubled live register
set cost more in occupancy than it saved. Removed; `bp_opt` is now
unconditionally scalar. Final 100-epoch result: `gpu-opt` and `gpu-img`
essentially tied (0.17% gap, within noise). Investigation details:
`sessions/2026-08-25-readme-cleanup-investigation-log.md`.

### Work-group tuning

Three parameters swept, none previously tuned past a first guess:
`fp_image` (`gpu-img`/`gpu-opt`) `{16,16,1}→{8,32,1}` (~5-7%),
`fp_buffer` (`gpu-buf`) `{16,16,1}→{4,64,1}` (>4× at 512³, the largest
single win in the pool15-01/AMD era of this project), and `fp_cpu`'s
`FP_TILE` `8→32` (~19% at 512³). All correctness-neutral by
construction — confirmed via `validate.py` at the float32 noise floor.
Overridable via `FP_IMAGE_LWS`/`FP_BUFFER_LWS`/`FP_TILE_ENV` env vars.
Full sweep data and reasoning: `sessions/2026-08-25-readme-cleanup-investigation-log.md`.

### Other findings (AABB, thread scaling, dataset facts)

Several smaller investigations, all with negative or neutral results,
kept as project history rather than in the README: AABB clipping still
hurts at 256³ when forced on (~4% regression, gate stays `W>512`); a
latent (currently inert) AABB axis-transposition bug found and fixed;
sphere-shaped AABB measured no headroom over the box clip; `bp_cpu`
thread scaling confirmed the Makefile default is already near-optimal;
`clFinish` batching (Phase E) measured no win on kale. Dataset facts
confirmed rather than assumed: full 360° scan (not short-scan), no
ground-truth volume in the HDF5 files, and a known CPU-vs-Python
sampling-ratio mismatch that leaves 512³ `MSE vs Python` unavailable
(infeasible to generate on pool15-01's 15GB RAM). Full writeups:
`sessions/2026-08-25-readme-cleanup-investigation-log.md`.

## Modes

| Mode | Flag | Description |
|------|------|-------------|
| CPU | `--mode cpu` | C + OpenMP, `-ffast-math`, incremental ray stepping |
| GPU buffer | `--mode gpu-buf` | OpenCL global buffers, manual bilinear/trilinear — naive baseline; slow at 512³ (no texture cache), chunked to avoid the driver watchdog |
| GPU image | `--mode gpu-img` | Hardware image2d_array + image3d sampler |
| GPU opt | `--mode gpu-opt` | Hardware sampler + float2 LUT + local mem (`bp_opt`); fp shared with `gpu-img` via `fp_image.cl` |

Both `gpu-img` and `gpu-opt` share the same `fp_image.cl` forward-projection
kernel (only their bp kernel differs — `bp_image.cl` vs `bp_opt`), so AABB
clipping (gated `W>512`, see "Other findings" above) applies to both
equally, not just `gpu-opt`.

`vol_img` precision (`gpu-img`/`gpu-opt`) defaults to **float32**; pass
`--half` to opt into `CL_HALF_FLOAT` for lower texture bandwidth at the cost
of ~3-decimal-digit quantization (was previously always-on, uncredited as an
accuracy tradeoff — see Validation).

## Files

```
src/
  main.c              — CLI: parse args, dispatch to cpu/gpu modes, save HDF5
  utils.c/h           — HDF5 load/save, get_time_sec()
  ct_cpu.c/h          — CPU: cone_weight, fp_cpu, bp_cpu, reconstruct_cpu
  ct_gpu.c/h          — OpenCL host: gpu_init, reconstruct_gpu, reconstruct_gpu_opt
kernels/
  bp_buffer.cl        — bp (buffer) + preprocess_proj + proj_divide + vol_update
  fp_buffer.cl        — fp (buffer): ray march + manual trilinear + AABB clipping
  bp_image.cl         — bp (image): hardware bilinear on image2d_array_t + float2 LUT
  fp_image.cl         — fp (image): hardware trilinear on image3d_t + AABB clipping
  bp_buffer_opt.cl    — bp_opt: image2d_array_t + float2 LUT + local mem
validate.py           — load HDF5 outputs, print MSE + outlier-location diagnostics (supports 256/512)
validate_ops.py       — per-operator fp/bp comparison vs Python reference, isolated from MLEM iteration
run_python_reference.py — full MLEM loop in Python (fp_func/bp_func), --epochs to match C runs
diag_fp.py             — fp_cpu vs Python fp_func comparison, parameterized for 256^3/512^3 (--data/--dump/--samples)
diag_voxel.py, diag_voxel2.py — one-off scripts from the boundary-rule/truncation bug hunt; kept for reference, not part of the regular workflow
Topic_2_CTreconstruction.py — original course-provided Python reference (fp_func/bp_func); superseded by run_python_reference.py for regular use, kept as the unmodified starting point
```

## Build

```bash
make
```

Requires: `gcc`, `libhdf5-dev`, `ocl-icd-opencl-dev`, `opencl-headers`.

```bash
sudo apt install libhdf5-dev ocl-icd-opencl-dev opencl-headers gcc make
```

## Run

Data is at `/lgrp/edu-2026-1-gpulab/` on lab machines.

### 256³
```bash
make run-cpu     EPOCHS=100
make run-gpu-buf EPOCHS=100
make run-gpu-img EPOCHS=100
make run-gpu-opt EPOCHS=100
python3 validate.py
```

### 512³
```bash
make run-cpu-512     EPOCHS=100
make run-gpu-buf-512 EPOCHS=100   # now unblocked; chunked launches avoid the watchdog hang
make run-gpu-img-512 EPOCHS=100
make run-gpu-opt-512 EPOCHS=100
python3 validate.py 512
```

Or run directly:
```bash
./build/ct_recon --data /lgrp/edu-2026-1-gpulab/proj_512_75.hdf5 \
                 --out output.hdf5 --mode gpu-opt --epochs 100 \
                 --samples 512 --kernels kernels
```

### Component tests (isolate fp/bp correctness, CPU only)
Both print their own elapsed time (`fp_cpu time: X.XXX s`). 512³ variants
exist since fp/bp cost differs a lot by dataset size.
```bash
make run-op-fp       # dumps fp_cpu.hdf5 (256^3)
make run-op-bp       # dumps bp_cpu.hdf5 (256^3)
make run-op-fp-512   # dumps fp_cpu_512.hdf5 (512^3, SAMPLES512 samples — slow, use SAMPLES512=64 for a quick check)
make run-op-bp-512   # dumps bp_cpu_512.hdf5 (512^3)
python3 validate_ops.py fp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump fp_cpu.hdf5
python3 validate_ops.py bp --data /lgrp/edu-2026-1-gpulab/proj_256_75.hdf5 --dump bp_cpu.hdf5
python3 diag_fp.py --data /lgrp/edu-2026-1-gpulab/proj_512_75.hdf5 --dump fp_cpu_512.hdf5 --samples 512
```

### Python reference (must match C run's `--epochs` for validate.py's "MSE vs Python" to be meaningful)
```bash
make run-python     EPOCHS=10   # 256^3, slow — Python fp is ~minutes/epoch
make run-python-512 EPOCHS=10   # 512^3, slower still; see Known Gaps for the
                                 # sample_ratio mismatch this doesn't fix
```

## Algorithm

MLEM multiplicative update:

```
v = ones(Nxz, Nxz, Ny)
bp_ones = bp( cone_weight( ones_proj ) )   # computed once before loop

for each epoch:
    b      = fp(v)                          # forward project current estimate
    ratio  = p0 / b    (b > 1e-3)          # measured / estimated
    ratio  = cone_weight(ratio)             # apply cone weight before bp
    v     *= bp(ratio) / bp_ones           # multiplicative MLEM update
```

`preprocess_proj`: flip H-axis + transpose `[np,H,W]→[np,W,H]` + `/voxelSize`
(applied to ratio and ones before every bp call).

### OSEM (Ordered Subsets EM) — `gpu-opt` only, `--subsets N`

perf-v2 Phase C1. Splits the angle set into `N` subsets and does one
volume update per subset instead of one per full angle pass — near-linear
convergence acceleration per unit work, at the cost of limit-cycling near
(not exactly onto) plain MLEM's fixed point. Default `--subsets 1` is
**provably identical** to plain MLEM — no permutation, one normalizer,
verified byte-identical (`max abs diff: 0.0`) against the unset-flag
default in testing, not just assumed equivalent.

```
v = ones(...)
bp_ones[s] = bp( cone_weight(ones_proj), subset s )   for s in 0..N   # once, before the loop

for each epoch (= one full pass over all N subsets):
    for each subset s:
        b       = fp(v)                          # full angle range every sub-iteration
        ratio   = p0 / b   (b > 1e-3)
        ratio   = cone_weight(ratio)
        v      *= bp(ratio, subset s) / bp_ones[s]   # only this subset's angles
```

`fp` still covers the full angle range every sub-iteration (`vol_img`
holds the whole current volume) — only the `bp`/update step is restricted
to the subset. An accepted implementation cost (`fp` isn't `1/N` the work
it could be), not a correctness issue; scoped this way because `gpu-opt`'s
`fp` is comparatively cheap and doing it per-subset would have meant
splitting `vol_img`/`ratio_img` handling further for limited additional
win.

**Angle ordering**: subset `s` gets `{s, s+N, s+2N, ...}` (interleaved —
each subset spans the full angular range, not a contiguous wedge), and
subsets are visited in a stride-coprime-to-N order chosen near the
golden-ratio point so consecutive sub-iterations are maximally separated
in angle. Applied once at load time as a physical permutation of the
projection stack (`compute_osem_permutation`/`permute_projections_inplace`
in `utils.c`), so subset `k` becomes exactly the contiguous angle range
`[k·np/N, (k+1)·np/N)` and the kernels only need an `(ip_start, ip_count)`
launch range, not per-work-item subset membership checks.

256³ only for now — at 512³ the per-subset normalizer array (`N` volume-sized
buffers instead of 1) doesn't fit alongside the rest of `gpu-opt`'s already
~4GB resident set on this project's cards; not implemented, see the plan.

**Measured result, report scale** (256³, **100 epochs**, `--log-convergence`,
log-likelihood vs matched wall-clock time — the metric MLEM/OSEM actually
optimizes, not just epoch count). An earlier 30-epoch smoke test (kept below
for the record) found S=5 leading early and S=15 overtaking it — the full
100-epoch run shows that was still an early transient:

| Wall-clock time | S=1 | S=3 | S=5 | S=15 | S=25 |
|---|---|---|---|---|---|
| 10 s | −482,720 | −481,297 | **−481,109** | −480,543 | −480,577 |
| 15 s | −481,109 | −480,391 | −480,304 | **−479,652** | −479,285 |
| 20 s | −480,652 (plateaued) | −480,013 | −479,942 | −479,305 | **−478,917** |
| 30 s | −480,652 (plateaued) | −479,701 | −479,653 | −479,042 | **−478,651** |
| 36 s | −480,652 (plateaued) | −479,610 | −479,580 | −478,964 | **−478,571** |

(higher/less-negative is better.) **The real story only appears at full
epoch count: S=1 plateaus by ~20s and goes nowhere for the rest of the
run (it has converged to its 100-epoch ceiling), while every OSEM
configuration keeps improving for the entire 36s window and never
visibly plateaus.** S=25 is the worst performer before ~10s (25
sub-iterations per epoch means more per-epoch launch/normalizer
overhead before the noisier updates pay off) but is the clear winner
from ~15s onward and stays ahead through 36s — the opposite ranking
from what an early-epoch-only measurement would suggest. **The earlier
30-epoch finding ("S=5 leads early, S=15 overtakes") is confirmed as
real but incomplete — it was the early transient of a curve that, given
enough time, keeps favoring larger S.** Whether that trend continues
past 36s or S=25 also plateaus eventually is not established by this
data — report the time-matched curve, not a single "best S," and note
the time budget it was measured under.

**Regression tests, both passed exactly:**
1. `--subsets` unset vs the pre-OSEM baseline: MSE=8.423e-10 vs CPU,
   unchanged.
2. Explicit `--subsets 1` vs unset: `max abs diff: 0.0`, bit-identical.

An earlier 30-epoch smoke test and an S=3 angle-ordering investigation
(both superseded by / resolved in the 100-epoch result above) are kept
in `sessions/2026-08-25-readme-cleanup-investigation-log.md`.

## Optimizations

| Optimization | Where | Effect |
|---|---|---|
| `-ffast-math` + OpenMP `collapse(2)` | `fp_cpu`, `bp_cpu` | denormal flush; parallelism across detector pixels |
| Incremental ray stepping | `fp_cpu`, `fp_buffer.cl`, `fp_image.cl` | eliminates multiply per sample in ray march |
| image2d\_array\_t hardware bilinear | `bp_image.cl`, `bp_buffer_opt.cl` | texture cache + free HW interpolation |
| image3d\_t hardware trilinear | `fp_image.cl` | texture cache replaces manual 8-tap trilinear |
| float2 cos/sin LUT | `bp_buffer_opt.cl`, `bp_image.cl` | eliminates cos/sin per voxel per angle |
| Local memory LUT cache | `bp_buffer_opt.cl` | angle\_cs cooperatively loaded into `__local` |
| float4 vectorized divide/update | `bp_buffer.cl` | 4 elements/work-item via `vload4`/`vstore4` |
| Work-group `{4,4,16}` for bp kernels | `ct_gpu.c` | 16 contiguous z-threads → coalesced writes (volume layout `[x][y][z]`) |
| Fused cone\_weight + preprocess\_proj | `bp_buffer.cl` | one kernel pass instead of two |
| Precomputed R/T matrices | `ct_gpu.c`, `fp_image.cl` | eliminates cos/sin + matmul per work-item in fp |
| Half-precision `vol_img` (`CL_HALF_FLOAT`, opt-in via `--half`) | `fp_image.cl`, `ct_gpu.c` | halves texture bandwidth for volume reads; float\_to\_half kernel on GPU, no PCIe roundtrip; ~3-decimal-digit accuracy cost, so off by default |
| AABB slab ray clipping (gated W>512) | `fp_image.cl`, `fp_buffer.cl` | tightens per-ray sample range on large detectors; skips empty cone-beam edge rays |
| `native_recip` in bp kernels | `bp_buffer_opt.cl`, `bp_buffer.cl`, `bp_image.cl` | hardware SFU reciprocal, ~4× faster than IEEE division |
| ~~Unroll-x2 gated Nxz≥512~~ (removed) | `bp_buffer_opt.cl` | tried, measured harmful on this GPU — see "gpu-opt vs gpu-img" below |
| OMP\_NUM\_THREADS=nproc | `Makefile` cpu targets | uses all available cores on lab node |
| Ray tiling (`FP_TILE`, sample-index-outer loop) | `fp_cpu` | groups neighboring rays' 8-tap gathers close in time for cache reuse; 2.1× on fp_cpu at 512³ when introduced (tile=8), later swept to 32 for a further ~19% |
| `schedule(guided,4)` | `fp_cpu` | lower scheduling overhead than `dynamic` at 512³'s larger iteration count (75×1120=84000 work items), still handles AABB-clipped load imbalance |
| Work-group `{16,16,1}→{8,32,1}` for `fp_image` | `ct_gpu.c` | ~5-7% at both scales; swept 10 candidates, see "Work-group tuning" above. `FP_IMAGE_LWS` env override for further tuning |
| Skip float32 staging copy for `vol_img` upload | `ct_gpu.c` | float32 mode (default) copies `d_vol` straight to the image instead of through a same-format memcpy staging buffer; ~0.7% at 512³, correctness-neutral |
| Work-group `{16,16,1}→{4,64,1}` for `fp_buffer` | `ct_gpu.c` | 2.3-4.8× at 512³ (56.93s→4.87-10.19s/epoch, run-to-run variance) — the single largest win in this project; `fp_buffer`'s uncoalesced gather is far more work-group-sensitive than `fp_image`'s. `FP_BUFFER_LWS` env override |
| `FP_TILE` swept `8→32` | `fp_cpu` | ~19% further on top of the original tiling rewrite (13.58s vs 16.84s fp/epoch at 512³). `FP_TILE_ENV` env override |
