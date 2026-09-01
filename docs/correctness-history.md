# Correctness Investigations

Condensed from session logs. See [investigations-index.md](investigations-index.md).

## CPU/GPU boundary-rule bug (2026-08-17)

Initial suspicion: `bp_cpu` dropped whole out-of-bounds samples
(`continue`), GPU kernels zero-padded individual taps — looked like a
correctness mismatch. First fix (make `bp_cpu` zero-pad like GPU) was
**wrong** — didn't move the actual MSE number after a real run. Checked
empirically: `scipy`'s `RegularGridInterpolator(fill_value=0)` /
`map_coordinates(mode='constant',cval=0)` both zero the **whole
interpolation cell** the instant any corner is OOB, not per-tap. Reverted
CPU change; fixed the 3 GPU kernels to match instead
(`bp_buffer.cl`, `bp_image.cl`, `bp_buffer_opt.cl`).

## fp_cpu truncation bug (2026-08-17)

Even after the boundary fix, CPU-vs-GPU outlier persisted at the same
magnitude — sign the fix wasn't the real cause. Root-caused via dedicated
diagnostic scripts (`diag_voxel.py`, `diag_voxel2.py`, `diag_fp.py`)
isolating fp/bp from the full MLEM loop, comparing term-by-term against
the Python reference at single voxels. Found: `fp_cpu`'s ray march used
`(int)xi` (truncates toward zero) instead of `floorf(xi)` — for `xi` in
`(-1,0)`, `(int)xi` gives `0` instead of `-1`, wrongly passing an
unsigned bounds check and sampling a real voxel with a bogus weight for a
ray that should've missed the volume. GPU kernels already used `floor()`
correctly. Fixed; MSE dropped >4 orders of magnitude across every mode.

## Python-reference outlier-voxel bug (2026-08-27/28)

MSE vs the course Python reference (`Topic_2_CTreconstruction.py`) was
dominated by a handful of extreme outlier voxels (26 on GTX 680, 24 on
AMD Hawaii PRO, heavily overlapping indices, values up to 521 vs a
normal ~0-2.6 range).

Initial hypothesis (MLEM normalizer `bp_ones` near-zero somewhere,
causing divide blowup) — disproved directly via `python/diag_bp_ones.py`:
`bp_ones` is completely normal (364-443) at every outlier voxel, same
range as everywhere else.

Real mechanism: each outlier voxel's per-epoch multiplicative ratio is
consistently ~1.29-1.37 (backed out from `521.6699^(1/20)` etc. matching
observed finals from v0=1) — 20 unclamped MLEM iterations compound a
small persistent drift into a large final value. A real numerical-
stability gap in the unmodified reference script at ill-conditioned
voxels (likely sparse/near-tangent ray coverage), not a bug in this
project's C/GPU/CPU code — all 4 modes show identical MSE vs the
reference, meaning the reference's own output is the outlier.

**Fix attempt 1 (wrong, caught before a wasted run):** per-epoch ratio
clamp `[0.5, 2.0]`. A standalone Python simulation using the real
observed ratio (~1.3) showed the clamp never triggers (1.3 is already
inside range) — compounding continues regardless. Verified this held
even at a much tighter `[0.909, 1.1]` bound (still reaches ~6.7x after 20
epochs).

**Fix (real, shipped):** clamp `v0` itself to `[0, 5]` after each update,
in both `Topic_2_CTreconstruction.py` and its 512³ variant. Verified via
simulation against all 3 real observed ratios (1.29, 1.33, 1.37) — all
cap at exactly 5.0. Course staff (Guangpu Yang) explicitly approved
fixing the reference script after being emailed about the finding.

**Confirmed with a real 20-epoch run** on AMD Hawaii PRO (2026-08-28):
MSE vs Python Ref `6.371e-03` → `3.014e-04` (~21x better, matches the
~0.00030 estimate from manually excluding the 24 outliers weeks
earlier), zero outliers remain, `python_ref` max exactly `5.0000`.

One scare along the way: first post-run check used a stale `EPOCHS=10`
`output_cpu.hdf5` on pool15-01 (smoke test, not the real 100-epoch
reference) — caught because its `max` (1.4189) didn't match every other
documented value (1.7303) for that file. Re-ran against the correct
archived 100-epoch file for the real confirmed number.

## gpu-img/gpu-opt max-gap investigation (2026-08-26)

`gpu-img`/`gpu-opt` MSE-vs-CPU (1.949e-07 at 256³) is ~300x looser than
`gpu-buf`'s (6.436e-10) — worth understanding properly since course
grading criteria explicitly requires GPU speed not compromise quality.

**First hypothesis, wrong, caught by measurement:** `1/U²` geometric
singularity (a documented comment already suspected this). Wrote
`diag_maxgap.py` to check directly rather than assert — measured min|U|
at the worst voxel = 4.10 (82.1% of SOD), max `1/U²` amplification only
1.5x. No blowup possible. Re-checked at 512³: min|U|=9.57 (90.9% of
SOD), 1.2x amplification. Ruled out at both scales.

**Also checked and ruled out:** missing `+0.5f` half-texel offset before
`read_imagef` (grepped every call site in `fp_image.cl`/`bp_image.cl`/
`bp_buffer_opt.cl` — all present and consistent).

**Real finding:** a one-voxel spatial displacement of sharp features —
mass roughly conserved across the displaced voxel pair (~2% difference),
not a magnitude error. Consistent with the GCN hardware sampler
quantizing the interpolation fraction to a small number of bits (~8,
common folklore figure) — fits every observed fact (identical in smooth
regions, one-voxel shift only at sharp edges, worse at large radius) but
was explicitly **not independently verified against AMD hardware docs**
— stated as a data-confirmed pattern, not a proven mechanism.

Affects 21/16.7M voxels at 256³, 138/134M at 512³ (>100×RMS),
concentrated at the FOV edge (100% beyond 0.8×FOV radius at 512³, vs 62%
at 256³ — got *more* concentrated at higher resolution). RMS as % of
signal range *improves* with resolution: 0.0255% (256³) → 0.0026%
(512³). Genuinely good result — the GPU path gets relatively more
accurate as the problem gets larger.

`gpu-buf` (manual float32 trilinear, no hardware sampler) has none of
this — recommended when bit-level CPU fidelity matters more than speed.

### Hybrid-precision investigation (2026-08-26, not shipped)

Explored fixing the gap directly: `CLK_FILTER_NEAREST` sampler + manual
float32 bilinear blend in `bp_image.cl`/`bp_buffer_opt.cl`, gated by
radius + gradient threshold, delivered as a `-DHYBRID_PRECISION`
compile flag (verified byte-identical to baseline when unset).

Measured before building: contrary to the initial guess (fp should
dominate error, more reads/ray), **bp dominates by ~100,000x** (fp RMS
3.665e-10, bp RMS 3.986e-5) — fp's many samples per ray average out
sampler error, bp's one sample per angle doesn't.

**Real result: net negative.** MSE got *worse* with more manual coverage
(not better), the same voxel `(244,66,17)` disagreed by the same amount
at every coverage level, `-cl-fast-relaxed-math` ruled out as the cause
(byte-identical with/without). Conclusion: "MSE vs CPU" isn't "MSE vs
ground truth" — both the CPU path and the hardware sampler are
approximations of the true continuous integral; nothing guarantees the
manual scheme is closer everywhere, and there's no ground-truth volume
in this project's datasets to arbitrate. Not shipped, gated off by
default — `gpu-buf` already satisfies the "no compromise" requirement
without touching kernel code.

Real reusable byproducts: `gpu_op_fp`/`gpu_op_bp` (per-operator GPU
isolation infra, reused later by the gpu-buf speed investigation),
`diag_op_attribution.py`, and one real bug fix (`gpu_op_fp` missing a
`build_RT_buffers` call, causing 6.7% NaN pixels).

## Closing the gpu-img/gpu-opt 256³ gap (2026-09-01)

Reviewer feedback set an accuracy bar of "MSE at 1e-8 level is good".
Exactly one reported number missed it: `gpu-img`/`gpu-opt` at 256³,
`1.1278e-07`. (512³ already passed at `1.232e-09`; `gpu-buf` passed
everywhere; OSEM's `1.061e-04` was explicitly excused as a different
update path.)

**Resolved.** `FP_TEX_EXACT=1` reaches `2.524e-09` — 45× better, clearing
the bar — at 21.34 s vs the 14.65 s baseline (1.46×). Defaults unchanged.

### What worked, and why

The error was in the FORWARD projection, not backprojection. The hardware
sampler bundles two separable things: a 3D-tiled texture cache (the source
of `gpu-img`'s speed) and a fixed-function interpolation unit whose blend
weights carry less than float32 precision (the source of its error).
`CLK_FILTER_NEAREST` returns exact stored texels with no blending, so eight
nearest fetches plus a manual float32 trilinear keep the cache and drop the
lossy blend.

Two implementations reach identical accuracy:
- `--fp-buf` routes fp through `gpu-buf`'s kernel: `2.524e-09`, 26.82 s.
- `FP_TEX_EXACT=1` keeps the texture cache: `2.524e-09`, 21.34 s.

The 5.5 s difference is the texture cache, and it confirms the diagnosis —
`--fp-buf`'s cost was the abandoned cache, not the float32 math. The two are
not bit-identical (8955 vs 8958 outliers) since texture storage round-trips
through the image format.

### Four falsified hypotheses

Each was tested directly and killed by its own measurement. Recorded because
the failures constrain the explanation more than the success does.

1. **Backprojection sampler (ported hybrid-precision to current code).**
   Full-coverage manual float32 blend in `bp_image`/`bp_opt`, sampler fully
   bypassed: `1.128e-07` → `1.061e-07`, only 6%, at 55-58% slower. This also
   re-tested the 2026-08-26 negative result at 100 epochs rather than 10 —
   the suspicion that its conclusion was an artifact of the short-run regime
   was itself wrong; it holds at both.
   *Why it failed:* single-operator attribution had measured bp error as
   ~100,000× fp's (`3.986e-5` vs `3.665e-10`), but that is instantaneous
   error at one iteration. FP feeds the ratio driving every MLEM update, so
   its error re-enters the loop each epoch; BP's averages over 75 angles.
   Over 100 iterations the ranking inverts.

2. **`bp_ones` amplification / FOV mask.** Theory: near-zero `bp_ones` at
   FOV-edge voxels amplifies tiny differences through `v *= bp_ratio/bp_ones`.
   Swept the threshold over {1e-4, 1e-3, 1e-2} relative to max — all three
   **bit-identical** to baseline. Then measured `bp_ones` directly
   (`BP_ONES_DUMP` + `python/diag_bp_ones.py`): range **364 to 443**, minimum
   at **82% of max**, zero voxels below 1e-10. At the worst-diff voxel
   `bp_ones = 414` (93% of max), where `1/bp_ones` *attenuates* by 400×.
   The precondition simply does not exist in this geometry; the `> 1e-10f`
   guard has never fired. An epsilon-floor variant fails for the same reason.

3. **IEEE intrinsics.** Replacing `native_recip`/`native_sqrt` with
   correctly-rounded `/` and `sqrt()`: MSE `1.128e-07`, unchanged to four
   digits, same worst voxel. Costs nothing (14.72 s vs 14.65 s) and buys
   nothing. The error is in how the volume is *sampled*, not in the geometry
   arithmetic.

4. **Two-phase warm start.** Proposed running most epochs fast and a final
   clean-FP "polish", on the premise that MLEM's contraction means only late
   iterations set final precision. Falsified by measuring fp-buf-only MSE at
   increasing epoch counts: `5.202e-04` (5 ep), `2.986e-04` (10), `1.978e-04`
   (15), `1.026e-04` (25) — still five orders above the converged `2.524e-09`.
   MLEM has not converged by 25 epochs from any starting point, so a short
   clean tail cannot wash away earlier drift. MLEM is multiplicative: each
   update scales what is already there.

## AMD vs NVIDIA MSE gap — open question, not resolved

At 256³, AMD Hawaii PRO's `gpu-buf` MSE (`6.436e-10`) is ~5.6x worse than
GTX 680's (`1.148e-10`) — real gap, larger than expected for `gpu-buf`
specifically, since `gpu-buf` never touches a hardware texture sampler
(manual float32 interpolation only) — "texture-sampler rounding" can't
explain this part of the gap. `-cl-fast-relaxed-math` (always on,
`src/ct_gpu.c`) is a plausible cause — the OpenCL spec leaves its exact
numerical behavior vendor-defined. A prior investigation confirmed it
does NOT explain `gpu-img`/`gpu-opt`'s *within-GPU* precision floor on a
single machine (rebuilt with the flag stripped, byte-identical output —
see `experimentation/mixed-precision-backprojection/kernels/bp_image.cl`), but that
test never compared the same build across AMD vs NVIDIA — the actual
open question. **Not resolved** — would need a real AMD-hardware rebuild
with the flag stripped to test directly.

## Validation tables (100 epochs, both datasets, all four modes)

Raw min/max/mean/MSE numbers for report tables. `python_ref` rows are
the course Python reference (`Topic_2_CTreconstruction.py`); pre/post
refer to the v0-clamp fix above.

**AMD Hawaii PRO, 256³ (unfixed python_ref):**
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   247.7164 0.0070    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=6.371e-03
gpu-buf     0.0000   1.7307   0.0067    0    0  MSE=6.436e-10  MSE=6.371e-03  max=0.0511
gpu-img     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=6.371e-03  max=1.1490
gpu-opt     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=6.371e-03  max=1.1490
```

**AMD Hawaii PRO, 256³ (v0-clamp fix applied, 2026-08-28, real 20-epoch python_ref run):**
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   5.0000   0.0069    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=3.014e-04
gpu-buf     0.0000   1.7307   0.0067    0    0  MSE=6.436e-10  MSE=3.014e-04  max=0.0511
gpu-img     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=3.016e-04  max=1.1490
gpu-opt     0.0000   1.8741   0.0067    0    0  MSE=1.949e-07  MSE=3.016e-04  max=1.1490
```

**AMD Hawaii PRO, 512³:**
```
Mode       min      max     mean   nan  inf  MSE vs CPU     MSE vs Python Ref
cpu      0.0000   1.0055   0.0330    0    0  (reference)
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.087e-11  max=0.0156
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=6.849e-10  max=0.0169
```

**NVIDIA GTX 680, 256³ (unfixed python_ref):**
```
Mode        min      max      mean   nan  inf  MSE vs CPU     MSE vs Python Ref
python_ref  0.0000   521.6699 0.0070    0    0  (python ref)
cpu         0.0000   1.7303   0.0067    0    0  (reference)    MSE=2.061e-02
gpu-buf     0.0000   1.7292   0.0067    0    0  MSE=1.148e-10  MSE=2.061e-02  max=0.0203
gpu-img     0.0000   1.6577   0.0067    0    0  MSE=1.128e-07  MSE=2.061e-02  max=0.8966
gpu-opt     0.0000   1.6577   0.0067    0    0  MSE=1.128e-07  MSE=2.061e-02  max=0.8966
```

**NVIDIA GTX 680, 512³:** no python_ref (OOM on both machines, see
project-state.md).
```
Mode       min      max     mean   nan  inf  MSE vs CPU     MSE vs Python Ref
cpu      0.0000   1.0055   0.0330    0    0  (reference)
gpu-buf  0.0000   1.0054   0.0330    0    0  MSE=9.534e-11  max=0.0114
gpu-img  0.0000   1.0054   0.0330    0    0  MSE=1.232e-09  max=0.0186
gpu-opt  0.0000   1.0054   0.0330    0    0  MSE=1.232e-09  max=0.0186
```

MSE vs CPU, GTX 680: 256³ `1.1477e-10` (gpu-buf) / `1.1278e-07`
(gpu-img/gpu-opt); 512³ `9.534e-11` / `1.232e-09`. RMS as % of signal
range: 256³ 0.0194%, 512³ 0.0026% (gpu-img/gpu-opt) — improves at higher
resolution. `gpu-buf`'s margin over gpu-img/gpu-opt is scale-dependent,
not fixed: ~980x tighter at 256³, only ~13x tighter at 512³ — gpu-buf is
already at the float32 noise floor with nowhere to improve, while
gpu-img/gpu-opt's systematic displacement error (see max-gap section
above) is diluted by more total voxels at 512³.

Pure-Python fp/bp (scipy `RegularGridInterpolator` rebuilt per-angle, no
vectorization/GPU): ~4690s/epoch at 256³ on GTX 680, script default
`sample_ratio=2`. Reduced to `sample_ratio=1` (call-site only, matches
C/GPU modes' default sample count of `n_samples=Nxz=256` at 256³) —
measured ~3679s/epoch, only ~21% faster (most cost is fixed per-pixel
Python/interpreter overhead across the 1.3M-pixel loop, not proportional
to ray length). Full 100 epochs at this rate ≈ 4.3 days; `Epochs` set to
20 (~20hrs) instead, run under `tmux -s topic2`, completed 2026-08-27.
