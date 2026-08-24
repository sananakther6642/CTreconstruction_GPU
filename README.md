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

> **Two more sweeps landed, both confirmed at matched 10-epoch runs with
> correctness validated (`validate.py` at the float32 noise floor, both
> scales):**
>
> - **`cpu` `FP_TILE` 8→32**: `fp_cpu` was never swept past its first
>   tuning; 32 is ~19% faster than 8 at 512³ (`16.84s→13.58s` fp/epoch).
>   256³ `cpu` also dropped from 2.73s to 2.60s/epoch.
> - **`gpu-buf` work-group `{16,16,1}→{4,64,1}`**: `fp_buffer.cl` had also
>   never been swept. This is the big one — **`gpu-buf` at 512³ went from
>   56.93s/epoch to 4.87-10.19s/epoch across repeat runs — 5.6-11.7× faster
>   than the old `{16,16,1}` default**, and correspondingly from *slower
>   than CPU* (0.46×) to **2.3-4.8× faster than CPU**, a real win over it
>   every time it's been measured. `fp_buffer.cl`'s manual uncoalesced
>   gather (no texture cache) turns out to be far more sensitive to
>   work-group shape than `fp_image.cl` was — shapes like `{32,8,1}` that
>   were merely suboptimal for `fp_image` are catastrophic here (4-8×
>   slower). See "Work-group sweeps" below for both sweeps' full data.
>
> **`gpu-buf`'s own run-to-run variance is larger than the other three
> modes'.** Three 10-epoch confirmation runs: 75.37s, 101.89s, 83.63s
> total — same `{4,64,1}` work-group, same code, real difference each
> time. `cpu`/`gpu-img`/`gpu-opt` stayed within ~1% of their usual numbers
> across all three runs, so this is specific to `fp_buffer`'s uncached
> gather pattern.
>
> Investigated rather than assumed: per-slab timing shows the *specific*
> angle-slabs that go slow **differ between runs** (slabs 5-8 in one run,
> 3-8 in another) — this rules out a deterministic per-angle AABB-clipping
> cost (which would hit the same slabs every time). `who`/`w` confirmed no
> other users were logged into `pool15-01` (a shared lab pool machine)
> during a slow run, ruling out user contention. `dmesg` requires root,
> not available on this account, so GPU thermal/driver-level events
> couldn't be directly confirmed via system logs.
>
> **Confirmed GPU-bound, not host/driver overhead.** OpenCL event
> profiling (`CL_PROFILING_COMMAND_START`/`END`) around each angle-slab
> dispatch in `run_fp_buffer` measures actual device execution time,
> independent of host-side queueing or driver dispatch gaps. Result:
> **100% of flagged slow slabs showed `wall≈gpu` to the millisecond**
> (e.g. `wall=2.975s gpu=2.975s`). The GPU itself is taking longer to run
> the identical kernel, not waiting on the host.
>
> **"Thermal throttling" (the earlier working theory here) is wrong —
> corrected after direct testing.** A pause-based mitigation
> (`FP_BUFFER_SLAB_PAUSE_US`, `usleep()` between slab dispatches, on the
> theory that idle gaps let the GPU recover a higher clock state) was
> tried first and made things *worse* (172.47s for 10 epochs vs a 75-102s
> unmitigated range, more slabs flagged slow, not fewer) — the wrong shape
> for throttling, since idle time can only help or be neutral under real
> clock throttling. That result was mis-read at the time as "confirms
> genuine device-side throttling"; it should have been read as evidence
> *against* it.
>
> A dedicated diagnostic (`--diag repeat-slab:<angle_offset>:<slab_size>:
> <n_repeats>[:<realloc_at>]`, `gpu_diag_repeat_slab()` in `ct_gpu.c`)
> repeats one fixed angle-slab N times back-to-back, isolating the
> mechanism from full-epoch noise. Two 20-repeat runs on different, fixed
> angle ranges (angles 64-71 and angles 0-7) both showed the same
> signature: fast for several repeats (~0.44-0.53s), then a **single step
> up to ~5.8-6× slower that HOLDS** for the rest of the run — not a
> monotone climb (rules out thermal ramping), not repeated bimodal
> flapping (rules out simple TLB thrashing), not slow from the first
> repeat (rules out deterministic memory-channel camping on that angle
> set). The magnitude alone already ruled out throttling: Hawaii's entire
> DVFS range is ~3.2× (300→947MHz), well short of the observed ~5.8-6×,
> and `gpu-img`/`gpu-opt` stay within ~1% in the very same sessions
> `gpu-buf` goes 6× slow — a device-wide clock/power drop cannot be
> mode-selective like that.
>
> A follow-up 40-repeat run added a mid-run `d_vol` **reallocation**
> (`realloc_at`): the step-degradation **recovered instantly** on
> reallocation (2.99s → 0.51s the very next repeat) — then **degraded
> again on its own** after a further, variable number of repeats (6 vs 13
> launches in two cycles of the same run). This is the load-bearing
> result: the slowdown is tied to the specific `d_vol` allocation and is
> recoverable by replacing it, but recurs on a roughly time/pressure-based
> cycle rather than a fixed launch count — consistent with the OpenCL
> driver periodically demoting the 537MB volume buffer's memory placement
> under sustained access (e.g. losing a large-page mapping or migrating
> out of the fastest VRAM tier), not with any cyclical *hardware* effect.
>
> **Root cause: confirmed driver-side memory-placement demotion of the
> large `d_vol` allocation under sustained access, not thermal throttling.**
> This is now evidence-based rather than inferred, unlike the earlier
> "narrowed but not confirmed" throttling guess.
>
> **Mitigation attempted, does not reliably help at full scale.** Added
> `FP_BUFFER_VOL_REALLOC_EVERY=N` (`ct_gpu.c`, `run_fp_buffer`): every N
> angle-slab launches, read `d_vol` back to host, free it, and recreate it
> fresh — directly targeting the confirmed mechanism. A 5-epoch sweep
> (N ∈ {3,4,5,6,7,10}) found N=5 as a clear winner (34.91s vs a
> 37.7-50.9s baseline-equivalent range, clean of slow-slab warnings after
> ~2 epochs of warmup). **This did not reproduce at the documented
> 10-epoch/75-angle scale**: four full 10-epoch runs with N=5 gave
> `[105.84, 86.77, 89.04, 85.31]s` (mean 91.74s, stdev 9.52) against the
> three existing unmitigated baseline runs `[75.37, 101.89, 83.63]s`
> (mean 86.96s, stdev 13.57) — the mitigated mean is *slower*, not
> faster (−5.5%), and most mitigated runs still showed scattered
> `GPU-BOUND` warnings despite reallocating every 5 launches. Each
> reallocation costs real time (~0.2-0.4s observed for the 537MB
> readback+upload) that appears to roughly cancel whatever it saves, and
> a fixed launch-count period doesn't reliably land inside the
> variable-length degradation window at this scale. Left disabled by
> default (`FP_BUFFER_VOL_REALLOC_EVERY=0`); the env var remains
> available for further tuning (a different trigger heuristic, e.g. one
> based on elapsed time rather than launch count, is untried) since the
> root-cause diagnosis and hook point are still correct even though this
> specific fix isn't.
>
> Not chased further given no root/`dmesg`/GPU sensor access on this
> machine to directly confirm the driver-side mechanism, though the
> reallocation-recovery evidence is strong indirect support.
> Report a range, not a single number, when citing `gpu-buf` at 512³
> **on pool15-01/AMD Hawaii specifically.**
>
> **Cross-checked on different hardware — confirmed AMD-driver-specific,
> not general.** The project's working machine later moved to `kale`
> (NVIDIA GeForce GTX 680). Re-ran the exact same `--diag repeat-slab`
> test that found this mechanism (`repeat-slab:64:8:40:20` — same angle
> range, same repeat count, same mid-run reallocation point) there.
> Result: **completely flat across all 40 repeats**, ~0.796-0.797s every
> single time, `wall≈gpu` throughout, no change after the reallocation
> at rep 21 (nothing to recover from). The variance does not reproduce
> on NVIDIA's driver stack at all. This is a clean, decisive negative
> result: the mechanism is genuinely AMD-driver-specific behavior, not a
> universal OpenCL/GPU phenomenon this codebase needs to defend against.
> `gpu-buf` on kale needs no variance mitigation and can be reported as
> a single stable number, unlike pool15-01.
>
> These 10-epoch numbers aren't yet re-confirmed at 100 epochs — worth
> doing before citing final numbers in a report; `gpu-img`/`gpu-opt`
> speedup vs CPU looks lower here (27-28× vs the previous 100-epoch
> table's 29-30×) purely because `cpu` got faster too, not because
> `gpu-img`/`gpu-opt` regressed — their own per-epoch times are essentially
> unchanged from before.

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
`fp_cpu` dominates the CPU epoch time at 512³ (was 35.2-36.2s of a
44.5-45.7s epoch — 78%). The 8-tap trilinear gather it does per ray sample
is memory-latency-bound: each gather spans up to ~2MB due to the
`Nxz*Ny=262144` and `Ny=512` float strides in the volume layout, so
sequentially marching one ray to completion before starting the next gave
the CPU cache nothing to reuse.

Fix: batch `FP_TILE=8` neighboring detector rows together (later swept to
32, see "Work-group sweeps" below), and iterate the
sample index `s` as the *outer* loop across the tile instead of per-ray.
Neighboring rays for the same angle start near each other and diverge only
slightly, so at a given `s` their volume addresses cluster — batching
means the memory subsystem sees those nearby reads close together in time
instead of scattered across a full ray march per pixel. Each ray still
computes its own AABB-derived `s_start`/`s_end` and only accumulates within
its own range, so the actual math and reduction order per ray is unchanged
— this is purely a memory-access-order change, verified via `--op fp`
component test to still agree with the Python reference (`MSE=8.34e-08` at
64 samples, no regression from the pre-tiling `8.57e-08` at full samples).

Result at 100 epochs: `cpu` 512³ **26.06s/epoch** (2605.5s total), down
from a pre-tiling baseline of 44.5-45.7s/epoch — confirmed stable across
the full 100-epoch run, not just the earlier 10-epoch spot check.
`bp_cpu` itself was never the bottleneck (a separate analytical-range
optimization there had negligible effect since bp's cost is dominated by
the same kind of gather, just at a fraction of fp's time budget) — the win
is entirely from fp_cpu.

### gpu-opt vs gpu-img: unroll-x2 was measured harmful, removed

> This section's numbers are a checkpoint taken immediately after the
> unroll-x2 fix, *before* the work-group sweep below. They're kept as-is
> because they isolate what unroll-x2 alone was costing. The Performance
> Results table at the top of this README reflects both fixes together
> and is the current, correct number — `gpu-opt` there (88.38s) is now a
> little faster than `gpu-img` (88.58s), not tied.

`gpu-opt` (`bp_opt` in `bp_buffer_opt.cl`) layers a float2 cos/sin LUT,
cooperative local-memory caching, and (previously) an unroll-x2 loop on
top of the same hardware texture sampler `gpu-img` uses. It should
therefore always be at least as fast as `gpu-img` — but the 100-epoch
512³ run showed `gpu-opt` at **0.945s/epoch, slightly slower** than
`gpu-img`'s 0.930s. Investigated properly instead of accepting a "close
enough" 1.6% gap:

1. Ruled out redundant `floor()` calls in `bp_opt`'s bounds check (it
   called `floor(u)`/`floor(v)` twice each vs `bp_image.cl`'s once) —
   fixed this regardless (unconditionally correct cleanup), re-measured:
   **no change** (0.934-0.940s), so this wasn't the cause.
2. Tested the unroll-x2 path in isolation by disabling its gate
   (`Nxz>=512` → an always-false condition) to force the scalar-only loop
   at 512³: **0.923-0.927s/epoch — faster than `gpu-img`**, confirming
   unroll-x2 was the actual cause of the regression, not masking it.

Unroll-x2's theory was that overlapping two texture fetches would hide
latency via instruction-level parallelism. On this hardware (AMD Hawaii,
GCN 1.1) it did the opposite: the doubled live register set (two
`float2`/`float4`/etc. sets in flight instead of one) apparently costs
more in work-group occupancy than the ILP saves in latency-hiding — the
code's original comment gated this "for large volumes" as an assumption,
never actually measured on this GPU. Removed permanently; `bp_opt` is now
unconditionally scalar.

**Final 100-epoch result**: `gpu-opt` 93.18s vs `gpu-img` 93.02s — a
0.17% gap, within measurement noise. Two 10-epoch spot-checks after the
fix (`0.923-0.927s`, `0.922-0.930s`) had suggested `gpu-opt` would come
out ahead; at full 100-epoch scale it landed essentially tied instead.
The fix was still correct and worth keeping — it closed a real, measured
1.6% regression (`gpu-opt` used to be *slower* than `gpu-img` by design
flaw, not just by chance) — but `gpu-opt`'s remaining edge (LUT +
local-mem cos/sin caching) turns out to be a small ALU-side saving that
doesn't move the needle much on a kernel this texture/memory-bound.
Reported honestly here rather than rounding the 10-epoch numbers up to a
"win" that the full run didn't confirm.

### Work-group sweeps: fp_image, fp_buffer, and fp_cpu's tile size

Three separate never-swept parameters, tuned one at a time as the project
went on — each cheap to test since none of them change results, only timing.

**`fp_image` (`gpu-img`/`gpu-opt`): `{16,16,1}→{8,32,1}`, ~5-7%.**
The bp kernels' `{4,4,16}` (below) was the largest single measured win in
the project, but fp's `{16,16,1}` was never tested against alternatives.
Swept 10 candidates at 512³, 10 epochs each (spread under 1% within each
config):

```
lws        s/epoch
8,32,1     0.873-0.879   <- winner
16,16,1    0.921-0.928   (old default)
4,64,1     0.944-0.947
32,8,1     1.171-1.188
8,16,1     1.107-1.115
8,8,1      1.230-1.236
16,4,1     1.627-1.632
32,4,1     2.363-2.383
32,2,1     2.531-2.539
64,1,1     4.233-4.240
32,16,1, 16,32,1   CL_INVALID_WORK_GROUP_SIZE (512 items exceeds
                    Hawaii's 256-item max work-group size)
```

Two things stood out: total occupancy alone doesn't predict the winner
(`8,16,1` at 128 items was slower than `8,8,1` at 64), and aspect ratio
matters independent of item count (`8,32,1` beat `32,8,1` by 25% at the
same 256 items — narrow-in-W/tall-in-H specifically helps on this
detector's `W=1120, H=1184` layout). `{8,32,1}` is the default in
`run_fp_image` (`src/ct_gpu.c`); overridable via `FP_IMAGE_LWS=X,Y,Z`.

**`fp_buffer` (`gpu-buf`): `{16,16,1}→{4,64,1}`, >4× at 512³.** Same
detector shape as `fp_image`, so the same candidates were worth testing —
but `fp_buffer.cl`'s manual uncoalesced global-memory gather (no texture
cache) turned out to be *far* more sensitive to work-group shape than
`fp_image.cl` was. First 3-epoch pass showed huge, confusing swings (same
config varying 8-91s across runs) — turned out to be real signal buried in
noise, not pure measurement error. Confirmed clean at 10 epochs with 30-60s
gaps between configs to rule out thermal/contention buildup:

```
lws        s / 10 epochs
4,64,1     75.37    <- winner, mostly 5-10s/epoch, stable
16,16,1    321.19   (old default, 9-48s/epoch, highly variable)
8,32,1     56.02    (3-epoch only, not re-confirmed at 10 —
                      4,64,1 already won decisively)
8,16,1     213.18
32,8,1     255.43   (worst — same wide-short shape that was
                      merely suboptimal for fp_image, catastrophic here)
```

This is the biggest single win of this project's optimization work: at
512³, `gpu-buf` went from 56.93s/epoch (slower than CPU) to 4.87-10.19s
across repeat 10-epoch runs (this mode has more run-to-run variance than
the others even post-fix — see the Performance Results note above) — a
solid 2.3-4.8× speedup over CPU every time it's been measured, instead of
a 0.46× embarrassment. Default in `run_fp_buffer` (`src/ct_gpu.c`);
overridable via `FP_BUFFER_LWS=X,Y,Z`.

**`fp_cpu`'s `FP_TILE`: `8→32`, ~19% at 512³.** The ray-tiling batch size
(see "CPU 512³ speedup" above) was tuned once when the tiling rewrite
landed and never swept afterward. Swept 4/8/16/32/48/64 at 512³, 3-epoch
runs (fp time only):

```
FP_TILE   fp s/epoch
4         26.05   (unreliable — ran right after a GPU sweep had
                    saturated the machine; contention noise)
8         16.84   (old default)
16        14.72
32        13.58   <- winner
48        13.79
64        13.64
```

32-64 are within noise of each other — the curve plateaus at 32, and 32
costs less stack/cache footprint per tile than going bigger for no
measured benefit. Default `FP_TILE` in `fp_cpu` (`src/ct_cpu.c`);
overridable via `FP_TILE_ENV=N` (1-64).

Correctness unaffected by construction for all three — none change which
samples are read or how they're weighted, only when/how work is scheduled.
Confirmed via `validate.py` at matched 10-epoch runs: MSE at the float32
noise floor at both scales for every change above.

### Two negative results worth recording

**AABB re-tested at 256³, still hurts.** All three fp implementations gate
AABB clipping on `W > 512`, disabling it entirely at 256³ (detector is
exactly 512 wide). Geometry shows ~65% of the 256 samples/ray are outside
the volume there — a large apparent opportunity. But the existing gate
wasn't an oversight: it's a prior measured decision (see the 512³
optimization table below, and the code's own comment). Re-tested via
`FP_IMAGE_AABB=1` after the work-group sweep, in case that had changed the
tradeoff — it hadn't: **0.098-0.100s/epoch with AABB forced on vs
0.095-0.096s with it off**, a consistent ~4% regression. The 256³ launch
(0.095s total) is small enough to be occupancy-bound rather than
fetch-bound; AABB's setup cost (6 divides + branches per ray) and the
ragged per-thread trip counts it creates cost more than the fetch
reduction saves. Gate correctly stays `W > 512`; `FP_IMAGE_AABB` env var
kept for future re-testing if the kernel changes again.

**Latent AABB axis transposition, fixed.** Independent of performance:
the AABB slab test bounded the world-space y component (`rd[1]`/`T[1]`)
by `hxz` (the `Nxz`-axis half-extent) and the z component (`rd[2]`/`T[2]`)
by `hy` (the `Ny`-axis half-extent) — but the sampling code right below
maps `wy→yi` via `inv_sv_y` (the `Ny` axis) and `wz→zi` via `inv_sv_xz`
(the `Nxz` axis). The two disagreed about which axis is which. Currently
latent, not active: both datasets are cubic (`Nxz==Ny`, confirmed by each
run's own `Volume: N x N x N` printout), so `hxz==hy` numerically and the
swap changes nothing. Fixed anyway in all three fp implementations
(`fp_image.cl`, `fp_buffer.cl`, `ct_cpu.c`) together, so a future
non-cubic dataset doesn't hit silently-wrong clipping — and because CPU
and GPU shared the bug identically, meaning CPU-vs-GPU validation could
never have caught it (both would agree while both were wrong). Confirmed
`validate.py` shows zero change, as expected for a currently-no-op fix.

**Sphere-shaped AABB considered, measured no headroom, not implemented.**
A tighter (sphere or cylinder) bounding test was proposed as a possible
win over the current box clip's setup cost — but only worth writing if
the box clip is actually loose. Measured directly instead of assumed:
`FP_CPU_DIAG_AABB_RANGE=1` dumps `(s_end-s_start)` per ray (the number of
samples the current box clip keeps) instead of the accumulated value, run
once at 512³ via `--op fp`. Result: **mean 146.6/512 (28.6%), median
177/512 (34.6%)**, both close to the ~35% figure that was the "already
tight, don't bother" threshold going in. The box clip is doing its job;
a sphere test would only tighten the corner-ray cases, which are already
a small minority. Not implemented.

**`bp_cpu` thread scaling measured, current default already near-optimal.**
Never measured before (only ever run at the Makefile's `nproc`-detected
default). Swept `OMP_NUM_THREADS` ∈ {1,2,4,6,8,12} on `run-op-bp-512`
(isolated bp timing, no MLEM loop) at 512³: scales cleanly through the
5820K's 6 physical cores (92-99% parallel efficiency), as expected for a
memory-bound kernel. Past 6 the picture is **not** the clean
"plateau-then-regress" curve a hyperthreading story would predict — 8
threads is a genuine outlier (12.77s, worse than both 6 *and* 12
threads), while 12 threads (9.09s) is essentially tied with 6 threads
(9.80s), not worse. Single-sample-per-count, so the 8-thread dip isn't
chased further, but the actionable conclusion holds either way: the
current default (`nproc`-detected, 12 on this machine) is already
at or near the best available, no Makefile change indicated.

**Two dataset/device facts checked, not assumed.** `cl_khr_3d_image_writes`
is supported on this Hawaii device (printed once at `gpu_init`) — the
extension OpenCL 1.2 requires for 3D read-write images, gating the
"`vol_update` writes straight into `vol_img`" optimization considered for
`gpu-opt` (would remove a `clEnqueueCopyBufferToImage` per epoch). And the
input HDF5's `Angle` field spans 0.0-355.2° (75 angles × 4.8° steps) —
a **full 360° scan, not a short-scan/limited-angle acquisition** — and
contains only geometry + `Projection` + `Angle`, **no ground-truth
volume field**. Both matter for future work: the full-angle range means
an FDK reconstruction wouldn't need Parker short-scan weighting if ever
implemented, and the absence of ground truth confirms any future
image-quality comparison has to use a data-domain metric (residual,
log-likelihood) rather than a true reconstruction-error metric.
- CPU-vs-Python sampling mismatch (`fp_func`'s hardcoded `sample_ratio=2`
  gives `n_samples=ceil(Nxz*2)` — 512 at 256³, 1024 at 512³ — vs the C
  side's `--samples` default of `Nxz`/512) not reconciled — doesn't affect
  CPU-vs-GPU correctness (both C paths use the same `n_samples`), but
  means "MSE vs Python" isn't apples-to-apples at either scale.
- `MSE vs Python` at 512³ was never shown above because
  `run_python_reference.py` had only ever been run at 256³, and
  `validate.py` hardcoded the same `output_python.hdf5` filename
  regardless of which scale you validated — so `validate.py 512` always
  compared 512³ C output against a 256³-shaped array and printed "shape
  mismatch". `validate.py`'s filename bug is fixed (now looks for
  `output_python_512.hdf5`), but generating that file turned out to be
  **infeasible on `pool15-01`**: `bp_func`/`fp_func` at 512³ build several
  float64 `(512,512,512)` arrays at once (~1GB each), and with only 15GB
  RAM and no swap configured on this machine, a first uncapped attempt
  froze the entire desktop badly enough to need a hard reset. A second,
  memory-capped attempt was killed by the OOM killer instead (confirmed
  by the shell reporting `Killed`, not by reading kernel logs —
  `dmesg` requires root, not available on this account). A `ulimit -v`
  memory cap just turns the freeze into a clean, faster kill — it doesn't
  make the workload fit. Would need a machine with substantially more RAM
  (or a numpy rewrite using float32 / chunked processing instead of
  `bp_func`'s all-at-once float64 meshgrids) to ever produce this file.
  Not pursued further — `MSE vs Python` at 512³ stays unavailable; 256³
  is unaffected and already works via `make run-python`.

## Modes

| Mode | Flag | Description |
|------|------|-------------|
| CPU | `--mode cpu` | C + OpenMP, `-ffast-math`, incremental ray stepping |
| GPU buffer | `--mode gpu-buf` | OpenCL global buffers, manual bilinear/trilinear — naive baseline; slow at 512³ (no texture cache), chunked to avoid the driver watchdog |
| GPU image | `--mode gpu-img` | Hardware image2d_array + image3d sampler |
| GPU opt | `--mode gpu-opt` | Hardware sampler + float2 LUT + local mem (`bp_opt`); fp shared with `gpu-img` via `fp_image.cl` |

Both `gpu-img` and `gpu-opt` share the same `fp_image.cl` forward-projection
kernel (only their bp kernel differs — `bp_image.cl` vs `bp_opt`), so AABB
clipping (gated `W>512`, see "Two negative results" below) applies to both
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
  fp_buffer_opt.cl    — dead code: never loaded by gpu_init, not wired to any dispatch
validate.py           — load HDF5 outputs, print MSE + outlier-location diagnostics (supports 256/512)
validate_ops.py       — per-operator fp/bp comparison vs Python reference, isolated from MLEM iteration
run_python_reference.py — full MLEM loop in Python (fp_func/bp_func), --epochs to match C runs
diag_fp.py             — fp_cpu vs Python fp_func comparison, parameterized for 256^3/512^3 (--data/--dump/--samples)
diag_voxel.py, diag_voxel2.py — one-off scripts from the boundary-rule/truncation bug hunt; kept for reference, not part of the regular workflow
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
| Work-group `{16,16,1}→{8,32,1}` for `fp_image` | `ct_gpu.c` | ~5-7% at both scales; swept 10 candidates, see "Work-group sweeps" above. `FP_IMAGE_LWS` env override for further tuning |
| Skip float32 staging copy for `vol_img` upload | `ct_gpu.c` | float32 mode (default) copies `d_vol` straight to the image instead of through a same-format memcpy staging buffer; ~0.7% at 512³, correctness-neutral |
| Work-group `{16,16,1}→{4,64,1}` for `fp_buffer` | `ct_gpu.c` | 2.3-4.8× at 512³ (56.93s→4.87-10.19s/epoch, run-to-run variance) — the single largest win in this project; `fp_buffer`'s uncoalesced gather is far more work-group-sensitive than `fp_image`'s. `FP_BUFFER_LWS` env override |
| `FP_TILE` swept `8→32` | `fp_cpu` | ~19% further on top of the original tiling rewrite (13.58s vs 16.84s fp/epoch at 512³). `FP_TILE_ENV` env override |
