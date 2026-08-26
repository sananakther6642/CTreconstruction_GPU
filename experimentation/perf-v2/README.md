# perf-v2 (unmerged, branch `perf-v2`)

## What it is

A broader algorithmic investigation beyond plain MLEM: OSEM (Ordered
Subsets EM, `--subsets N`) is the one part of this that shipped and is
merged into `features`/`main` (see the main README's OSEM section). The
other phases below (FDK-as-initializer, Huber prior, log-domain momentum,
SQS) were explored, measured, and **not merged** — either negative results
or, for SQS, not implemented after a written cost/benefit evaluation.

This branch (`perf-v2`) has no shared git history with `features` (an old
divergence, never rebased), so its source changes can't be cleanly
diffed/merged — this file preserves the real measured findings as
documentation. To see the actual code, check out the `perf-v2` branch
directly.

## Findings

### FDK initialization — `--init fdk`, and its measured overlap with OSEM

perf-v2 Phase C2. Single-pass analytic reconstruction (Ram-Lak ramp filter +
cosine weight, CPU FFT, parallelized with OpenMP; single GPU `bp_opt` pass)
used as `v0` instead of `ones`. `--init fdk` works with any `--mode`/
`--subsets` combination — it spins up its own temporary `GPU_MODE_OPT`
context regardless of the main mode, since `bp_opt` is `gpu-opt`-only.
0.80s total at 256³ (filter+backprojection, 24 CPU threads). FDK's ramp
filter undershoots and produces negative voxels (a known property, not a
bug) — **49.17% of voxels needed clamping** to a `1e-6` positive floor,
mandatory since MLEM's multiplicative update can never recover a voxel
that starts ≤0.

**Measured full 2×2 ({ones, fdk}-init × {MLEM, OSEM S=5}), 256³, 20
epochs, log-likelihood** — the plan explicitly asked for this measurement
rather than assuming FDK-init and OSEM's gains simply add:

| Epoch | ones+MLEM | fdk+MLEM | ones+OSEM(S=5) | fdk+OSEM(S=5) |
|---|---|---|---|---|
| 1 | −20,017,427 | **−587,751** | −20,017,427 | **−587,751** |
| 5 | −524,214 | **−520,645** | −497,197 | **−495,211** |
| 10 | −523,152 | **−519,822** | −483,565 | −483,478 |
| 20 | −499,712 | **−497,293** | **−480,635** | −480,951 |

(higher/less-negative is better; bold = better of the pair at that
epoch.) **Confirms the plan's predicted mechanism exactly:**
- Under plain **MLEM**, FDK-init is a clear, large win — ~34× better
  log-likelihood at epoch 1 alone, and stays ahead through epoch 20.
- Under **OSEM**, the gap **collapses almost immediately**: by epoch 2
  the two are nearly tied, and from ~epoch 11 onward **ones-init
  slightly overtakes FDK-init** (−480,635 vs −480,951 at epoch 20).

FDK-init and OSEM accelerate the *same* early-convergence phase, so
their gains do not stack — once OSEM is doing the fast-early-progress
job, FDK's head start is redundant, and FDK's own artifacts (the 49%
clamped-negative voxels) may introduce a small bias OSEM's early
sub-iterations spend some work correcting, which is the likely
explanation for ones-init's late, small lead over fdk-init under OSEM.

**Re-run at report scale (50 epochs) confirms and sharpens this — the
OSEM gap does not close back up with more epochs, it widens:**

| Epoch | ones+MLEM | fdk+MLEM | ones+OSEM(S=5) | fdk+OSEM(S=5) |
|---|---|---|---|---|
| 20 | −499,712 | **−497,293** | **−480,635** | −480,951 |
| 50 (final) | −483,091 | **−483,057** | **−479,645** | −480,087 |

By epoch 50, ones-init's lead over fdk-init under OSEM has grown from
−480,635 vs −480,951 (epoch 20) to −479,645 vs −480,087 — fdk-init is
not just redundant under OSEM, it measurably holds OSEM back over a
full run, consistent with the clamped-voxel-bias explanation above.
Under plain MLEM the two nearly converge to the same final value by
epoch 50 (−483,091 vs −483,057) — FDK's early-epoch head start still
doesn't change where plain MLEM ends up, it just gets there faster.
**Practical guidance unchanged and now better supported: use
`--init fdk` with plain MLEM (`--subsets 1`) for a real win; skip it
when already using OSEM (`--subsets >1`) — it adds ~0.8s of setup for
a measured net cost there, not just no benefit.**

### OSL-MLEM with a Huber prior — `--beta B`, image quality on sparse views

perf-v2 Phase C3. One-Step-Late MLEM (Green 1990): `v *= bp_ratio /
(bp_ones + beta_eff·prior_grad)` instead of plain `v *= bp_ratio/bp_ones`.
`prior_grad` is a 6-neighbour Huber-potential gradient (quadratic below
`--beta-delta`, linear-and-bounded above it — edge-preserving, no TV
staircasing), read from the plain volume buffer (exact integer
neighbours, no benefit from the image sampler's interpolation here).
`beta_eff = beta/subsets`, so the penalty is applied once per epoch's
worth of work under OSEM, not once per sub-iteration. `--beta 0`
(default) runs the original unregularized `vol_update`/`vol_update_img`
kernels completely unchanged — not a `beta=0` branch of new code, the
literal pre-C3 code path — confirmed byte-identical
(`max abs diff: 0.0`) in testing. `gpu-opt` only; refused at the CLI for
other modes. Overhead is negligible (~7.3-7.5s vs ~7.3s baseline at 50
epochs, 256³ — well under the plan's 10% budget).

**Calibration was not obvious and needed measuring, not guessing.** An
initial sweep at `--beta` ∈ {1e-3, 1e-2, 1e-1} (chosen by analogy to
typical regularization weights) showed the *correct direction* — total
variation decreased monotonically — but a *negligible magnitude*
(changes only in the 4th decimal place). Root cause: `bp_ones`'s real
scale was never measured. Added a diagnostic (`PRINT_BP_ONES_STATS=1`)
and found `bp_ones ≈ 364–443` (mean 413) at 256³ — `beta*prior_grad`
needs to be a meaningful *fraction* of that to matter, and Huber-clipped
gradients are bounded by `±beta_delta` (default 0.01) per neighbour, so
the original β range was roughly 4 orders of magnitude too small.

**Re-swept at the corrected scale** (`--beta` ∈ {100, 1000, 5000}, 256³,
50 epochs, against the `bp_ones≈413` calibration), measuring total
variation (mean absolute neighbour difference, a roughness proxy) and a
40³ central-region std-dev:

| β | Total variation | Center std | Global max | Verdict |
|---|---|---|---|---|
| 0 | 0.006880 | 0.119440 | 1.7489 | baseline |
| 100 | 0.005074 (**−26%**) | 0.112110 | 1.5856 | clean smoothing |
| 1000 | 0.003460 (**−50%**) | 0.091516 | 1.8698 | strongest clean result |
| 5000 | 0.019694 (**+186%**) | 0.248329 | 2.7265 | **unstable — overshoot** |

No NaN/inf at any β tested — the mandatory denominator clamp held even
in the unstable case, preventing an outright blow-up, but β=5000's total
variation and center std both *increased* well past baseline (worse,
not better) and its global max jumped to 2.73 from a baseline ~1.75 —
a real, measured OSL failure mode (the plan's flagged "negative/near-zero
denominator" risk, manifesting here as an overshoot rather than a NaN
once the clamp is in place).

**Correction after re-running at report scale (100 epochs, same β
values, plus a β=500 bisection point): β=1000 does not actually
converge — the 50-epoch sweep above stopped too early to see it.**
`rel_change` (the per-epoch relative volume change from
`--log-convergence`) should decay toward zero as MLEM/OSL settles; at
100 epochs:

| β | epoch-100 loglik | epoch-100 residual | epoch-100 rel_change | verdict |
|---|---|---|---|---|
| 0 | −480652 | 0.05109 | 0.00182 | converges cleanly |
| 100 | −480610 | 0.05041 | 0.00130 | converges cleanly, matches baseline |
| 500 | −480762 | 0.06843 | 0.02595 | still converging, but a real residual/rel_change floor ~15-20× baseline |
| 1000 | −480958 | 0.08196 | **0.09174** | **stuck in a limit cycle, not converging** |
| 5000 | −490585 | 0.47983 | 0.76373 | clearly diverging (residual climbs monotonically epoch 5→100) |

β=0 and β=100 both decay `rel_change` below 0.002 by epoch 100 — real
convergence. β=1000's `rel_change` never drops below ~0.09 for the
entire second half of the run (oscillating, not decaying) — this was
invisible at 50 epochs, where it still looked like it was on a
downward trend. β=500 sits in between: residual keeps decreasing
monotonically (real convergence, just slower/noisier than β≤100), so
it's a genuine — if less clean — working point, not a failure mode.

**Corrected working range at 256³ (`--beta-delta` default 0.01): β≤100
converges cleanly; β=500 converges but with visibly more residual
noise; β≥1000 does not converge (limit cycle at 1000, outright
divergence by 5000).** The earlier "β=1000 is the strongest clean
result" verdict is retracted — it was a 50-epoch snapshot of a
trajectory that hadn't revealed its own instability yet, a reminder
that *convergence* claims need enough epochs to actually observe
convergence (or its absence), not just a improving trend. Sweep
`--beta-delta` too before trusting these exact values on a different
dataset/scale — it wasn't re-tuned here, only its default was used
throughout.

### Log-domain momentum — `--gamma G`, negative result (implemented, not usable)

perf-v2 Phase C5 (planned as a "coin flip", 35-55% chance of a usable
result). Log-domain (multiplicative) Nesterov-style extrapolation:
`v_{k+1} = u_k·(u_k/v_k)^gamma`, applied once per epoch after the full
OSEM subset loop. Positive by construction — no clamp needed, unlike
naive additive momentum which can drive voxels negative under MLEM's
multiplicative update and get stuck there. `--gamma 0` (default) skips
the kernel and extra buffer entirely — exactly the pre-C5 code path,
confirmed byte-identical (`max abs diff: 0.0`).

**Rejected after measurement: unstable at every tested `gamma`, no
usable operating point found.** Swept `gamma` ∈ {0.1, 0.2, 0.3, 0.5} at
256³, 30 epochs, `--subsets 1`:

| γ | max | mean |
|---|---|---|
| 0 (baseline) | 1.7489 | 0.006644 |
| 0.1 | 2.3471 (+34%) | 0.006633 |
| 0.2 | 1.8661 | 0.006628 |
| 0.3 | 1.8661 | 0.006619 |
| 0.5 | 20.9745 (**+12×**) | 0.006496 |

No NaN/inf/negative values at any γ — the positivity guarantee holds as
designed — but the max-value overshoot is present from the smallest γ
tested and does not grow monotonically (γ=0.1 spikes worse than
γ=0.2/0.3), then jumps 12× at γ=0.5. Convergence logging at γ=0.5 shows
why: log-likelihood/residual settle into low-level non-monotone
ringing by epoch ~18-30 instead of smooth convergence, consistent with
momentum overshoot compounding on a small set of voxels near the
positivity floor each epoch rather than a global divergence. The
0.2/0.3 near-identical max (1.8661 both) was verified as a genuine
coincidence of two different outlier voxels via `md5sum` (different
files) and `argmax` (different voxel locations), not a stale-output
artifact.

**Verdict:** implementation is correct and the positivity argument is
sound, but standalone log-domain momentum is not stable enough to use
at any γ tested. Left in the tree at `γ=0` default (zero cost, honest
negative result) rather than reverted.

### SQS — evaluated, not implemented (C6)

perf-v2 Phase C6. Separable Quadratic Surrogates was in scope as an
alternative MLEM-acceleration algorithm; evaluated in writing per the
plan rather than built, for four reasons:

1. **Wrong objective.** SQS targets penalized *weighted least squares*.
   This project's noise model is Poisson (photon-counting CT), which
   MLEM/OSEM target directly. Using SQS here means optimizing the wrong
   likelihood, not just a different algorithm for the same one.
2. **New system-matrix machinery required.** SQS needs a precomputed
   curvature volume, which means kernels that expose individual
   system-matrix entries — nothing in the current fp/bp kernels does
   this (they only ever produce the *projected* forward/back operators,
   never per-ray-per-voxel weights). This is new kernel infrastructure,
   not a parameter added to existing ones — unlike C1/C3/C5, which all
   reused the existing fp/bp/vol_update kernels with small additions.
3. **No capability gained.** SQS's headline property — guaranteed
   monotone objective decrease — is a property plain MLEM already has.
   There is nothing SQS adds on that axis for this project.
4. **Worse convergence rate than what's already shipped.** SQS's
   per-iteration convergence is typically slower than OSEM's, and OSEM
   (C1) is implemented, validated, and measured in this repo already.

Estimated cost to implement properly: 3+ days, for an algorithm that
targets the wrong noise model and is expected to underperform the
OSEM already in the tree. Not implemented.

## Optimizations
