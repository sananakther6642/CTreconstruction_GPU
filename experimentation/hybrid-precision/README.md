# hybrid-precision (unmerged investigation, branch `hybrid-precision`)

## Goal

`gpu-img`/`gpu-opt` use the hardware texture sampler (`CLK_FILTER_LINEAR`),
which trades precision for speed vs `gpu-buf`'s manual float32
interpolation. This investigation asked: can a hybrid mode — fp16-speed
sampler backprojection with an fp32 manual-interpolation correction layer
— close that precision gap without giving up the speed of the hardware
sampler entirely?

## What's here

- `src/ct_gpu.c`, `src/ct_gpu.h`, `src/main.c` — full files from the branch,
  including the `HYBRID_PRECISION` env/build-flag-gated code path,
  `gpu_op_fp`/`gpu_op_bp` (component-test isolation infra, reused later by
  the `gpu-buf-speed` investigation for fp/bp cost-split measurement),
  and `--op fp|bp` extended to GPU modes.
- `kernels/bp_buffer_opt.cl`, `kernels/bp_image.cl` — the manual fp32
  fallback/correction kernels added for the hybrid path.

These are reference copies (not a working checkout — build system paths
assume the whole repo, not this subfolder). To actually build/run this
mode, check out the `hybrid-precision` branch directly.

## Finding: redundant, not shipped

`gpu-buf` (manual float32 interpolation, no hardware sampler) already
satisfies the project's precision requirement — MSE vs CPU `1.1477e-10`,
~1000x tighter than `gpu-img`/`gpu-opt`'s `1.1278e-07` — with zero kernel
changes and no speed/complexity tradeoff. The hybrid mode's own measured
residual error still had at least one voxel with genuine unresolved
disagreement (no independent ground truth available to arbitrate), so it
wasn't a clean win over simply using `gpu-buf` when precision matters.

Real, durable byproducts kept independent of the hybrid mode not shipping:
- `gpu_op_fp`/`gpu_op_bp`: reusable per-operator (fp vs bp) isolation
  infrastructure, not previously possible in this codebase.
- `diag_op_attribution.py`: per-operator error attribution tool (see
  `hybrid-precision` branch).
- One real bug found and fixed as a byproduct: `gpu_op_fp` was missing a
  `build_RT_buffers` call, producing NaN — would not have been caught
  without building this isolation infra. This bug never affected
  `features` — it lives entirely inside `gpu_op_fp`, a function that only
  exists on this branch.

## Disposition

Not merged. Kept gated behind `HYBRID_PRECISION` (unset by default, zero
runtime cost when off) on its own branch. `gpu-buf` already satisfies the
"no compromise" precision requirement, so this is a recorded investigation
for the report, not pending work.
