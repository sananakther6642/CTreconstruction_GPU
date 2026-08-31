#ifndef CT_CPU_H
#define CT_CPU_H

#include "utils.h"

/*
 * Backprojection (CPU)
 * proj:   [num_projs * detector_height * detector_width]  (pre-weighted)
 * volume: [Nxz * Nxz * Ny]  output, must be zeroed by caller
 *
 * `restrict` (C2 perf-algorithmic): every caller (src/ct_cpu.c:501,510,523
 * and src/main.c) passes distinct, non-overlapping buffers -- verified
 * before adding this. Without it, GCC must assume `volume`'s writes
 * (strip[iz] += ...) could alias `proj`'s reads (slice[...]), blocking
 * LICM of the loop-invariant row-pointer/weight computations in bp_cpu's
 * inner iz loop.
 */
void bp_cpu(const float *restrict proj, float *restrict volume, const CBpara *p);

/*
 * Forward projection (CPU)
 * volume: [Nxz * Nxz * Ny]  input
 * proj:   [num_projs * detector_height * detector_width]  output
 *
 * `restrict`: same rationale as bp_cpu above -- callers always pass
 * distinct buffers.
 */
void fp_cpu(const float *restrict volume, float *restrict proj, const CBpara *p);

/*
 * Apply cone-weight to projections in-place.
 * Called once before iterative loop (on measured projections).
 */
void cone_weight_cpu(float *proj, const CBpara *p);

/*
 * Iterative MLEM-style reconstruction.
 * proj_measured: original measured projections [num_projs * H * W]
 * volume:        initial estimate (caller provides ones), output is here
 * epochs:        number of iterations
 * conv_log: path for per-epoch convergence CSV (loglik/residual/rel_change),
 *           or NULL to disable (default; no extra cost when NULL).
 */
void reconstruct_cpu(const float *proj_measured, float *volume,
                     const CBpara *p, int epochs, const char *conv_log);

#endif /* CT_CPU_H */
