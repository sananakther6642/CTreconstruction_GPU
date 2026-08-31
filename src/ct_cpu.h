#ifndef CT_CPU_H
#define CT_CPU_H

#include "utils.h"

/*
 * Backprojection (CPU)
 * proj:   [num_projs * detector_height * detector_width]  (pre-weighted)
 * volume: [Nxz * Nxz * Ny]  output, must be zeroed by caller
 */
void bp_cpu(const float *proj, float *volume, const CBpara *p);

/*
 * Forward projection (CPU)
 * volume: [Nxz * Nxz * Ny]  input
 * proj:   [num_projs * detector_height * detector_width]  output
 */
void fp_cpu(const float *volume, float *proj, const CBpara *p);

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
