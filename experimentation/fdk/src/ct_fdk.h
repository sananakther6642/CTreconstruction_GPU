#ifndef CT_FDK_H
#define CT_FDK_H

#include "utils.h"
#include "ct_gpu.h"

/*
 * FDK (Feldkamp-Davis-Kress) cone-beam reconstruction.
 *
 * Single-pass analytical method:
 *   1. Weight each projection by cosine factor
 *   2. Ramp-filter each detector row (Ram-Lak, CPU Cooley-Tukey FFT)
 *   3. Backproject all filtered projections once (GPU bp_opt kernel)
 *
 * proj_measured: [num_projs * H * W] host memory (not modified)
 * volume:        [Nxz * Nxz * Ny]   host memory (output, init to 0)
 */
void reconstruct_fdk(CLState *cl, const CBpara *p,
                     const float *proj_measured, float *volume);

#endif /* CT_FDK_H */
