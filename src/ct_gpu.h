#ifndef CT_GPU_H
#define CT_GPU_H

#include "utils.h"

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

typedef enum {
    GPU_MODE_BUFFER = 0,
    GPU_MODE_IMAGE  = 1,
    GPU_MODE_OPT    = 2   /* optimized: LUT + local mem + float4 + unroll */
} GPUMode;

/* OpenCL context and pre-compiled kernels */
typedef struct {
    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       ctx;
    cl_command_queue queue;

    /* buffer-mode kernels */
    cl_program prog_buffer;
    cl_kernel  k_bp_buf;
    cl_kernel  k_fp_buf;
    cl_kernel  k_divide;
    cl_kernel  k_update;
    cl_kernel  k_preproc;
    cl_kernel  k_preproc_img;  /* perf-v2 Phase B1: writes straight to an image, no intermediate buffer copy */
    cl_kernel  k_cone_hw;

    /* image-mode kernels */
    cl_program prog_image;
    cl_kernel  k_bp_img;
    cl_kernel  k_fp_img;
    cl_kernel  k_f2h;      /* float_to_half: float vol buffer → half buf for vol_img */

    /* optimized kernels */
    cl_program prog_opt;
    cl_kernel  k_bp_opt;

    /* precomputed R/T for fp_image — built once per reconstruct call */
    cl_mem d_R_mats;  /* [num_projs * 9] float */
    cl_mem d_T_vecs;  /* [num_projs * 3] float */

    GPUMode mode;
} CLState;

/* Initialize OpenCL, compile kernels. Returns 0 on success. */
int  gpu_init(CLState *cl, GPUMode mode, const char *kernel_dir);

/* Free all OpenCL resources */
void gpu_cleanup(CLState *cl);

/*
 * Full iterative reconstruction on GPU.
 * proj_measured: [num_projs * H * W] host memory
 * volume:        [Nxz * Nxz * Ny]   host memory (init to 1, output here)
 * conv_log: path for per-epoch convergence CSV (loglik/residual/rel_change),
 *           or NULL to disable (default; no extra cost when NULL).
 */
void reconstruct_gpu(CLState *cl, const CBpara *p,
                     const float *proj_measured, float *volume,
                     int epochs, const char *conv_log);

/* Optimized reconstruction (LUT + local mem + float4 + loop unroll) */
void reconstruct_gpu_opt(CLState *cl, const CBpara *p,
                         const float *proj_measured, float *volume,
                         int epochs, const char *conv_log);

/*
 * perf-v2 Phase A2/A3 diagnostic: repeat one fixed fp_buffer angle-slab
 * n_repeats times, printing wall + GPU-event-profiled time per repeat.
 * Distinguishes thermal throttling (monotone degradation) from
 * TLB/page-residency effects (bimodal) from channel camping (uniform).
 * angle_offset/slab_size select which angles form the slab -- rerun with
 * a different range to test slab-index vs angle-value sensitivity (A3).
 * realloc_at (0 = never): after this many repeats, free and recreate
 * d_vol from the same host data, then continue -- tests whether the
 * observed one-time step-degradation is tied to the buffer allocation
 * itself (recovers after realloc) or to external GPU/driver state
 * (doesn't recover).
 * cl must already be gpu_init'd in GPU_MODE_BUFFER. Diagnostic only --
 * writes nothing, runs no epoch loop.
 */
void gpu_diag_repeat_slab(CLState *cl, const CBpara *p, const float *volume,
                           int angle_offset, int slab_size, int n_repeats,
                           int realloc_at);

#endif /* CT_GPU_H */
