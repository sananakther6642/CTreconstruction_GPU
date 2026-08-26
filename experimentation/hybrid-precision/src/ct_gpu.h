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
    cl_kernel  k_divide_preproc_img; /* perf-v2 Phase B1+B2: fuses proj_divide + preprocess, writes straight to ratio_img */
    cl_kernel  k_update_img; /* perf-v2 Phase B4: vol_update that also writes straight to vol_img (float32 mode only) */
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
    int has_fp16; /* cl_khr_fp16 support -- gates --half; false on e.g. NVIDIA GTX 680 */
    int hybrid_precision; /* HYBRID_PRECISION env var, read once at gpu_init --
                            * whether k_bp_img/k_bp_opt were compiled with the
                            * two extra threshold args (see bp_image.cl /
                            * bp_buffer_opt.cl). Remembered so call sites know
                            * whether to bind them, without re-reading getenv
                            * or guessing from kernel arg count. */
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

/*
 * Optimized reconstruction (LUT + local mem + float4 + loop unroll).
 *
 * perf-v2 Phase C1: subsets implements Ordered Subsets EM (OSEM).
 * subsets=1 (default) is EXACTLY the pre-OSEM MLEM path -- no
 * permutation, one full-angle normalizer, ip_start=0/ip_count=num_projs
 * every sub-iteration. subsets=S>1 requires the caller to have already
 * permuted p->angles AND proj_measured with the SAME permutation (see
 * utils.h compute_osem_permutation/permute_projections_inplace) so that
 * subset k is exactly the contiguous angle range
 * [k*num_projs/S, (k+1)*num_projs/S).
 *
 * --epochs convention: one epoch is one full pass over all S subsets
 * (S sub-iterations), matching plain MLEM's "one epoch = one full-angle
 * update" in total work done, NOT in wall-clock number of volume
 * updates -- --epochs 20 --subsets 5 does 100 sub-iterations total,
 * the same fp/bp work as --epochs 100 --subsets 1.
 */
void reconstruct_gpu_opt(CLState *cl, const CBpara *p,
                         const float *proj_measured, float *volume,
                         int epochs, const char *conv_log, int subsets);

/*
 * Component test, GPU version of the --op fp|bp CPU path (see main.c).
 * Runs a single fp or bp call on all-ones input, no MLEM iteration --
 * isolates operator precision so it can be compared against gpu-buf's
 * manual (exact) interpolation. cl must already be gpu_init'd in
 * GPU_MODE_IMAGE or GPU_MODE_OPT (fp) / any GPU mode (bp).
 *
 * gpu_op_fp:  proj_out must be pre-allocated, size num_projs*H*W floats.
 *             volume is used as-is (caller fills with 1.0f for the
 *             all-ones case, matching --op fp's CPU behavior).
 * gpu_op_bp:  volume_out must be pre-allocated, size Nxz*Nxz*Ny floats.
 *             Internally fills a raw-ones projection buffer, applies the
 *             same cone_weight+flip+transpose preprocessing run_preprocess
 *             does inside reconstruct_gpu (matching --op bp's CPU
 *             cone_weight_cpu + manual layout transform), then bp's it --
 *             so both are bp(cone_weight(ones)), not raw bp(ones).
 */
void gpu_op_fp(CLState *cl, const CBpara *p, const float *volume, float *proj_out);
void gpu_op_bp(CLState *cl, const CBpara *p, float *volume_out);

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
