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
 */
void reconstruct_gpu(CLState *cl, const CBpara *p,
                     const float *proj_measured, float *volume,
                     int epochs);

/* Optimized reconstruction (LUT + local mem + float4 + loop unroll) */
void reconstruct_gpu_opt(CLState *cl, const CBpara *p,
                         const float *proj_measured, float *volume,
                         int epochs);

#endif /* CT_GPU_H */
