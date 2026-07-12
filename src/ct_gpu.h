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
    GPU_MODE_IMAGE  = 1
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
    cl_kernel  k_cone;
    cl_kernel  k_divide;
    cl_kernel  k_update;

    /* image-mode kernels */
    cl_program prog_image;
    cl_kernel  k_bp_img;
    cl_kernel  k_fp_img;

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

#endif /* CT_GPU_H */
