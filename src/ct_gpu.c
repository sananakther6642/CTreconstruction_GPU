#include "ct_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Helper macros ──────────────────────────────────────────────────────── */
#define CL_CHECK(err, msg) \
    do { if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s\n", (err), (msg)); \
        exit(1); } } while(0)

/* ── Load a text file into a malloc'd buffer ────────────────────────────── */
static char *load_source(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open kernel: %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);
    char *src = (char *)malloc(len + 1);
    fread(src, 1, len, f);
    src[len] = '\0';
    fclose(f);
    return src;
}

/* ── Compile a program and print build log on error ─────────────────────── */
static cl_program build_program(cl_context ctx, cl_device_id dev,
                                  const char *src)
{
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src,
                                                NULL, &err);
    CL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char *log = (char *)malloc(log_sz);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(1);
    }
    return prog;
}

/* ── Concatenate two source strings (for sharing utility kernels) ────────── */
static char *concat_src(const char *a, const char *b)
{
    size_t la = strlen(a), lb = strlen(b);
    char *out = (char *)malloc(la + lb + 1);
    memcpy(out, a, la);
    memcpy(out + la, b, lb);
    out[la+lb] = '\0';
    return out;
}

/* ── gpu_init ─────────────────────────────────────────────────────────────── */
int gpu_init(CLState *cl, GPUMode mode, const char *kernel_dir)
{
    cl_int err;
    cl->mode = mode;

    /* Platform + device */
    err = clGetPlatformIDs(1, &cl->platform, NULL);
    CL_CHECK(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_GPU, 1, &cl->device, NULL);
    CL_CHECK(err, "clGetDeviceIDs");

    /* Print device name */
    char dev_name[256];
    clGetDeviceInfo(cl->device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("OpenCL device: %s\n", dev_name);

    cl->ctx   = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    cl->queue = clCreateCommandQueue(cl->ctx, cl->device,
                                     CL_QUEUE_PROFILING_ENABLE, &err);
#pragma GCC diagnostic pop
    CL_CHECK(err, "clCreateCommandQueue");

    /* Build paths */
    char path_bp_buf[512], path_fp_buf[512];
    char path_bp_img[512], path_fp_img[512];
    char path_bp_opt[512], path_fp_opt[512];
    snprintf(path_bp_buf, sizeof(path_bp_buf), "%s/bp_buffer.cl",     kernel_dir);
    snprintf(path_fp_buf, sizeof(path_fp_buf), "%s/fp_buffer.cl",     kernel_dir);
    snprintf(path_bp_img, sizeof(path_bp_img), "%s/bp_image.cl",      kernel_dir);
    snprintf(path_fp_img, sizeof(path_fp_img), "%s/fp_image.cl",      kernel_dir);
    snprintf(path_bp_opt, sizeof(path_bp_opt), "%s/bp_buffer_opt.cl", kernel_dir);
    snprintf(path_fp_opt, sizeof(path_fp_opt), "%s/fp_buffer_opt.cl", kernel_dir);

    /* ── Buffer-mode program (bp + fp + utilities all in one) ── */
    {
        char *src_bp  = load_source(path_bp_buf);
        char *src_fp  = load_source(path_fp_buf);
        char *combined = concat_src(src_bp, src_fp);
        cl->prog_buffer = build_program(cl->ctx, cl->device, combined);
        free(src_bp); free(src_fp); free(combined);

        cl->k_bp_buf = clCreateKernel(cl->prog_buffer, "bp_buffer", &err);
        CL_CHECK(err, "bp_buffer");
        cl->k_fp_buf = clCreateKernel(cl->prog_buffer, "fp_buffer", &err);
        CL_CHECK(err, "fp_buffer");
        cl->k_cone   = clCreateKernel(cl->prog_buffer, "cone_weight", &err);
        CL_CHECK(err, "cone_weight");
        cl->k_divide = clCreateKernel(cl->prog_buffer, "proj_divide", &err);
        CL_CHECK(err, "proj_divide");
        cl->k_update  = clCreateKernel(cl->prog_buffer, "vol_update", &err);
        CL_CHECK(err, "vol_update");
        cl->k_preproc = clCreateKernel(cl->prog_buffer, "preprocess_proj", &err);
        CL_CHECK(err, "preprocess_proj");
    }

    /* ── Image-mode program (also used by OPT for fp_image) ── */
    if (mode == GPU_MODE_IMAGE || mode == GPU_MODE_OPT) {
        char *src_bp  = load_source(path_bp_img);
        char *src_fp  = load_source(path_fp_img);
        char *combined = concat_src(src_bp, src_fp);
        cl->prog_image = build_program(cl->ctx, cl->device, combined);
        free(src_bp); free(src_fp); free(combined);

        cl->k_bp_img = clCreateKernel(cl->prog_image, "bp_image", &err);
        CL_CHECK(err, "bp_image");
        cl->k_fp_img = clCreateKernel(cl->prog_image, "fp_image", &err);
        CL_CHECK(err, "fp_image");
    }

    /* ── Optimized program ── */
    if (mode == GPU_MODE_OPT) {
        char *src_bp   = load_source(path_bp_opt);
        char *src_fp   = load_source(path_fp_opt);
        /* also need utility kernels (cone_weight, proj_divide, vol_update) */
        char *src_util = load_source(path_bp_buf);
        char *tmp      = concat_src(src_bp, src_fp);
        char *combined = concat_src(tmp, src_util);
        cl->prog_opt = build_program(cl->ctx, cl->device, combined);
        free(src_bp); free(src_fp); free(src_util); free(tmp); free(combined);

        cl->k_bp_opt = clCreateKernel(cl->prog_opt, "bp_opt", &err);
        CL_CHECK(err, "bp_opt");
        cl->k_bp_ang = clCreateKernel(cl->prog_opt, "bp_angle_parallel", &err);
        CL_CHECK(err, "bp_angle_parallel");
        cl->k_fp_opt = clCreateKernel(cl->prog_opt, "fp_opt", &err);
        CL_CHECK(err, "fp_opt");
        /* reuse utility kernels from buffer prog */
        cl->k_cone   = clCreateKernel(cl->prog_opt, "cone_weight", &err);
        CL_CHECK(err, "cone_weight (opt)");
        cl->k_divide = clCreateKernel(cl->prog_opt, "proj_divide", &err);
        CL_CHECK(err, "proj_divide (opt)");
        cl->k_update = clCreateKernel(cl->prog_opt, "vol_update", &err);
        CL_CHECK(err, "vol_update (opt)");
    }

    return 0;
}

void gpu_cleanup(CLState *cl)
{
    if (cl->mode != GPU_MODE_OPT) {
        clReleaseKernel(cl->k_bp_buf); clReleaseKernel(cl->k_fp_buf);
        clReleaseKernel(cl->k_cone);   clReleaseKernel(cl->k_divide);
        clReleaseKernel(cl->k_update);
        clReleaseProgram(cl->prog_buffer);
    }
    if (cl->mode == GPU_MODE_IMAGE) {
        clReleaseKernel(cl->k_bp_img); clReleaseKernel(cl->k_fp_img);
        clReleaseProgram(cl->prog_image);
    }
    if (cl->mode == GPU_MODE_OPT) {
        clReleaseKernel(cl->k_bp_opt); clReleaseKernel(cl->k_bp_ang);
        clReleaseKernel(cl->k_fp_opt); clReleaseKernel(cl->k_cone);
        clReleaseKernel(cl->k_divide); clReleaseKernel(cl->k_update);
        clReleaseKernel(cl->k_fp_img); clReleaseKernel(cl->k_bp_img);
        clReleaseProgram(cl->prog_opt);
        clReleaseProgram(cl->prog_image);
    }
    clReleaseCommandQueue(cl->queue);
    clReleaseContext(cl->ctx);
}

/*
 * run_preprocess — GPU equivalent of Python's bp_func first line:
 *   proj[:, ::-1, :].transpose(0,2,1) / voxelSize
 *
 * src: [np*H*W]  (as loaded from HDF5, row-major [ip][ih][iw])
 * dst: [np*W*H]  (col-major [ip][iw][ih], ready for bp kernels)
 */
static void run_preprocess(CLState *cl, const CBpara *p,
                            cl_mem src, cl_mem dst)
{
    cl_int err;
    cl_kernel k = cl->k_preproc;
    int W = p->detector_width, H = p->detector_height, np = p->num_projs;
    float vs = (float)p->voxelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &src);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dst);
    clSetKernelArg(k, 2, sizeof(int),    &W);
    clSetKernelArg(k, 3, sizeof(int),    &H);
    clSetKernelArg(k, 4, sizeof(float),  &vs);

    size_t gws[3] = {(size_t)W, (size_t)H, (size_t)np};
    size_t lws[3] = {16, 16, 1};
    for (int d = 0; d < 3; d++)
        if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0, NULL, NULL);
    CL_CHECK(err, "preprocess_proj enqueue");
}

/* ── Internal: run backprojection on GPU (buffer mode) ─────────────────── */
static void run_bp_buffer(CLState *cl, const CBpara *p,
                           cl_mem d_proj, cl_mem d_angles,
                           cl_mem d_vol)
{
    cl_int err;
    cl_kernel k = cl->k_bp_buf;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W = p->detector_width,   H  = p->detector_height;
    int np = p->num_projs;
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_proj);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_angles);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_vol);
    clSetKernelArg(k, 3, sizeof(int),    &Nxz);
    clSetKernelArg(k, 4, sizeof(int),    &Ny);
    clSetKernelArg(k, 5, sizeof(int),    &W);
    clSetKernelArg(k, 6, sizeof(int),    &H);
    clSetKernelArg(k, 7, sizeof(int),    &np);
    clSetKernelArg(k, 8, sizeof(float),  &SOD);
    clSetKernelArg(k, 9, sizeof(float),  &SDD);
    clSetKernelArg(k,10, sizeof(float),  &vs);
    clSetKernelArg(k,11, sizeof(float),  &px);

    size_t gws[3] = {(size_t)Nxz, (size_t)Nxz, (size_t)Ny};
    size_t lws[3] = {8, 8, 4};  /* tune for target GPU */
    /* round up to multiple of lws */
    for (int d=0;d<3;d++) {
        if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
    }
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0,NULL,NULL);
    CL_CHECK(err, "bp_buffer enqueue");
}

/* ── Internal: run forward projection on GPU (buffer mode) ─────────────── */
static void run_fp_buffer(CLState *cl, const CBpara *p,
                           cl_mem d_vol, cl_mem d_angles,
                           cl_mem d_proj)
{
    cl_int err;
    cl_kernel k = cl->k_fp_buf;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W = p->detector_width,   H  = p->detector_height;
    int np = p->num_projs;
    int n_samples = (int)ceilf(Nxz * 2.f);
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_vol);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_angles);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_proj);
    clSetKernelArg(k, 3, sizeof(int),    &Nxz);
    clSetKernelArg(k, 4, sizeof(int),    &Ny);
    clSetKernelArg(k, 5, sizeof(int),    &W);
    clSetKernelArg(k, 6, sizeof(int),    &H);
    clSetKernelArg(k, 7, sizeof(int),    &np);
    clSetKernelArg(k, 8, sizeof(int),    &n_samples);
    clSetKernelArg(k, 9, sizeof(float),  &SOD);
    clSetKernelArg(k,10, sizeof(float),  &SDD);
    clSetKernelArg(k,11, sizeof(float),  &vs);
    clSetKernelArg(k,12, sizeof(float),  &px);

    size_t gws[3] = {(size_t)W, (size_t)H, (size_t)np};
    size_t lws[3] = {16, 16, 1};
    for (int d=0;d<3;d++) {
        if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
    }
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0,NULL,NULL);
    CL_CHECK(err, "fp_buffer enqueue");
}

/* ── Internal: run backprojection (image mode) ──────────────────────────── */
static void run_bp_image(CLState *cl, const CBpara *p,
                          cl_mem proj_img, cl_mem d_angles, cl_mem d_vol)
{
    cl_int err;
    cl_kernel k = cl->k_bp_img;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W = p->detector_width,   H  = p->detector_height;
    int np = p->num_projs;
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &proj_img);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_angles);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_vol);
    clSetKernelArg(k, 3, sizeof(int),    &Nxz);
    clSetKernelArg(k, 4, sizeof(int),    &Ny);
    clSetKernelArg(k, 5, sizeof(int),    &W);
    clSetKernelArg(k, 6, sizeof(int),    &H);
    clSetKernelArg(k, 7, sizeof(int),    &np);
    clSetKernelArg(k, 8, sizeof(float),  &SOD);
    clSetKernelArg(k, 9, sizeof(float),  &SDD);
    clSetKernelArg(k,10, sizeof(float),  &vs);
    clSetKernelArg(k,11, sizeof(float),  &px);

    size_t gws[3] = {(size_t)Nxz, (size_t)Nxz, (size_t)Ny};
    size_t lws[3] = {8, 8, 4};
    for (int d=0;d<3;d++) {
        if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
    }
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0,NULL,NULL);
    CL_CHECK(err, "bp_image enqueue");
}

/* ── Internal: run forward projection (image mode) ─────────────────────── */
static void run_fp_image(CLState *cl, const CBpara *p,
                          cl_mem vol_img, cl_mem d_angles, cl_mem d_proj)
{
    cl_int err;
    cl_kernel k = cl->k_fp_img;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W = p->detector_width,   H  = p->detector_height;
    int np = p->num_projs;
    int n_samples = (int)ceilf(Nxz * 2.f);
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &vol_img);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_angles);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_proj);
    clSetKernelArg(k, 3, sizeof(int),    &Nxz);
    clSetKernelArg(k, 4, sizeof(int),    &Ny);
    clSetKernelArg(k, 5, sizeof(int),    &W);
    clSetKernelArg(k, 6, sizeof(int),    &H);
    clSetKernelArg(k, 7, sizeof(int),    &np);
    clSetKernelArg(k, 8, sizeof(int),    &n_samples);
    clSetKernelArg(k, 9, sizeof(float),  &SOD);
    clSetKernelArg(k,10, sizeof(float),  &SDD);
    clSetKernelArg(k,11, sizeof(float),  &vs);
    clSetKernelArg(k,12, sizeof(float),  &px);

    size_t gws[3] = {(size_t)W, (size_t)H, (size_t)np};
    size_t lws[3] = {16, 16, 1};
    for (int d=0;d<3;d++) {
        if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
    }
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0,NULL,NULL);
    CL_CHECK(err, "fp_image enqueue");
}

/* ── reconstruct_gpu ─────────────────────────────────────────────────────── */
void reconstruct_gpu(CLState *cl, const CBpara *p,
                     const float *proj_measured, float *volume,
                     int epochs)
{
    cl_int err;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W   = p->detector_width,  H  = p->detector_height;
    int np  = p->num_projs;

    size_t vol_bytes  = (size_t)Nxz * Nxz * Ny * sizeof(float);
    size_t proj_bytes = (size_t)np  * H   * W  * sizeof(float);

    /* Convert angles to float */
    float *ang_f = (float *)malloc(np * sizeof(float));
    for (int i=0;i<np;i++) ang_f[i] = (float)p->angles[i];

    /* ── Allocate device buffers ── */
    cl_mem d_proj_meas = clCreateBuffer(cl->ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                         proj_bytes, (void*)proj_measured, &err);
    CL_CHECK(err, "d_proj_meas");

    cl_mem d_angles = clCreateBuffer(cl->ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                      np*sizeof(float), ang_f, &err);
    CL_CHECK(err, "d_angles");

    cl_mem d_vol = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                   vol_bytes, volume, &err);
    CL_CHECK(err, "d_vol");

    /* proj_prep: [np*W*H] — preprocessed (flip+transpose+scale) layout for bp */
    cl_mem d_proj_b    = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_proj_b");
    cl_mem d_ratio     = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_ratio");
    cl_mem d_ratio_prep= clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_ratio_prep");
    cl_mem d_proj_prep = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_proj_prep");
    cl_mem d_bp_ratio  = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, vol_bytes,  NULL, &err);
    CL_CHECK(err, "d_bp_ratio");
    cl_mem d_bp_ones   = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, vol_bytes,  NULL, &err);
    CL_CHECK(err, "d_bp_ones");

    /*
     * Preprocess measured projections once:
     *   flip H-axis + transpose [np,H,W]→[np,W,H] + /voxelSize
     * Result in d_proj_prep — used as bp target each epoch.
     * d_proj_meas kept raw for the divide step (ratio = p0/b).
     */
    run_preprocess(cl, p, d_proj_meas, d_proj_prep);
    clFinish(cl->queue);

    /* Apply cone-weight to d_proj_prep in-place */
    {
        cl_kernel k = cl->k_cone;
        float SDD=(float)p->SDD, px=(float)p->pixelSize;
        clSetKernelArg(k,0,sizeof(cl_mem),&d_proj_prep);
        clSetKernelArg(k,1,sizeof(int),&W);
        clSetKernelArg(k,2,sizeof(int),&H);
        clSetKernelArg(k,3,sizeof(float),&SDD);
        clSetKernelArg(k,4,sizeof(float),&px);
        size_t gws[3]={(size_t)W,(size_t)H,(size_t)np};
        size_t lws[3]={16,16,1};
        for (int d=0;d<3;d++) if(gws[d]%lws[d]) gws[d]+=lws[d]-gws[d]%lws[d];
        err=clEnqueueNDRangeKernel(cl->queue,k,3,NULL,gws,lws,0,NULL,NULL);
        CL_CHECK(err,"cone_weight enqueue");
        clFinish(cl->queue);
    }

    /* ── Precompute bp(ones) ── */
    {
        cl_mem d_ones_raw  = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        cl_mem d_ones_prep = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        float one = 1.f;
        err = clEnqueueFillBuffer(cl->queue, d_ones_raw, &one, sizeof(float),
                                  0, proj_bytes, 0, NULL, NULL);
        CL_CHECK(err, "fill ones");
        clFinish(cl->queue);
        run_preprocess(cl, p, d_ones_raw, d_ones_prep);
        clFinish(cl->queue);

        float zero = 0.f;
        clEnqueueFillBuffer(cl->queue, d_bp_ones, &zero, sizeof(float),
                            0, vol_bytes, 0, NULL, NULL);
        clFinish(cl->queue);

        if (cl->mode == GPU_MODE_BUFFER) {
            run_bp_buffer(cl, p, d_ones_prep, d_angles, d_bp_ones);
        } else {
            cl_image_format fmt = {CL_R, CL_FLOAT};
            cl_image_desc desc = {0};
            desc.image_type       = CL_MEM_OBJECT_IMAGE2D_ARRAY;
            desc.image_width      = (size_t)W;
            desc.image_height     = (size_t)H;
            desc.image_array_size = (size_t)np;
            float *ones_h = (float*)malloc(proj_bytes);
            for (size_t i=0;i<(size_t)np*H*W;i++) ones_h[i]=1.f;
            cl_mem ones_img = clCreateImage(cl->ctx,
                CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, &fmt, &desc, ones_h, &err);
            CL_CHECK(err,"ones_img");
            free(ones_h);
            run_bp_image(cl, p, ones_img, d_angles, d_bp_ones);
            clReleaseMemObject(ones_img);
        }
        clFinish(cl->queue);
        clReleaseMemObject(d_ones_raw);
        clReleaseMemObject(d_ones_prep);
        printf("  bp(ones) computed.\n");
    }

    /* ── Image objects for image mode ── */
    cl_mem proj_img_array = NULL;
    cl_mem vol_img        = NULL;
    cl_image_format img_fmt = {CL_R, CL_FLOAT};

    if (cl->mode == GPU_MODE_IMAGE) {
        /* proj image array */
        cl_image_desc desc = {0};
        desc.image_type       = CL_MEM_OBJECT_IMAGE2D_ARRAY;
        desc.image_width      = (size_t)W;
        desc.image_height     = (size_t)H;
        desc.image_array_size = (size_t)np;
        proj_img_array = clCreateImage(cl->ctx,
            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, &img_fmt, &desc,
            (void*)proj_measured, &err);
        CL_CHECK(err,"proj_img_array");
    }

    /* ── Iterative loop ─────────────────────────────────────────────────── */
    int proj_n = np * H * W;
    int vol_n  = Nxz * Nxz * Ny;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t_ep = get_time_sec();

        /* forward project: b = F(v0) */
        if (cl->mode == GPU_MODE_BUFFER) {
            run_fp_buffer(cl, p, d_vol, d_angles, d_proj_b);
        } else {
            /* Rebuild volume image from current d_vol */
            float *vol_h = (float*)malloc(vol_bytes);
            clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0, vol_bytes, vol_h, 0,NULL,NULL);
            if (vol_img) clReleaseMemObject(vol_img);
            cl_image_desc vdesc = {0};
            vdesc.image_type   = CL_MEM_OBJECT_IMAGE3D;
            vdesc.image_width  = (size_t)Ny;   /* z */
            vdesc.image_height = (size_t)Nxz;  /* y */
            vdesc.image_depth  = (size_t)Nxz;  /* x */
            vol_img = clCreateImage(cl->ctx,
                CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, &img_fmt, &vdesc, vol_h, &err);
            CL_CHECK(err,"vol_img");
            free(vol_h);
            run_fp_image(cl, p, vol_img, d_angles, d_proj_b);
        }
        clFinish(cl->queue);

        /* ratio = p0 / b  (both in raw [np,H,W] layout) */
        {
            cl_kernel k = cl->k_divide;
            clSetKernelArg(k,0,sizeof(cl_mem),&d_proj_meas);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_proj_b);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_ratio);
            clSetKernelArg(k,3,sizeof(int),   &proj_n);
            size_t gws=(size_t)proj_n;
            err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
            CL_CHECK(err,"proj_divide");
        }
        clFinish(cl->queue);

        /* preprocess ratio → d_ratio_prep before passing to bp */
        run_preprocess(cl, p, d_ratio, d_ratio_prep);
        clFinish(cl->queue);

        /* bp_ratio = bp(ratio_prep) */
        {
            float zero=0.f;
            clEnqueueFillBuffer(cl->queue,d_bp_ratio,&zero,sizeof(float),0,vol_bytes,0,NULL,NULL);
            if (cl->mode == GPU_MODE_BUFFER) {
                run_bp_buffer(cl, p, d_ratio_prep, d_angles, d_bp_ratio);
            } else {
                float *ratio_h = (float*)malloc(proj_bytes);
                clEnqueueReadBuffer(cl->queue,d_ratio_prep,CL_TRUE,0,proj_bytes,ratio_h,0,NULL,NULL);
                cl_image_desc rdesc={0};
                rdesc.image_type=CL_MEM_OBJECT_IMAGE2D_ARRAY;
                rdesc.image_width=(size_t)W; rdesc.image_height=(size_t)H;
                rdesc.image_array_size=(size_t)np;
                cl_mem ratio_img=clCreateImage(cl->ctx,
                    CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,&img_fmt,&rdesc,ratio_h,&err);
                CL_CHECK(err,"ratio_img");
                free(ratio_h);
                run_bp_image(cl, p, ratio_img, d_angles, d_bp_ratio);
                clFinish(cl->queue);
                clReleaseMemObject(ratio_img);
            }
        }
        clFinish(cl->queue);

        /* v0 *= bp_ratio / bp_ones */
        {
            cl_kernel k = cl->k_update;
            clSetKernelArg(k,0,sizeof(cl_mem),&d_vol);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_bp_ratio);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ones);
            clSetKernelArg(k,3,sizeof(int),   &vol_n);
            size_t gws=(size_t)vol_n;
            err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
            CL_CHECK(err,"vol_update");
        }
        clFinish(cl->queue);
        printf("  epoch %3d/%d  %.3f s\n", epoch+1, epochs, get_time_sec()-t_ep);
    }

    /* Read back result */
    clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0, vol_bytes, volume, 0,NULL,NULL);

    /* Cleanup */
    clReleaseMemObject(d_proj_meas);
    clReleaseMemObject(d_proj_prep);
    clReleaseMemObject(d_angles);
    clReleaseMemObject(d_vol);
    clReleaseMemObject(d_proj_b);
    clReleaseMemObject(d_ratio);
    clReleaseMemObject(d_ratio_prep);
    clReleaseMemObject(d_bp_ratio);
    clReleaseMemObject(d_bp_ones);
    if (proj_img_array) clReleaseMemObject(proj_img_array);
    if (vol_img)        clReleaseMemObject(vol_img);
    free(ang_f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * reconstruct_gpu_opt — uses optimized kernels (bp_opt + fp_opt)
 *
 * Key differences from reconstruct_gpu:
 *  - Passes float2 angle_cs (cos/sin LUT) instead of raw angle array
 *  - Passes __local scratch buffer sized W*H floats to bp_opt
 *  - Uses k_bp_opt / k_fp_opt
 * ═══════════════════════════════════════════════════════════════════════════ */
void reconstruct_gpu_opt(CLState *cl, const CBpara *p,
                         const float *proj_measured, float *volume,
                         int epochs)
{
    cl_int err;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W   = p->detector_width,  H  = p->detector_height;
    int np  = p->num_projs;

    size_t vol_bytes  = (size_t)Nxz * Nxz * Ny * sizeof(float);
    size_t proj_bytes = (size_t)np  * H   * W  * sizeof(float);

    /* Build float2 cos/sin LUT + raw float angles for fp_image */
    float *ang_cs = (float *)malloc(np * 2 * sizeof(float));
    float *ang_f  = (float *)malloc(np * sizeof(float));
    for (int i = 0; i < np; i++) {
        ang_cs[2*i]   = (float)cos(p->angles[i]);
        ang_cs[2*i+1] = (float)sin(p->angles[i]);
        ang_f[i]      = (float)p->angles[i];
    }

    /* Device buffers */
    cl_mem d_ang_cs = clCreateBuffer(cl->ctx,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, np*2*sizeof(float), ang_cs, &err);
    CL_CHECK(err, "d_ang_cs");
    cl_mem d_angles = clCreateBuffer(cl->ctx,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, np*sizeof(float), ang_f, &err);
    CL_CHECK(err, "d_angles opt");

    cl_mem d_proj_meas = clCreateBuffer(cl->ctx,
        CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, proj_bytes, (void*)proj_measured, &err);
    CL_CHECK(err, "d_proj_meas opt");

    cl_mem d_vol = clCreateBuffer(cl->ctx,
        CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, vol_bytes, volume, &err);
    CL_CHECK(err, "d_vol opt");

    cl_mem d_proj_b    = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_proj_b opt");
    cl_mem d_ratio     = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_ratio opt");
    cl_mem d_ratio_prep= clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_ratio_prep opt");
    cl_mem d_proj_prep = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_proj_prep opt");
    cl_mem d_bp_ratio  = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, vol_bytes, NULL, &err);
    CL_CHECK(err, "d_bp_ratio opt");
    cl_mem d_bp_ones   = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, vol_bytes, NULL, &err);
    CL_CHECK(err, "d_bp_ones opt");

    /* Preprocess measured projections once: flip+transpose+scale → d_proj_prep */
    run_preprocess(cl, p, d_proj_meas, d_proj_prep);
    clFinish(cl->queue);

    /* lmem unused in new bp_opt (image sampler replaces local cache) */
    size_t lmem_bytes = 16;

    /* image format for proj image array */
    cl_image_format img_fmt = {CL_R, CL_FLOAT};

    /* Apply cone-weight to d_proj_prep in-place */
    {
        cl_kernel k = cl->k_cone;
        float SDD=(float)p->SDD, px=(float)p->pixelSize;
        clSetKernelArg(k,0,sizeof(cl_mem),&d_proj_prep);
        clSetKernelArg(k,1,sizeof(int),&W);
        clSetKernelArg(k,2,sizeof(int),&H);
        clSetKernelArg(k,3,sizeof(float),&SDD);
        clSetKernelArg(k,4,sizeof(float),&px);
        size_t gws[3]={(size_t)W,(size_t)H,(size_t)np};
        size_t lws[3]={16,16,1};
        for(int d=0;d<3;d++) if(gws[d]%lws[d]) gws[d]+=lws[d]-gws[d]%lws[d];
        err=clEnqueueNDRangeKernel(cl->queue,k,3,NULL,gws,lws,0,NULL,NULL);
        CL_CHECK(err,"cone_weight opt");
        clFinish(cl->queue);
    }

    /* ── bp_ones: create image of all-ones, run bp_opt ── */
    {
        float zero=0.f;
        clEnqueueFillBuffer(cl->queue,d_bp_ones,&zero,sizeof(float),0,vol_bytes,0,NULL,NULL);
        clFinish(cl->queue);

        float *ones_h = (float*)malloc(proj_bytes);
        for (size_t i = 0; i < (size_t)np*H*W; i++) ones_h[i] = 1.f;
        cl_image_desc idesc={0};
        idesc.image_type=CL_MEM_OBJECT_IMAGE2D_ARRAY;
        idesc.image_width=(size_t)W; idesc.image_height=(size_t)H;
        idesc.image_array_size=(size_t)np;
        cl_mem ones_img = clCreateImage(cl->ctx,
            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, &img_fmt, &idesc, ones_h, &err);
        CL_CHECK(err,"ones_img opt");
        free(ones_h);

        cl_kernel k = cl->k_bp_opt;
        int ns=np; float SOD=(float)p->SOD,SDD=(float)p->SDD;
        float vs=(float)p->voxelSize,px=(float)p->pixelSize;
        clSetKernelArg(k,0,sizeof(cl_mem),&ones_img);
        clSetKernelArg(k,1,sizeof(cl_mem),&d_ang_cs);
        clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ones);
        clSetKernelArg(k,3,lmem_bytes,NULL);
        clSetKernelArg(k,4,sizeof(int),&Nxz);
        clSetKernelArg(k,5,sizeof(int),&Ny);
        clSetKernelArg(k,6,sizeof(int),&W);
        clSetKernelArg(k,7,sizeof(int),&H);
        clSetKernelArg(k,8,sizeof(int),&ns);
        clSetKernelArg(k,9,sizeof(float),&SOD);
        clSetKernelArg(k,10,sizeof(float),&SDD);
        clSetKernelArg(k,11,sizeof(float),&vs);
        clSetKernelArg(k,12,sizeof(float),&px);
        size_t gws[3]={(size_t)Nxz,(size_t)Nxz,(size_t)Ny};
        size_t lws[3]={8,8,4};
        for(int d=0;d<3;d++) if(gws[d]%lws[d]) gws[d]+=lws[d]-gws[d]%lws[d];
        err=clEnqueueNDRangeKernel(cl->queue,k,3,NULL,gws,lws,0,NULL,NULL);
        CL_CHECK(err,"bp_opt ones");
        clFinish(cl->queue);
        clReleaseMemObject(ones_img);
        printf("  bp_opt(ones) computed.\n");
    }

    int proj_n = np*H*W;
    int vol_n  = Nxz*Nxz*Ny;
    int n_samples = (int)ceilf(Nxz*2.f);
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    /* Persistent image2d_array for ratio — reused every epoch via CopyBufferToImage */
    cl_image_desc rdesc={0};
    rdesc.image_type=CL_MEM_OBJECT_IMAGE2D_ARRAY;
    rdesc.image_width=(size_t)W; rdesc.image_height=(size_t)H;
    rdesc.image_array_size=(size_t)np;
    cl_mem ratio_img = clCreateImage(cl->ctx, CL_MEM_READ_WRITE, &img_fmt, &rdesc, NULL, &err);
    CL_CHECK(err,"ratio_img persistent");

    /* Persistent image3D for volume — rebuilt each epoch */
    cl_mem vol_img = NULL;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t_ep = get_time_sec();

        /* ── fp: use fp_image with image3D volume (texture cache) ── */
        {
            float *vol_h = (float*)malloc(vol_bytes);
            clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0, vol_bytes, vol_h, 0,NULL,NULL);
            if (vol_img) clReleaseMemObject(vol_img);
            cl_image_desc vdesc={0};
            vdesc.image_type=CL_MEM_OBJECT_IMAGE3D;
            vdesc.image_width=(size_t)Ny;
            vdesc.image_height=(size_t)Nxz;
            vdesc.image_depth=(size_t)Nxz;
            vol_img = clCreateImage(cl->ctx,
                CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, &img_fmt, &vdesc, vol_h, &err);
            CL_CHECK(err,"vol_img opt");
            free(vol_h);
            run_fp_image(cl, p, vol_img, d_angles, d_proj_b);
        }
        clFinish(cl->queue);

        /* ── ratio = p0/b ── */
        {
            cl_kernel k = cl->k_divide;
            clSetKernelArg(k,0,sizeof(cl_mem),&d_proj_meas);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_proj_b);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_ratio);
            clSetKernelArg(k,3,sizeof(int),&proj_n);
            size_t gws=(size_t)proj_n;
            err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
            CL_CHECK(err,"divide opt");
        }
        clFinish(cl->queue);

        /* preprocess ratio → d_ratio_prep */
        run_preprocess(cl, p, d_ratio, d_ratio_prep);
        clFinish(cl->queue);

        /* ── copy d_ratio_prep buffer → ratio_img (no host roundtrip) ── */
        {
            size_t origin[3]={0,0,0};
            size_t region[3]={(size_t)W,(size_t)H,(size_t)np};
            err=clEnqueueCopyBufferToImage(cl->queue, d_ratio_prep, ratio_img,
                                           0, origin, region, 0, NULL, NULL);
            CL_CHECK(err,"CopyBufferToImage ratio");
        }
        clFinish(cl->queue);

        /* ── bp_opt(ratio_img) ── */
        {
            float zero=0.f;
            clEnqueueFillBuffer(cl->queue,d_bp_ratio,&zero,sizeof(float),0,vol_bytes,0,NULL,NULL);
            cl_kernel k = cl->k_bp_opt;
            clSetKernelArg(k,0,sizeof(cl_mem),&ratio_img);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_ang_cs);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ratio);
            clSetKernelArg(k,3,lmem_bytes,NULL);
            clSetKernelArg(k,4,sizeof(int),&Nxz);
            clSetKernelArg(k,5,sizeof(int),&Ny);
            clSetKernelArg(k,6,sizeof(int),&W);
            clSetKernelArg(k,7,sizeof(int),&H);
            clSetKernelArg(k,8,sizeof(int),&np);
            clSetKernelArg(k,9,sizeof(float),&SOD);
            clSetKernelArg(k,10,sizeof(float),&SDD);
            clSetKernelArg(k,11,sizeof(float),&vs);
            clSetKernelArg(k,12,sizeof(float),&px);
            size_t gws[3]={(size_t)Nxz,(size_t)Nxz,(size_t)Ny};
            size_t lws[3]={8,8,4};
            for(int d=0;d<3;d++) if(gws[d]%lws[d]) gws[d]+=lws[d]-gws[d]%lws[d];
            err=clEnqueueNDRangeKernel(cl->queue,k,3,NULL,gws,lws,0,NULL,NULL);
            CL_CHECK(err,"bp_opt ratio");
        }
        clFinish(cl->queue);

        /* ── update ── */
        {
            cl_kernel k = cl->k_update;
            clSetKernelArg(k,0,sizeof(cl_mem),&d_vol);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_bp_ratio);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ones);
            clSetKernelArg(k,3,sizeof(int),&vol_n);
            size_t gws=(size_t)vol_n;
            err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
            CL_CHECK(err,"update opt");
        }
        clFinish(cl->queue);
        printf("  epoch %3d/%d  %.3f s\n", epoch+1, epochs, get_time_sec()-t_ep);
    }
    if (vol_img) clReleaseMemObject(vol_img);
    clReleaseMemObject(ratio_img);

    clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0, vol_bytes, volume, 0,NULL,NULL);

    clReleaseMemObject(d_ang_cs);
    clReleaseMemObject(d_angles);
    clReleaseMemObject(d_proj_meas);
    clReleaseMemObject(d_proj_prep);
    clReleaseMemObject(d_vol);
    clReleaseMemObject(d_proj_b);
    clReleaseMemObject(d_ratio);
    clReleaseMemObject(d_ratio_prep);
    clReleaseMemObject(d_bp_ratio);
    clReleaseMemObject(d_bp_ones);
    free(ang_cs);
    free(ang_f);
}
