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

/* ── Precompute R/T matrices for fp_image kernel ───────────────────────────
 * phi1=-pi/2, phi2=pi/2 are constants; R = R3(angle)*R2(phi2)*R1(phi1)
 * Stores result in cl->d_R_mats ([np*9]) and cl->d_T_vecs ([np*3]).
 */
static void build_RT_buffers(CLState *cl, const CBpara *p)
{
    int np = p->num_projs;
    float *R_host = (float *)malloc(np * 9 * sizeof(float));
    float *T_host = (float *)malloc(np * 3 * sizeof(float));

    /* R1(phi1=-pi/2, rot X): {{1,0,0},{0,0,1},{0,-1,0}} */
    /* R2(phi2=+pi/2, rot Z): {{0,-1,0},{1,0,0},{0,0,1}} */
    /* R12 = R2*R1 */
    float R12[3][3] = {
        { 0.f,  0.f, -1.f},
        { 1.f,  0.f,  0.f},
        { 0.f, -1.f,  0.f}
    };

    for (int i = 0; i < np; i++) {
        float angle = (float)p->angles[i];
        float ca = cosf(angle), sa = sinf(angle);
        /* R3(angle) */
        float R3[3][3] = {{ca,-sa,0.f},{sa,ca,0.f},{0.f,0.f,1.f}};
        /* R = R3 * R12 */
        float R[3][3];
        for (int r=0;r<3;r++) for (int c=0;c<3;c++) {
            R[r][c] = 0.f;
            for (int k=0;k<3;k++) R[r][c] += R3[r][k]*R12[k][c];
        }
        for (int r=0;r<3;r++) for (int c=0;c<3;c++)
            R_host[i*9 + r*3 + c] = R[r][c];
        T_host[i*3+0] = (float)p->SOD * ca;
        T_host[i*3+1] = (float)p->SOD * sa;
        T_host[i*3+2] = 0.f;
    }

    cl_int err;
    cl->d_R_mats = clCreateBuffer(cl->ctx,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, np*9*sizeof(float), R_host, &err);
    CL_CHECK(err, "d_R_mats");
    cl->d_T_vecs = clCreateBuffer(cl->ctx,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, np*3*sizeof(float), T_host, &err);
    CL_CHECK(err, "d_T_vecs");

    free(R_host);
    free(T_host);
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
    char path_bp_opt[512];
    snprintf(path_bp_buf, sizeof(path_bp_buf), "%s/bp_buffer.cl",     kernel_dir);
    snprintf(path_fp_buf, sizeof(path_fp_buf), "%s/fp_buffer.cl",     kernel_dir);
    snprintf(path_bp_img, sizeof(path_bp_img), "%s/bp_image.cl",      kernel_dir);
    snprintf(path_fp_img, sizeof(path_fp_img), "%s/fp_image.cl",      kernel_dir);
    snprintf(path_bp_opt, sizeof(path_bp_opt), "%s/bp_buffer_opt.cl", kernel_dir);

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
        cl->k_divide   = clCreateKernel(cl->prog_buffer, "proj_divide", &err);
        CL_CHECK(err, "proj_divide");
        cl->k_update   = clCreateKernel(cl->prog_buffer, "vol_update", &err);
        CL_CHECK(err, "vol_update");
        cl->k_preproc  = clCreateKernel(cl->prog_buffer, "preprocess_proj", &err);
        CL_CHECK(err, "preprocess_proj");
        cl->k_cone_hw  = clCreateKernel(cl->prog_buffer, "cone_weight_hw", &err);
        CL_CHECK(err, "cone_weight_hw");
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
        cl->k_f2h    = clCreateKernel(cl->prog_image, "float_to_half", &err);
        CL_CHECK(err, "float_to_half");
    }

    /* ── Optimized program ── */
    if (mode == GPU_MODE_OPT) {
        char *src_bp   = load_source(path_bp_opt);
        /* also need utility kernels (cone_weight, proj_divide, vol_update) */
        char *src_util = load_source(path_bp_buf);
        char *combined = concat_src(src_bp, src_util);
        cl->prog_opt = build_program(cl->ctx, cl->device, combined);
        free(src_bp); free(src_util); free(combined);

        cl->k_bp_opt   = clCreateKernel(cl->prog_opt, "bp_opt", &err);
        CL_CHECK(err, "bp_opt");
        cl->k_divide   = clCreateKernel(cl->prog_opt, "proj_divide", &err);
        CL_CHECK(err, "proj_divide (opt)");
        cl->k_update   = clCreateKernel(cl->prog_opt, "vol_update", &err);
        CL_CHECK(err, "vol_update (opt)");
        cl->k_preproc  = clCreateKernel(cl->prog_opt, "preprocess_proj", &err);
        CL_CHECK(err, "preprocess_proj (opt)");
        cl->k_cone_hw  = clCreateKernel(cl->prog_opt, "cone_weight_hw", &err);
        CL_CHECK(err, "cone_weight_hw (opt)");
    }

    return 0;
}

void gpu_cleanup(CLState *cl)
{
    if (cl->mode != GPU_MODE_OPT) {
        clReleaseKernel(cl->k_bp_buf); clReleaseKernel(cl->k_fp_buf);
        clReleaseKernel(cl->k_divide); clReleaseKernel(cl->k_update);
        clReleaseKernel(cl->k_preproc); clReleaseKernel(cl->k_cone_hw);
        clReleaseProgram(cl->prog_buffer);
    }
    if (cl->mode == GPU_MODE_IMAGE) {
        clReleaseKernel(cl->k_bp_img); clReleaseKernel(cl->k_fp_img);
        clReleaseKernel(cl->k_f2h);
        clReleaseProgram(cl->prog_image);
    }
    if (cl->mode == GPU_MODE_OPT) {
        clReleaseKernel(cl->k_bp_opt);
        clReleaseKernel(cl->k_divide); clReleaseKernel(cl->k_update);
        clReleaseKernel(cl->k_preproc); clReleaseKernel(cl->k_cone_hw);
        clReleaseKernel(cl->k_fp_img); clReleaseKernel(cl->k_bp_img);
        clReleaseKernel(cl->k_f2h);
        clReleaseProgram(cl->prog_opt);
        clReleaseProgram(cl->prog_image);
    }
    clReleaseCommandQueue(cl->queue);
    clReleaseContext(cl->ctx);
}

/*
 * run_preprocess — fused: cone_weight + flip + transpose + /voxelSize in one kernel pass.
 * src: [np*H*W]  (row-major [ip][ih][iw])
 * dst: [np*W*H]  (col-major [ip][iw][ih], ready for bp kernels)
 */
/* Fused: cone_weight + flip + transpose + /voxelSize in one pass */
static void run_preprocess(CLState *cl, const CBpara *p,
                            cl_mem src, cl_mem dst)
{
    cl_int err;
    cl_kernel k = cl->k_preproc;
    int W = p->detector_width, H = p->detector_height, np = p->num_projs;
    float vs  = (float)p->voxelSize;
    float SDD = (float)p->SDD;
    float px  = (float)p->pixelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &src);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dst);
    clSetKernelArg(k, 2, sizeof(int),    &W);
    clSetKernelArg(k, 3, sizeof(int),    &H);
    clSetKernelArg(k, 4, sizeof(float),  &vs);
    clSetKernelArg(k, 5, sizeof(float),  &SDD);
    clSetKernelArg(k, 6, sizeof(float),  &px);

    size_t gws[3] = {(size_t)W, (size_t)H, (size_t)np};
    size_t lws[3] = {16, 16, 1};
    for (int d = 0; d < 3; d++)
        if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0, NULL, NULL);
    CL_CHECK(err, "preprocess_proj enqueue");
}


/* ── Internal: run backprojection on GPU (buffer mode) ─────────────────── */
static void run_bp_buffer(CLState *cl, const CBpara *p,
                           cl_mem d_proj, cl_mem d_ang_cs,
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
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_ang_cs);
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

    size_t lws[3] = {4, 4, 16};  /* 16 contiguous z-threads → coalesced writes */
    size_t gws_full[3] = {(size_t)Nxz, (size_t)Nxz, (size_t)Ny};
    for (int d=0;d<3;d++)
        if (gws_full[d] % lws[d]) gws_full[d] += lws[d] - gws_full[d] % lws[d];

    /* Chunk the z-dimension into slabs so no single kernel launch runs long
     * enough to trip the GPU watchdog (bp_buffer does 4 uncoalesced global
     * loads/angle/voxel with no texture cache — at 512^3 one unchunked
     * launch can exceed the ~10s driver timeout and reset the GPU). */
    const size_t Z_SLAB = 64;
    int slab_idx = 0;
    for (size_t z0 = 0; z0 < gws_full[2]; z0 += Z_SLAB) {
        size_t slab = (gws_full[2] - z0 < Z_SLAB) ? (gws_full[2] - z0) : Z_SLAB;
        size_t offset[3] = {0, 0, z0};
        size_t gws[3]    = {gws_full[0], gws_full[1], slab};
        double t0 = get_time_sec();
        err = clEnqueueNDRangeKernel(cl->queue, k, 3, offset, gws, lws, 0, NULL, NULL);
        CL_CHECK(err, "bp_buffer enqueue (slab)");
        err = clFinish(cl->queue);
        CL_CHECK(err, "bp_buffer slab finish");
        double dt = get_time_sec() - t0;
        /* flag any slab that takes much longer than a healthy slab should —
         * pinpoints WHICH slab stalls within a slow epoch, since epoch
         * totals alone can't distinguish "one bad slab" from "uniformly
         * slower this epoch" */
        if (dt > 2.0)
            printf("    [bp_buffer slab %d, z0=%zu] %.3f s (SLOW)\n", slab_idx, z0, dt);
        slab_idx++;
    }
}

/* ── Internal: run forward projection on GPU (buffer mode) ─────────────── */
static void run_fp_buffer(CLState *cl, const CBpara *p,
                           cl_mem d_vol, cl_mem d_proj)
{
    cl_int err;
    cl_kernel k = cl->k_fp_buf;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W = p->detector_width,   H  = p->detector_height;
    int np = p->num_projs;
    int n_samples = p->n_samples;
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_vol);
    clSetKernelArg(k, 1, sizeof(cl_mem), &cl->d_R_mats);
    clSetKernelArg(k, 2, sizeof(cl_mem), &cl->d_T_vecs);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_proj);
    clSetKernelArg(k, 4, sizeof(int),    &Nxz);
    clSetKernelArg(k, 5, sizeof(int),    &Ny);
    clSetKernelArg(k, 6, sizeof(int),    &W);
    clSetKernelArg(k, 7, sizeof(int),    &H);
    clSetKernelArg(k, 8, sizeof(int),    &np);
    clSetKernelArg(k, 9, sizeof(int),    &n_samples);
    clSetKernelArg(k,10, sizeof(float),  &SOD);
    clSetKernelArg(k,11, sizeof(float),  &SDD);
    clSetKernelArg(k,12, sizeof(float),  &vs);
    clSetKernelArg(k,13, sizeof(float),  &px);

    /* Swept 16,16,1 / 8,32,1 / 4,64,1 / 32,8,1 / 8,16,1 at 512^3 (10-epoch
     * confirmation, after an earlier noisy 3-epoch pass suggested the
     * same trend): 4,64,1 gave 75.37s/10ep vs 321.19s/10ep for the old
     * 16,16,1 default -- >4x faster, and far more stable epoch-to-epoch
     * (mostly 5-10s vs 9-48s for the old default). 32,8,1 and 8,16,1 were
     * catastrophically worse (~4-8x slower), matching the same wide-short
     * shape being bad on this GPU that fp_image's sweep also found.
     * fp_buffer.cl's manual uncoalesced global-memory gather (no texture
     * cache, unlike fp_image) is far more sensitive to work-group shape
     * than fp_image was. FP_BUFFER_LWS=X,Y,Z still overrides for further
     * testing — keep Z=1, the angle-slab chunking below requires lws[2]
     * to divide ANG_SLAB=8 evenly. */
    size_t lws[3] = {4, 64, 1};
    const char *lws_env = getenv("FP_BUFFER_LWS");
    if (lws_env) {
        unsigned long a=4, b=64, c=1;
        if (sscanf(lws_env, "%lu,%lu,%lu", &a, &b, &c) == 3) {
            lws[0]=a; lws[1]=b; lws[2]=c;
        }
    }
    size_t gws_full[3] = {(size_t)W, (size_t)H, (size_t)np};
    for (int d=0;d<3;d++)
        if (gws_full[d] % lws[d]) gws_full[d] += lws[d] - gws_full[d] % lws[d];

    /* Chunk over angles (dim 2, lws=1) — same watchdog-avoidance rationale
     * as run_bp_buffer; fp_buffer's trilinear_buf is also uncoalesced. */
    const size_t ANG_SLAB = 8;
    int slab_idx = 0;
    for (size_t p0 = 0; p0 < gws_full[2]; p0 += ANG_SLAB) {
        size_t slab = (gws_full[2] - p0 < ANG_SLAB) ? (gws_full[2] - p0) : ANG_SLAB;
        size_t offset[3] = {0, 0, p0};
        size_t gws[3]    = {gws_full[0], gws_full[1], slab};
        double t0 = get_time_sec();
        err = clEnqueueNDRangeKernel(cl->queue, k, 3, offset, gws, lws, 0, NULL, NULL);
        CL_CHECK(err, "fp_buffer enqueue (slab)");
        err = clFinish(cl->queue);
        CL_CHECK(err, "fp_buffer slab finish");
        double dt = get_time_sec() - t0;
        if (dt > 2.0)
            printf("    [fp_buffer slab %d, p0=%zu] %.3f s (SLOW)\n", slab_idx, p0, dt);
        slab_idx++;
    }
}

/* ── Internal: run backprojection (image mode) ──────────────────────────── */
static void run_bp_image(CLState *cl, const CBpara *p,
                          cl_mem proj_img, cl_mem d_ang_cs, cl_mem d_vol)
{
    cl_int err;
    cl_kernel k = cl->k_bp_img;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W = p->detector_width,   H  = p->detector_height;
    int np = p->num_projs;
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &proj_img);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_ang_cs);
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
    size_t lws[3] = {4, 4, 16};
    for (int d=0;d<3;d++) {
        if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
    }
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0,NULL,NULL);
    CL_CHECK(err, "bp_image enqueue");
}

/* ── Internal: run forward projection (image mode) ─────────────────────── */
static void run_fp_image(CLState *cl, const CBpara *p,
                          cl_mem vol_img, cl_mem d_proj)
{
    cl_int err;
    cl_kernel k = cl->k_fp_img;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W = p->detector_width,   H  = p->detector_height;
    int np = p->num_projs;
    int n_samples = p->n_samples;
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &vol_img);
    clSetKernelArg(k, 1, sizeof(cl_mem), &cl->d_R_mats);
    clSetKernelArg(k, 2, sizeof(cl_mem), &cl->d_T_vecs);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_proj);
    clSetKernelArg(k, 4, sizeof(int),    &Nxz);
    clSetKernelArg(k, 5, sizeof(int),    &Ny);
    clSetKernelArg(k, 6, sizeof(int),    &W);
    clSetKernelArg(k, 7, sizeof(int),    &H);
    clSetKernelArg(k, 8, sizeof(int),    &np);
    clSetKernelArg(k, 9, sizeof(int),    &n_samples);
    clSetKernelArg(k,10, sizeof(float),  &SOD);
    clSetKernelArg(k,11, sizeof(float),  &SDD);
    clSetKernelArg(k,12, sizeof(float),  &vs);
    clSetKernelArg(k,13, sizeof(float),  &px);
    /* AABB ray clipping helps on large detectors (many edge pixels miss
     * volume) but hurts at 256^3: re-measured after the work-group sweep
     * above in case that changed the tradeoff (it didn't). Forcing AABB on
     * at 256^3 via FP_IMAGE_AABB=1 gave 0.098-0.100s/epoch vs 0.095-0.096s
     * with it off — a consistent ~4% regression despite geometry showing
     * ~65% of samples per ray are outside the volume there. The kernel is
     * small enough at this scale (0.095s launch) that AABB's setup cost
     * (6 divides + branches before the march) and the ragged per-thread
     * trip counts it creates cost more than the fetch reduction saves.
     * Gate stays W>512; FP_IMAGE_AABB kept for future re-testing. */
    int use_aabb = (W > 512) ? 1 : 0;
    const char *aabb_env = getenv("FP_IMAGE_AABB");
    if (aabb_env) use_aabb = atoi(aabb_env);
    clSetKernelArg(k,14, sizeof(int), &use_aabb);

    size_t gws[3] = {(size_t)W, (size_t)H, (size_t)np};
    /* {8,32,1} measured ~5% faster than the previous {16,16,1} at 512^3 on
     * this hardware (AMD Hawaii): 0.873-0.879s/epoch vs 0.921-0.928s over
     * a 10-candidate sweep (64,1,1 through 32,16,1; the latter and
     * {16,32,1} exceed the device's 256-item max work-group size and fail
     * with CL_INVALID_WORK_GROUP_SIZE). Aspect ratio matters independent
     * of total group size: {8,32,1} beat {32,8,1} by 25%, and {8,16,1}
     * (half the items, same 1:4 ratio) was 20% slower than {8,32,1} — so
     * this is not just "more occupancy wins", the W-narrow/H-tall shape
     * specifically helps on this detector layout (W=1120, H=1184). */
    size_t lws[3] = {8, 32, 1};
    /* FP_IMAGE_LWS=X,Y,Z overrides the work-group shape for further
     * sweeping without a rebuild. */
    const char *lws_env = getenv("FP_IMAGE_LWS");
    if (lws_env) {
        unsigned long a=8, b=32, c=1;
        if (sscanf(lws_env, "%lu,%lu,%lu", &a, &b, &c) == 3) {
            lws[0]=a; lws[1]=b; lws[2]=c;
        }
    }
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

    build_RT_buffers(cl, p);

    /* Build float2 cos/sin LUT for bp_buffer */
    float *ang_cs = (float *)malloc(np * 2 * sizeof(float));
    float *ang_f  = (float *)malloc(np * sizeof(float));
    for (int i=0;i<np;i++) {
        ang_cs[2*i]   = (float)cos(p->angles[i]);
        ang_cs[2*i+1] = (float)sin(p->angles[i]);
        ang_f[i]      = (float)p->angles[i];
    }

    /* ── Allocate device buffers ── */
    cl_mem d_proj_meas = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                         proj_bytes, (void*)proj_measured, &err);
    CL_CHECK(err, "d_proj_meas");

    cl_mem d_ang_cs = clCreateBuffer(cl->ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                      np*2*sizeof(float), ang_cs, &err);
    CL_CHECK(err, "d_ang_cs buf");
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

    /* d_proj_meas stays raw — cone weight applied to ratio after divide, matching Python */

    /* ── Precompute bp(ones) ── */
    {
        cl_mem d_ones_raw  = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        cl_mem d_ones_prep = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        float one = 1.f;
        err = clEnqueueFillBuffer(cl->queue, d_ones_raw, &one, sizeof(float),
                                  0, proj_bytes, 0, NULL, NULL);
        CL_CHECK(err, "fill ones");
        run_preprocess(cl, p, d_ones_raw, d_ones_prep);  /* fused: cone_weight + flip + transpose */

        float zero = 0.f;
        clEnqueueFillBuffer(cl->queue, d_bp_ones, &zero, sizeof(float),
                            0, vol_bytes, 0, NULL, NULL);

        if (cl->mode == GPU_MODE_BUFFER) {
            run_bp_buffer(cl, p, d_ones_prep, d_ang_cs, d_bp_ones);
        } else {
            cl_image_format fmt = {CL_R, CL_FLOAT};
            cl_image_desc desc = {0};
            desc.image_type       = CL_MEM_OBJECT_IMAGE2D_ARRAY;
            desc.image_width      = (size_t)H;
            desc.image_height     = (size_t)W;
            desc.image_array_size = (size_t)np;
            cl_mem ones_img = clCreateImage(cl->ctx, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err);
            CL_CHECK(err,"ones_img");
            size_t oorigin[3]={0,0,0};
            size_t oregion[3]={(size_t)H,(size_t)W,(size_t)np};
            err = clEnqueueCopyBufferToImage(cl->queue, d_ones_prep, ones_img,
                                             0, oorigin, oregion, 0, NULL, NULL);
            CL_CHECK(err,"CopyBufferToImage ones_prep");
            run_bp_image(cl, p, ones_img, d_ang_cs, d_bp_ones);
            clReleaseMemObject(ones_img);
        }
        clFinish(cl->queue);  /* wait for bp_ones before epoch loop */
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

        /* persistent vol image3D — CL_HALF_FLOAT halves texture bandwidth but
         * quantizes to ~3 decimal digits; use --half to opt in, default float32 */
        cl_image_format vol_fmt_img = p->use_half_vol
            ? (cl_image_format){CL_R, CL_HALF_FLOAT}
            : (cl_image_format){CL_R, CL_FLOAT};
        cl_image_desc vdesc={0};
        vdesc.image_type=CL_MEM_OBJECT_IMAGE3D;
        vdesc.image_width=(size_t)Ny; vdesc.image_height=(size_t)Nxz; vdesc.image_depth=(size_t)Nxz;
        vol_img = clCreateImage(cl->ctx, CL_MEM_READ_WRITE, &vol_fmt_img, &vdesc, NULL, &err);
        CL_CHECK(err,"vol_img persistent");
    }

    /* staging buffer for vol_img upload (gpu-img mode); only needed for
     * --half, since float_to_half must write somewhere before the image
     * copy. In float32 mode d_vol is copied straight into vol_img below —
     * the old code copied d_vol into this buffer first via a same-format
     * clEnqueueCopyBuffer memcpy that did nothing but add ~1.07GB/epoch of
     * redundant traffic at 512^3 (~0.7% of a 930ms epoch — real but small). */
    cl_mem d_vol_half_img = NULL;
    if (cl->mode == GPU_MODE_IMAGE && p->use_half_vol) {
        int vol_n_img = Nxz * Nxz * Ny;
        d_vol_half_img = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE,
                                        (size_t)vol_n_img * sizeof(cl_half), NULL, &err);
        CL_CHECK(err,"d_vol_half_img");
    }

    /* persistent ratio image2d_array for gpu-img mode */
    cl_mem ratio_img_buf = NULL;
    (void)0; /* removed unused d_ratio_half_img */
    if (cl->mode == GPU_MODE_IMAGE) {
        cl_image_desc rdesc={0};
        rdesc.image_type=CL_MEM_OBJECT_IMAGE2D_ARRAY;
        rdesc.image_width=(size_t)H; rdesc.image_height=(size_t)W;
        rdesc.image_array_size=(size_t)np;
        ratio_img_buf = clCreateImage(cl->ctx, CL_MEM_READ_WRITE, &img_fmt, &rdesc, NULL, &err);
        CL_CHECK(err,"ratio_img_buf persistent");
    }

    /* ── Iterative loop ─────────────────────────────────────────────────── */
    int proj_n = np * H * W;
    int vol_n  = Nxz * Nxz * Ny;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t_ep = get_time_sec();

        /* forward project: b = F(v0) */
        if (cl->mode == GPU_MODE_BUFFER) {
            run_fp_buffer(cl, p, d_vol, d_proj_b);
        } else {
            size_t vorigin[3]={0,0,0};
            size_t vregion[3]={(size_t)Ny,(size_t)Nxz,(size_t)Nxz};
            if (p->use_half_vol) {
                cl_kernel k = cl->k_f2h;
                clSetKernelArg(k, 0, sizeof(cl_mem), &d_vol);
                clSetKernelArg(k, 1, sizeof(cl_mem), &d_vol_half_img);
                clSetKernelArg(k, 2, sizeof(int),    &vol_n);
                size_t gws = ((size_t)vol_n + 255) / 256 * 256;
                size_t lws = 256;
                err = clEnqueueNDRangeKernel(cl->queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
                CL_CHECK(err,"float_to_half vol img");
                err=clEnqueueCopyBufferToImage(cl->queue, d_vol_half_img, vol_img,
                                               0, vorigin, vregion, 0,NULL,NULL);
                CL_CHECK(err,"CopyBufferToImage vol img");
            } else {
                /* float32: copy d_vol straight into vol_img, skipping the
                 * same-format staging buffer entirely. */
                err=clEnqueueCopyBufferToImage(cl->queue, d_vol, vol_img,
                                               0, vorigin, vregion, 0,NULL,NULL);
                CL_CHECK(err,"CopyBufferToImage vol img float32");
            }
            run_fp_image(cl, p, vol_img, d_proj_b);
        }

        /* ratio = p0 / b */
        {
            cl_kernel k = cl->k_divide;
            clSetKernelArg(k,0,sizeof(cl_mem),&d_proj_meas);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_proj_b);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_ratio);
            clSetKernelArg(k,3,sizeof(int),   &proj_n);
            size_t gws=((size_t)proj_n + 3) / 4;
            err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
            CL_CHECK(err,"proj_divide");
        }

        run_preprocess(cl, p, d_ratio, d_ratio_prep);  /* fused: cone_weight + flip + transpose */

        /* bp_ratio = bp(ratio_prep). No zero-fill needed: bp_buffer/bp_image
         * kernels unconditionally overwrite every voxel in range (plain
         * assignment, not accumulation), so pre-clearing is dead work. */
        {
            if (cl->mode == GPU_MODE_BUFFER) {
                run_bp_buffer(cl, p, d_ratio_prep, d_ang_cs, d_bp_ratio);
            } else {
                size_t rorigin[3]={0,0,0};
                size_t rregion[3]={(size_t)H,(size_t)W,(size_t)np};
                err=clEnqueueCopyBufferToImage(cl->queue, d_ratio_prep, ratio_img_buf,
                                               0, rorigin, rregion, 0,NULL,NULL);
                CL_CHECK(err,"CopyBufferToImage ratio img");
                run_bp_image(cl, p, ratio_img_buf, d_ang_cs, d_bp_ratio);
            }
        }

        /* v0 *= bp_ratio / bp_ones */
        {
            cl_kernel k = cl->k_update;
            clSetKernelArg(k,0,sizeof(cl_mem),&d_vol);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_bp_ratio);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ones);
            clSetKernelArg(k,3,sizeof(int),   &vol_n);
            size_t gws=((size_t)vol_n + 3) / 4;
            err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
            CL_CHECK(err,"vol_update");
        }
        clFinish(cl->queue);  /* single sync per epoch — for timing */
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
    if (d_vol_half_img) clReleaseMemObject(d_vol_half_img);
    if (ratio_img_buf)  clReleaseMemObject(ratio_img_buf);
    clReleaseMemObject(d_ang_cs);
    clReleaseMemObject(cl->d_R_mats);
    clReleaseMemObject(cl->d_T_vecs);
    free(ang_cs);
    free(ang_f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * reconstruct_gpu_opt — uses optimized bp kernel (bp_opt)
 *
 * Key differences from reconstruct_gpu:
 *  - Passes float2 angle_cs (cos/sin LUT) instead of raw angle array
 *  - Passes __local scratch buffer sized W*H floats to bp_opt
 *  - Uses k_bp_opt for backprojection and fp_image for forward projection
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

    /* local mem for angle_cs LUT cache in bp_opt */
    size_t lmem_bytes = (size_t)np * sizeof(cl_float2);

    /* image format for proj image array */
    cl_image_format img_fmt = {CL_R, CL_FLOAT};

    build_RT_buffers(cl, p);

    /* d_proj_meas stays raw — cone weight applied to ratio after divide, matching Python */

    /* ── bp_ones: preprocess all-ones → image, run bp_opt ── */
    {
        cl_mem d_ones_raw  = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        CL_CHECK(err,"d_ones_raw opt");
        cl_mem d_ones_prep = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        CL_CHECK(err,"d_ones_prep opt");
        float one=1.f;
        clEnqueueFillBuffer(cl->queue, d_ones_raw, &one, sizeof(float), 0, proj_bytes, 0, NULL, NULL);
        run_preprocess(cl, p, d_ones_raw, d_ones_prep);  /* fused: cone_weight + flip + transpose */
        clReleaseMemObject(d_ones_raw);

        float zero=0.f;
        clEnqueueFillBuffer(cl->queue,d_bp_ones,&zero,sizeof(float),0,vol_bytes,0,NULL,NULL);

        cl_image_desc idesc={0};
        idesc.image_type=CL_MEM_OBJECT_IMAGE2D_ARRAY;
        idesc.image_width=(size_t)H; idesc.image_height=(size_t)W;
        idesc.image_array_size=(size_t)np;
        cl_mem ones_img = clCreateImage(cl->ctx, CL_MEM_READ_WRITE, &img_fmt, &idesc, NULL, &err);
        CL_CHECK(err,"ones_img opt");
        size_t oorigin[3]={0,0,0};
        size_t oregion[3]={(size_t)H,(size_t)W,(size_t)np};
        err = clEnqueueCopyBufferToImage(cl->queue, d_ones_prep, ones_img,
                                         0, oorigin, oregion, 0, NULL, NULL);
        CL_CHECK(err,"CopyBufferToImage ones_prep opt");
        clReleaseMemObject(d_ones_prep);

        {
            cl_kernel k = cl->k_bp_opt;
            float SOD=(float)p->SOD,SDD=(float)p->SDD;
            float vs=(float)p->voxelSize,px=(float)p->pixelSize;
            clSetKernelArg(k,0,sizeof(cl_mem),&ones_img);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_ang_cs);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ones);
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
            size_t lws[3]={4,4,16};
            for(int d=0;d<3;d++) if(gws[d]%lws[d]) gws[d]+=lws[d]-gws[d]%lws[d];
            err=clEnqueueNDRangeKernel(cl->queue,k,3,NULL,gws,lws,0,NULL,NULL);
            CL_CHECK(err,"bp_opt ones");
        }
        clFinish(cl->queue);  /* wait for bp_opt(ones) before epoch loop */
        clReleaseMemObject(ones_img);
        printf("  bp_opt(ones) computed.\n");
    }

    int proj_n = np*H*W;
    int vol_n  = Nxz*Nxz*Ny;
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    /* Persistent image2d_array for ratio */
    cl_image_format half_fmt = {CL_R, CL_HALF_FLOAT};
    cl_image_desc rdesc={0};
    rdesc.image_type=CL_MEM_OBJECT_IMAGE2D_ARRAY;
    rdesc.image_width=(size_t)H; rdesc.image_height=(size_t)W;
    rdesc.image_array_size=(size_t)np;
    cl_mem ratio_img = clCreateImage(cl->ctx, CL_MEM_READ_WRITE, &img_fmt, &rdesc, NULL, &err);
    CL_CHECK(err,"ratio_img persistent");

    /* vol_img precision: --half opts into CL_HALF_FLOAT (bandwidth win,
     * ~3-decimal-digit quantization floor); default float32 for accuracy */
    cl_image_format vol_fmt = p->use_half_vol ? half_fmt : img_fmt;
    cl_image_desc vdesc={0};
    vdesc.image_type=CL_MEM_OBJECT_IMAGE3D;
    vdesc.image_width=(size_t)Ny;
    vdesc.image_height=(size_t)Nxz;
    vdesc.image_depth=(size_t)Nxz;
    cl_mem vol_img = clCreateImage(cl->ctx, CL_MEM_READ_WRITE, &vol_fmt, &vdesc, NULL, &err);
    CL_CHECK(err,"vol_img persistent");

    /* Staging buffer: only needed for --half (float_to_half must write
     * somewhere before the image copy). In float32 mode d_vol is copied
     * straight into vol_img in the epoch loop below — no staging buffer,
     * no allocation. The old unconditional allocation + same-format
     * clEnqueueCopyBuffer memcpy did nothing in float32 mode but add
     * ~1.07GB/epoch of redundant traffic at 512^3 (~0.7% of a 930ms
     * epoch — real but small). */
    cl_mem d_vol_half = NULL;
    if (p->use_half_vol) {
        d_vol_half = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE,
                                    (size_t)vol_n * sizeof(cl_half), NULL, &err);
        CL_CHECK(err,"d_vol_half");
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t_ep = get_time_sec();

        /* ── float d_vol → vol_img (GPU-side, no PCIe) ── */
        size_t origin[3]={0,0,0};
        size_t region[3]={(size_t)Ny,(size_t)Nxz,(size_t)Nxz};
        if (p->use_half_vol) {
            cl_kernel k = cl->k_f2h;
            clSetKernelArg(k, 0, sizeof(cl_mem), &d_vol);
            clSetKernelArg(k, 1, sizeof(cl_mem), &d_vol_half);
            clSetKernelArg(k, 2, sizeof(int),    &vol_n);
            size_t gws = ((size_t)vol_n + 255) / 256 * 256;
            size_t lws = 256;
            err = clEnqueueNDRangeKernel(cl->queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
            CL_CHECK(err,"float_to_half vol");
            err=clEnqueueCopyBufferToImage(cl->queue, d_vol_half, vol_img,
                                           0, origin, region, 0,NULL,NULL);
            CL_CHECK(err,"CopyBufferToImage vol half");
        } else {
            err=clEnqueueCopyBufferToImage(cl->queue, d_vol, vol_img,
                                           0, origin, region, 0,NULL,NULL);
            CL_CHECK(err,"CopyBufferToImage vol float32");
        }
        run_fp_image(cl, p, vol_img, d_proj_b);

        /* ── ratio = p0/b ── */
        {
            cl_kernel k = cl->k_divide;
            clSetKernelArg(k,0,sizeof(cl_mem),&d_proj_meas);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_proj_b);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_ratio);
            clSetKernelArg(k,3,sizeof(int),&proj_n);
            size_t gws=((size_t)proj_n + 3) / 4;
            err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
            CL_CHECK(err,"divide opt");
        }
        run_preprocess(cl, p, d_ratio, d_ratio_prep);

        /* ── copy d_ratio_prep buffer → ratio_img ── */
        {
            size_t origin[3]={0,0,0};
            size_t region[3]={(size_t)H,(size_t)W,(size_t)np};
            err=clEnqueueCopyBufferToImage(cl->queue, d_ratio_prep, ratio_img,
                                           0, origin, region, 0, NULL, NULL);
            CL_CHECK(err,"CopyBufferToImage ratio");
        }

        /* ── bp_opt(ratio_img) ── */
        {
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
            size_t lws[3]={4,4,16};
            for(int d=0;d<3;d++) if(gws[d]%lws[d]) gws[d]+=lws[d]-gws[d]%lws[d];
            err=clEnqueueNDRangeKernel(cl->queue,k,3,NULL,gws,lws,0,NULL,NULL);
            CL_CHECK(err,"bp_opt ratio");
        }

        /* ── update (float4 vectorized) ── */
        {
            cl_kernel k = cl->k_update;
            clSetKernelArg(k,0,sizeof(cl_mem),&d_vol);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_bp_ratio);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ones);
            clSetKernelArg(k,3,sizeof(int),&vol_n);
            size_t gws=((size_t)vol_n + 3) / 4;
            err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
            CL_CHECK(err,"update opt");
        }
        clFinish(cl->queue);
        printf("  epoch %3d/%d  %.3f s\n", epoch+1, epochs, get_time_sec()-t_ep);
    }
    clReleaseMemObject(vol_img);
    if (d_vol_half) clReleaseMemObject(d_vol_half);
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
    clReleaseMemObject(cl->d_R_mats);
    clReleaseMemObject(cl->d_T_vecs);
    free(ang_cs);
    free(ang_f);
}
