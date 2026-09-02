#include "ct_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* ── Helper macros ──────────────────────────────────────────────────────── */
#define CL_CHECK(err, msg) \
    do { if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s\n", (err), (msg)); \
        exit(1); } } while(0)

static cl_program build_program_opts(cl_context ctx, cl_device_id dev,
                                       const char *src, const char *extra_opts);

/* ── Diagnostic: per-kernel resource usage ────────────────────────────
 * OpenCL exposes no direct register count, but CL_KERNEL_PRIVATE_MEM_SIZE
 * (spilled registers) and a CL_KERNEL_WORK_GROUP_SIZE lower than the
 * device max both indicate register pressure worth addressing. Behind
 * CT_DIAG_KERNEL_INFO=1 since it adds several blocking API calls. */
static void diag_print_kernel_info(cl_kernel k, const char *name, cl_device_id dev)
{
    if (!k) return;
    size_t wg_size = 0, pref_multiple = 0;
    cl_ulong local_mem = 0, private_mem = 0;
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE,
                              sizeof(wg_size), &wg_size, NULL);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                              sizeof(pref_multiple), &pref_multiple, NULL);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE,
                              sizeof(local_mem), &local_mem, NULL);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE,
                              sizeof(private_mem), &private_mem, NULL);
    printf("    [kernel-info] %-24s max_wg=%zu pref_mult=%zu local_mem=%llu B private_mem=%llu B\n",
           name, wg_size, pref_multiple,
           (unsigned long long)local_mem, (unsigned long long)private_mem);
}

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
    return build_program_opts(ctx, dev, src, "-cl-fast-relaxed-math");
}

/* extra_opts is appended after the always-on -cl-fast-relaxed-math, e.g.
 * "-DHAVE_FP16" when the device supports cl_khr_fp16 (see gpu_init) --
 * lets a kernel source #ifdef out device-unsupported code paths instead
 * of failing the whole program's build. */
static cl_program build_program_opts(cl_context ctx, cl_device_id dev,
                                       const char *src, const char *extra_opts)
{
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src,
                                                NULL, &err);
    CL_CHECK(err, "clCreateProgramWithSource");

    char opts[256];
    snprintf(opts, sizeof(opts), "-cl-fast-relaxed-math %s", extra_opts ? extra_opts : "");
    err = clBuildProgram(prog, 1, &dev, opts, NULL, NULL);
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

    /* cl_khr_3d_image_writes gates the float32 vol_img fusion path
     * (OpenCL 1.2 has no 3D read-write images without it). cl_khr_fp16
     * gates --half's float_to_half kernel -- present on Hawaii, absent
     * on GTX 680, so without this check --half's kernel source fails to
     * build and takes down the whole image-mode program, not just
     * --half. Remembered on CLState so main.c can refuse --half cleanly. */
    {
        size_t ext_sz = 0;
        clGetDeviceInfo(cl->device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_sz);
        char *ext_str = (char *)malloc(ext_sz + 1);
        clGetDeviceInfo(cl->device, CL_DEVICE_EXTENSIONS, ext_sz, ext_str, NULL);
        ext_str[ext_sz] = '\0';
        int has_3d_image_writes = (strstr(ext_str, "cl_khr_3d_image_writes") != NULL);
        printf("  cl_khr_3d_image_writes: %s\n", has_3d_image_writes ? "yes" : "no");
        cl->has_fp16 = (strstr(ext_str, "cl_khr_fp16") != NULL);
        printf("  cl_khr_fp16: %s\n", cl->has_fp16 ? "yes" : "no");
        free(ext_str);
    }

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
        cl->k_divide_preproc_img = clCreateKernel(cl->prog_buffer, "divide_preprocess_img", &err);
        CL_CHECK(err, "divide_preprocess_img");
        cl->k_update_img = clCreateKernel(cl->prog_buffer, "vol_update_img", &err);
        CL_CHECK(err, "vol_update_img");
        cl->k_cone_hw  = clCreateKernel(cl->prog_buffer, "cone_weight_hw", &err);
        CL_CHECK(err, "cone_weight_hw");
    }

    /* ── Image-mode program (also used by OPT for fp_image) ── */
    if (mode == GPU_MODE_IMAGE || mode == GPU_MODE_OPT) {
        char *src_bp  = load_source(path_bp_img);
        char *src_fp  = load_source(path_fp_img);
        char *combined = concat_src(src_bp, src_fp);
        cl->prog_image = build_program_opts(cl->ctx, cl->device, combined,
                                            cl->has_fp16 ? "-DHAVE_FP16" : "");
        free(src_bp); free(src_fp); free(combined);

        cl->k_bp_img = clCreateKernel(cl->prog_image, "bp_image", &err);
        CL_CHECK(err, "bp_image");
        cl->k_fp_img = clCreateKernel(cl->prog_image, "fp_image", &err);
        CL_CHECK(err, "fp_image");
        /* bp_image + vol_update_img fusion -- created here regardless of
         * IMAGE vs OPT (cheap), but only ever launched from gpu-img. */
        cl->k_bp_img_update = clCreateKernel(cl->prog_image, "bp_image_update", &err);
        CL_CHECK(err, "bp_image_update");
        /* float_to_half only exists in the compiled program when
         * HAVE_FP16 was defined (see kernels/fp_image.cl's #ifdef) --
         * k_f2h stays NULL on devices without cl_khr_fp16, and --half
         * is refused in main.c before this would ever be dereferenced. */
        cl->k_f2h = cl->has_fp16 ? clCreateKernel(cl->prog_image, "float_to_half", &err) : NULL;
        if (cl->has_fp16) CL_CHECK(err, "float_to_half");
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

    /* Phase 0b diagnostic (see diag_print_kernel_info above). Off by
     * default -- opt in with CT_DIAG_KERNEL_INFO=1. */
    {
        const char *diag_env = getenv("CT_DIAG_KERNEL_INFO");
        if (diag_env && atoi(diag_env) != 0) {
            printf("  [kernel-info] per-kernel resource usage:\n");
            diag_print_kernel_info(cl->k_bp_buf,  "bp_buffer",  cl->device);
            diag_print_kernel_info(cl->k_fp_buf,  "fp_buffer",  cl->device);
            diag_print_kernel_info(cl->k_bp_img,  "bp_image",   cl->device);
            diag_print_kernel_info(cl->k_fp_img,  "fp_image",   cl->device);
            diag_print_kernel_info(cl->k_bp_opt,  "bp_opt",     cl->device);
            diag_print_kernel_info(cl->k_divide,  "proj_divide", cl->device);
            diag_print_kernel_info(cl->k_update,  "vol_update", cl->device);
            diag_print_kernel_info(cl->k_update_img, "vol_update_img", cl->device);
            diag_print_kernel_info(cl->k_bp_img_update, "bp_image_update", cl->device);
        }
    }

    return 0;
}

void gpu_cleanup(CLState *cl)
{
    if (cl->mode != GPU_MODE_OPT) {
        clReleaseKernel(cl->k_bp_buf); clReleaseKernel(cl->k_fp_buf);
        clReleaseKernel(cl->k_divide); clReleaseKernel(cl->k_update);
        clReleaseKernel(cl->k_preproc);
        clReleaseKernel(cl->k_divide_preproc_img); clReleaseKernel(cl->k_update_img);
        clReleaseKernel(cl->k_cone_hw);
        clReleaseProgram(cl->prog_buffer);
    }
    if (cl->mode == GPU_MODE_IMAGE) {
        clReleaseKernel(cl->k_bp_img); clReleaseKernel(cl->k_fp_img);
        clReleaseKernel(cl->k_bp_img_update);
        if (cl->k_f2h) clReleaseKernel(cl->k_f2h);
        clReleaseProgram(cl->prog_image);
    }
    if (cl->mode == GPU_MODE_OPT) {
        clReleaseKernel(cl->k_bp_opt);
        clReleaseKernel(cl->k_divide); clReleaseKernel(cl->k_update);
        clReleaseKernel(cl->k_preproc);
        clReleaseKernel(cl->k_divide_preproc_img); clReleaseKernel(cl->k_update_img);
        clReleaseKernel(cl->k_cone_hw);
        clReleaseKernel(cl->k_fp_img); clReleaseKernel(cl->k_bp_img);
        clReleaseKernel(cl->k_bp_img_update);
        if (cl->k_f2h) clReleaseKernel(cl->k_f2h);
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

/*
 * Fuses proj_divide (ratio=p0/b) with cone_weight + flip + transpose,
 * writing straight into an image2d_array_t (ratio_img) instead of a
 * buffer -- replaces three steps (divide, preprocess, buffer-to-image
 * copy) with one kernel launch. See divide_preprocess_img in
 * kernels/bp_buffer.cl for the kernel itself.
 *
 * ip_start/ip_count: OSEM support, mirrors run_fp_image's pattern. */
static void run_divide_preprocess_img(CLState *cl, const CBpara *p,
                                       cl_mem d_proj_meas, cl_mem d_proj_b,
                                       cl_mem dst_img, int ip_start, int ip_count)
{
    cl_int err;
    cl_kernel k = cl->k_divide_preproc_img;
    int W = p->detector_width, H = p->detector_height;
    float vs  = (float)p->voxelSize;
    float SDD = (float)p->SDD;
    float px  = (float)p->pixelSize;

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_proj_meas);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_proj_b);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dst_img);
    clSetKernelArg(k, 3, sizeof(int),    &W);
    clSetKernelArg(k, 4, sizeof(int),    &H);
    clSetKernelArg(k, 5, sizeof(float),  &vs);
    clSetKernelArg(k, 6, sizeof(float),  &SDD);
    clSetKernelArg(k, 7, sizeof(float),  &px);
    clSetKernelArg(k, 8, sizeof(int),    &ip_start);
    clSetKernelArg(k, 9, sizeof(int),    &ip_count);

    size_t gws[3] = {(size_t)W, (size_t)H, (size_t)ip_count};
    size_t lws[3] = {16, 16, 1};
    for (int d = 0; d < 3; d++)
        if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
    size_t offset[3] = {0, 0, (size_t)ip_start};
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, offset, gws, lws, 0, NULL, NULL);
    CL_CHECK(err, "divide_preprocess_img enqueue");
}


/* ── Internal: run backprojection on GPU (buffer mode) ─────────────────── */
/* OSEM: ip_start/ip_count select a contiguous angle subrange for bp's
 * internal angle loop; this function's z-slab chunking is over voxels,
 * unrelated to angles, so no host-side offset-launch changes needed. */
static void run_bp_buffer(CLState *cl, const CBpara *p,
                           cl_mem d_proj, cl_mem d_ang_cs,
                           cl_mem d_vol, int ip_start, int ip_count)
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
    clSetKernelArg(k,12, sizeof(int),    &ip_start);
    clSetKernelArg(k,13, sizeof(int),    &ip_count);

    /* iz is dim 0 (fastest local id), ix is dim 2 (slowest) -- see the
     * kernel's own comment at bp_buffer.cl. lws[0] must stay a divisor
     * of Z_SLAB=64 below (the z-slab chunking requires it). */
    size_t lws[3] = {16, 4, 4};
    {
        const char *bp_lws_env = getenv("BP_LWS");
        if (bp_lws_env) {
            unsigned long a=16, b=4, c=4;
            if (sscanf(bp_lws_env, "%lu,%lu,%lu", &a, &b, &c) == 3) {
                lws[0]=a; lws[1]=b; lws[2]=c;
            }
        }
    }
    size_t gws_full[3] = {(size_t)Ny, (size_t)Nxz, (size_t)Nxz};
    for (int d=0;d<3;d++)
        if (gws_full[d] % lws[d]) gws_full[d] += lws[d] - gws_full[d] % lws[d];

    /* Chunk the z-dimension into slabs so no single kernel launch runs
     * long enough to trip the GPU watchdog (bp_buffer's uncoalesced
     * global loads can exceed the driver timeout at 512^3 unchunked). */
    const size_t Z_SLAB = 64;

    /* Mirrors FP_BUFFER_SKIP_SLAB_FINISH (run_fp_buffer, above): the
     * per-slab clFinish here also enables the slow-slab diagnostic.
     * Off by default so the diagnostic stays on. */
    int skip_slab_finish = 0;
    {
        const char *skip_env = getenv("BP_BUFFER_SKIP_SLAB_FINISH");
        if (skip_env && atoi(skip_env) != 0) skip_slab_finish = 1;
    }

    int slab_idx = 0;
    for (size_t z0 = 0; z0 < gws_full[0]; z0 += Z_SLAB) {
        size_t slab = (gws_full[0] - z0 < Z_SLAB) ? (gws_full[0] - z0) : Z_SLAB;
        size_t offset[3] = {z0, 0, 0};
        size_t gws[3]    = {slab, gws_full[1], gws_full[2]};

        if (skip_slab_finish) {
            /* batched path: enqueue only, no per-slab sync/timing/watchdog.
             * The queue is in-order, so the final clFinish after this loop
             * (called by the caller before reading results) still ensures
             * completion -- just without per-slab visibility. */
            cl_event evt;
            err = clEnqueueNDRangeKernel(cl->queue, k, 3, offset, gws, lws, 0, NULL, &evt);
            CL_CHECK(err, "bp_buffer enqueue (slab, batched)");
            clReleaseEvent(evt);
        } else {
            double t0 = get_time_sec();
            err = clEnqueueNDRangeKernel(cl->queue, k, 3, offset, gws, lws, 0, NULL, NULL);
            CL_CHECK(err, "bp_buffer enqueue (slab)");
            err = clFinish(cl->queue);
            CL_CHECK(err, "bp_buffer slab finish");
            double dt = get_time_sec() - t0;
            /* pinpoints which slab stalls within a slow epoch */
            if (dt > 2.0)
                printf("    [bp_buffer slab %d, z0=%zu] %.3f s (SLOW)\n", slab_idx, z0, dt);
        }
        slab_idx++;
    }
}

/* ── Internal: run forward projection on GPU (buffer mode) ─────────────── */
/* OSEM: ip_start/ip_count select a contiguous angle
 * subrange, chunked the same way as the full-range path (ANG_SLAB=8 per
 * launch, watchdog avoidance). Same rounding-edge-case fix as
 * run_fp_image: the kernel's guard checks "ip >= ip_start + ip_count",
 * not "ip >= num_projs", since a slab's rounded-up gws can exceed the
 * subset's actual end while staying under num_projs. */
static void run_fp_buffer(CLState *cl, const CBpara *p,
                           cl_mem *d_vol_ptr, cl_mem d_proj,
                           int ip_start, int ip_count)
{
    cl_int err;
    cl_kernel k = cl->k_fp_buf;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W = p->detector_width,   H  = p->detector_height;
    int np = p->num_projs;
    int n_samples = p->n_samples;
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;
    cl_mem d_vol = *d_vol_ptr;

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
    /* AABB is ~6% SLOWER on gpu-buf, unlike fp_image/fp_cpu where it's a
     * real win -- gpu-buf's cost is dominated by uncoalesced memory
     * access, which AABB's sample-range narrowing does nothing for.
     * Default off unconditionally (not W>512, inherited from fp_image
     * without being re-verified here). FP_IMAGE_AABB still overrides. */
    int use_aabb = 0;
    {
        const char *aabb_env = getenv("FP_IMAGE_AABB");
        if (aabb_env) use_aabb = atoi(aabb_env);
    }
    clSetKernelArg(k,14, sizeof(int),    &use_aabb);
    clSetKernelArg(k,15, sizeof(int),    &ip_start);
    clSetKernelArg(k,16, sizeof(int),    &ip_count);

    /* Work-group shape does not transfer across GPU architectures: swept
     * separately on Hawaii (GCN, 64-wide wavefront, optimum {4,64,1}) and
     * GTX 680 (Kepler, 32-wide warp, optimum {2,16,2}) -- re-tune per
     * platform rather than reusing one card's numbers. FP_BUFFER_LWS=X,Y,Z
     * still overrides; keep Z a divisor of ANG_SLAB=8 below. */
    size_t lws[3] = {2, 16, 2};
    const char *lws_env = getenv("FP_BUFFER_LWS");
    if (lws_env) {
        unsigned long a=4, b=64, c=1;
        if (sscanf(lws_env, "%lu,%lu,%lu", &a, &b, &c) == 3) {
            lws[0]=a; lws[1]=b; lws[2]=c;
        }
    }
    size_t gws_full[3] = {(size_t)W, (size_t)H, (size_t)ip_count};
    for (int d=0;d<3;d++)
        if (gws_full[d] % lws[d]) gws_full[d] += lws[d] - gws_full[d] % lws[d];

    /* Chunk over angles (dim 2, lws=1) -- same watchdog-avoidance
     * rationale as run_bp_buffer. p0 ranges over the subset's local
     * offsets; the launch offset is ip_start+p0 so get_global_id(2)
     * lands on the correct absolute angle index. */
    const size_t ANG_SLAB = 8;
    int slab_idx = 0;
    /* FP_BUFFER_SLAB_PAUSE_US: host-side sleep between slab launches, in
     * case idle windows let a throttled GPU recover. Off by default. */
    long slab_pause_us = 0;
    {
        const char *pause_env = getenv("FP_BUFFER_SLAB_PAUSE_US");
        if (pause_env) slab_pause_us = atol(pause_env);
    }
    /* The per-slab clFinish below also enables the slow-slab diagnostic
     * (each slab individually timed) -- skipping it saves a small amount
     * but silently disables that diagnostic. Off by default. */
    int skip_slab_finish = 0;
    {
        const char *skip_env = getenv("FP_BUFFER_SKIP_SLAB_FINISH");
        if (skip_env && atoi(skip_env) != 0) skip_slab_finish = 1;
    }
    /* gpu-buf's run-to-run variance on Hawaii (see README) is root-caused
     * to memory-clock DVFS, not thermal throttling or contention -- GPU
     * event profiling confirmed slow slabs are GPU-bound. Not driver-side
     * buffer-placement demotion, an earlier theory this superseded.
     *
     * FP_BUFFER_VOL_REALLOC_EVERY: mitigation attempt -- every N
     * angle-slabs, free and recreate d_vol from the same data. DOES NOT
     * RELIABLY BEAT BASELINE at real MLEM scale (looked like a win in a
     * short sweep, measured slower over full runs) -- off by default,
     * kept as an opt-in env var for further tuning. */
    int realloc_every = 0;
    {
        const char *re_env = getenv("FP_BUFFER_VOL_REALLOC_EVERY");
        if (re_env) realloc_every = atoi(re_env);
    }
    static long total_slab_launches = 0;
    size_t vol_bytes = (size_t)Nxz * Nxz * Ny * sizeof(float);
    float *vol_scratch = (realloc_every > 0) ? (float *)malloc(vol_bytes) : NULL;

    for (size_t p0 = 0; p0 < gws_full[2]; p0 += ANG_SLAB) {
        if (realloc_every > 0 && total_slab_launches > 0 &&
            total_slab_launches % realloc_every == 0) {
            double t_realloc0 = get_time_sec();
            err = clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0, vol_bytes, vol_scratch, 0, NULL, NULL);
            CL_CHECK(err, "fp_buffer vol readback (realloc mitigation)");
            clReleaseMemObject(d_vol);
            d_vol = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                    vol_bytes, vol_scratch, &err);
            CL_CHECK(err, "fp_buffer d_vol realloc (mitigation)");
            clSetKernelArg(k, 0, sizeof(cl_mem), &d_vol);
            *d_vol_ptr = d_vol;
            printf("    [fp_buffer] reallocated d_vol at total_slab_launches=%ld (%.3fs)\n",
                   total_slab_launches, get_time_sec() - t_realloc0);
        }

        size_t slab = (gws_full[2] - p0 < ANG_SLAB) ? (gws_full[2] - p0) : ANG_SLAB;
        size_t offset[3] = {0, 0, (size_t)ip_start + p0};
        size_t gws[3]    = {gws_full[0], gws_full[1], slab};

        if (skip_slab_finish) {
            /* batched path: enqueue only, no per-slab sync/timing/watchdog */
            cl_event evt;
            err = clEnqueueNDRangeKernel(cl->queue, k, 3, offset, gws, lws, 0, NULL, &evt);
            CL_CHECK(err, "fp_buffer enqueue (slab, batched)");
            clReleaseEvent(evt);
        } else {
            cl_event evt;
            double t0 = get_time_sec();
            err = clEnqueueNDRangeKernel(cl->queue, k, 3, offset, gws, lws, 0, NULL, &evt);
            CL_CHECK(err, "fp_buffer enqueue (slab)");
            err = clFinish(cl->queue);
            CL_CHECK(err, "fp_buffer slab finish");
            double dt = get_time_sec() - t0;
            if (dt > 2.0) {
                cl_ulong t_start = 0, t_end = 0;
                cl_int perr1 = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
                                                        sizeof(t_start), &t_start, NULL);
                cl_int perr2 = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
                                                        sizeof(t_end), &t_end, NULL);
                if (perr1 == CL_SUCCESS && perr2 == CL_SUCCESS) {
                    double gpu_dt = (double)(t_end - t_start) * 1e-9;
                    printf("    [fp_buffer slab %d, p0=%zu] wall=%.3fs gpu=%.3fs %s (SLOW)\n",
                           slab_idx, p0, dt, gpu_dt,
                           (dt - gpu_dt > 0.5) ? "HOST-SIDE-GAP" : "GPU-BOUND");
                } else {
                    printf("    [fp_buffer slab %d, p0=%zu] %.3f s (SLOW, profiling query failed)\n",
                           slab_idx, p0, dt);
                }
            }
            clReleaseEvent(evt);
        }
        slab_idx++;
        total_slab_launches++;
        if (slab_pause_us > 0) usleep((useconds_t)slab_pause_us);
    }
    if (skip_slab_finish) {
        err = clFinish(cl->queue);
        CL_CHECK(err, "fp_buffer batched finish");
    }

    if (vol_scratch) free(vol_scratch);
}

/*
 * ── Diagnostic: repeat one fp_buffer angle-slab N times ─────────────────
 * Isolates one fixed angle-slab and launches it N times back-to-back,
 * timing each with GPU-side event profiling, to distinguish thermal
 * throttling (monotone degradation) from TLB/page-residency effects
 * (bimodal) from channel camping (uniform) by the resulting shape.
 * angle_offset lets the same slab-index slot be tested with a different
 * angle range. realloc_at (0 = never): free and recreate d_vol after
 * this many repeats, to test whether degradation is tied to the
 * allocation itself vs external GPU/driver state.
 */
static void run_diag_repeat_slab(CLState *cl, const CBpara *p,
                                  const float *volume,
                                  int angle_offset, int slab_size, int n_repeats,
                                  int realloc_at)
{
    cl_int err;
    cl_kernel k = cl->k_fp_buf;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W = p->detector_width,   H  = p->detector_height;
    int np = p->num_projs;
    int n_samples = p->n_samples;
    float SOD=(float)p->SOD, SDD=(float)p->SDD;
    float vs=(float)p->voxelSize, px=(float)p->pixelSize;

    if (angle_offset < 0 || angle_offset + slab_size > np) {
        fprintf(stderr, "run_diag_repeat_slab: angle range [%d,%d) out of "
                        "bounds for np=%d\n", angle_offset, angle_offset+slab_size, np);
        return;
    }

    size_t vol_bytes  = (size_t)Nxz * Nxz * Ny * sizeof(float);
    size_t proj_bytes = (size_t)np * H * W * sizeof(float);
    cl_mem d_proj_scratch = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_proj_scratch (diag)");

    cl_mem d_vol = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                   vol_bytes, (void*)volume, &err);
    CL_CHECK(err, "d_vol (diag)");

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_vol);
    clSetKernelArg(k, 1, sizeof(cl_mem), &cl->d_R_mats);
    clSetKernelArg(k, 2, sizeof(cl_mem), &cl->d_T_vecs);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_proj_scratch);
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

    size_t lws[3] = {4, 64, 1};
    const char *lws_env = getenv("FP_BUFFER_LWS");
    if (lws_env) {
        unsigned long a=4, b=64, c=1;
        if (sscanf(lws_env, "%lu,%lu,%lu", &a, &b, &c) == 3) {
            lws[0]=a; lws[1]=b; lws[2]=c;
        }
    }
    size_t gws_full[3] = {(size_t)W, (size_t)H, (size_t)np};
    for (int d=0; d<2; d++)
        if (gws_full[d] % lws[d]) gws_full[d] += lws[d] - gws_full[d] % lws[d];

    printf("=== diag repeat-slab: angles [%d,%d), %d repeats, lws={%zu,%zu,%zu} ===\n",
           angle_offset, angle_offset + slab_size, n_repeats, lws[0], lws[1], lws[2]);

    size_t offset[3] = {0, 0, (size_t)angle_offset};
    size_t gws[3]     = {gws_full[0], gws_full[1], (size_t)slab_size};

    for (int rep = 0; rep < n_repeats; rep++) {
        if (realloc_at > 0 && rep == realloc_at) {
            clReleaseMemObject(d_vol);
            d_vol = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                    vol_bytes, (void*)volume, &err);
            CL_CHECK(err, "d_vol realloc (diag)");
            clSetKernelArg(k, 0, sizeof(cl_mem), &d_vol);
            printf("  -- reallocated d_vol before rep %d --\n", rep+1);
        }

        cl_event evt;
        double t0 = get_time_sec();
        err = clEnqueueNDRangeKernel(cl->queue, k, 3, offset, gws, lws, 0, NULL, &evt);
        CL_CHECK(err, "diag repeat-slab enqueue");
        err = clFinish(cl->queue);
        CL_CHECK(err, "diag repeat-slab finish");
        double dt = get_time_sec() - t0;

        cl_ulong t_start = 0, t_end = 0;
        double gpu_dt = -1.0;
        if (clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL) == CL_SUCCESS &&
            clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(t_end),   &t_end,   NULL) == CL_SUCCESS) {
            gpu_dt = (double)(t_end - t_start) * 1e-9;
        }
        printf("  rep %3d/%d  wall=%.4fs  gpu=%.4fs\n", rep+1, n_repeats, dt, gpu_dt);
        clReleaseEvent(evt);
    }

    clReleaseMemObject(d_vol);
    clReleaseMemObject(d_proj_scratch);
}

/* ── Internal: run backprojection (image mode) ──────────────────────────── */
/* OSEM: ip_start/ip_count select a contiguous angle
 * subrange for bp's internal angle loop. */
static void run_bp_image(CLState *cl, const CBpara *p,
                          cl_mem proj_img, cl_mem d_ang_cs, cl_mem d_vol,
                          int ip_start, int ip_count)
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
    clSetKernelArg(k,12, sizeof(int),    &ip_start);
    clSetKernelArg(k,13, sizeof(int),    &ip_count);

    size_t gws[3] = {(size_t)Nxz, (size_t)Nxz, (size_t)Ny};
    size_t lws[3] = {4, 4, 16};  /* see BP_LWS note in run_bp_buffer above */
    {
        const char *bp_lws_env = getenv("BP_LWS");
        if (bp_lws_env) {
            unsigned long a=4, b=4, c=16;
            if (sscanf(bp_lws_env, "%lu,%lu,%lu", &a, &b, &c) == 3) {
                lws[0]=a; lws[1]=b; lws[2]=c;
            }
        }
    }
    for (int d=0;d<3;d++) {
        if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
    }
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0,NULL,NULL);
    CL_CHECK(err, "bp_image enqueue");
}

/* ── Internal: run forward projection (image mode) ─────────────────────── */
/*
 * OSEM: ip_start/ip_count select a contiguous angle subrange. Launches
 * with a global work-offset of ip_start in dim 2 and a work-size of
 * ip_count rounded up to lws[2] -- the rounding can push gws[2] past
 * ip_count while still under num_projs, so the kernel guards
 * "ip >= ip_start + ip_count" (fp_image.cl), not just "ip >= num_projs".
 */
static void run_fp_image(CLState *cl, const CBpara *p,
                          cl_mem vol_img, cl_mem d_proj,
                          int ip_start, int ip_count)
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
    /* AABB ray clipping helps on large detectors but hurts at 256^3
     * (~4% regression): the kernel is small enough there that AABB's
     * setup cost outweighs the fetch reduction. Gate stays W>512;
     * FP_IMAGE_AABB overrides for testing. */
    int use_aabb = (W > 512) ? 1 : 0;
    const char *aabb_env = getenv("FP_IMAGE_AABB");
    if (aabb_env) use_aabb = atoi(aabb_env);
    clSetKernelArg(k,14, sizeof(int), &use_aabb);
    clSetKernelArg(k,15, sizeof(int), &ip_start);
    clSetKernelArg(k,16, sizeof(int), &ip_count);
    /* FP_TEX_EXACT=1: keep the 3D texture cache but replace the sampler's
     * fixed-function blend with an IEEE float32 one -- eight
     * CLK_FILTER_NEAREST fetches plus a manual trilinear (see
     * kernels/fp_image.cl's trilinear_tex).
     *
     * The forward projection is where the sampler's interpolation error
     * actually matters: fp produces the ratio driving every MLEM update, so
     * its error re-enters the loop each epoch, while bp's averages out over
     * 75 angles. Measured on GTX 680, 100 epochs, MSE vs CPU:
     *   256^3  1.128e-07 -> 2.524e-09 (45x)
     *            gpu-img 14.65s -> 21.34s, gpu-opt 14.20s -> 20.48s
     *   512^3  1.232e-09 -> 5.528e-10 (2.2x)
     * Off by default; unset leaves the hardware-filtered path untouched. */
    int tex_exact = 0;
    {
        const char *te = getenv("FP_TEX_EXACT");
        if (te) tex_exact = atoi(te) != 0;
    }
    clSetKernelArg(k,17, sizeof(int), &tex_exact);

    size_t gws[3] = {(size_t)W, (size_t)H, (size_t)ip_count};
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
    /* Re-swept on NVIDIA GTX 680 (32-wide warp) after the
     * Unlike fp_buffer, this shape did not need re-tuning after the
     * hardware switch -- {8,32,1} is still best on GTX 680 too; a sweep
     * of alternatives at 256^3 confirmed all worse. */
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
    size_t offset[3] = {0, 0, (size_t)ip_start};
    err = clEnqueueNDRangeKernel(cl->queue, k, 3, offset, gws, lws, 0,NULL,NULL);
    CL_CHECK(err, "fp_image enqueue");
}

/* Public entry for the repeat-slab variance diagnostic: builds R/T
 * buffers, repeats one fixed angle-slab n_repeats times. Diagnostic
 * only -- no epoch loop, no output. */
void gpu_diag_repeat_slab(CLState *cl, const CBpara *p, const float *volume,
                           int angle_offset, int slab_size, int n_repeats,
                           int realloc_at)
{
    build_RT_buffers(cl, p);

    run_diag_repeat_slab(cl, p, volume, angle_offset, slab_size, n_repeats, realloc_at);

    clReleaseMemObject(cl->d_R_mats);
    clReleaseMemObject(cl->d_T_vecs);
}

/* ── reconstruct_gpu ─────────────────────────────────────────────────────── */
void reconstruct_gpu(CLState *cl, const CBpara *p,
                     const float *proj_measured, float *volume,
                     int epochs, const char *conv_log)
{
    cl_int err;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W   = p->detector_width,  H  = p->detector_height;
    int np  = p->num_projs;
    /* Needed by the fused bp_image_update call below (float32
     * GPU_MODE_IMAGE), inlined into the epoch loop directly. */
    float SOD = (float)p->SOD, SDD = (float)p->SDD;
    float vs  = (float)p->voxelSize, px = (float)p->pixelSize;

    size_t vol_bytes  = (size_t)Nxz * Nxz * Ny * sizeof(float);
    size_t proj_bytes = (size_t)np  * H   * W  * sizeof(float);

    build_RT_buffers(cl, p);

    /* Build float2 cos/sin LUT for bp_buffer */
    float *ang_cs = (float *)malloc(np * 2 * sizeof(float));
    for (int i=0;i<np;i++) {
        ang_cs[2*i]   = (float)cos(p->angles[i]);
        ang_cs[2*i+1] = (float)sin(p->angles[i]);
    }

    /* ── Allocate device buffers ── */
    cl_mem d_proj_meas = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                         proj_bytes, (void*)proj_measured, &err);
    CL_CHECK(err, "d_proj_meas");

    cl_mem d_ang_cs = clCreateBuffer(cl->ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                      np*2*sizeof(float), ang_cs, &err);
    CL_CHECK(err, "d_ang_cs buf");

    cl_mem d_vol = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                   vol_bytes, volume, &err);
    CL_CHECK(err, "d_vol");

    /* proj_prep: [np*W*H] — preprocessed (flip+transpose+scale) layout for bp */
    cl_mem d_proj_b    = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_proj_b");
    /* d_ratio/d_ratio_prep are only used in GPU_MODE_BUFFER -- image
     * mode fuses divide+preprocess straight into ratio_img_buf. Gating
     * allocation on the actual mode avoids wasting ~759MiB VRAM in
     * gpu-img/gpu-opt at 512^3. */
    cl_mem d_ratio = NULL, d_ratio_prep = NULL;
    if (cl->mode == GPU_MODE_BUFFER) {
        d_ratio = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        CL_CHECK(err, "d_ratio");
        d_ratio_prep = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        CL_CHECK(err, "d_ratio_prep");
    }
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
            run_bp_buffer(cl, p, d_ones_prep, d_ang_cs, d_bp_ones, 0, np);
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
            run_bp_image(cl, p, ones_img, d_ang_cs, d_bp_ones, 0, np);
            clReleaseMemObject(ones_img);
        }
        clFinish(cl->queue);  /* wait for bp_ones before epoch loop */
        clReleaseMemObject(d_ones_raw);
        clReleaseMemObject(d_ones_prep);
        printf("  bp(ones) computed.\n");
    }

    /* ── Image objects for image mode ── */
    cl_mem vol_img        = NULL;
    cl_image_format img_fmt = {CL_R, CL_FLOAT};

    if (cl->mode == GPU_MODE_IMAGE) {
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
     * copy. In float32 mode d_vol is copied straight into vol_img below. */
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

    /* vol_img fusion: seed vol_img once with the initial volume
     * (float32 mode only; --half still uses its own per-epoch
     * float_to_half+copy path). After this, vol_update_img keeps vol_img
     * current at the end of every epoch, so the per-epoch
     * clEnqueueCopyBufferToImage this loop used to do at the START of
     * every epoch (~1.07GB/epoch at 512^3) is removed entirely below. */
    if (cl->mode == GPU_MODE_IMAGE && !p->use_half_vol) {
        size_t vorigin0[3]={0,0,0};
        size_t vregion0[3]={(size_t)Ny,(size_t)Nxz,(size_t)Nxz};
        err = clEnqueueCopyBufferToImage(cl->queue, d_vol, vol_img,
                                         0, vorigin0, vregion0, 0, NULL, NULL);
        CL_CHECK(err, "seed vol_img float32");
    }

    /* --log-convergence scratch: host-side readback buffers, only allocated
     * when logging is on (default NULL, zero extra cost otherwise). b_host
     * holds fp(v) read back right after fp; v_prev_host holds the volume
     * from the start of the epoch, for rel_change. */
    float *b_host      = conv_log ? (float *)malloc((size_t)proj_n * sizeof(float)) : NULL;
    float *v_prev_host = conv_log ? (float *)malloc((size_t)vol_n  * sizeof(float)) : NULL;
    float *v_cur_host  = conv_log ? (float *)malloc((size_t)vol_n  * sizeof(float)) : NULL;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t_ep = get_time_sec();

        if (conv_log)
            clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0,
                                 (size_t)vol_n * sizeof(float), v_prev_host, 0, NULL, NULL);

        /* forward project: b = F(v0). OSEM is implemented in
         * reconstruct_gpu_opt only; this path always uses the full
         * angle range (ip_start=0/ip_count=np), equivalent to
         * --subsets 1. */
        if (cl->mode == GPU_MODE_BUFFER) {
            run_fp_buffer(cl, p, &d_vol, d_proj_b, 0, np);
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
            }
            /* float32 mode: no copy needed here -- vol_img was seeded
             * before the loop and is kept current by vol_update_img at
             * the end of every epoch (Phase B4). --half still copies
             * every epoch (above), since float_to_half is a real format
             * conversion vol_update_img doesn't do. */
            run_fp_image(cl, p, vol_img, d_proj_b, 0, np);
        }

        if (conv_log)
            clEnqueueReadBuffer(cl->queue, d_proj_b, CL_TRUE, 0,
                                 (size_t)proj_n * sizeof(float), b_host, 0, NULL, NULL);

        /* bp_ratio = bp(ratio_prep). No zero-fill needed: bp kernels
         * unconditionally overwrite every voxel in range.
         *
         * Image mode fuses proj_divide with preprocess into one kernel
         * writing straight into ratio_img_buf, no intermediate buffer.
         * Buffer mode still needs proj_divide's plain-buffer output.
         *
         * float32 GPU_MODE_IMAGE takes a further-fused path below
         * (bp_image_update, kernels/bp_image.cl) that skips the separate
         * d_bp_ratio buffer and vol_update_img launch, applying the
         * update and vol_img write in one kernel invocation. --half and
         * GPU_MODE_BUFFER fall through to the un-fused branch below. */
        if (cl->mode == GPU_MODE_IMAGE && !p->use_half_vol) {
            run_divide_preprocess_img(cl, p, d_proj_meas, d_proj_b, ratio_img_buf, 0, np);

            cl_kernel k = cl->k_bp_img_update;
            clSetKernelArg(k, 0, sizeof(cl_mem), &ratio_img_buf);
            clSetKernelArg(k, 1, sizeof(cl_mem), &d_ang_cs);
            clSetKernelArg(k, 2, sizeof(cl_mem), &d_vol);
            clSetKernelArg(k, 3, sizeof(cl_mem), &d_bp_ones);
            clSetKernelArg(k, 4, sizeof(cl_mem), &vol_img);
            clSetKernelArg(k, 5, sizeof(int),    &Nxz);
            clSetKernelArg(k, 6, sizeof(int),    &Ny);
            clSetKernelArg(k, 7, sizeof(int),    &W);
            clSetKernelArg(k, 8, sizeof(int),    &H);
            clSetKernelArg(k, 9, sizeof(int),    &np);
            clSetKernelArg(k,10, sizeof(float),  &SOD);
            clSetKernelArg(k,11, sizeof(float),  &SDD);
            clSetKernelArg(k,12, sizeof(float),  &vs);
            clSetKernelArg(k,13, sizeof(float),  &px);
            int ip_start0 = 0;
            clSetKernelArg(k,14, sizeof(int),    &ip_start0);
            clSetKernelArg(k,15, sizeof(int),    &np);

            size_t gws[3] = {(size_t)Nxz, (size_t)Nxz, (size_t)Ny};
            size_t lws[3] = {4, 4, 16};  /* same shape as run_bp_image */
            {
                const char *bp_lws_env = getenv("BP_LWS");
                if (bp_lws_env) {
                    unsigned long a=4, b=4, c=16;
                    if (sscanf(bp_lws_env, "%lu,%lu,%lu", &a, &b, &c) == 3) {
                        lws[0]=a; lws[1]=b; lws[2]=c;
                    }
                }
            }
            for (int d=0;d<3;d++)
                if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
            err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0, NULL, NULL);
            CL_CHECK(err, "bp_image_update");
        } else {
            if (cl->mode == GPU_MODE_BUFFER) {
                cl_kernel k = cl->k_divide;
                clSetKernelArg(k,0,sizeof(cl_mem),&d_proj_meas);
                clSetKernelArg(k,1,sizeof(cl_mem),&d_proj_b);
                clSetKernelArg(k,2,sizeof(cl_mem),&d_ratio);
                clSetKernelArg(k,3,sizeof(int),   &proj_n);
                size_t gws=((size_t)proj_n + 3) / 4;
                err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
                CL_CHECK(err,"proj_divide");
                run_preprocess(cl, p, d_ratio, d_ratio_prep);  /* fused: cone_weight + flip + transpose */
                run_bp_buffer(cl, p, d_ratio_prep, d_ang_cs, d_bp_ratio, 0, np);
            } else {
                /* GPU_MODE_IMAGE && use_half_vol -- the only other case
                 * that reaches here (see the comment above). */
                run_divide_preprocess_img(cl, p, d_proj_meas, d_proj_b, ratio_img_buf, 0, np);
                run_bp_image(cl, p, ratio_img_buf, d_ang_cs, d_bp_ratio, 0, np);
            }

            /* v0 *= bp_ratio / bp_ones. GPU_MODE_IMAGE reaching this
             * point is always --half (plain float32 took the fused
             * branch above) -- --half keeps plain vol_update since its
             * vol_img refreshes via float_to_half+copy instead. */
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
        double t_ep_total = get_time_sec() - t_ep;
        printf("  epoch %3d/%d  %.3f s\n", epoch+1, epochs, t_ep_total);

        if (conv_log) {
            clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0,
                                 (size_t)vol_n * sizeof(float), v_cur_host, 0, NULL, NULL);
            log_convergence(conv_log, epoch, t_ep_total,
                             proj_measured, b_host, (size_t)proj_n,
                             v_cur_host, (epoch == 0) ? NULL : v_prev_host, (size_t)vol_n);
        }
    }

    /* Read back result */
    clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0, vol_bytes, volume, 0,NULL,NULL);

    if (b_host)      free(b_host);
    if (v_prev_host) free(v_prev_host);
    if (v_cur_host)  free(v_cur_host);

    /* Cleanup */
    clReleaseMemObject(d_proj_meas);
    clReleaseMemObject(d_vol);
    clReleaseMemObject(d_proj_b);
    if (d_ratio)      clReleaseMemObject(d_ratio);
    if (d_ratio_prep) clReleaseMemObject(d_ratio_prep);
    clReleaseMemObject(d_bp_ratio);
    clReleaseMemObject(d_bp_ones);
    if (vol_img)        clReleaseMemObject(vol_img);
    if (d_vol_half_img) clReleaseMemObject(d_vol_half_img);
    if (ratio_img_buf)  clReleaseMemObject(ratio_img_buf);
    clReleaseMemObject(d_ang_cs);
    clReleaseMemObject(cl->d_R_mats);
    clReleaseMemObject(cl->d_T_vecs);
    free(ang_cs);
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
                         int epochs, const char *conv_log, int subsets)
{
    cl_int err;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W   = p->detector_width,  H  = p->detector_height;
    int np  = p->num_projs;
    int S   = (subsets > 0) ? subsets : 1;
    if (S > np) S = 1; /* degenerate guard: more subsets than angles makes no sense */

    /* OSEM: subset k is the contiguous angle range [subset_start[k],
     * subset_start[k]+subset_count[k]) -- valid only because the caller
     * has already permuted p->angles/proj_measured (see utils.h). */
    int *subset_start = (int *)malloc((size_t)S * sizeof(int));
    int *subset_count  = (int *)malloc((size_t)S * sizeof(int));
    {
        int pos = 0;
        for (int s = 0; s < S; s++) {
            int count = np / S + (s < np % S ? 1 : 0);
            subset_start[s] = pos;
            subset_count[s] = count;
            pos += count;
        }
    }

    size_t vol_bytes  = (size_t)Nxz * Nxz * Ny * sizeof(float);
    size_t proj_bytes = (size_t)np  * H   * W  * sizeof(float);

    /* Build float2 cos/sin LUT for fp_image */
    float *ang_cs = (float *)malloc(np * 2 * sizeof(float));
    for (int i = 0; i < np; i++) {
        ang_cs[2*i]   = (float)cos(p->angles[i]);
        ang_cs[2*i+1] = (float)sin(p->angles[i]);
    }

    /* Device buffers */
    cl_mem d_ang_cs = clCreateBuffer(cl->ctx,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, np*2*sizeof(float), ang_cs, &err);
    CL_CHECK(err, "d_ang_cs");

    cl_mem d_proj_meas = clCreateBuffer(cl->ctx,
        CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, proj_bytes, (void*)proj_measured, &err);
    CL_CHECK(err, "d_proj_meas opt");

    cl_mem d_vol = clCreateBuffer(cl->ctx,
        CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, vol_bytes, volume, &err);
    CL_CHECK(err, "d_vol opt");

    cl_mem d_proj_b    = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
    CL_CHECK(err, "d_proj_b opt");
    /* Kernel fusion: d_ratio and d_ratio_prep both removed --
     * run_divide_preprocess_img now fuses divide+preprocess and writes
     * straight into ratio_img, no intermediate buffers needed. */
    cl_mem d_bp_ratio  = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, vol_bytes, NULL, &err);
    CL_CHECK(err, "d_bp_ratio opt");
    /* OSEM: one normalizer bp(cone_weight(ones)) per
     * subset -- each subset's sensitivity map differs since it only sums
     * its own angle range. S=1 (default) allocates exactly one buffer,
     * identical to the pre-OSEM single-normalizer behavior. 256^3 only
     * for now (S=1..25 costs 32-800MB total there); the plan's VRAM
     * analysis found this infeasible at 512^3 without fp16 normalizers,
     * not implemented here -- see the plan's C1 section. */
    cl_mem *d_bp_ones = (cl_mem *)malloc((size_t)S * sizeof(cl_mem));
    for (int s = 0; s < S; s++) {
        d_bp_ones[s] = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, vol_bytes, NULL, &err);
        CL_CHECK(err, "d_bp_ones[s] opt");
    }

    /* local mem for angle_cs LUT cache in bp_opt */
    size_t lmem_bytes = (size_t)np * sizeof(cl_float2);

    /* image format for proj image array */
    cl_image_format img_fmt = {CL_R, CL_FLOAT};

    build_RT_buffers(cl, p);

    /* d_proj_meas stays raw — cone weight applied to ratio after divide, matching Python */

    /* ── bp_ones: preprocess all-ones → image, run bp_opt once per subset ──
     * OSEM: total precompute cost across all S subsets equals
     * one full-angle bp_ones (each subset only sums its own 1/S of the
     * angles) -- the ones_img/preprocess step is subset-independent and
     * done once; only the bp_opt call and its ip_start/ip_count vary. */
    {
        cl_mem d_ones_raw  = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        CL_CHECK(err,"d_ones_raw opt");
        cl_mem d_ones_prep = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, proj_bytes, NULL, &err);
        CL_CHECK(err,"d_ones_prep opt");
        float one=1.f;
        clEnqueueFillBuffer(cl->queue, d_ones_raw, &one, sizeof(float), 0, proj_bytes, 0, NULL, NULL);
        run_preprocess(cl, p, d_ones_raw, d_ones_prep);  /* fused: cone_weight + flip + transpose */
        clReleaseMemObject(d_ones_raw);

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

        for (int s = 0; s < S; s++) {
            float zero=0.f;
            clEnqueueFillBuffer(cl->queue,d_bp_ones[s],&zero,sizeof(float),0,vol_bytes,0,NULL,NULL);

            cl_kernel k = cl->k_bp_opt;
            float SOD=(float)p->SOD,SDD=(float)p->SDD;
            float vs=(float)p->voxelSize,px=(float)p->pixelSize;
            clSetKernelArg(k,0,sizeof(cl_mem),&ones_img);
            clSetKernelArg(k,1,sizeof(cl_mem),&d_ang_cs);
            clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ones[s]);
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
            clSetKernelArg(k,13,sizeof(int),&subset_start[s]);
            clSetKernelArg(k,14,sizeof(int),&subset_count[s]);
            size_t gws[3]={(size_t)Nxz,(size_t)Nxz,(size_t)Ny};
            size_t lws[3]={4,4,16};  /* see BP_LWS note in run_bp_buffer above */
            {
                const char *bp_lws_env = getenv("BP_LWS");
                if (bp_lws_env) {
                    unsigned long a=4, b=4, c=16;
                    if (sscanf(bp_lws_env, "%lu,%lu,%lu", &a, &b, &c) == 3) {
                        lws[0]=a; lws[1]=b; lws[2]=c;
                    }
                }
            }
            for(int d=0;d<3;d++) if(gws[d]%lws[d]) gws[d]+=lws[d]-gws[d]%lws[d];
            err=clEnqueueNDRangeKernel(cl->queue,k,3,NULL,gws,lws,0,NULL,NULL);
            CL_CHECK(err,"bp_opt ones");
        }
        clFinish(cl->queue);  /* wait for all subsets' bp_ones before epoch loop */
        clReleaseMemObject(ones_img);
        printf("  bp_opt(ones) computed (%d subset%s).\n", S, S==1?"":"s");
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

    /* vol_img fusion: seed vol_img once with the initial volume
     * (float32 mode only). After this, vol_update_img keeps vol_img
     * current at the end of every epoch, so the per-epoch
     * clEnqueueCopyBufferToImage this loop used to do at the START of
     * every epoch (~1.07GB/epoch at 512^3) is removed entirely below. */
    if (!p->use_half_vol) {
        size_t vorigin0[3]={0,0,0};
        size_t vregion0[3]={(size_t)Ny,(size_t)Nxz,(size_t)Nxz};
        err = clEnqueueCopyBufferToImage(cl->queue, d_vol, vol_img,
                                         0, vorigin0, vregion0, 0, NULL, NULL);
        CL_CHECK(err, "seed vol_img float32 opt");
    }

    /* --log-convergence scratch: only allocated when logging is on (default
     * NULL, zero extra cost otherwise). See reconstruct_gpu for the pattern. */
    float *b_host      = conv_log ? (float *)malloc((size_t)proj_n * sizeof(float)) : NULL;
    float *v_prev_host = conv_log ? (float *)malloc((size_t)vol_n  * sizeof(float)) : NULL;
    float *v_cur_host  = conv_log ? (float *)malloc((size_t)vol_n  * sizeof(float)) : NULL;

    /*
     * OSEM: one epoch = one full pass over all S
     * subsets. S=1 makes the inner loop run exactly once with
     * ip_start=0/ip_count=np, i.e. byte-identical to the pre-OSEM
     * MLEM loop -- the only structural difference from before is this
     * added (degenerate, S=1) inner loop.
     *
     * fp is now restricted to the subset's ip_start/ip_count (used to
     * cover the full angle range every sub-iteration -- an accepted
     * S-fold cost). Bit-identical to the full-range version: d_proj_b
     * slices outside the subset now hold stale values from a previous
     * sub-iteration instead of a fresh fp(v_current), but those slices
     * are only ever consumed by run_divide_preprocess_img below (which
     * still runs full-range and writes all of ratio_img), and bp_opt
     * is launched with this same ip_start/ip_count -- it never reads
     * ratio_img outside the subset. The stale fp values are computed,
     * written into ratio_img, and discarded; the update is unchanged.
     */
    for (int epoch = 0; epoch < epochs; epoch++) {
        double t_ep = get_time_sec();

        for (int s = 0; s < S; s++) {
            int ip_start = subset_start[s];
            int ip_count = subset_count[s];

            if (conv_log && s == 0)
                clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0,
                                     (size_t)vol_n * sizeof(float), v_prev_host, 0, NULL, NULL);

            if (p->use_half_vol) {
                /* ── float d_vol → vol_img (GPU-side, no PCIe) ── */
                size_t origin[3]={0,0,0};
                size_t region[3]={(size_t)Ny,(size_t)Nxz,(size_t)Nxz};
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
            }
            /* float32 mode: no copy needed here -- vol_img was seeded before
             * the loop and is kept current by vol_update_img at the end of
             * every sub-iteration (Phase B4). fp now restricted to this
             * subset -- see function-level comment above for why this is
             * bit-identical to the previous full-range call. */
            run_fp_image(cl, p, vol_img, d_proj_b, ip_start, ip_count);

            if (conv_log && s == 0)
                clEnqueueReadBuffer(cl->queue, d_proj_b, CL_TRUE, 0,
                                     (size_t)proj_n * sizeof(float), b_host, 0, NULL, NULL);

            /* Kernel fusion: fuses ratio=p0/b with cone_weight+flip+
             * transpose+write-to-image into one kernel -- no d_ratio
             * intermediate buffer, no buffer->image copy (was ~0.80GB/epoch
             * at 512^3 plus two kernel launches). Same math, same values.
             *
             * OSEM step 2 (see the function-level comment above for step
             * 1): now restricted to this subset's ip_start/ip_count, the
             * other half of the S-fold saving -- d_proj_b outside the
             * subset already only holds stale fp() values (step 1), so
             * computing ratio/cone-weight/write-to-image for them was
             * pure waste. ratio_img's out-of-subset slices are simply
             * never written this sub-iteration instead of being
             * (re)written with a value nothing reads: ratio_img is read
             * only by bp_opt below, which is itself launched with this
             * same ip_start/ip_count and never touches other slices. */
            run_divide_preprocess_img(cl, p, d_proj_meas, d_proj_b, ratio_img, ip_start, ip_count);

            /* ── bp_opt(ratio_img), this subset's angle range only ── */
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
                clSetKernelArg(k,13,sizeof(int),&ip_start);
                clSetKernelArg(k,14,sizeof(int),&ip_count);
                size_t gws[3]={(size_t)Nxz,(size_t)Nxz,(size_t)Ny};
                size_t lws[3]={4,4,16};  /* see BP_LWS note in run_bp_buffer above */
                {
                    const char *bp_lws_env = getenv("BP_LWS");
                    if (bp_lws_env) {
                        unsigned long a=4, b=4, c=16;
                        if (sscanf(bp_lws_env, "%lu,%lu,%lu", &a, &b, &c) == 3) {
                            lws[0]=a; lws[1]=b; lws[2]=c;
                        }
                    }
                }
                for(int d=0;d<3;d++) if(gws[d]%lws[d]) gws[d]+=lws[d]-gws[d]%lws[d];
                err=clEnqueueNDRangeKernel(cl->queue,k,3,NULL,gws,lws,0,NULL,NULL);
                CL_CHECK(err,"bp_opt ratio");
            }

            /* ── update, this subset's normalizer ──
             * vol_img fusion: float32 mode uses vol_update_img, which
             * also writes straight into vol_img (replacing the copy that
             * used to run at the top of the NEXT sub-iteration). --half
             * keeps plain vol_update since its vol_img is refreshed via
             * float_to_half+copy above instead. */
            if (!p->use_half_vol) {
                cl_kernel k = cl->k_update_img;
                clSetKernelArg(k,0,sizeof(cl_mem),&d_vol);
                clSetKernelArg(k,1,sizeof(cl_mem),&d_bp_ratio);
                clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ones[s]);
                clSetKernelArg(k,3,sizeof(cl_mem),&vol_img);
                clSetKernelArg(k,4,sizeof(int),&Nxz);
                clSetKernelArg(k,5,sizeof(int),&Ny);
                clSetKernelArg(k,6,sizeof(int),&vol_n);
                size_t gws=((size_t)vol_n + 3) / 4;
                err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
                CL_CHECK(err,"update_img opt");
            } else {
                cl_kernel k = cl->k_update;
                clSetKernelArg(k,0,sizeof(cl_mem),&d_vol);
                clSetKernelArg(k,1,sizeof(cl_mem),&d_bp_ratio);
                clSetKernelArg(k,2,sizeof(cl_mem),&d_bp_ones[s]);
                clSetKernelArg(k,3,sizeof(int),&vol_n);
                size_t gws=((size_t)vol_n + 3) / 4;
                err=clEnqueueNDRangeKernel(cl->queue,k,1,NULL,&gws,NULL,0,NULL,NULL);
                CL_CHECK(err,"update opt");
            }
        }

        clFinish(cl->queue);
        double t_ep_total = get_time_sec() - t_ep;
        printf("  epoch %3d/%d  %.3f s%s\n", epoch+1, epochs, t_ep_total,
               (S>1) ? "  (S subsets)" : "");

        if (conv_log) {
            clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0,
                                 (size_t)vol_n * sizeof(float), v_cur_host, 0, NULL, NULL);
            log_convergence(conv_log, epoch, t_ep_total,
                             proj_measured, b_host, (size_t)proj_n,
                             v_cur_host, (epoch == 0) ? NULL : v_prev_host, (size_t)vol_n);
        }
    }
    clReleaseMemObject(vol_img);
    if (d_vol_half) clReleaseMemObject(d_vol_half);
    clReleaseMemObject(ratio_img);

    clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0, vol_bytes, volume, 0,NULL,NULL);

    if (b_host)      free(b_host);
    if (v_prev_host) free(v_prev_host);
    if (v_cur_host)  free(v_cur_host);

    clReleaseMemObject(d_ang_cs);
    clReleaseMemObject(d_proj_meas);
    clReleaseMemObject(d_vol);
    clReleaseMemObject(d_proj_b);
    clReleaseMemObject(d_bp_ratio);
    for (int s = 0; s < S; s++) clReleaseMemObject(d_bp_ones[s]);
    free(d_bp_ones);
    free(subset_start);
    free(subset_count);
    clReleaseMemObject(cl->d_R_mats);
    clReleaseMemObject(cl->d_T_vecs);
    free(ang_cs);
}
