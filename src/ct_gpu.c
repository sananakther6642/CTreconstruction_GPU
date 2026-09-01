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

/* ── Phase 0b diagnostic: per-kernel resource usage ─────────────────────
 * On Kepler (GTX 680, SM 3.0) full occupancy needs <=32 registers/thread
 * (65536 regs / 2048 max threads per SMX); 63 regs/thread (the
 * architectural max) caps you at ~50% occupancy. The OpenCL API exposes
 * no direct register count, but CL_KERNEL_PRIVATE_MEM_SIZE (spilled
 * registers, i.e. local/private memory beyond the register file) and
 * CL_KERNEL_WORK_GROUP_SIZE (the max work-group size the compiler will
 * allow given its actual register allocation, which drops below the
 * device max once registers/thread * threads/group exceeds 65536) are
 * the two numbers that let us infer register pressure without vendor
 * extensions. A CL_KERNEL_WORK_GROUP_SIZE lower than the device's
 * CL_DEVICE_MAX_WORK_GROUP_SIZE, or nonzero private mem, both indicate
 * register pressure worth addressing (see the perf-algorithmic plan's
 * Phase 1/G4 items). Printed once at init behind CT_DIAG_KERNEL_INFO=1
 * since it adds several small blocking clGetKernelWorkGroupInfo calls. */
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

/* ── gpu-buf DVFS diagnosis: sample GPU clock state via sysfs ────────────
 * Tests the mclk-DVFS hypothesis for the gpu-buf slab slowdown (see the
 * README "gpu-buf run-to-run variance" section and the root-cause comment
 * on run_fp_buffer below): fp_buffer is bandwidth-bound, and Hawaii's
 * memory clock has an 8.3x DPM range (1250->150 MHz), unlike the ~3.2x
 * core-clock range the earlier "not thermal" argument used. If slow slabs
 * consistently sample a low mclk DPM state and fast slabs sample the top
 * state, that confirms DVFS rather than the previously-assumed d_vol
 * buffer-placement demotion.
 *
 * Reads /sys/class/drm/cardN/device/pp_dpm_sclk, pp_dpm_mclk, and
 * hwmon/hwmonM/temp1_input -- all readable without root on the AMD driver.
 * Linux-only
 * (no equivalent on the macOS dev machine); the caller must be behind
 * FP_BUFFER_CLOCK_PROBE=1 so this costs nothing by default. card_idx picks
 * which /sys/class/drm/card<N> to read (default 0; override via
 * FP_BUFFER_CLOCK_PROBE_CARD for multi-GPU machines).
 *
 * Output is deliberately compact -- one line, appended to the existing
 * slow-slab printf -- e.g. "sclk=947M* mclk=150M* temp=71.2C" where '*'
 * marks the DPM table's currently-active entry. */
#ifdef __linux__
static void sample_gpu_clocks(char *out, size_t n)
{
    out[0] = '\0';
    int card_idx = 0;
    {
        const char *card_env = getenv("FP_BUFFER_CLOCK_PROBE_CARD");
        if (card_env) card_idx = atoi(card_env);
    }

    char active_sclk[32] = "?", active_mclk[32] = "?";
    char path[256];

    snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/pp_dpm_sclk", card_idx);
    FILE *f = fopen(path, "r");
    if (f) {
        char line[128];
        while (fgets(line, sizeof(line), f)) {
            if (strchr(line, '*')) {
                char freq[32] = "?";
                sscanf(line, "%*d: %31s", freq);
                snprintf(active_sclk, sizeof(active_sclk), "%s", freq);
                break;
            }
        }
        fclose(f);
    }

    snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/pp_dpm_mclk", card_idx);
    f = fopen(path, "r");
    if (f) {
        char line[128];
        while (fgets(line, sizeof(line), f)) {
            if (strchr(line, '*')) {
                char freq[32] = "?";
                sscanf(line, "%*d: %31s", freq);
                snprintf(active_mclk, sizeof(active_mclk), "%s", freq);
                break;
            }
        }
        fclose(f);
    }

    double temp_c = -1.0;
    for (int hw = 0; hw < 4 && temp_c < 0.0; hw++) {
        snprintf(path, sizeof(path),
                 "/sys/class/drm/card%d/device/hwmon/hwmon%d/temp1_input", card_idx, hw);
        f = fopen(path, "r");
        if (f) {
            long milli_c = 0;
            if (fscanf(f, "%ld", &milli_c) == 1) temp_c = milli_c / 1000.0;
            fclose(f);
        }
    }

    if (temp_c >= 0.0)
        snprintf(out, n, "sclk=%s mclk=%s temp=%.1fC", active_sclk, active_mclk, temp_c);
    else
        snprintf(out, n, "sclk=%s mclk=%s temp=?", active_sclk, active_mclk);
}
#endif

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

    /* Check cl_khr_3d_image_writes -- gates the float32 vol_img fusion path
     * (vol_update writing directly into vol_img instead of a separate
     * clEnqueueCopyBufferToImage each epoch). OpenCL 1.2 has no 3D
     * read-write images without this extension. Printed once at init,
     * not gated behind an env var since it costs nothing. */
    /* Also check cl_khr_fp16 -- required for --half mode's
     * float_to_half kernel (kernels/fp_image.cl). Confirmed present on
     * the original AMD Hawaii target but ABSENT on this project's other
     * target, an NVIDIA GTX 680 -- without this check, --half
     * mode's kernel source fails to compile there and takes down the
     * whole image-mode program build, not just --half. Remembered on
     * CLState so main.c can refuse --half cleanly instead of crashing,
     * and passed as -DHAVE_FP16 to the image-mode program build below. */
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
        /* G3 fusion (bp_image + vol_update_img) -- kernel lives in the
         * same source (bp_image.cl) so it's created here regardless of
         * IMAGE vs OPT, but only ever launched from the plain gpu-img
         * path (reconstruct_gpu). Cheap to create unconditionally,
         * consistent with how the other kernels in this block are
         * handled. */
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
 * Kernel fusion: fuses proj_divide (ratio=p0/b) with cone_weight +
 * flip + transpose, writing straight into an image2d_array_t (ratio_img)
 * instead of a buffer. Replaces what was three steps (proj_divide into a
 * d_ratio buffer, preprocess into a d_ratio_prep buffer,
 * clEnqueueCopyBufferToImage into the image) with one kernel launch.
 * d_ratio/d_ratio_prep were both write-once/read-once and never used
 * elsewhere. An intermediate B1-only version that fused just the
 * image-write step (without the divide) was tried first and superseded
 * by this fully-fused version once B2 landed -- removed rather than kept
 * as dead code. See divide_preprocess_img in kernels/bp_buffer.cl for the
 * kernel and its math-equivalence note against the original two kernels.
 */
/* ip_start/ip_count: OSEM support, mirrors run_fp_image's pattern (see
 * that function's comment). np is still needed for gws sizing at S=1
 * (ip_count==np then), but the kernel itself only sees ip_start/ip_count. */
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
/* OSEM: ip_start/ip_count select a contiguous angle
 * subrange for bp's internal angle loop (kernel change only -- this
 * function's own z-slab chunking is unrelated, it's over voxels, not
 * angles, so no host-side offset-launch changes needed here). */
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

    /* Axis reassignment (see the kernel's own comment at bp_buffer.cl):
     * iz is now dim 0 (fastest local id), ix is dim 2 (slowest). {4,4,16}
     * was swept and won repeatedly, but every candidate in that sweep
     * kept ix as dim 0 -- the sweep never tested *which axis* is fastest,
     * only work-group *shape*. Kept the same shape {4,4,16} here, just
     * reinterpreted: lws[0] (was ix's 4) is now iz's chunk width, lws[2]
     * (was iz's 16) is now ix's. Re-sweep BP_LWS after this change lands
     * -- the old sweep's conclusion doesn't transfer across an axis swap.
     * Keep lws[0] a divisor of Z_SLAB=64 below (1,2,4,8,16,32,64) — the
     * z-slab chunking requires it, same constraint as before just moved
     * to dim 0. */
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

    /* Chunk the z-dimension (now dim 0) into slabs so no single kernel
     * launch runs long enough to trip the GPU watchdog (bp_buffer does 4
     * uncoalesced global loads/angle/voxel with no texture cache — at
     * 512^3 one unchunked launch can exceed the ~10s driver timeout and
     * reset the GPU). */
    const size_t Z_SLAB = 64;

    /* Mirrors FP_BUFFER_SKIP_SLAB_FINISH (run_fp_buffer, above): the
     * per-slab clFinish here is also what makes the SLOW-slab diagnostic
     * possible (each slab individually timed). Skipping it (enqueue all
     * slabs, sync once at the end) trades that diagnostic for fewer
     * pipeline drains -- at 256^3, Z_SLAB=64 means only 4 slabs/call, so
     * the expected win is small (a few percent at most). Off by default
     * so the diagnostic stays on unless explicitly opted out of. */
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
            /* flag any slab that takes much longer than a healthy slab should —
             * pinpoints WHICH slab stalls within a slow epoch, since epoch
             * totals alone can't distinguish "one bad slab" from "uniformly
             * slower this epoch" */
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
    /* Was hardcoded "if (W > 512)" inside fp_buffer.cl. Measured on
     * kale at 512^3 (W=1120, so the old gate was ON): AABB is ~6%
     * SLOWER on gpu-buf (5.10-5.11s/epoch off vs 5.40-5.41s/epoch on,
     * 3-epoch runs). Unlike fp_image/fp_cpu (where AABB is a real win --
     * CPU measured ~26% faster / ~32% on fp alone at the same
     * resolution), gpu-buf's cost is dominated by uncoalesced memory
     * access, which AABB's sample-range narrowing does nothing for; its
     * 6-divide setup cost is pure overhead here. Default now OFF
     * unconditionally (not W>512 -- that threshold was inherited from
     * fp_image without being re-verified for this kernel, and turned
     * out wrong for it). FP_IMAGE_AABB override still works, for
     * further testing. */
    int use_aabb = 0;
    {
        const char *aabb_env = getenv("FP_IMAGE_AABB");
        if (aabb_env) use_aabb = atoi(aabb_env);
    }
    clSetKernelArg(k,14, sizeof(int),    &use_aabb);
    clSetKernelArg(k,15, sizeof(int),    &ip_start);
    clSetKernelArg(k,16, sizeof(int),    &ip_count);

    /* Hardware target switched from AMD Hawaii PRO,
     * GCN 1.1, 64-wide wavefront) to NVIDIA GTX 680 (Kepler,
     * 32-wide warp) -- work-group tuning does not transfer between them,
     * confirmed by direct sweep rather than assumed.
     *
     * AMD Hawaii PRO history: swept 16,16,1 / 8,32,1 / 4,64,1 /
     * 32,8,1 / 8,16,1 at 512^3 (10-epoch confirmation) -- 4,64,1 gave
     * 75.37s/10ep vs 321.19s/10ep for the old 16,16,1 default, >4x
     * faster. Wide-short shapes (32,8,1 / 8,16,1) were catastrophic
     * (~4-8x slower) on that GPU.
     *
     * GTX 680 re-sweep: 4,64,1 (the Hawaii winner) was NOT best
     * here -- 10.53s/10ep @ 256^3. Swept 2,32,1 / 4,32,1 / 2,16,2 /
     * 8,32,1 / 1,32,1 / 2,16,1 / 4,16,2 / 1,16,2 / 2,8,2: 2,16,2 won
     * (9.77s, ~7.2% faster than 4,64,1), with 2,32,1 close behind
     * (9.83s). 8,32,1 was worst (12.88s, ~30% slower) -- the same
     * wide-short-shape penalty as on Hawaii, but the specific optimum
     * shifted with the warp width, confirming this needs re-tuning per
     * GPU rather than reusing one card's numbers. FP_BUFFER_LWS=X,Y,Z
     * still overrides for further testing on other hardware -- keep Z a
     * divisor of ANG_SLAB=8, the angle-slab chunking below requires it. */
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

    /* Chunk over angles (dim 2, lws=1) — same watchdog-avoidance rationale
     * as run_bp_buffer; fp_buffer's trilinear_buf is also uncoalesced.
     * p0 ranges over the SUBSET's local offsets [0, gws_full[2]); the
     * actual launch offset below is ip_start+p0 so the kernel's own
     * get_global_id(2) lands on the correct absolute angle index.
     *
     * FP_BUFFER_ANG_SLAB: opt-in override, tested as a second mclk-DVFS
     * mitigation angle (see FP_BUFFER_SKIP_SLAB_FINISH below) on the
     * theory that fewer, larger slabs means fewer sync points and fewer
     * idle gaps for the GPU to settle into a low mclk state between.
     *
     * TESTED AND REJECTED -- made things worse, not better. ANG_SLAB=16
     * (vs default 8) with FP_BUFFER_SKIP_SLAB_FINISH=1 already on,
     * 30-epoch 512^3 run: 694.79s total (23.16s/epoch mean), max epoch
     * 63.9s -- vs SKIP_SLAB_FINISH alone's 618.00s (20.60s/epoch mean),
     * max epoch 36.1s. ~12% SLOWER on average and nearly 2x the peak
     * epoch cost. No watchdog reset occurred (the risk flagged below did
     * not materialize at 16), but the result doesn't help: a larger slab
     * means a burst's cost compounds across more angles before any
     * chance of mid-slab recovery, which outweighs having fewer sync
     * points. Default (8) is the right choice -- FP_BUFFER_SKIP_SLAB_FINISH
     * alone remains the recommended mitigation, this knob does not add to
     * it. See docs/performance-history.md's DVFS section for the full
     * comparison.
     *
     * CAUTION (why this was risky to test, still applies to any further
     * testing at higher values): raising ANG_SLAB trades away watchdog
     * headroom. At 512^3 under the degraded (150MHz) DVFS state, one
     * already-observed slab cost ~11.3s wall (see the DVFS root-cause
     * comment below) against a driver timeout of ~10s (the same limit
     * run_bp_buffer's chunking comment cites) -- ANG_SLAB=8 is already
     * close to that ceiling WHILE degraded. Raising it multiplies a
     * single launch's worst-case wall time by the same factor, precisely
     * during the degraded state this is meant to help with -- a burst
     * hitting a larger slab risks a real watchdog reset (GPU-wide, not
     * just a slow epoch), not just a slow one. Treat any value beyond 16
     * as unverified until run long enough to hit a slow-tier slab and
     * confirm no reset occurs. */
    size_t ANG_SLAB = 8;
    {
        const char *ang_slab_env = getenv("FP_BUFFER_ANG_SLAB");
        if (ang_slab_env) {
            long v = atol(ang_slab_env);
            if (v > 0) ANG_SLAB = (size_t)v;
        }
    }
    if (ANG_SLAB != 8) {
        fprintf(stderr, "    [fp_buffer] WARNING: FP_BUFFER_ANG_SLAB=%zu overrides the "
                        "default (8) -- untested against the ~10s driver watchdog under "
                        "the degraded (150MHz) DVFS state; see this variable's comment.\n",
                ANG_SLAB);
    }
    int slab_idx = 0;
    /* FP_BUFFER_SLAB_PAUSE_US: experimental mitigation attempt, from before
     * the mechanism below was understood -- inserts a host-side sleep
     * between slab launches. Superseded by FP_BUFFER_SKIP_SLAB_FINISH
     * (below), which is the mitigation that actually measured a real win.
     * Left in, off by default (0), for further testing. */
    long slab_pause_us = 0;
    {
        const char *pause_env = getenv("FP_BUFFER_SLAB_PAUSE_US");
        if (pause_env) slab_pause_us = atol(pause_env);
    }
    /* FP_BUFFER_SKIP_SLAB_FINISH: RECOMMENDED for real runs (see the DVFS
     * root-cause comment below) -- batches all slab enqueues with one
     * clFinish at the end instead of syncing after every slab. Measured
     * ~29% faster on average on AMD Hawaii PRO (20.60s/epoch vs
     * 29.14s/epoch unmitigated, 30-epoch runs) and reduced both the
     * frequency and peak severity of the mclk-DVFS bursts described
     * below, though it does not eliminate them -- some slabs still
     * degrade even fully packed. Off by default only because it disables
     * the per-slab SLOW/HOST-SIDE-GAP diagnostic print (each slab's own
     * clFinish is what makes that timing possible) -- opt in for
     * production runs, leave off only when actively re-diagnosing this
     * variance. */
    int skip_slab_finish = 0;
    {
        const char *skip_env = getenv("FP_BUFFER_SKIP_SLAB_FINISH");
        if (skip_env && atoi(skip_env) != 0) skip_slab_finish = 1;
    }
    /* ── gpu-buf variance root cause (re-diagnosed 2026-09-01) ──────────
     * See README "gpu-buf run-to-run variance" for the full history. This
     * replaces an earlier theory (kept below, superseded) that pinned it
     * on driver-side memory-placement demotion of the 537MB d_vol buffer.
     *
     * CONFIRMED MECHANISM: Hawaii's memory-clock DVFS periodically drops
     * mclk from 1500 MHz to 150 MHz (10x) during sustained fp_buffer
     * execution. Direct evidence via FP_BUFFER_CLOCK_PROBE=1
     * (sample_gpu_clocks, above): every slow slab across a 30-epoch,
     * ~35-event run sampled mclk=150MHz; every fast baseline sampled
     * mclk=1500MHz. Zero overlap. sclk and temperature both held flat
     * throughout (1010MHz core clock never moved, 49-58C) -- rules out
     * core-clock throttling and thermal throttling as the trigger; only
     * mclk moves. gpu-img/gpu-opt stay flat in the same sessions gpu-buf
     * degrades, consistent with fp_buffer's uncoalesced buffer access
     * being bandwidth-bound (hit hard by a 10x memory-bandwidth cut)
     * while gpu-img/gpu-opt read through the texture cache (largely
     * insensitive to mclk state). A --diag repeat-slab run (60 fixed-slab
     * repeats, at the corrected lws={2,16,2} -- see that function's lws
     * comment -- with NO reallocation anywhere in the run) showed the
     * SAME slab oscillate between the fast and slow cost tier repeatedly
     * and spontaneously; this is what falsifies the earlier "holds until
     * d_vol is freed and recreated" theory below.
     *
     * power_dpm_force_performance_level (the sysfs knob to pin the GPU at
     * its top DPM state) exists but is not writable without root on this
     * machine (confirmed: permission denied), so the mechanism can't be
     * force-disabled at the driver level. FP_BUFFER_SKIP_SLAB_FINISH
     * (above) is the mitigation that actually helps, by closing the idle
     * gap between slab launches that may otherwise let the GPU settle
     * into the low-power state -- real but partial (~29% faster mean,
     * bursts reduced not eliminated), which also indicates the trigger is
     * only partly gap-related.
     *
     * ── superseded theory (kept for the record) ──────────────────────
     * Earlier diagnosis: ruled out other-user contention and per-angle
     * geometry, couldn't check dmesg (no root). GPU-side event profiling
     * (CL_PROFILING_COMMAND_START/END, already enabled on the queue via
     * CL_QUEUE_PROFILING_ENABLE) confirmed slow slabs are GPU-bound
     * (wall==gpu to the ms), and a follow-up diagnostic (--diag
     * repeat-slab) found what looked like the mechanism: NOT thermal
     * throttling (Hawaii's ~3.2x CORE-clock DVFS range can't produce the
     * observed ~5.8-6x jump -- the error in this argument is that it used
     * core-clock range on a kernel that turned out to be memory-clock
     * bound, where the DPM range is 8.3x, comfortably covering it), and
     * gpu-img/gpu-opt stay stable in the same sessions gpu-buf goes slow.
     * Repeating one fixed angle-slab many times APPEARED to show a step
     * degradation to ~6x cost that held, then recovered instantly when
     * d_vol was freed and recreated -- and degraded again after a
     * further, variable number of launches (6-13 in testing). This was
     * attributed to the OpenCL driver periodically demoting the 537MB
     * d_vol buffer's memory placement under sustained access. Two things
     * this missed: that repeat-slab diagnostic ran at lws={4,64,1} while
     * production fp_buffer used {2,16,2} (a different work-group shape
     * than the kernel actually showing the variance -- fixed, see that
     * function's comment), and a longer, corrected-lws repeat-slab run
     * showed the same slab recovering spontaneously with NO reallocation
     * at all, which a one-shot allocation-tied demotion cannot produce.
     *
     * FP_BUFFER_VOL_REALLOC_EVERY: mitigation attempt under the old
     * theory -- every N angle-slabs, read the current d_vol contents back
     * to host, free the buffer, and recreate it fresh from that same
     * data, on the theory that this resets whatever driver-side state
     * causes the (apparent) demotion.
     *
     * DID NOT RELIABLY BEAT BASELINE even under the old theory -- kept
     * off by default (0). A 5-epoch sweep at N in {3,4,5,6,7,10} showed
     * N=5 as a clear winner (34.91s vs a 37.7-50.9s baseline-equivalent
     * range, clean of slow-slab warnings after 2 epochs of warmup) -- but
     * that did not reproduce at the full 10-epoch/75-angle scale used for
     * the documented baseline. Four full 10-epoch runs with N=5 gave
     * [105.84, 86.77, 89.04, 85.31]s (mean 91.74s, stdev 9.52) against the
     * three existing unmitigated baseline runs [75.37, 101.89, 83.63]s
     * (mean 86.96s, stdev 13.57) -- the mitigated mean is *slower*, not
     * faster (-5.5%). Makes sense in hindsight: it was intervening on a
     * mechanism (buffer placement) that was never the actual cause, so
     * each reallocation (~0.2-0.4s, readback+upload of 537MB) was pure
     * overhead with no corresponding benefit.
     *
     * Left in as an opt-in env var since the hook point (swap d_vol mid-
     * run) could still be useful for other purposes, but it is not a
     * variance mitigation -- use FP_BUFFER_SKIP_SLAB_FINISH for that. */
    int realloc_every = 0;
    {
        const char *re_env = getenv("FP_BUFFER_VOL_REALLOC_EVERY");
        if (re_env) realloc_every = atoi(re_env);
    }
    static long total_slab_launches = 0;
    size_t vol_bytes = (size_t)Nxz * Nxz * Ny * sizeof(float);
    float *vol_scratch = (realloc_every > 0) ? (float *)malloc(vol_bytes) : NULL;

    /* FP_BUFFER_CLOCK_PROBE: DVFS diagnosis (see sample_gpu_clocks above).
     * Off by default -- zero cost when unset. Linux-only. */
    int clock_probe = 0;
#ifdef __linux__
    {
        const char *probe_env = getenv("FP_BUFFER_CLOCK_PROBE");
        if (probe_env) clock_probe = atoi(probe_env);
    }
    if (clock_probe && slab_idx == 0) {
        char clocks[128];
        sample_gpu_clocks(clocks, sizeof(clocks));
        printf("    [fp_buffer clock-probe] baseline %s\n", clocks);
    }
#endif

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
                char clocks[128] = "";
#ifdef __linux__
                if (clock_probe) sample_gpu_clocks(clocks, sizeof(clocks));
#endif
                if (perr1 == CL_SUCCESS && perr2 == CL_SUCCESS) {
                    double gpu_dt = (double)(t_end - t_start) * 1e-9;
                    printf("    [fp_buffer slab %d, p0=%zu] wall=%.3fs gpu=%.3fs %s (SLOW)%s%s\n",
                           slab_idx, p0, dt, gpu_dt,
                           (dt - gpu_dt > 0.5) ? "HOST-SIDE-GAP" : "GPU-BOUND",
                           clocks[0] ? " " : "", clocks);
                } else {
                    printf("    [fp_buffer slab %d, p0=%zu] %.3f s (SLOW, profiling query failed)%s%s\n",
                           slab_idx, p0, dt, clocks[0] ? " " : "", clocks);
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
 *
 * Used to isolate the gpu-buf variance root cause -- now confirmed as
 * Hawaii memory-clock (mclk) DVFS (see run_fp_buffer's root-cause comment
 * above and README "gpu-buf run-to-run variance" for the full evidence).
 * Combined with FP_BUFFER_CLOCK_PROBE=1, a 60-repeat run at the corrected
 * lws={2,16,2} (see run_fp_buffer's lws comment -- this diagnostic
 * previously ran at a mismatched {4,64,1}, now fixed) showed the SAME
 * fixed slab oscillate between the fast and slow cost tier repeatedly and
 * spontaneously across the run, with realloc_at unused (0) throughout --
 * i.e. no d_vol reallocation ever happened, yet speed recovered on its
 * own multiple times. That result falsified an earlier theory (see the
 * superseded section in run_fp_buffer's comment) that this was a one-shot
 * d_vol buffer-placement demotion recoverable only by reallocation.
 *
 * This isolates ONE fixed angle-slab and launches it N times back-to-back,
 * timing each with GPU-side event profiling -- still useful for
 * characterizing the oscillation pattern itself (frequency, duty cycle)
 * even with the mechanism now identified:
 *
 *   - Throttling predicts MONOTONE degradation (die heating under load) --
 *     ruled out; instead see spontaneous fast<->slow oscillation.
 *   - TLB/page-residency predicts BIMODAL switching between two cost
 *     levels -- consistent with what's observed (two clear cost tiers,
 *     tracking mclk's two DPM states directly).
 *   - Memory-channel camping predicts UNIFORMLY slow, every time (fully
 *     deterministic for that angle set) -- ruled out by fast reps.
 *
 * angle_offset lets the same test be re-run with a different angle range
 * in the same slab-index slot (does slowness follow slab INDEX or ANGLE
 * VALUE? -- moot now that the trigger is confirmed to be the GPU's own
 * clock state rather than anything angle- or slab-specific, but the knob
 * is left in for further characterization).
 *
 * realloc_at (0 = never): after this many repeats have completed, free
 * d_vol and create a fresh buffer from the same host data, then continue.
 * No longer expected to matter for the variance itself (mclk state is
 * independent of d_vol's lifetime) -- left in as a diagnostic knob, not a
 * mitigation; see FP_BUFFER_SKIP_SLAB_FINISH in run_fp_buffer for the
 * mitigation that actually helps.
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
    /* Args 14-16 (use_aabb, ip_start, ip_count) were never set here --
     * fp_buffer's real signature (see run_fp_buffer above) has 17 args,
     * not 14. Left unset, they caused CL_INVALID_KERNEL_ARGS (-52) at
     * enqueue -- the bug blocking this diagnostic on pool15-01. use_aabb
     * off (matches run_fp_buffer's default, see that function's AABB
     * comment); ip_start/ip_count cover the single fixed
     * angle_offset/slab_size range this diagnostic launches in one shot
     * (no angle-slab chunking here, unlike run_fp_buffer). */
    int diag_use_aabb = 0;
    {
        const char *aabb_env = getenv("FP_IMAGE_AABB");
        if (aabb_env) diag_use_aabb = atoi(aabb_env);
    }
    clSetKernelArg(k,14, sizeof(int),    &diag_use_aabb);
    clSetKernelArg(k,15, sizeof(int),    &angle_offset);
    clSetKernelArg(k,16, sizeof(int),    &slab_size);

    /* Was {4,64,1} -- the GTX 680's old default -- while production
     * run_fp_buffer has used {2,16,2} since the Hawaii/GTX680 re-sweep
     * (see that function's comment). Every repeat-slab result taken before
     * this fix (including the d_vol-demotion root-cause finding) was
     * measured under a DIFFERENT work-group shape than the kernel that
     * actually shows the variance, so those numbers are not comparable to
     * production timings. Fixed to match run_fp_buffer's default. */
    size_t lws[3] = {2, 16, 2};
    const char *lws_env = getenv("FP_BUFFER_LWS");
    if (lws_env) {
        unsigned long a=2, b=16, c=2;
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

    double t_diag_start = get_time_sec();
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
        /* t_since_start: seconds since this diag began, so bursts can be
         * correlated by time against an external clock/temp sampler
         * (e.g. run_hawaii_dvfs_diagnosis.sh) polling sysfs independently. */
        printf("  rep %3d/%d  t=%.3fs  wall=%.4fs  gpu=%.4fs\n",
               rep+1, n_repeats, get_time_sec() - t_diag_start, dt, gpu_dt);
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
 * OSEM: ip_start/ip_count select a contiguous angle
 * subrange. Launches with a global work-offset of ip_start in dim 2 and
 * a global work-size of ip_count (rounded up to lws[2]) instead of the
 * full num_projs -- get_global_id(2) then naturally ranges over
 * [ip_start, ip_start+ip_count) inside the kernel.
 *
 * The rounding-up-to-lws step is exactly the bug risk the original investigation plan
 * flagged before any code was written: rounding ip_count up can push
 * gws[2] past ip_count while still being < num_projs, so a guard of
 * "ip >= num_projs" alone would NOT catch the extra rounded-up
 * work-items -- they'd process angles beyond the subset. Fixed by
 * passing ip_start/ip_count into the kernel itself and guarding
 * "ip >= ip_start + ip_count" there (see fp_image.cl), not just
 * "ip >= num_projs".
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
    clSetKernelArg(k,15, sizeof(int), &ip_start);
    clSetKernelArg(k,16, sizeof(int), &ip_count);

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
     * hardware switch from AMD Hawaii PRO. Unlike fp_buffer, this shape did
     * NOT need re-tuning -- {8,32,1} (the Hawaii-era default) is still
     * the best on GTX 680 too. Swept {2,32,1}/{2,16,2}/{4,32,1}/{1,32,1}/
     * {4,16,1} at 256^3, gpu-opt, 10 epochs: every alternative was
     * worse (4,32,1 closest at -6.5%, 1,32,1 catastrophic at -87%
     * i.e. nearly 2x slower). Negative result, kept as-is. */
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

/*
 * ── gpu_diag_repeat_slab ────────────────────────────────────────────────
 * Public entry for the repeat-slab variance diagnostic. Builds R/T buffers,
 * repeats one fixed angle-slab n_repeats times (optionally reallocating
 * d_vol partway through -- see run_diag_repeat_slab for realloc_at).
 * Does not run any epoch loop and does not write output -- diagnostic only.
 */
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
    /* Needed by the G3 fused bp_image_update call below (float32
     * GPU_MODE_IMAGE) -- run_bp_image/run_bp_buffer compute their own
     * copies of these locally since they're separate functions; this one
     * is inlined directly into reconstruct_gpu's epoch loop instead. */
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
    /* P1b (perf-512): d_ratio/d_ratio_prep are only ever bound to a
     * kernel inside the GPU_MODE_BUFFER branch below (proj_divide's
     * output and run_preprocess's output feeding run_bp_buffer) --
     * image mode fuses divide+preprocess straight into ratio_img_buf
     * and never touches these. Were allocated unconditionally
     * regardless of mode: 2 * proj_bytes = 758.8 MiB wasted in
     * gpu-img/gpu-opt at 512^3 (379.4 MiB each), on a card already at
     * 97.7% occupancy in gpu-img -- this is a genuine VRAM leak in
     * everything but buffer mode, not just an optimization. Gating
     * allocation (and the matching release below) on the actual mode
     * frees that headroom, which is what makes OSEM S>=2 feasible at
     * 512^3 (previously S=1's own precompute block already peaked near
     * the card's 4037 MiB limit -- see the 256^3-only VRAM comment near
     * reconstruct_gpu_opt). */
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

        /* forward project: b = F(v0)
         * OSEM: reconstruct_gpu (buffer/image modes) does not
         * get real OSEM subsetting in this pass -- always the full angle
         * range, equivalent to --subsets 1. OSEM is implemented in
         * reconstruct_gpu_opt only; ip_start=0/ip_count=np here keeps
         * this path's behavior byte-identical to before. */
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

        /* bp_ratio = bp(ratio_prep). No zero-fill needed: bp_buffer/bp_image
         * kernels unconditionally overwrite every voxel in range (plain
         * assignment, not accumulation), so pre-clearing is dead work.
         *
         * Kernel fusion: image mode fuses proj_divide (ratio=p0/b)
         * with preprocess (cone_weight+flip+transpose) into one kernel
         * that writes straight into ratio_img_buf -- no d_ratio
         * intermediate buffer, no buffer->image copy. Buffer mode is
         * unaffected: it still needs proj_divide's plain-buffer output
         * for run_preprocess/run_bp_buffer, which don't use an image.
         *
         * G3 fusion (bp_image_update, kernels/bp_image.cl): float32
         * GPU_MODE_IMAGE takes a further-fused path below instead of
         * this block -- it skips the separate d_bp_ratio buffer and
         * vol_update_img launch entirely, computing the backprojection
         * sum and immediately applying v *= bp_ratio/bp_ones and the
         * vol_img write in the same kernel invocation, at the same
         * voxel. See that kernel's comment for why this is scoped to
         * float32 GPU_MODE_IMAGE only (not --half, not gpu-opt/bp_opt,
         * not the bp(ones) precompute) -- this function (reconstruct_gpu)
         * is never called for GPU_MODE_OPT, so the only cases reaching
         * the un-fused branches below are GPU_MODE_BUFFER and
         * GPU_MODE_IMAGE with --half. */
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

            /* v0 *= bp_ratio / bp_ones. GPU_MODE_IMAGE reaching this point
             * is always --half (the plain-float32 case took the G3 fused
             * branch above) -- --half keeps plain vol_update since its
             * vol_img is refreshed via float_to_half+copy above instead,
             * so vol_update_img (the unfused float32 update+image-write)
             * is now dead code and has been removed; use it again only
             * if a case that needs it (float32, non-fused) is reintroduced.
             * Buffer mode has no vol_img at all, so it always uses plain
             * vol_update too. */
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

    /* OSEM: subset k is the contiguous angle range
     * [subset_start[k], subset_start[k]+subset_count[k]) -- valid ONLY
     * because the caller has already permuted p->angles/proj_measured
     * (see utils.h) so that interleaved, max-separated subsets become
     * contiguous ranges. Computed once here since it's needed both for
     * the per-subset normalizer precompute and every epoch's inner loop. */
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
