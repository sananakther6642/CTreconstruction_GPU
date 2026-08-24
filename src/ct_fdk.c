#include "ct_fdk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#define CL_CHECK(err, msg) \
    do { if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s\n", (err), (msg)); \
        exit(1); } } while(0)

/* ── Cooley-Tukey radix-2 FFT (in-place, complex interleaved) ──────────────
 * data: [2*n] floats, even indices = real, odd = imag.
 * n must be power of 2. invert=1 for IFFT (includes 1/n normalization).
 */
static void fft_1d(float *data, int n, int invert)
{
    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = data[2*i], ti = data[2*i+1];
            data[2*i]   = data[2*j]; data[2*i+1] = data[2*j+1];
            data[2*j]   = tr;        data[2*j+1] = ti;
        }
    }
    /* Butterfly stages */
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2.f * (float)M_PI / (float)len * (invert ? -1.f : 1.f);
        float wRe = cosf(ang), wIm = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float uRe = 1.f, uIm = 0.f;
            for (int j = 0; j < len/2; j++) {
                int a = i+j, b = i+j+len/2;
                float aRe = data[2*a], aIm = data[2*a+1];
                float bRe = data[2*b]*uRe - data[2*b+1]*uIm;
                float bIm = data[2*b]*uIm + data[2*b+1]*uRe;
                data[2*a]   = aRe + bRe; data[2*a+1] = aIm + bIm;
                data[2*b]   = aRe - bRe; data[2*b+1] = aIm - bIm;
                float nuRe = uRe*wRe - uIm*wIm;
                uIm = uRe*wIm + uIm*wRe;
                uRe = nuRe;
            }
        }
    }
    if (invert)
        for (int i = 0; i < 2*n; i++) data[i] /= (float)n;
}

/* ── Next power of 2 >= n ──────────────────────────────────────────────── */
static int next_pow2(int n)
{
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ── FDK pre-processing: cosine weight + ramp filter all projections ───────
 *
 * FDK cosine weight: w(u,v) = SDD / sqrt(SDD^2 + u^2 + v^2)
 * Ram-Lak ramp filter: multiply by |omega| in frequency domain, per row.
 *
 * Input/output: proj_filtered [np * H * W] (caller allocates, same layout as
 * proj_measured [ip][iv][iu]).
 *
 * perf-v2 Phase C2: parallelized with OpenMP -- this loop was serial on the
 * source `fdk` branch and was ~1.5s of FDK's 1.57s total at 256^3 (the GPU
 * backprojection pass is only ~0.03-0.9s). Every (ip,iv) row is fully
 * independent (each row's FFT scratch is self-contained, ramp[] is
 * read-only), so this is a clean collapse(2) with one scratch buffer per
 * thread instead of the original single shared allocation.
 */
static void fdk_filter(const CBpara *p, const float *proj_in, float *proj_out)
{
    int W  = p->detector_width;
    int H  = p->detector_height;
    int np = p->num_projs;
    float SDD = (float)p->SDD;
    float px  = (float)p->pixelSize;

    /* FFT size: next power of 2 >= 2*W (zero-pad to avoid circular wrap) */
    int Nfft = next_pow2(2 * W);

    /* Ram-Lak ramp: H[k] = k / (Nfft * Δu)
     * Discretized |ω_k| where ω_k = k/(Nfft*Δu) cycles/mm.
     * IFFT (with built-in 1/Nfft) then gives the correct FBP convolution.
     * Computed once, read-only inside the parallel loop below. */
    float *ramp = (float *)malloc(Nfft * sizeof(float));
    float ramp_scale = 1.0f / ((float)Nfft * px);
    ramp[0] = 0.f;
    for (int k = 1; k <= Nfft/2; k++) {
        ramp[k]        = (float)k * ramp_scale;
        ramp[Nfft - k] = ramp[k];
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ip = 0; ip < np; ip++) {
        for (int iv = 0; iv < H; iv++) {
            /* perf-v2: per-thread scratch (was one shared `row` buffer on
             * the serial version -- must be private under collapse(2)). */
            float *row = (float *)malloc(2 * Nfft * sizeof(float));

            const float *src = proj_in  + (size_t)ip*H*W + (size_t)iv*W;
            float       *dst = proj_out + (size_t)ip*H*W + (size_t)iv*W;

            /* Fill complex buffer with cosine-weighted row, zero-padded */
            memset(row, 0, 2 * Nfft * sizeof(float));
            for (int iu = 0; iu < W; iu++) {
                float u = ((float)iu + 0.5f - W * 0.5f) * px;
                float v = ((float)iv + 0.5f - H * 0.5f) * px;
                float cw = SDD / sqrtf(SDD*SDD + u*u + v*v);
                row[2*iu]   = src[iu] * cw;
                row[2*iu+1] = 0.f;
            }

            /* Forward FFT */
            fft_1d(row, Nfft, 0);

            /* Multiply by ramp |omega| */
            for (int k = 0; k < Nfft; k++) {
                row[2*k]   *= ramp[k];
                row[2*k+1] *= ramp[k];
            }

            /* Inverse FFT */
            fft_1d(row, Nfft, 1);

            /* Copy real part back */
            for (int iu = 0; iu < W; iu++)
                dst[iu] = row[2*iu];

            free(row);
        }
    }

    free(ramp);
}

/* ── reconstruct_fdk ────────────────────────────────────────────────────────
 *
 * Uses CLState already initialized in GPU_MODE_OPT.
 * Volume is zeroed here internally before the accumulating bp pass; caller
 * need not pre-zero it (unlike the source fdk branch's contract, which
 * required a pre-zeroed volume -- shifted the responsibility here since
 * this is the safer default and matches how the rest of the codebase's
 * reconstruct_* entry points already work: caller provides *some* initial
 * volume, this function does what it needs internally).
 */
void reconstruct_fdk(CLState *cl, const CBpara *p,
                     const float *proj_measured, float *volume)
{
    cl_int err;
    int Nxz = p->Volumen_num_xz, Ny = p->Volumen_num_y;
    int W   = p->detector_width,  H  = p->detector_height;
    int np  = p->num_projs;

    size_t vol_n      = (size_t)Nxz * Nxz * Ny;
    size_t vol_bytes  = vol_n * sizeof(float);
    size_t proj_bytes = (size_t)np  * H   * W  * sizeof(float);

    memset(volume, 0, vol_bytes);

    /* ── Step 1: cosine weight + ramp filter (CPU, parallelized) ── */
    printf("  FDK: filtering projections (CPU, %d threads)...\n", omp_get_max_threads());
    float *proj_filtered = (float *)malloc(proj_bytes);
    fdk_filter(p, proj_measured, proj_filtered);

    /* ── Step 2: preprocess filtered projections (flip+transpose) ──
     * Cone weight already baked into the filtered projections by
     * fdk_filter, and FDK's ramp-filter scaling already accounts for
     * voxelSize normalization -- so this is flip+transpose only, done on
     * CPU since it's one-shot (not worth a GPU kernel launch for a single
     * pass at init time). */
    float *proj_prep = (float *)malloc(proj_bytes);
    for (int ip = 0; ip < np; ip++)
        for (int iw = 0; iw < W; iw++)
            for (int ih = 0; ih < H; ih++)
                proj_prep[(size_t)ip*W*H + (size_t)iw*H + ih] =
                    proj_filtered[(size_t)ip*H*W + (size_t)(H-1-ih)*W + iw];
    free(proj_filtered);

    /* ── Step 3: build cos/sin LUT ──
     * perf-v2 Phase C2: the source fdk branch also called
     * build_RT_buffers(cl, p) here, building cl->d_R_mats/d_T_vecs -- but
     * those are only read by fp_image/fp_buffer (forward projection).
     * FDK does a single backprojection pass via bp_opt, which only reads
     * angle_cs (verified: no d_R_mats/d_T_vecs reference anywhere in
     * kernels/bp_buffer_opt.cl). That call was dead work left over from
     * copy-pasting the MLEM setup pattern; dropped here, along with the
     * matching d_R_mats/d_T_vecs release at the end of this function that
     * would otherwise double-release memory objects when reconstruct_fdk
     * is used standalone (main.c's --mode fdk path does not otherwise
     * call build_RT_buffers first, so those fields would be uninitialized
     * cl_mem handles). Also side-steps build_RT_buffers being `static` in
     * ct_gpu.c and not exposed to this translation unit. */
    float *ang_cs = (float *)malloc(np * 2 * sizeof(float));
    for (int i = 0; i < np; i++) {
        ang_cs[2*i]   = (float)cos(p->angles[i]);
        ang_cs[2*i+1] = (float)sin(p->angles[i]);
    }

    /* ── Step 4: upload to GPU ── */
    cl_mem d_proj_prep = clCreateBuffer(cl->ctx,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, proj_bytes, proj_prep, &err);
    CL_CHECK(err, "d_proj_prep fdk");
    free(proj_prep);

    cl_mem d_ang_cs = clCreateBuffer(cl->ctx,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, np*2*sizeof(float), ang_cs, &err);
    CL_CHECK(err, "d_ang_cs fdk");
    free(ang_cs);

    cl_mem d_vol = clCreateBuffer(cl->ctx,
        CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, vol_bytes, volume, &err);
    CL_CHECK(err, "d_vol fdk");

    /* ── Step 5: upload proj_prep as image2d_array for bp_opt ── */
    cl_image_format img_fmt = {CL_R, CL_FLOAT};
    cl_image_desc idesc = {0};
    idesc.image_type       = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    idesc.image_width      = (size_t)H;
    idesc.image_height     = (size_t)W;
    idesc.image_array_size = (size_t)np;
    cl_mem proj_img = clCreateImage(cl->ctx, CL_MEM_READ_WRITE, &img_fmt, &idesc, NULL, &err);
    CL_CHECK(err, "proj_img fdk");
    size_t origin[3] = {0,0,0};
    size_t region[3] = {(size_t)H, (size_t)W, (size_t)np};
    err = clEnqueueCopyBufferToImage(cl->queue, d_proj_prep, proj_img,
                                     0, origin, region, 0, NULL, NULL);
    CL_CHECK(err, "CopyBufferToImage fdk");

    /* ── Step 6: single backprojection pass ──
     * perf-v2 Phase C2: bp_opt now takes ip_start/ip_count (added for
     * OSEM, Phase C1) -- pass the full range (0, np), i.e. no subsetting,
     * matching FDK's single-pass-over-all-angles design. */
    printf("  FDK: backprojecting (GPU)...\n");
    {
        cl_kernel k = cl->k_bp_opt;
        size_t lmem_bytes = (size_t)np * sizeof(cl_float2);
        float SOD = (float)p->SOD, SDD = (float)p->SDD;
        float px  = (float)p->pixelSize;
        float fdk_vs = (float)p->voxelSize;
        int ip_start = 0;

        clSetKernelArg(k, 0, sizeof(cl_mem), &proj_img);
        clSetKernelArg(k, 1, sizeof(cl_mem), &d_ang_cs);
        clSetKernelArg(k, 2, sizeof(cl_mem), &d_vol);
        clSetKernelArg(k, 3, lmem_bytes,     NULL);
        clSetKernelArg(k, 4, sizeof(int),    &Nxz);
        clSetKernelArg(k, 5, sizeof(int),    &Ny);
        clSetKernelArg(k, 6, sizeof(int),    &W);
        clSetKernelArg(k, 7, sizeof(int),    &H);
        clSetKernelArg(k, 8, sizeof(int),    &np);
        clSetKernelArg(k, 9, sizeof(float),  &SOD);
        clSetKernelArg(k,10, sizeof(float),  &SDD);
        clSetKernelArg(k,11, sizeof(float),  &fdk_vs);
        clSetKernelArg(k,12, sizeof(float),  &px);
        clSetKernelArg(k,13, sizeof(int),    &ip_start);
        clSetKernelArg(k,14, sizeof(int),    &np);

        size_t gws[3] = {(size_t)Nxz, (size_t)Nxz, (size_t)Ny};
        size_t lws[3] = {8, 8, 4};
        for (int d = 0; d < 3; d++)
            if (gws[d] % lws[d]) gws[d] += lws[d] - gws[d] % lws[d];
        err = clEnqueueNDRangeKernel(cl->queue, k, 3, NULL, gws, lws, 0, NULL, NULL);
        CL_CHECK(err, "bp_opt fdk");
    }
    clFinish(cl->queue);

    /* ── Step 7: read back ── */
    clEnqueueReadBuffer(cl->queue, d_vol, CL_TRUE, 0, vol_bytes, volume, 0, NULL, NULL);

    /* ── Step 8: clamp to a small positive floor ──
     * perf-v2 Phase C2, MANDATORY per the plan: FDK's ramp filter
     * undershoots and produces negative voxel values (a known FBP/FDK
     * property, not a bug). If this volume is ever used as an MLEM
     * initializer (v0 *= ratio each epoch), a voxel that starts at <= 0
     * can NEVER recover -- MLEM's multiplicative update leaves zero (or
     * negative-times-negative-becomes-positive-but-wrong) voxels stuck
     * forever. Clamping here, once, at the source, means every caller of
     * reconstruct_fdk gets a safe volume automatically instead of having
     * to remember to clamp themselves. */
    {
        const float eps = 1e-6f;
        size_t n_clamped = 0;
        for (size_t i = 0; i < vol_n; i++) {
            if (volume[i] < eps) { volume[i] = eps; n_clamped++; }
        }
        if (n_clamped > 0)
            printf("  FDK: clamped %zu/%zu voxels (%.2f%%) to %.0e (ramp-filter undershoot)\n",
                   n_clamped, vol_n, 100.0 * (double)n_clamped / (double)vol_n, (double)eps);
    }

    clReleaseMemObject(proj_img);
    clReleaseMemObject(d_proj_prep);
    clReleaseMemObject(d_ang_cs);
    clReleaseMemObject(d_vol);
}
