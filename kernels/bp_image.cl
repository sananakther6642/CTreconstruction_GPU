/*
 * bp_image.cl — Backprojection using OpenCL image2D objects.
 *
 * Each projection slice is stored as a CL_MEM_OBJECT_IMAGE2D.
 * Hardware bilinear sampler replaces the manual interpolation in bp_buffer.cl.
 *
 * Image coords: sampler normalized=false, filter=linear, address=clamp_to_edge.
 * Pixel (u, v) maps to image coord (u + W/2, v + H/2) in texel space.
 */

/* CLK_ADDRESS_CLAMP returns 0 outside image bounds — eliminates per-sample
 * bounds check while keeping correct zero-padding at detector edges. */
__constant sampler_t samp =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP           |
    CLK_FILTER_LINEAR;

#ifdef HYBRID_PRECISION
/* ── HYBRID_PRECISION: manual float32 bilinear instead of the hardware
 * sampler's quantized blend ──────────────────────────────────────────
 *
 * Re-opened 2026-09-01. A prior version of this experiment (branch
 * `hybrid-precision`, 2026-08-26) measured NET NEGATIVE and was not
 * shipped -- but that entire measurement was taken at 10 EPOCHS, where
 * its own baseline was already 8.42e-10, i.e. ~134x better than the
 * 1.1278e-07 at 100 epochs that actually misses the project's accuracy
 * bar. The regime that matters was never tested; that, not a flaw in the
 * blend math, is why this is being re-run rather than repeated. The
 * original blend was verified correct three independent ways (hand
 * derivation, a numpy sim matching bilinear_buf to 1e-6, and a
 * byte-identical baseline reproduction with the gate disabled) -- that
 * verification is inherited here unchanged, only the call sites moved.
 *
 * CLK_FILTER_NEAREST returns the exact stored texel with no hardware
 * blending, so four nearest fetches + a manual float32 blend reproduce
 * bp_buffer.cl's bilinear_buf. Reads still go through the texture cache,
 * so the spatial-locality advantage over gpu-buf's raw global loads is
 * retained; only the fixed-function filter stage is bypassed.
 *
 * Unlike the original (which inlined this twice), the four corner
 * fetches live in one helper shared by bp_image and bp_image_update --
 * the same four reads still serve both the gradient-check decision and
 * the returned value, so there is no extra fetch cost from factoring it
 * out. image axes are (col=v-ish, row=u-ish) matching the callers'
 * texel_u=vf+0.5 / texel_v=uf+0.5 convention, so (v, u) order into
 * read_imagef is required to hit the same four corners bilinear_buf does. */
__constant sampler_t samp_exact =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP           |
    CLK_FILTER_NEAREST;

static float bp_sample_hybrid(__read_only image2d_array_t proj_images,
                               int ip, float uf, float vf, int u0, int v0,
                               float texel_u, float texel_v,
                               int in_radius_band, float grad_thresh)
{
    if (in_radius_band) {
        float c00 = read_imagef(proj_images, samp_exact, (float4)(v0 + 0.5f, u0 + 0.5f, (float)ip, 0.f)).x;
        float c10 = read_imagef(proj_images, samp_exact, (float4)(v0 + 0.5f, u0 + 1.5f, (float)ip, 0.f)).x;
        float c01 = read_imagef(proj_images, samp_exact, (float4)(v0 + 1.5f, u0 + 0.5f, (float)ip, 0.f)).x;
        float c11 = read_imagef(proj_images, samp_exact, (float4)(v0 + 1.5f, u0 + 1.5f, (float)ip, 0.f)).x;
        float cmin = fmin(fmin(c00,c10), fmin(c01,c11));
        float cmax = fmax(fmax(c00,c10), fmax(c01,c11));
        if (cmax - cmin > grad_thresh) {
            float du = uf - u0, dv = vf - v0;
            return c00*(1-du)*(1-dv) + c10*du*(1-dv)
                 + c01*(1-du)*dv     + c11*du*dv;
        }
    }
    return read_imagef(proj_images, samp, (float4)(texel_u, texel_v, (float)ip, 0.f)).x;
}
#endif /* HYBRID_PRECISION */

/* OSEM: see bp_buffer.cl's kernel comment for the
 * full ip_start/ip_count/M_PI_F-num_projs-cancellation rationale --
 * identical here. */
__kernel void bp_image(
    __read_only  image2d_array_t proj_images, /* [num_projs][W][H] as 2D array */
    __global const float2 *angle_cs,          /* [num_projs] (.x=cos,.y=sin) LUT */
    __global       float *volume,
    int   Nxz,
    int   Ny,
    int   W,
    int   H,
    int   num_projs,
    float SOD,
    float SDD,
    float voxelSize,
    float pixelSize,
    int   ip_start,
    int   ip_count
#ifdef HYBRID_PRECISION
    /* radius_frac: fraction of Nxz/2 beyond which the manual path is
     * considered at all -- 62%/100% of >100xRMS outliers sit beyond
     * 0.8xFOV radius, so the interior majority skips the check.
     * grad_thresh: min |max-min| across the 4 corner texels before
     * paying for the blend; skips flat regions inside the band too.
     * Set radius_frac tiny + grad_thresh 0 for full coverage. Both
     * clamped host-side (ct_gpu.c), not trusted raw from getenv. */
    , float radius_frac
    , float grad_thresh
#endif
)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if (ix >= Nxz || iy >= Nxz || iz >= Ny) return;

    float radius   = Nxz * 0.5f - 0.5f;
    float radius_z = Ny  * 0.5f - 0.5f;

    float xpr = ((float)ix - radius)   * voxelSize;
    float ypr = ((float)iy - radius)   * voxelSize;
    float zpr = ((float)iz - radius_z) * voxelSize;

#ifdef HYBRID_PRECISION
    /* Cheap radius gate, once per voxel (not per angle) -- matches
     * diag_maxgap.py's radial_dist_from_center_xy metric exactly. */
    float cx = (float)ix - radius, cy = (float)iy - radius;
    float r2 = cx*cx + cy*cy;
    float rg = radius_frac * (Nxz * 0.5f);
    int in_radius_band = (r2 > rg*rg);
#endif

    float sum = 0.f;

    for (int ip = ip_start; ip < ip_start + ip_count; ip++) {
        float2 cs = angle_cs[ip];
        float ca = cs.x, sa = cs.y;

        float U  = SOD + ypr*sa + xpr*ca;
        float inv_U = native_recip(U);
        float t  = ypr*ca - xpr*sa;
        float ai = SDD * t * inv_U;
        float bi = zpr * SDD * inv_U;

        /*
         * Convert (ai, bi) mm coords to texel coords.
         * Python: a = -(iw - (W-1)/2)*pixelSize  →  iw = -(a/pixelSize) + (W-1)/2
         *         b =  (ih - (H-1)/2)*pixelSize  →  ih =  (b/pixelSize) + (H-1)/2
         * Image2D uses (col, row) = (iw, ih), offset +0.5 for center-of-texel.
         */
        float uf = -(ai / pixelSize) + (W - 1) * 0.5f;
        float vf =  (bi / pixelSize) + (H - 1) * 0.5f;

        /* Whole-cell zero rule matching the Python reference (see bp_buffer.cl):
         * CLK_ADDRESS_CLAMP zero-pads individual texels, which differs from
         * scipy's all-or-nothing fill_value=0 — gate explicitly instead. */
        int u0 = (int)floor(uf), v0 = (int)floor(vf);
        if (u0 < 0 || u0+1 >= W || v0 < 0 || v0+1 >= H) continue;

        float texel_u = vf + 0.5f;
        float texel_v = uf + 0.5f;
#ifdef HYBRID_PRECISION
        float sval = bp_sample_hybrid(proj_images, ip, uf, vf, u0, v0,
                                       texel_u, texel_v, in_radius_band, grad_thresh);
        sum += sval * (SOD*SOD) * inv_U * inv_U;
#else
        float4 val = read_imagef(proj_images, samp, (float4)(texel_u, texel_v, (float)ip, 0.f));
        sum += val.x * (SOD*SOD) * inv_U * inv_U;
#endif
    }

    sum *= M_PI_F / (float)num_projs;

    int out_ix = Nxz - 1 - ix;
    int out_iy = Nxz - 1 - iy;
    volume[out_ix * Nxz * Ny + out_iy * Ny + iz] = sum;
}

/*
 * G3 (perf-algorithmic): bp_image + vol_update_img fused.
 *
 * bp_image writes d_bp_ratio[out_ix*Nxz*Ny+out_iy*Ny+iz]; vol_update_img
 * immediately reads that exact element and nothing else -- a pure
 * element-wise, index-identical dependency (bp_buffer.cl's
 * vol_update_img comment already documents the identical index
 * decomposition this kernel writes directly). Fusing removes one full
 * d_bp_ratio write + read (128 MB/epoch at 256^3, 1.07 GB at 512^3) and
 * one launch.
 *
 * Scope: float32 vol_img mode ONLY, and only the in-epoch-loop update
 * call (ct_gpu.c's reconstruct_gpu, cl->mode==GPU_MODE_IMAGE &&
 * !use_half_vol). NOT used for: (a) the bp(ones) precompute (there is no
 * "ones" update to divide by, it's computing the normalizer itself --
 * see ct_gpu.c:1127, unchanged, still calls plain bp_image into
 * d_bp_ones), (b) --half mode (its vol_img refresh goes through a
 * separate float_to_half+copy path incompatible with this fusion, see
 * ct_gpu.c's own comment at the vol_update_img/vol_update branch), and
 * (c) gpu-opt/bp_opt (a different, already locally-cached kernel --
 * fusing that is a separate, riskier follow-up, not this change).
 *
 * Scalar (not vec4) stores to volume/vol_img -- vol_update_img's vec4
 * form doesn't survive this kernel's 3D voxel indexing, but per
 * bp_buffer.cl's divide_preprocess_img precedent, 32 consecutive scalar
 * lanes coalesce identically to vstore4.
 */
__kernel void bp_image_update(
    __read_only  image2d_array_t proj_images,
    __global const float2 *angle_cs,
    __global       float *volume,      /* read-write, in place */
    __global const float *bp_ones,
    __write_only image3d_t vol_img,
    int   Nxz,
    int   Ny,
    int   W,
    int   H,
    int   num_projs,
    float SOD,
    float SDD,
    float voxelSize,
    float pixelSize,
    int   ip_start,
    int   ip_count,
    float ones_thresh,  /* FOV mask cutoff -- see src/utils.h */
    int   mask_on       /* 1 = zero below threshold, 0 = legacy freeze */
#ifdef HYBRID_PRECISION
    /* radius_frac: fraction of Nxz/2 beyond which the manual path is
     * considered at all -- 62%/100% of >100xRMS outliers sit beyond
     * 0.8xFOV radius, so the interior majority skips the check.
     * grad_thresh: min |max-min| across the 4 corner texels before
     * paying for the blend; skips flat regions inside the band too.
     * Set radius_frac tiny + grad_thresh 0 for full coverage. Both
     * clamped host-side (ct_gpu.c), not trusted raw from getenv. */
    , float radius_frac
    , float grad_thresh
#endif
)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if (ix >= Nxz || iy >= Nxz || iz >= Ny) return;

    float radius   = Nxz * 0.5f - 0.5f;
    float radius_z = Ny  * 0.5f - 0.5f;

    float xpr = ((float)ix - radius)   * voxelSize;
    float ypr = ((float)iy - radius)   * voxelSize;
    float zpr = ((float)iz - radius_z) * voxelSize;

#ifdef HYBRID_PRECISION
    /* Cheap radius gate, once per voxel (not per angle) -- matches
     * diag_maxgap.py's radial_dist_from_center_xy metric exactly. */
    float cx = (float)ix - radius, cy = (float)iy - radius;
    float r2 = cx*cx + cy*cy;
    float rg = radius_frac * (Nxz * 0.5f);
    int in_radius_band = (r2 > rg*rg);
#endif

    float sum = 0.f;

    for (int ip = ip_start; ip < ip_start + ip_count; ip++) {
        float2 cs = angle_cs[ip];
        float ca = cs.x, sa = cs.y;

        float U  = SOD + ypr*sa + xpr*ca;
        float inv_U = native_recip(U);
        float t  = ypr*ca - xpr*sa;
        float ai = SDD * t * inv_U;
        float bi = zpr * SDD * inv_U;

        float uf = -(ai / pixelSize) + (W - 1) * 0.5f;
        float vf =  (bi / pixelSize) + (H - 1) * 0.5f;

        int u0 = (int)floor(uf), v0 = (int)floor(vf);
        if (u0 < 0 || u0+1 >= W || v0 < 0 || v0+1 >= H) continue;

        float texel_u = vf + 0.5f;
        float texel_v = uf + 0.5f;
#ifdef HYBRID_PRECISION
        float sval = bp_sample_hybrid(proj_images, ip, uf, vf, u0, v0,
                                       texel_u, texel_v, in_radius_band, grad_thresh);
        sum += sval * (SOD*SOD) * inv_U * inv_U;
#else
        float4 val = read_imagef(proj_images, samp, (float4)(texel_u, texel_v, (float)ip, 0.f));
        sum += val.x * (SOD*SOD) * inv_U * inv_U;
#endif
    }

    sum *= M_PI_F / (float)num_projs;

    /* Same flip as plain bp_image; bp_ones was itself computed via
     * plain bp_image so it carries the identical flip -- indexing both
     * volume and bp_ones at (out_ix,out_iy,iz) is consistent with the
     * existing (unfused) bp_image + vol_update_img pairing. */
    int out_ix = Nxz - 1 - ix;
    int out_iy = Nxz - 1 - iy;
    int idx = out_ix * Nxz * Ny + out_iy * Ny + iz;

    float denom = bp_ones[idx];
    float v     = volume[idx];
    /* FOV mask: zero below threshold, don't freeze -- see bp_buffer.cl's
     * vol_update note and src/utils.h. */
    float out_v = (denom > ones_thresh) ? v * sum / denom
                                        : (mask_on ? 0.f : v);
    volume[idx] = out_v;

    /* vol_img layout: width=Ny(z), height=Nxz(y), depth=Nxz(x) -- same
     * convention as fp_image.cl's read and vol_update_img's write. */
    write_imagef(vol_img, (int4)(iz, out_iy, out_ix, 0), (float4)(out_v, 0.f, 0.f, 0.f));
}

/* ── Utility kernels reused from buffer version (included via host) ──────
 * cone_weight, proj_divide, vol_update are identical and shared.
 * They operate on plain buffers so no image variant is needed for them.
 */
