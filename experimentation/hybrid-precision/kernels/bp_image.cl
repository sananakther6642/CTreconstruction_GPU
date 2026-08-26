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
/* perf-v2 hybrid-precision -- MEASURED NEGATIVE RESULT, kept off by
 * default (HYBRID_PRECISION unset). Do not enable without re-reading
 * this comment.
 *
 * Hypothesis going in: CLK_FILTER_LINEAR's hardware bilinear blend
 * quantizes its interpolation weight to a small number of fractional
 * bits (fixed-function texture-filter silicon, not a bug) -- measured
 * via diag_op_attribution.py to be the dominant source of gpu-img/
 * gpu-opt's MSE-vs-CPU gap (bp RMS 3.99e-5 vs fp's 3.67e-10, ~100,000x
 * larger). CLK_FILTER_NEAREST returns the exact stored texel with no
 * hardware blending, so four nearest fetches + a manual float32 blend
 * SHOULD reproduce bp_buffer.cl's bilinear_buf exactly and be strictly
 * more accurate wherever it fires.
 *
 * The manual-blend math was independently verified correct three ways
 * (hand-derivation of the coordinate/corner mapping, a numpy simulation
 * matching bilinear_buf to 1e-6 across random inputs, and a kale test
 * confirming HYBRID_GRAD_THRESH set high enough to disable the gate --
 * 1e6 -- reproduces the pre-hybrid baseline output BYTE-FOR-BYTE). The
 * gate logic is also confirmed correct: it is not a stuck-always-on bug.
 *
 * Measured on kale (256^3, 10 epochs), varying coverage from a tiny
 * radius_frac=0.99 band up to radius_frac=0.001 (effectively the whole
 * volume): MSE-vs-CPU got WORSE with MORE manual-blend coverage, not
 * better (baseline 8.42e-10 -> full-coverage 1.16e-9, ~37% worse), while
 * epoch time got 37-53% SLOWER. The single worst voxel across every
 * coverage level was identically (244,66,17) with an identical diff
 * (0.0734) regardless of how much of the volume used the manual path --
 * i.e. this specific voxel's error is intrinsic to the manual blend
 * itself, not a gate-tuning artifact.
 *
 * -cl-fast-relaxed-math theory: TESTED AND RULED OUT. Rebuilt the image/
 * opt programs with that flag entirely stripped (HYBRID_PRECISION_NO_
 * FASTMATH=1) -- output was byte-identical to the flag-on run at every
 * digit, including this exact voxel. Not the cause.
 *
 * Actual likely explanation: "MSE vs CPU" was never really "MSE vs
 * ground truth" -- it's agreement with one specific numerical scheme
 * (manual float32 bilinear interpolation). Both the CPU path and the
 * hardware sampler are approximations of the true continuous
 * backprojection integral; bilinear interpolation itself carries bias,
 * and nothing guarantees the manual scheme sits closer to the true
 * value than the hardware sampler's rounding does at every voxel. At
 * (244,66,17) they appear to genuinely disagree by a real (not buggy)
 * amount, with no ground truth available in this dataset (see README's
 * Files section -- proj_*.hdf5 has no reference volume) to say which
 * is actually more correct. Not chased further: the cost (real,
 * measured 37-53% slowdown) already outweighs the tiny gap this was
 * meant to close (0.0026-0.0255% of signal range, see README), and the
 * "which is more correct" question would need an independent ground
 * truth this project's data doesn't have.
 *
 * Kept in the codebase (not reverted) as a measured negative result, in
 * the same spirit as unroll-x2, AABB-at-256, and
 * FP_BUFFER_VOL_REALLOC_EVERY elsewhere in this project -- zero cost
 * when off (this whole block preprocesses away, confirmed byte-identical
 * via `gcc -E` diff against the pre-hybrid source), real information if
 * anyone revisits it. Gated behind a compile-time define (see gpu_init's
 * HYBRID_PRECISION build flag, ct_gpu.c) rather than a new mode. */
__constant sampler_t samp_exact =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP           |
    CLK_FILTER_NEAREST;
/* The four corner fetches + gradient check + manual blend are inlined
 * directly into bp_image's angle loop below (not factored into a helper)
 * so the same four reads serve both the gradient-check decision and the
 * fallback value -- a helper would either duplicate the fetches or need
 * an awkward out-param for cmin/cmax. uf/vf at the call site are
 * bp_image's already-offset coords (same value as bp_buffer.cl's
 * bilinear_buf's internal u/v, post "+(W-1)*0.5f"/"+(H-1)*0.5f"); image
 * axes are (col=v-ish, row=u-ish) matching bp_image's own texel_u=vf+0.5/
 * texel_v=uf+0.5 convention, so passing (v, u) order to read_imagef (not
 * (u, v)) is required to hit the same four corners bilinear_buf does. */
#endif /* HYBRID_PRECISION */

/* perf-v2 Phase C1 (OSEM): see bp_buffer.cl's kernel comment for the
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
    /* radius_frac: fraction of Nxz/2 beyond which the gradient check (and
     * possibly the manual fallback) is tried at all -- measured 62%/100%
     * of the real error's >100xRMS outliers sit beyond 0.8xFOV radius, so
     * gating here skips the cheap-but-pointless check for the ~interior
     * majority of voxels where hardware and manual already agree.
     * grad_thresh: minimum |max-min| across the 4 corner texels (same
     * units as the projection data) before paying for the manual blend --
     * skips flat regions inside the radius band too. Both clamped host-
     * side (see ct_gpu.c) rather than trusted raw from getenv. */
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
    /* Cheap radius gate, computed once per voxel (not per angle) -- exact
     * match to diag_maxgap.py's own radial_dist_from_center_xy metric. */
    float cx = (float)ix - radius, cy = (float)iy - radius;
    float r2 = cx*cx + cy*cy;
    float rgate2 = (radius_frac * (Nxz * 0.5f)) * (radius_frac * (Nxz * 0.5f));
    int in_radius_band = (r2 > rgate2);
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
        float val_hw;
        if (in_radius_band) {
            /* Only pay for the 4 nearest-fetch corner reads (used both to
             * decide AND, if triggered, as the final answer) inside the
             * radius band -- the interior majority never reads these. */
            float c00 = read_imagef(proj_images, samp_exact, (float4)(v0 + 0.5f, u0 + 0.5f, (float)ip, 0.f)).x;
            float c10 = read_imagef(proj_images, samp_exact, (float4)(v0 + 0.5f, u0 + 1.5f, (float)ip, 0.f)).x;
            float c01 = read_imagef(proj_images, samp_exact, (float4)(v0 + 1.5f, u0 + 0.5f, (float)ip, 0.f)).x;
            float c11 = read_imagef(proj_images, samp_exact, (float4)(v0 + 1.5f, u0 + 1.5f, (float)ip, 0.f)).x;
            float cmin = fmin(fmin(c00,c10), fmin(c01,c11));
            float cmax = fmax(fmax(c00,c10), fmax(c01,c11));
            if (cmax - cmin > grad_thresh) {
                float du = uf - u0, dv = vf - v0;
                val_hw = c00*(1-du)*(1-dv) + c10*du*(1-dv) + c01*(1-du)*dv + c11*du*dv;
            } else {
                val_hw = read_imagef(proj_images, samp, (float4)(texel_u, texel_v, (float)ip, 0.f)).x;
            }
        } else {
            val_hw = read_imagef(proj_images, samp, (float4)(texel_u, texel_v, (float)ip, 0.f)).x;
        }
        sum += val_hw * (SOD*SOD) * inv_U * inv_U;
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

/* ── Utility kernels reused from buffer version (included via host) ──────
 * cone_weight, proj_divide, vol_update are identical and shared.
 * They operate on plain buffers so no image variant is needed for them.
 */
