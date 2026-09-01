/*
 * fp_image.cl — Forward projection using OpenCL image3D for the volume.
 * Also contains float_to_half utility kernel for half-precision vol_img upload.
 */

/* HAVE_FP16 is passed as a host build option (-DHAVE_FP16) only when
 * CL_DEVICE_EXTENSIONS reports cl_khr_fp16 support. Guards this kernel
 * out entirely on devices without it (e.g. this project's NVIDIA GTX 680
 * target, alongside the original AMD Hawaii target which does support
 * it) -- without this guard, the whole program source (fp_image +
 * bp_image combined) fails to build, not just --half mode, since OpenCL
 * C compiles the full translation unit even for kernels never launched. */
#ifdef HAVE_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* Convert float volume buffer to half buffer for bandwidth-efficient vol_img upload.
 * vol_img stored as CL_HALF_FLOAT → halves texture bandwidth in fp_image.
 * read_imagef auto-converts half→float on read (no kernel change needed). */
__kernel void float_to_half(__global const float *src, __global half *dst, int n)
{
    int i = get_global_id(0);
    if (i < n) dst[i] = (half)src[i];
}
#endif /* HAVE_FP16 */

/*
 * fp_image.cl — Forward projection using OpenCL image3D for the volume.
 *
 * Volume stored as CL_MEM_OBJECT_IMAGE3D (read-only).
 * Hardware trilinear sampler replaces manual trilinear_buf.
 *
 * Image3D layout matches volume [Nxz][Nxz][Ny]:
 *   width  = Ny   (z dimension)
 *   height = Nxz  (y dimension)
 *   depth  = Nxz  (x dimension)
 * Texel coord = (zi+0.5, yi+0.5, xi+0.5) in texel space.
 *
 * R_mats: [num_projs * 9] row-major rotation matrices precomputed on host.
 * T_vecs: [num_projs * 3] translation vectors precomputed on host.
 * Eliminates angle2pose_img() per work-item (cos/sin + 2x matmul per pixel).
 */

__constant sampler_t vol_samp =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP           |
    CLK_FILTER_LINEAR;

/* ── FP_TEX_EXACT: texture cache, IEEE float32 blend ────────────────────
 * Decouples the two things the hardware sampler bundles together:
 *   - its 3D-tiled TEXTURE CACHE, which is where gpu-img's speed comes
 *     from (gpu-buf is slow because plain __global reads miss that cache,
 *     not because its math is expensive), and
 *   - its fixed-function INTERPOLATION unit, whose blend weights carry
 *     less than float32 precision and which is where the accuracy loss
 *     comes from.
 * CLK_FILTER_NEAREST returns the exact stored texel with no blending, so
 * eight nearest fetches plus a manual float32 trilinear keep the cache
 * and drop the lossy blend.
 *
 * Measured context (kale, 256^3, 100 epochs, MSE vs CPU):
 *   baseline hardware-filtered      1.128e-07   14.65 s
 *   --fp-buf (fp via gpu-buf path)  2.524e-09   26.82 s
 * --fp-buf buys 45x accuracy but pays 1.8x because it abandons the
 * texture cache entirely. This path aims to keep that accuracy at a
 * smaller time cost by keeping the cache. Whether it does is empirical:
 * 8 fetches + an ALU blend replace 1 filtered fetch, so it is NOT
 * expected to match the 14.65 s baseline.
 *
 * Coordinate convention: the volume image is indexed (z, y, x) and the
 * caller passes coord = (zi+0.5, yi+0.5, xi+0.5), so texel centres sit
 * at integer+0.5. floor(c - 0.5) gives the lower neighbour and the
 * remainder is the blend weight. Getting this wrong shifts the volume by
 * half a voxel -- exactly the class of error this exists to remove.
 *
 * Uses CLK_ADDRESS_CLAMP to match vol_samp above, so out-of-range
 * behaviour at the volume border is identical to the path it replaces. */
__constant sampler_t vol_samp_exact =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP           |
    CLK_FILTER_NEAREST;

static float trilinear_tex(__read_only image3d_t vol, float3 c)
{
    float3 base = floor(c - 0.5f);
    float3 f    = (c - 0.5f) - base;          /* blend weights in [0,1) */
    float bx = base.x, by = base.y, bz = base.z;

    /* +0.5f centres each fetch inside its texel so FILTER_NEAREST cannot
     * land on a boundary and round unpredictably. */
    float c000 = read_imagef(vol, vol_samp_exact, (float4)(bx+0.5f, by+0.5f, bz+0.5f, 0.f)).x;
    float c100 = read_imagef(vol, vol_samp_exact, (float4)(bx+1.5f, by+0.5f, bz+0.5f, 0.f)).x;
    float c010 = read_imagef(vol, vol_samp_exact, (float4)(bx+0.5f, by+1.5f, bz+0.5f, 0.f)).x;
    float c110 = read_imagef(vol, vol_samp_exact, (float4)(bx+1.5f, by+1.5f, bz+0.5f, 0.f)).x;
    float c001 = read_imagef(vol, vol_samp_exact, (float4)(bx+0.5f, by+0.5f, bz+1.5f, 0.f)).x;
    float c101 = read_imagef(vol, vol_samp_exact, (float4)(bx+1.5f, by+0.5f, bz+1.5f, 0.f)).x;
    float c011 = read_imagef(vol, vol_samp_exact, (float4)(bx+0.5f, by+1.5f, bz+1.5f, 0.f)).x;
    float c111 = read_imagef(vol, vol_samp_exact, (float4)(bx+1.5f, by+1.5f, bz+1.5f, 0.f)).x;

    float c00 = c000 + f.x * (c100 - c000);
    float c10 = c010 + f.x * (c110 - c010);
    float c01 = c001 + f.x * (c101 - c001);
    float c11 = c011 + f.x * (c111 - c011);
    float c0  = c00  + f.y * (c10  - c00);
    float c1  = c01  + f.y * (c11  - c01);
    return c0 + f.z * (c1 - c0);
}

/*
 * OSEM: ip_start/ip_count select a contiguous angle
 * subrange. The host launches with a global work-offset of ip_start in
 * dim 2 and a work-size of ip_count rounded up to a multiple of lws[2]
 * (see run_fp_image in ct_gpu.c) -- that rounding is exactly why the
 * guard below checks "ip >= ip_start + ip_count", NOT "ip >= num_projs":
 * rounding ip_count up can push get_global_id(2) past ip_start+ip_count
 * while still being < num_projs, and a num_projs-only guard would let
 * those extra rounded-up work-items silently process angles beyond the
 * intended subset.
 */
__kernel void fp_image(
    __read_only  image3d_t       volume_img,   /* [Nxz][Nxz][Ny] as 3D image */
    __constant   float          *R_mats,       /* [num_projs * 9] row-major R per angle */
    __constant   float          *T_vecs,       /* [num_projs * 3] T per angle */
    __global     float          *proj,
    int   Nxz,
    int   Ny,
    int   W,
    int   H,
    int   num_projs,
    int   n_samples,
    float SOD,
    float SDD,
    float voxelSize,
    float pixelSize,
    int   use_aabb,  /* 1 = clip ray to volume AABB; 0 = full n_samples */
    int   ip_start,
    int   ip_count,
    int   tex_exact   /* 1 = NEAREST fetches + IEEE float32 blend (see
                       * trilinear_tex above); 0 = hardware filtering */
)
{
    int iu = get_global_id(0);
    int iv = get_global_id(1);
    int ip = get_global_id(2);

    if (iu >= W || iv >= H || ip >= ip_start + ip_count) return;

    float sVoxel_xz = Nxz * voxelSize;
    float sVoxel_y  = Ny  * voxelSize;
    float dist_max  = sVoxel_xz * 1.42f;
    float near_t    = fmax(0.f, SOD - dist_max);
    float far_t     = fmin(SOD * 2.f, SOD + dist_max);
    float dt        = (far_t - near_t) / (float)(n_samples - 1);

    __constant float *R = R_mats + ip * 9;
    __constant float *T = T_vecs + ip * 3;

    float uu = ((float)iu + 0.5f - W * 0.5f) * pixelSize;
    float vv = ((float)iv + 0.5f - H * 0.5f) * pixelSize;

    float dirs[3] = {uu/SDD, vv/SDD, 1.f};
    float rd[3];
    for (int k=0;k<3;k++)
        rd[k] = R[k*3+0]*dirs[0] + R[k*3+1]*dirs[1] + R[k*3+2]*dirs[2];

    float rd_norm  = sqrt(rd[0]*rd[0] + rd[1]*rd[1] + rd[2]*rd[2]);
    float step_val = dt * rd_norm;

    float inv_sv_xz = (float)Nxz / sVoxel_xz;
    float inv_sv_y  = (float)Ny  / sVoxel_y;
    float shift_xz  = 0.5f * Nxz - 0.5f;
    float shift_y   = 0.5f * Ny  - 0.5f;

    int s_start = 0, s_end = n_samples;
    /* AABB ray clipping: compute tighter sample range for large detectors.
     * Uniform branch (use_aabb same for all threads) — zero divergence cost. */
    if (use_aabb) {
        float hxz = 0.5f * Nxz * voxelSize;
        float hy  = 0.5f * Ny  * voxelSize;
        float ox0 = T[0], oy0 = T[1], oz0 = T[2];
        float tmin = near_t, tmax = far_t;
        if (fabs(rd[0]) > 1e-6f) {
            float t1 = (-hxz - ox0) / rd[0], t2 = (hxz - ox0) / rd[0];
            if (t1 > t2) { float tmp=t1; t1=t2; t2=tmp; }
            tmin = fmax(tmin, t1); tmax = fmin(tmax, t2);
        }
        /* component 1 (rd[1], oy0) maps to yi via inv_sv_y (the Ny-sized
         * axis, see below), so its AABB half-extent must be hy, not hxz —
         * and component 2 (rd[2], oz0) maps to zi via inv_sv_xz, so it
         * needs hxz, not hy. Was transposed (harmless while Nxz==Ny on
         * both datasets, but wrong for a future non-cubic volume). Fixed
         * identically in ct_cpu.c and fp_buffer.cl so CPU/GPU stay in
         * lockstep. */
        if (fabs(rd[1]) > 1e-6f) {
            float t1 = (-hy - oy0) / rd[1], t2 = (hy - oy0) / rd[1];
            if (t1 > t2) { float tmp=t1; t1=t2; t2=tmp; }
            tmin = fmax(tmin, t1); tmax = fmin(tmax, t2);
        }
        if (fabs(rd[2]) > 1e-6f) {
            float t1 = (-hxz - oz0) / rd[2], t2 = (hxz - oz0) / rd[2];
            if (t1 > t2) { float tmp=t1; t1=t2; t2=tmp; }
            tmin = fmax(tmin, t1); tmax = fmin(tmax, t2);
        }
        if (tmin >= tmax) { proj[ip * H * W + iv * W + iu] = 0.f; return; }
        s_start = max(0,        (int)((tmin - near_t) / dt));
        s_end   = min(n_samples,(int)((tmax - near_t) / dt) + 1);
    }

    float wx = T[0] + rd[0] * (near_t + s_start * dt);
    float wy = T[1] + rd[1] * (near_t + s_start * dt);
    float wz = T[2] + rd[2] * (near_t + s_start * dt);
    float dox = rd[0] * dt, doy = rd[1] * dt, doz = rd[2] * dt;

    float val = 0.f;
    for (int s = s_start; s < s_end; s++) {
        float xi = wx * inv_sv_xz + shift_xz;
        float yi = wy * inv_sv_y  + shift_y;
        float zi = wz * inv_sv_xz + shift_xz;

        if (xi >= 0.f && xi < (float)(Nxz - 1) &&
            yi >= 0.f && yi < (float)(Ny  - 1) &&
            zi >= 0.f && zi < (float)(Nxz - 1)) {
            float4 coord = (float4)(zi + 0.5f, yi + 0.5f, xi + 0.5f, 0.f);
            val += tex_exact
                     ? trilinear_tex(volume_img, (float3)(coord.x, coord.y, coord.z))
                     : read_imagef(volume_img, vol_samp, coord).x;
        }
        wx += dox; wy += doy; wz += doz;
    }
    val *= step_val;

    proj[ip * H * W + iv * W + iu] = val;
}
