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

/* OSEM: ip_start/ip_count select the angle subrange (see bp_buffer.cl). */
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
        float4 val = read_imagef(proj_images, samp, (float4)(texel_u, texel_v, (float)ip, 0.f));
        sum += val.x * (SOD*SOD) * inv_U * inv_U;
    }

    sum *= M_PI_F / (float)num_projs;

    int out_ix = Nxz - 1 - ix;
    int out_iy = Nxz - 1 - iy;
    volume[out_ix * Nxz * Ny + out_iy * Ny + iz] = sum;
}

/*
 * bp_image + vol_update_img fused: bp_image writes an element that
 * vol_update_img immediately reads and nothing else, so fusing removes
 * one full buffer write+read and one kernel launch.
 *
 * Scope: float32 vol_img mode only, in-epoch update call
 * (GPU_MODE_IMAGE && !use_half_vol). Not used for the bp(ones) precompute
 * (still plain bp_image), --half mode (separate float_to_half path), or
 * gpu-opt/bp_opt (different, already locally-cached kernel).
 *
 * Scalar (not vec4) stores -- coalesces the same as vstore4 across 32
 * consecutive lanes, per divide_preprocess_img's precedent.
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
    int   ip_count
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
        float4 val = read_imagef(proj_images, samp, (float4)(texel_u, texel_v, (float)ip, 0.f));
        sum += val.x * (SOD*SOD) * inv_U * inv_U;
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
    float out_v = (denom > 1e-10f) ? v * sum / denom : v;
    volume[idx] = out_v;

    /* vol_img layout: width=Ny(z), height=Nxz(y), depth=Nxz(x) -- same
     * convention as fp_image.cl's read and vol_update_img's write. */
    write_imagef(vol_img, (int4)(iz, out_iy, out_ix, 0), (float4)(out_v, 0.f, 0.f, 0.f));
}

/* ── Utility kernels reused from buffer version (included via host) ──────
 * cone_weight, proj_divide, vol_update are identical and shared.
 * They operate on plain buffers so no image variant is needed for them.
 */
