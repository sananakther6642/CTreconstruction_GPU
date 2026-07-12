/*
 * bp_image.cl — Backprojection using OpenCL image2D objects.
 *
 * Each projection slice is stored as a CL_MEM_OBJECT_IMAGE2D.
 * Hardware bilinear sampler replaces the manual interpolation in bp_buffer.cl.
 *
 * Image coords: sampler normalized=false, filter=linear, address=clamp_to_edge.
 * Pixel (u, v) maps to image coord (u + W/2, v + H/2) in texel space.
 */

__constant sampler_t samp =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE   |
    CLK_FILTER_LINEAR;

__kernel void bp_image(
    __read_only  image2d_array_t proj_images, /* [num_projs][W][H] as 2D array */
    __global const float *angles,
    __global       float *volume,
    int   Nxz,
    int   Ny,
    int   W,
    int   H,
    int   num_projs,
    float SOD,
    float SDD,
    float voxelSize,
    float pixelSize
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

    for (int ip = 0; ip < num_projs; ip++) {
        float angle = angles[ip];
        float ca = cos(angle), sa = sin(angle);

        float U  = SOD + ypr*sa + xpr*ca;
        float t  = ypr*ca - xpr*sa;
        float ai = SDD * t / U;
        float bi = zpr * SDD / U;

        /*
         * Convert (ai, bi) mm coords to texel coords.
         * Python: a = -(iw - (W-1)/2)*pixelSize  →  iw = -(a/pixelSize) + (W-1)/2
         *         b =  (ih - (H-1)/2)*pixelSize  →  ih =  (b/pixelSize) + (H-1)/2
         * Image2D uses (col, row) = (iw, ih), offset +0.5 for center-of-texel.
         */
        float texel_u = -(ai / pixelSize) + (W - 1) * 0.5f + 0.5f;
        float texel_v =  (bi / pixelSize) + (H - 1) * 0.5f + 0.5f;

        float4 val = read_imagef(proj_images, samp, (float4)(texel_u, texel_v, (float)ip, 0.f));
        sum += val.x * (SOD*SOD) / (U*U);
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
