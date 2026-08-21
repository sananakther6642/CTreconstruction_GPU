/*
 * bp_buffer_opt.cl — Optimized backprojection kernel.
 *
 * Combines image2d_array_t hardware texture sampler (free bilinear + texture
 * cache, same as gpu-img) with a precomputed float2 cos/sin LUT (eliminates
 * per-voxel transcendental calls in the angle loop).
 *
 * This is the "fast path": image sampler wins on bandwidth, LUT wins on ALU.
 */

/* CLK_ADDRESS_CLAMP returns 0 outside image bounds — eliminates per-sample
 * bounds check while keeping correct zero-padding at detector edges. */
__constant sampler_t samp =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP           |
    CLK_FILTER_LINEAR;

__kernel void bp_opt(
    __read_only  image2d_array_t proj_images, /* [num_projs][W][H] */
    __global const float2       *angle_cs,    /* [num_projs] (.x=cos, .y=sin) */
    __global       float        *volume,      /* [Nxz * Nxz * Ny] */
    __local        float2       *lcs,         /* local cache: num_projs float2 (caller allocates) */
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
    /* Cooperatively load angle_cs into local memory.
     * Each work-item loads one or more angles; barrier before use.
     * Reduces global reads from num_projs per work-item to num_projs/lsize. */
    int lid   = get_local_id(0)
              + get_local_id(1) * get_local_size(0)
              + get_local_id(2) * get_local_size(0) * get_local_size(1);
    int lsize = get_local_size(0) * get_local_size(1) * get_local_size(2);
    for (int i = lid; i < num_projs; i += lsize)
        lcs[i] = angle_cs[i];
    barrier(CLK_LOCAL_MEM_FENCE);

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
    float sod2 = SOD * SOD;
    float inv_px = 1.f / pixelSize;
    float half_W = (W - 1) * 0.5f;
    float half_H = (H - 1) * 0.5f;
    float sdd_inv_px = SDD * inv_px;
    float zpr_sdd_inv_px = zpr * sdd_inv_px;

    for (int ip = 0; ip < num_projs; ip++) {
        float2 cs = lcs[ip];
        float U = SOD + ypr*cs.y + xpr*cs.x;
        float inv_U = native_recip(U);
        float term = (ypr*cs.x - xpr*cs.y) * sdd_inv_px;
        float uf = half_W - term * inv_U;
        int u0 = (int)floor(uf);
        if ((unsigned)u0 >= (unsigned)(W - 1)) continue;

        float vf = half_H + zpr_sdd_inv_px * inv_U;
        int v0 = (int)floor(vf);
        if ((unsigned)v0 >= (unsigned)(H - 1)) continue;

        float4 val = read_imagef(proj_images, samp, (float4)(vf + 0.5f, uf + 0.5f, (float)ip, 0.f));
        sum += val.x * sod2 * (inv_U * inv_U);
    }

    sum *= M_PI_F / (float)num_projs;

    int out_ix = Nxz - 1 - ix;
    int out_iy = Nxz - 1 - iy;
    volume[out_ix * Nxz * Ny + out_iy * Ny + iz] = sum;
}

/*
 * bp_angle_parallel — kept for ABI; not used in main reconstruction path.
 */
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

static void atomic_add_float(__global float *addr, float val)
{
    union { float f; int i; } old, new_;
    do {
        old.f = *addr;
        new_.f = old.f + val;
    } while (atomic_cmpxchg((__global int*)addr, old.i, new_.i) != old.i);
}

__kernel void bp_angle_parallel(
    __read_only  image2d_array_t proj_images,
    __global const float2       *angle_cs,
    __global       float        *volume,
    int   Nxz,
    int   Ny,
    int   W,
    int   H,
    int   num_projs,
    float SOD,
    float SDD,
    float voxelSize,
    float pixelSize,
    float scale
)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2) % Ny;
    int ip = get_global_id(2) / Ny;

    if (ix >= Nxz || iy >= Nxz || iz >= Ny || ip >= num_projs) return;

    float radius   = Nxz * 0.5f - 0.5f;
    float radius_z = Ny  * 0.5f - 0.5f;

    float xpr = ((float)ix - radius)   * voxelSize;
    float ypr = ((float)iy - radius)   * voxelSize;
    float zpr = ((float)iz - radius_z) * voxelSize;

    float2 cs = angle_cs[ip];
    float ca = cs.x, sa = cs.y;
    float U  = SOD + ypr*sa + xpr*ca;
    float t  = ypr*ca - xpr*sa;
    float ai = SDD * t / U;
    float bi = zpr * SDD / U;

    float uf = -(ai / pixelSize) + (W - 1) * 0.5f;
    float vf =  (bi / pixelSize) + (H - 1) * 0.5f;
    if (uf < 0.f || uf >= (float)(W - 1) || vf < 0.f || vf >= (float)(H - 1))
        return;
    float texel_u = vf + 0.5f;
    float texel_v = uf + 0.5f;

    float4 val = read_imagef(proj_images, samp,
                             (float4)(texel_u, texel_v, (float)ip, 0.f));
    float contrib = val.x * (SOD * SOD) / (U * U) * scale;

    int out_ix = Nxz-1-ix, out_iy = Nxz-1-iy;
    atomic_add_float(&volume[out_ix*Nxz*Ny + out_iy*Ny + iz], contrib);
}
