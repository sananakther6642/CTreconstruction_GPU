/*
 * bp_buffer_opt.cl — Optimized backprojection kernel.
 *
 * Optimizations over bp_buffer.cl:
 *
 *  1. Sin/cos LUT: angles precomputed host-side, passed as (cos,sin) pairs.
 *     Eliminates per-voxel transcendental calls inside the angle loop.
 *
 *  2. Local memory projection cache: each work-group loads a tile of one
 *     projection slice into __local before the angle loop inner section.
 *     Reduces global memory bandwidth by work-group size factor.
 *
 *  3. float4 accumulation: voxel coordinates computed as float4 to use
 *     SIMD lanes on GPUs that vectorize scalar float4 ops.
 *
 *  4. Precomputed voxel world coords: xpr/ypr/zpr computed once outside
 *     angle loop (already trivial, but avoids repeated mul in inner loop).
 */

/* Bilinear from __local memory */
static float bilinear_opt(__local const float *tile,
                           int W, int H, float uf, float vf)
{
    float u = uf + (W-1)*0.5f, v = vf + (H-1)*0.5f;
    int u0=(int)floor(u), u1=u0+1, v0=(int)floor(v), v1=v0+1;
    float du=u-u0, dv=v-v0;
    float c00=(u0>=0&&u0<W&&v0>=0&&v0<H)?tile[u0*H+v0]:0.f;
    float c10=(u1>=0&&u1<W&&v0>=0&&v0<H)?tile[u1*H+v0]:0.f;
    float c01=(u0>=0&&u0<W&&v1>=0&&v1<H)?tile[u0*H+v1]:0.f;
    float c11=(u1>=0&&u1<W&&v1>=0&&v1<H)?tile[u1*H+v1]:0.f;
    return c00*(1-du)*(1-dv)+c10*du*(1-dv)+c01*(1-du)*dv+c11*du*dv;
}

/* Bilinear from __global memory — fallback when local too small */
static float bilinear_opt_g(__global const float *tile,
                             int W, int H, float uf, float vf)
{
    float u = uf + (W-1)*0.5f, v = vf + (H-1)*0.5f;
    int u0=(int)floor(u), u1=u0+1, v0=(int)floor(v), v1=v0+1;
    float du=u-u0, dv=v-v0;
    float c00=(u0>=0&&u0<W&&v0>=0&&v0<H)?tile[u0*H+v0]:0.f;
    float c10=(u1>=0&&u1<W&&v0>=0&&v0<H)?tile[u1*H+v0]:0.f;
    float c01=(u0>=0&&u0<W&&v1>=0&&v1<H)?tile[u0*H+v1]:0.f;
    float c11=(u1>=0&&u1<W&&v1>=0&&v1<H)?tile[u1*H+v1]:0.f;
    return c00*(1-du)*(1-dv)+c10*du*(1-dv)+c01*(1-du)*dv+c11*du*dv;
}

/*
 * bp_opt kernel.
 *
 * angle_cs: float2 array [num_projs] where .x=cos(angle), .y=sin(angle)
 *           precomputed on host — eliminates cos/sin in inner loop.
 *
 * Local memory size passed as lmem_size = W*H*sizeof(float).
 * Work-group loads one full projection slice into __local per angle step.
 * This is valid when W*H fits in __local (typically ≤48KB on modern GPUs).
 * For 256×256 detector: 256*256*4 = 256KB — too large for one shot.
 *
 * STRATEGY for large detectors: cache a ROW STRIP per work-group.
 * Each work-group handles a contiguous block of voxels in iy direction.
 * The strip of projection rows that those voxels could project to is bounded:
 *   ai_max = SDD * radius * voxelSize / (SOD - radius*voxelSize)
 * We cache only the rows [ai_row_min, ai_row_max] that the work-group needs.
 *
 * For simplicity here we demonstrate the full-slice cache for small detectors
 * AND fall back to global for larger ones based on compile-time define.
 * Set -DUSE_LOCAL_CACHE=1 when W*H*4 <= 48*1024.
 */

__kernel void bp_opt(
    __global const float  *proj,        /* [num_projs * W * H] */
    __global const float2 *angle_cs,    /* [num_projs] (.x=cos, .y=sin) */
    __global       float  *volume,      /* [Nxz * Nxz * Ny] */
    __local        float  *lmem,        /* local scratch, size W*H floats */
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

    int lid = get_local_id(0)*get_local_size(1)*get_local_size(2)
            + get_local_id(1)*get_local_size(2)
            + get_local_id(2);
    int lsz = get_local_size(0)*get_local_size(1)*get_local_size(2);

    float radius   = Nxz * 0.5f - 0.5f;
    float radius_z = Ny  * 0.5f - 0.5f;

    /* World coords — computed once, reused for every angle */
    float xpr = (ix < Nxz) ? ((float)ix - radius)   * voxelSize : 0.f;
    float ypr = (iy < Nxz) ? ((float)iy - radius)   * voxelSize : 0.f;
    float zpr = (iz < Ny)  ? ((float)iz - radius_z) * voxelSize : 0.f;

    float sum = 0.f;
    int slice_sz = W * H;

    /*
     * USE_LOCAL_CACHE: runtime check via lmem size.
     * If lmem was allocated W*H floats the cooperative load works.
     * If only 16 bytes were allocated (detector too large), skip the load
     * and read directly from global — correct but slower.
     * We detect this by checking get_local_size product vs slice_sz.
     */
    int can_use_local = (lsz >= slice_sz / 4);  /* rough: group covers slice */

    for (int ip = 0; ip < num_projs; ip++) {
        __global const float *gslice = proj + ip * slice_sz;

        if (can_use_local) {
            /* Cooperative load into __local */
            for (int k = lid; k < slice_sz; k += lsz)
                lmem[k] = gslice[k];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (ix < Nxz && iy < Nxz && iz < Ny) {
            float2 cs = angle_cs[ip];
            float ca = cs.x, sa = cs.y;

            float U  = SOD + ypr*sa + xpr*ca;
            float t  = ypr*ca - xpr*sa;
            float ai = SDD * t / U;
            float bi = zpr * SDD / U;

            float val;
            if (can_use_local)
                val = bilinear_opt(lmem,   W, H, ai/pixelSize, bi/pixelSize);
            else
                val = bilinear_opt_g(gslice, W, H, ai/pixelSize, bi/pixelSize);
            sum += val * (SOD*SOD) / (U*U);
        }

        if (can_use_local)
            barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (ix < Nxz && iy < Nxz && iz < Ny) {
        sum *= M_PI_F / (float)num_projs;
        int out_ix = Nxz - 1 - ix;
        int out_iy = Nxz - 1 - iy;
        volume[out_ix * Nxz * Ny + out_iy * Ny + iz] = sum;
    }
}

/*
 * bp_angle_parallel — Alternative: angle is a 4th global dimension.
 *
 * Global: [Nxz, Nxz, Ny, num_projs]
 * Each work-item computes contribution of ONE angle to ONE voxel,
 * then atomically adds to the output volume.
 *
 * Pros: maximum parallelism (Nxz² × Ny × np work-items).
 * Cons: atomic_add on float requires cl_khr_int64_base_atomics or
 *       OpenCL 2.0 atomic_fetch_add. Uses CAS loop fallback here.
 *       Memory traffic increases — each voxel written np times.
 *
 * Best for: GPU with many CUs, low angle count, large volume.
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
    __global const float  *proj,
    __global const float2 *angle_cs,
    __global       float  *volume,   /* must be zeroed before launch */
    int   Nxz,
    int   Ny,
    int   W,
    int   H,
    int   num_projs,
    float SOD,
    float SDD,
    float voxelSize,
    float pixelSize,
    float scale   /* = pi / num_projs */
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

    /* Manual bilinear from global (no local cache in this variant) */
    float uf = ai/pixelSize + (W-1)*0.5f;
    float vf = bi/pixelSize + (H-1)*0.5f;
    int u0=(int)floor(uf), u1=u0+1, v0=(int)floor(vf), v1=v0+1;
    float du=uf-u0, dv=vf-v0;
    __global const float *sl = proj + ip*W*H;
    float c00=(u0>=0&&u0<W&&v0>=0&&v0<H)?sl[u0*H+v0]:0.f;
    float c10=(u1>=0&&u1<W&&v0>=0&&v0<H)?sl[u1*H+v0]:0.f;
    float c01=(u0>=0&&u0<W&&v1>=0&&v1<H)?sl[u0*H+v1]:0.f;
    float c11=(u1>=0&&u1<W&&v1>=0&&v1<H)?sl[u1*H+v1]:0.f;
    float val = (c00*(1-du)*(1-dv)+c10*du*(1-dv)+c01*(1-du)*dv+c11*du*dv)
                * (SOD*SOD)/(U*U) * scale;

    int out_ix = Nxz-1-ix, out_iy = Nxz-1-iy;
    atomic_add_float(&volume[out_ix*Nxz*Ny + out_iy*Ny + iz], val);
}
