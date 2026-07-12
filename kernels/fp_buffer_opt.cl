/*
 * fp_buffer_opt.cl — Optimized forward projection kernel.
 *
 * Optimizations over fp_buffer.cl:
 *
 *  1. Sin/cos LUT: angle_cs float2 array (.x=cos, .y=sin) — no trig per pixel.
 *
 *  2. float4 ray stepping: pt advances by dt*rd as float4, avoids separate
 *     scalar adds for x/y/z each step.
 *
 *  3. Manual loop unrolling (#pragma unroll 4): compiler hint to pipeline
 *     4 trilinear samples simultaneously (reduces loop overhead, exposes ILP).
 *
 *  4. Volume index precomputed scale/offset: sVoxel and nVoxel factors
 *     turned into multiply-add constants computed once per work-item.
 *
 *  5. Early-exit: if point is outside volume bounds skip trilinear entirely
 *     (density = 0 anyway) — avoids 8 conditional reads per oob sample.
 */

static float trilinear_opt_fast(__global const float *vol,
                                 int Nxz, int Ny,
                                 float xi, float yi, float zi)
{
    /* Fast bounds check — avoids per-corner if/else in hot path */
    if (xi < 0.f || xi >= (float)(Nxz-1) ||
        yi < 0.f || yi >= (float)(Ny-1)  ||
        zi < 0.f || zi >= (float)(Nxz-1))
        return 0.f;

    int x0=(int)xi, y0=(int)yi, z0=(int)zi;
    int x1=x0+1,   y1=y0+1,   z1=z0+1;
    float dx=xi-x0, dy=yi-y0, dz=zi-z0;

    int s1 = Nxz*Ny, s2 = Ny;   /* strides */
    return vol[x0*s1+y0*s2+z0]*(1-dx)*(1-dy)*(1-dz)
         + vol[x1*s1+y0*s2+z0]*   dx *(1-dy)*(1-dz)
         + vol[x0*s1+y1*s2+z0]*(1-dx)*   dy *(1-dz)
         + vol[x1*s1+y1*s2+z0]*   dx *   dy *(1-dz)
         + vol[x0*s1+y0*s2+z1]*(1-dx)*(1-dy)*   dz
         + vol[x1*s1+y0*s2+z1]*   dx *(1-dy)*   dz
         + vol[x0*s1+y1*s2+z1]*(1-dx)*   dy *   dz
         + vol[x1*s1+y1*s2+z1]*   dx *   dy *   dz;
}

/* Build rotation matrix row (optimized — only the 3 dot products we need) */
static float3 rotate_dir(float2 cs_phi1, float2 cs_phi2, float2 cs_ang,
                          float3 d)
{
    /* R = R3(angle) @ R2(phi2) @ R1(phi1)
     * Fully expanded to avoid 3×3 matrix alloc on stack.
     * cs_phi1 = (cos(-pi/2), sin(-pi/2)) = (0, -1)
     * cs_phi2 = (cos(pi/2),  sin(pi/2))  = (0,  1)
     * These are constants — compiler will fold them. */
    float c1=cs_phi1.x, s1=cs_phi1.y;
    float c2=cs_phi2.x, s2=cs_phi2.y;
    float ca=cs_ang.x,  sa=cs_ang.y;

    /* R1 col vectors acting on d */
    float r1x = d.x;
    float r1y = c1*d.y - s1*d.z;
    float r1z = s1*d.y + c1*d.z;

    /* R2 acting on (r1x,r1y,r1z) */
    float r2x =  c2*r1x - s2*r1y;
    float r2y =  s2*r1x + c2*r1y;
    float r2z =  r1z;

    /* R3 acting on (r2x,r2y,r2z) */
    return (float3)(ca*r2x - sa*r2y,
                    sa*r2x + ca*r2y,
                    r2z);
}

__kernel void fp_opt(
    __global const float  *volume,
    __global const float2 *angle_cs,   /* [num_projs] (.x=cos, .y=sin) */
    __global       float  *proj,
    int   Nxz,
    int   Ny,
    int   W,
    int   H,
    int   num_projs,
    int   n_samples,
    float SOD,
    float SDD,
    float voxelSize,
    float pixelSize
)
{
    int iu = get_global_id(0);
    int iv = get_global_id(1);
    int ip = get_global_id(2);

    if (iu >= W || iv >= H || ip >= num_projs) return;

    float sVoxel_xz = Nxz * voxelSize;
    float sVoxel_y  = Ny  * voxelSize;
    float dist_max  = sVoxel_xz * 1.42f;
    float near_t    = fmax(0.f, SOD - dist_max);
    float far_t     = fmin(SOD * 2.f, SOD + dist_max);
    float dt        = (far_t - near_t) / (float)(n_samples - 1);

    /* Precompute index-space scale and offset once per work-item */
    float scale_xz = (float)Nxz / sVoxel_xz;
    float off_xz   = sVoxel_xz * 0.5f * scale_xz - 0.5f;
    float scale_y  = (float)Ny  / sVoxel_y;
    float off_y    = sVoxel_y  * 0.5f * scale_y  - 0.5f;

    float2 cs = angle_cs[ip];

    /* Constant phi1/phi2 — compiler folds these */
    float2 cs_phi1 = (float2)(0.f, -1.f);  /* cos(-pi/2), sin(-pi/2) */
    float2 cs_phi2 = (float2)(0.f,  1.f);  /* cos(pi/2),  sin(pi/2)  */

    float uu = ((float)iu + 0.5f - W*0.5f) * pixelSize;
    float vv = ((float)iv + 0.5f - H*0.5f) * pixelSize;

    float3 dirs = (float3)(uu/SDD, vv/SDD, 1.f);
    float3 rd   = rotate_dir(cs_phi1, cs_phi2, cs, dirs);
    float  rd_norm = length(rd);

    float3 T = (float3)(SOD*cs.x, SOD*cs.y, 0.f);

    /* Starting point at near_t */
    float3 pt = T + rd * near_t;
    float3 step = rd * dt;

    float val = 0.f;
    float step_contrib = dt * rd_norm;

    /* Unrolled 4× to expose instruction-level parallelism */
    int s = 0;
    int n4 = (n_samples / 4) * 4;

    for (; s < n4; s += 4) {
        float xi0 = pt.x*scale_xz + off_xz;
        float yi0 = pt.y*scale_y  + off_y;
        float zi0 = pt.z*scale_xz + off_xz;
        val += trilinear_opt_fast(volume, Nxz, Ny, xi0, yi0, zi0);
        pt += step;

        float xi1 = pt.x*scale_xz + off_xz;
        float yi1 = pt.y*scale_y  + off_y;
        float zi1 = pt.z*scale_xz + off_xz;
        val += trilinear_opt_fast(volume, Nxz, Ny, xi1, yi1, zi1);
        pt += step;

        float xi2 = pt.x*scale_xz + off_xz;
        float yi2 = pt.y*scale_y  + off_y;
        float zi2 = pt.z*scale_xz + off_xz;
        val += trilinear_opt_fast(volume, Nxz, Ny, xi2, yi2, zi2);
        pt += step;

        float xi3 = pt.x*scale_xz + off_xz;
        float yi3 = pt.y*scale_y  + off_y;
        float zi3 = pt.z*scale_xz + off_xz;
        val += trilinear_opt_fast(volume, Nxz, Ny, xi3, yi3, zi3);
        pt += step;
    }
    /* Remainder */
    for (; s < n_samples; s++) {
        float xi = pt.x*scale_xz + off_xz;
        float yi = pt.y*scale_y  + off_y;
        float zi = pt.z*scale_xz + off_xz;
        val += trilinear_opt_fast(volume, Nxz, Ny, xi, yi, zi);
        pt += step;
    }

    proj[ip * H * W + iv * W + iu] = val * step_contrib;
}
