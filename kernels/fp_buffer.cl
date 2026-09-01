/*
 * fp_buffer.cl — Forward projection kernel using OpenCL buffers.
 *
 * Work-item: one detector pixel (iu, iv) for one projection angle (ip).
 * Ray marching through the volume with trilinear interpolation.
 *
 * Volume layout:  [Nxz][Nxz][Ny]  (x outer, z inner)
 * Proj layout:    [num_projs][H][W]  → slice[iv*W + iu]
 */


static inline float trilinear_buf(__global const float *vol,
                                  int sNy, int Ny,
                                  float xi, float yi, float zi)
{
    int x0 = (int)xi, y0 = (int)yi, z0 = (int)zi;
    float dx = xi - (float)x0, dy = yi - (float)y0, dz = zi - (float)z0;
    float nx = 1.f - dx, ny_ = 1.f - dy, nz_ = 1.f - dz;
    int base = x0 * sNy + y0 * Ny + z0;

    return (vol[base]          * nx + vol[base + sNy]          * dx) * ny_ * nz_
         + (vol[base + Ny]     * nx + vol[base + sNy + Ny]     * dx) * dy  * nz_
         + (vol[base + 1]      * nx + vol[base + sNy + 1]      * dx) * ny_ * dz
         + (vol[base + Ny + 1] * nx + vol[base + sNy + Ny + 1] * dx) * dy  * dz;
}

/* OSEM: ip_start/ip_count select a contiguous angle
 * subrange. See fp_image.cl's kernel comment for why the guard checks
 * "ip >= ip_start + ip_count" rather than "ip >= num_projs" -- the host's
 * per-slab gws rounding (run_fp_buffer, ct_gpu.c) can otherwise let
 * extra rounded-up work-items process angles past the intended subset. */
__kernel void fp_buffer(
    __global const float *volume,    /* [Nxz * Nxz * Ny] */
    __constant float     *R_mats,    /* [num_projs * 9] row-major R per angle */
    __constant float     *T_vecs,    /* [num_projs * 3] T per angle */
    __global       float *proj,      /* [num_projs * H * W] output */
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
    int   ip_start,
    int   ip_count
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

    float rd_norm = native_sqrt(rd[0]*rd[0] + rd[1]*rd[1] + rd[2]*rd[2]);

    float inv_sv_xz = (float)Nxz / sVoxel_xz;
    float inv_sv_y  = (float)Ny  / sVoxel_y;
    float shift_xz  = 0.5f * Nxz - 0.5f;
    float shift_y   = 0.5f * Ny  - 0.5f;
    float step_val  = dt * rd_norm;

    /* AABB slab test: tighten sample range for large detectors */
    int s_start = 0, s_end = n_samples;
    if (W > 512) {
        float hxz = 0.5f * Nxz * voxelSize;
        float hy  = 0.5f * Ny  * voxelSize;
        float ox0 = T[0], oy0 = T[1], oz0 = T[2];
        float tmin = near_t, tmax = far_t;
        if (fabs(rd[0]) > 1e-6f) {
            float t1=(-hxz-ox0)/rd[0], t2=(hxz-ox0)/rd[0];
            if(t1>t2){float tmp=t1;t1=t2;t2=tmp;}
            tmin=fmax(tmin,t1); tmax=fmin(tmax,t2);
        }
        /* component 1 (rd[1],oy0) -> yi via inv_sv_y (Ny axis): half-extent
         * must be hy, not hxz. component 2 (rd[2],oz0) -> zi via
         * inv_sv_xz: needs hxz, not hy. Was transposed — see fp_image.cl
         * for the matching fix and full explanation. */
        if (fabs(rd[1]) > 1e-6f) {
            float t1=(-hy-oy0)/rd[1], t2=(hy-oy0)/rd[1];
            if(t1>t2){float tmp=t1;t1=t2;t2=tmp;}
            tmin=fmax(tmin,t1); tmax=fmin(tmax,t2);
        }
        if (fabs(rd[2]) > 1e-6f) {
            float t1=(-hxz-oz0)/rd[2], t2=(hxz-oz0)/rd[2];
            if(t1>t2){float tmp=t1;t1=t2;t2=tmp;}
            tmin=fmax(tmin,t1); tmax=fmin(tmax,t2);
        }
        if (tmin >= tmax) { proj[ip*H*W+iv*W+iu]=0.f; return; }
        s_start = max(0,        (int)((tmin-near_t)/dt));
        s_end   = min(n_samples,(int)((tmax-near_t)/dt)+1);
    }

    float wx = T[0] + rd[0] * (near_t + s_start * dt);
    float wy = T[1] + rd[1] * (near_t + s_start * dt);
    float wz = T[2] + rd[2] * (near_t + s_start * dt);
    float dox = rd[0] * dt, doy = rd[1] * dt, doz = rd[2] * dt;

    float val = 0.f;
    int sNy = Nxz * Ny;
    for (int s = s_start; s < s_end; s++) {
        float xi = wx * inv_sv_xz + shift_xz;
        float yi = wy * inv_sv_y  + shift_y;
        float zi = wz * inv_sv_xz + shift_xz;

        if (xi >= 0.f && xi < (float)(Nxz - 1) &&
            yi >= 0.f && yi < (float)(Ny  - 1) &&
            zi >= 0.f && zi < (float)(Nxz - 1)) {
            val += trilinear_buf(volume, sNy, Ny, xi, yi, zi);
        }
        wx += dox; wy += doy; wz += doz;
    }
    val *= step_val;

    /* proj stored [ip][iv][iu] matching Python proj[i][j] */
    proj[ip * H * W + iv * W + iu] = val;
}
