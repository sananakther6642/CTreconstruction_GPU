#include "ct_cpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

/* ── index helpers ─────────────────────────────────────────────────────── */
#define PROJ_IDX(p, h, w, H, W)  ((p)*(H)*(W) + (h)*(W) + (w))
#define VOL_IDX(x, y, z, Nxz, Ny) ((x)*(Nxz)*(Ny) + (y)*(Ny) + (z))

/*
 * prep_proj_for_bp: apply Python's pre-processing done at start of bp_func:
 *   proj[:, ::-1, :].transpose(0,2,1)  →  flip H-axis, then swap W↔H axes
 *   proj /= voxelSize
 *
 * Input  layout: [np][H][W]  (as loaded from HDF5, Python default)
 * Output layout: [np][W][H]  (col-major slices, as bp_cpu expects)
 *
 * dst may equal src only if a temp buffer is used — caller passes separate buf.
 */
static void prep_proj_for_bp(const float *src, float *dst,
                              int np, int W, int H, float voxelSize)
{
    for (int ip = 0; ip < np; ip++) {
        const float *s = src + ip * H * W;
        float       *d = dst + ip * W * H;
        for (int iw = 0; iw < W; iw++) {
            for (int ih = 0; ih < H; ih++) {
                d[iw * H + ih] = s[(H - 1 - ih) * W + iw] / voxelSize;
            }
        }
    }
}

/* ── cone weight ───────────────────────────────────────────────────────── */
void cone_weight_cpu(float *proj, const CBpara *p)
{
    int W = p->detector_width, H = p->detector_height;
    float SDD = (float)p->SDD;
    float px  = (float)p->pixelSize;

    for (int ip = 0; ip < p->num_projs; ip++) {
        float *slice = proj + ip * H * W;  /* [H][W] row-major per slice */
        for (int ih = 0; ih < H; ih++) {
            float v = ((float)(ih - (H-1)*0.5f)) * px;
            for (int iw = 0; iw < W; iw++) {
                float u = (-(float)(iw - (W-1)*0.5f)) * px;
                float w = SDD / sqrtf(SDD*SDD + u*u + v*v);
                slice[ih * W + iw] *= w;
            }
        }
    }
}

/* ── backprojection ────────────────────────────────────────────────────── */
/*
 * proj must already be preprocessed (flip+transpose+scale via prep_proj_for_bp).
 * Layout: [np][W][H] col-major slices.
 *
 * Optimizations vs naive version:
 *  1. OMP: angle loop parallelized, per-thread accumulation buffer avoids atomics
 *  2. Angle outer / voxels inner: each thread streams one full proj slice → cache
 *  3. x_sin/x_cos hoisted out of j/k loops
 *  4. 1/U multiply instead of division
 */
void bp_cpu(const float *proj, float *volume, const CBpara *p)
{
    int Nxz = p->Volumen_num_xz;
    int Ny   = p->Volumen_num_y;
    int W    = p->detector_width;
    int H    = p->detector_height;
    float SOD  = (float)p->SOD;
    float SDD  = (float)p->SDD;
    float vs   = (float)p->voxelSize;
    float px   = (float)p->pixelSize;
    int   np   = p->num_projs;
    float sod2 = SOD * SOD;

    float radius   = Nxz * 0.5f - 0.5f;
    float radius_z = Ny  * 0.5f - 0.5f;
    size_t vol_n   = (size_t)Nxz * Nxz * Ny;

    /* Precompute LUT: a_min and step for detector pixel → ai conversion */
    /* proj slice layout [W][H]: row iw col ih */
    /* a_min = a[0] = -(0 - (W-1)/2)*px, step = px (positive direction) */
    float a_min = -(0.f - (W-1)*0.5f) * px;  /* = (W-1)/2 * px */
    float da    = -px;                          /* a decreases as iw increases */
    float b_min =  (0.f - (H-1)*0.5f) * px;
    float db    =  px;

    /* Precompute cos/sin LUT once — avoids trig per voxel */
    float *ca_lut = (float *)malloc(np * sizeof(float));
    float *sa_lut = (float *)malloc(np * sizeof(float));
    for (int ip = 0; ip < np; ip++) {
        ca_lut[ip] = cosf((float)p->angles[ip]);
        sa_lut[ip] = sinf((float)p->angles[ip]);
    }

    float scale = (float)M_PI / (float)np;
    memset(volume, 0, vol_n * sizeof(float));

    /*
     * Parallelize over (ix, iy) voxel pairs.
     * Each (ix,iy) pair writes to a unique strip of the output volume
     * → no race conditions, no per-thread buffers, no reduction step.
     * Inner angle loop reads projection slices sequentially → cache friendly.
     */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ix = 0; ix < Nxz; ix++) {
        for (int iy = 0; iy < Nxz; iy++) {
            float xpr = ((float)ix - radius) * vs;
            float ypr = ((float)iy - radius) * vs;

            /* Accumulate contribution from all angles into a local array */
            float *strip = volume + VOL_IDX(Nxz-1-ix, Nxz-1-iy, 0, Nxz, Ny);

            for (int ip = 0; ip < np; ip++) {
                float ca   = ca_lut[ip], sa = sa_lut[ip];
                float t    = ypr*ca - xpr*sa;
                float U    = SOD + ypr*sa + xpr*ca;
                float Uinv = 1.f / U;
                float ai   = SDD * t * Uinv;
                float w    = sod2 * Uinv * Uinv;
                float zf   = SDD * Uinv;

                /* Whole-point zero rule matching the Python reference:
                 * both scipy.interpolate.RegularGridInterpolator(fill_value=0)
                 * (bp) and scipy.ndimage.map_coordinates(mode='constant',
                 * cval=0) (fp) return 0 for the ENTIRE query point the
                 * instant its interpolation cell touches outside the grid —
                 * verified empirically, they do not blend/zero-pad individual
                 * taps. GPU kernels currently zero-pad per-tap instead; that
                 * mismatch (not this CPU code) is the actual source of the
                 * edge-voxel outliers seen in validate.py. */
                float uf = (ai - a_min) / da;
                if (uf < 0.f || uf >= (float)(W-1)) continue;
                int u0 = (int)uf; float du = uf - u0;

                const float *slice = proj + ip * W * H;

                /* vf is affine in iz: vf(iz) = ((iz - radius_z)*vs*zf - b_min)/db.
                 * Solve the valid range [iz_lo, iz_hi) analytically instead of
                 * checking vf<0.f||vf>=(H-1) every iteration — at 512^3 (Ny=512)
                 * most (ip,ix,iy) combinations only have a fraction of iz in
                 * bounds (cone-beam geometry), so this skips the zpr/bi/vf
                 * computation entirely for iz outside the valid range rather
                 * than computing it and then discarding. */
                float vf_slope = vs * zf / db;
                float vf0      = (((0.f - radius_z) * vs * zf) - b_min) / db;
                int iz_lo, iz_hi;
                if (vf_slope > 1e-12f || vf_slope < -1e-12f) {
                    /* vf(iz) = vf0 + iz*vf_slope; solve 0 <= vf < H-1 */
                    float lo = -vf0 / vf_slope;
                    float hi = ((float)(H-1) - vf0) / vf_slope;
                    if (lo > hi) { float tmp = lo; lo = hi; hi = tmp; }
                    iz_lo = (int)ceilf(lo);
                    iz_hi = (int)ceilf(hi);       /* exclusive upper bound: vf<H-1 strictly */
                    if (iz_lo < 0)   iz_lo = 0;
                    if (iz_hi > Ny)  iz_hi = Ny;
                } else {
                    /* zf ~ 0: vf constant across all iz — either all in or all out */
                    int all_in = (vf0 >= 0.f && vf0 < (float)(H-1));
                    iz_lo = 0;
                    iz_hi = all_in ? Ny : 0;
                }

                for (int iz = iz_lo; iz < iz_hi; iz++) {
                    float zpr = ((float)iz - radius_z) * vs;
                    float bi  = zpr * zf;
                    float vf  = (bi - b_min) / db;
                    /* keep the check as a safety net against the boundary
                     * rounding in ceilf above — cost is negligible since the
                     * range is now tight */
                    if (vf < 0.f || vf >= (float)(H-1)) continue;
                    int v0 = (int)vf; float dv = vf - v0;

                    float val = (slice[u0*H+v0]   *(1-du) + slice[(u0+1)*H+v0]   *du)*(1-dv)
                              + (slice[u0*H+v0+1] *(1-du) + slice[(u0+1)*H+v0+1] *du)*dv;

                    strip[iz] += val * w;
                }
            }

            /* Scale strip in-place */
            for (int iz = 0; iz < Ny; iz++)
                strip[iz] *= scale;
        }
    }

    free(ca_lut); free(sa_lut);
}

/* ── forward projection ────────────────────────────────────────────────── */
/* 3×3 rotation + translation from angle (same as Python angle2pose) */
static void angle2pose(float SOD, float angle, float R[3][3], float T[3])
{
    float phi1 = -(float)M_PI / 2.f;
    float phi2 =  (float)M_PI / 2.f;

    /* R1: rotation about x by phi1 */
    float R1[3][3] = {
        {1,0,0},
        {0, cosf(phi1), -sinf(phi1)},
        {0, sinf(phi1),  cosf(phi1)}
    };
    /* R2: rotation about z by phi2 */
    float R2[3][3] = {
        { cosf(phi2), -sinf(phi2), 0},
        { sinf(phi2),  cosf(phi2), 0},
        {0,0,1}
    };
    /* R3: rotation about z by angle */
    float R3[3][3] = {
        { cosf(angle), -sinf(angle), 0},
        { sinf(angle),  cosf(angle), 0},
        {0,0,1}
    };
    /* rot = R3 @ R2 @ R1 */
    float tmp[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        tmp[i][j] = 0;
        for (int k=0;k<3;k++) tmp[i][j] += R3[i][k]*R2[k][j];
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        R[i][j] = 0;
        for (int k=0;k<3;k++) R[i][j] += tmp[i][k]*R1[k][j];
    }
    T[0] = SOD * cosf(angle);
    T[1] = SOD * sinf(angle);
    T[2] = 0.f;
}

void fp_cpu(const float *volume, float *proj, const CBpara *p)
{
    int Nxz = p->Volumen_num_xz;
    int Ny   = p->Volumen_num_y;
    int W    = p->detector_width;
    int H    = p->detector_height;
    float SOD = (float)p->SOD;
    float SDD = (float)p->SDD;
    float vs  = (float)p->voxelSize;
    float pxsz = (float)p->pixelSize;
    int   np  = p->num_projs;

    float sVoxel_xz = Nxz * vs;
    float sVoxel_y  = Ny  * vs;
    float dist_max  = sVoxel_xz * 1.42f;
    float near_t    = fmaxf(0.f, SOD - dist_max);
    float far_t     = fminf(SOD * 2.f, SOD + dist_max);
    int   n_samples = p->n_samples;
    float dt        = (far_t - near_t) / (float)(n_samples - 1);

    /* perf-v2 Phase A6: histogram actual AABB-clipped sample range instead
     * of assuming it -- see the dump site below for details. Read once. */
    int diag_aabb_range = 0;
    {
        const char *dr_env = getenv("FP_CPU_DIAG_AABB_RANGE");
        if (dr_env && atoi(dr_env) != 0) diag_aabb_range = 1;
    }

    /* constants for world→voxel mapping */
    float inv_sv_xz = (float)Nxz / sVoxel_xz;
    float inv_sv_y  = (float)Ny  / sVoxel_y;
    float shift_xz  = 0.5f * Nxz - 0.5f;
    float shift_y   = 0.5f * Ny  - 0.5f;

    /* precompute pose for all angles outside parallel region */
    float (*R_all)[3][3] = (float (*)[3][3])malloc(np * 9 * sizeof(float));
    float (*T_all)[3]    = (float (*)[3])   malloc(np * 3 * sizeof(float));
    for (int ip = 0; ip < np; ip++)
        angle2pose(SOD, (float)p->angles[ip], R_all[ip], T_all[ip]);

    memset(proj, 0, (size_t)np * H * W * sizeof(float));

    /* collapse(2): ip × iu gives np×W independent work items.
     * guided schedule: ray cost varies a lot per column (edge rays traverse
     * mostly empty space or get AABB-clipped short, center rays hit the
     * full volume) — static chunks leave threads idle waiting on the
     * slowest chunk. guided starts with large chunks (low scheduling
     * overhead while imbalance is still being discovered) and shrinks
     * toward the end (fine-grained load balancing for the tail) — cheaper
     * than dynamic's constant re-dispatch overhead at large np*W (75*1120
     * =84000 work items at 512^3, vs 75*512=38400 at 256^3). */
    /* Tile over iv: neighboring detector rows for the same (ip,iu) column
     * have rays that start near each other and diverge only slightly, so at
     * a given sample index s their volume-read addresses cluster. Batching
     * TILE rays together and iterating s in the outer loop (instead of one
     * ray fully marched at a time) means the memory subsystem sees nearby
     * reads close together in time instead of scattered across a full ray
     * march per pixel — improves cache/prefetcher reuse for the 8-tap
     * trilinear gather, which dominates fp_cpu's cost at 512^3 (8 reads x
     * up to 512 samples x 1.3M pixels x 75 angles, each gather spanning up
     * to ~2MB due to the Nxz*Ny and Ny strides). Correctness is unchanged:
     * each ray still computes its own s_start/s_end and does its own
     * per-sample bounds check exactly as before — this only reorders when
     * each ray's reads happen relative to its neighbors', not what is read
     * or how it's weighted. */
/* FP_TILE swept 4/8/16/32/48/64 at 512^3 (3-epoch runs, fp time only):
 *   4->26.05s(*)  8->16.84s  16->14.72s  32->13.58s  48->13.79s  64->13.64s
 * (*4's result carried contention noise from a preceding GPU sweep on the
 * same machine; treated as unreliable, not re-tested since the trend from
 * 8 onward is already clean and monotonic-then-flat). 32-64 are within
 * noise of each other — curve plateaus at 32, no benefit to going larger,
 * and 32 costs less stack/cache footprint per tile than 48/64. 32 is now
 * the default (was 8, a ~19% win: 16.84s->13.58s fp/epoch at 512^3).
 * Arrays sized to FP_TILE_MAX so FP_TILE_ENV can still override for
 * further testing without touching declarations. */
#define FP_TILE_MAX 64
    int FP_TILE = 32;
    {
        const char *tile_env = getenv("FP_TILE_ENV");
        if (tile_env) {
            int v = atoi(tile_env);
            if (v > 0 && v <= FP_TILE_MAX) FP_TILE = v;
        }
    }
    #pragma omp parallel for collapse(2) schedule(guided, 4)
    for (int ip = 0; ip < np; ip++) {
        for (int iu = 0; iu < W; iu++) {
            float uu = ((float)iu + 0.5f - W * 0.5f) * pxsz;
            float d0 = uu / SDD, d2 = 1.f;
            float (*R)[3] = R_all[ip];
            float *T      = T_all[ip];

            for (int iv0 = 0; iv0 < H; iv0 += FP_TILE) {
                int tile_n = (H - iv0 < FP_TILE) ? (H - iv0) : FP_TILE;

                float step_val[FP_TILE_MAX];
                int   s_start[FP_TILE_MAX], s_end[FP_TILE_MAX];
                float xi[FP_TILE_MAX], yi[FP_TILE_MAX], zi[FP_TILE_MAX];
                float dxi[FP_TILE_MAX], dyi[FP_TILE_MAX], dzi[FP_TILE_MAX];
                float val[FP_TILE_MAX];
                int   tile_s_lo = n_samples, tile_s_hi = 0;

                for (int t = 0; t < tile_n; t++) {
                    int iv = iv0 + t;
                    float vv = ((float)iv + 0.5f - H * 0.5f) * pxsz;
                    float d1 = vv / SDD;

                    float rdx = R[0][0]*d0 + R[0][1]*d1 + R[0][2]*d2;
                    float rdy = R[1][0]*d0 + R[1][1]*d1 + R[1][2]*d2;
                    float rdz = R[2][0]*d0 + R[2][1]*d1 + R[2][2]*d2;

                    float rd_norm = sqrtf(rdx*rdx + rdy*rdy + rdz*rdz);
                    step_val[t] = dt * rd_norm;

                    /* AABB slab clipping: tighten sample range for large
                     * detectors (many edge rays miss the volume entirely).
                     * Same gate (W>512) and math as fp_image.cl/fp_buffer.cl
                     * so CPU does the same work as GPU. */
                    int s0 = 0, s1 = n_samples;
                    if (W > 512) {
                        float hxz = 0.5f * Nxz * vs;
                        float hy  = 0.5f * Ny  * vs;
                        float tmin = near_t, tmax = far_t;
                        if (fabsf(rdx) > 1e-6f) {
                            float t1v = (-hxz - T[0]) / rdx, t2v = (hxz - T[0]) / rdx;
                            if (t1v > t2v) { float tmp=t1v; t1v=t2v; t2v=tmp; }
                            tmin = fmaxf(tmin, t1v); tmax = fminf(tmax, t2v);
                        }
                        /* rdy/T[1] maps to yi via inv_sv_y (Ny axis, see
                         * below) so its half-extent must be hy, not hxz;
                         * rdz/T[2] maps to zi via inv_sv_xz so it needs
                         * hxz, not hy. Was transposed — matching fix in
                         * fp_image.cl and fp_buffer.cl. */
                        if (fabsf(rdy) > 1e-6f) {
                            float t1v = (-hy - T[1]) / rdy, t2v = (hy - T[1]) / rdy;
                            if (t1v > t2v) { float tmp=t1v; t1v=t2v; t2v=tmp; }
                            tmin = fmaxf(tmin, t1v); tmax = fminf(tmax, t2v);
                        }
                        if (fabsf(rdz) > 1e-6f) {
                            float t1v = (-hxz - T[2]) / rdz, t2v = (hxz - T[2]) / rdz;
                            if (t1v > t2v) { float tmp=t1v; t1v=t2v; t2v=tmp; }
                            tmin = fmaxf(tmin, t1v); tmax = fminf(tmax, t2v);
                        }
                        if (tmin >= tmax) {
                            s0 = s1 = 0;  /* empty range: ray misses volume */
                        } else {
                            s0 = (int)fmaxf(0.f,        (tmin - near_t) / dt);
                            s1 = (int)fminf((float)n_samples, (tmax - near_t) / dt + 1.f);
                        }
                    }
                    s_start[t] = s0; s_end[t] = s1;
                    if (s0 < s1) {
                        if (s0 < tile_s_lo) tile_s_lo = s0;
                        if (s1 > tile_s_hi) tile_s_hi = s1;
                    }

                    float t0 = near_t + (float)s0 * dt;
                    float wx0 = T[0] + rdx * t0;
                    float wy0 = T[1] + rdy * t0;
                    float wz0 = T[2] + rdz * t0;

                    xi[t] = wx0 * inv_sv_xz + shift_xz;
                    yi[t] = wy0 * inv_sv_y  + shift_y;
                    zi[t] = wz0 * inv_sv_xz + shift_xz;

                    dxi[t] = (rdx * dt) * inv_sv_xz;
                    dyi[t] = (rdy * dt) * inv_sv_y;
                    dzi[t] = (rdz * dt) * inv_sv_xz;
                    val[t] = 0.f;
                }

                /* Outer loop over the union of the tile's sample ranges;
                 * each ray only accumulates while s is within its own
                 * [s_start,s_end) — same per-ray sample count as before,
                 * just interleaved with its tile neighbors' reads instead
                 * of run sequentially to completion one ray at a time. */
                for (int s = tile_s_lo; s < tile_s_hi; s++) {
                    for (int t = 0; t < tile_n; t++) {
                        if (s < s_start[t] || s >= s_end[t]) continue;

                        float curr_xi = xi[t];
                        float curr_yi = yi[t];
                        float curr_zi = zi[t];

                        /* floorf, not (int) truncation — see fp_cpu bug fix
                         * note: (int)xi truncates toward zero, wrongly
                         * passing the bounds check for xi in (-1,0). */
                        int x0 = (int)floorf(curr_xi), y0 = (int)floorf(curr_yi), z0 = (int)floorf(curr_zi);
                        if ((unsigned)x0 < (unsigned)(Nxz-1) &&
                            (unsigned)y0 < (unsigned)(Ny -1) &&
                            (unsigned)z0 < (unsigned)(Nxz-1)) {
                            float dx = curr_xi - (float)x0, dy = curr_yi - (float)y0, dz = curr_zi - (float)z0;
                            float nx = 1.f-dx, ny_ = 1.f-dy, nz_ = 1.f-dz;
                            int base = x0*(Nxz*Ny) + y0*Ny + z0;
                            int sNy  = Nxz*Ny;
                            val[t] += volume[base]         * nx * ny_ * nz_
                                    + volume[base+sNy]     * dx * ny_ * nz_
                                    + volume[base+Ny]      * nx * dy  * nz_
                                    + volume[base+sNy+Ny]  * dx * dy  * nz_
                                    + volume[base+1]       * nx * ny_ * dz
                                    + volume[base+sNy+1]   * dx * ny_ * dz
                                    + volume[base+Ny+1]    * nx * dy  * dz
                                    + volume[base+sNy+Ny+1]* dx * dy  * dz;
                        }
                        xi[t] = curr_xi + dxi[t];
                        yi[t] = curr_yi + dyi[t];
                        zi[t] = curr_zi + dzi[t];
                    }
                }

                for (int t = 0; t < tile_n; t++)
                    proj[ip*H*W + (iv0+t)*W + iu] = val[t] * step_val[t];

                /* perf-v2 Phase A6 diagnostic: dump (s_end-s_start) per ray
                 * instead of the accumulated value, so the actual AABB
                 * clip tightness can be histogrammed in Python rather than
                 * assumed. Only active with FP_CPU_DIAG_AABB_RANGE=1 and
                 * only meaningful when W>512 (the AABB gate) -- off by
                 * default, zero cost/behavior change otherwise. Use with
                 * --op fp so proj is dumped directly (no MLEM iteration
                 * folded in). */
                if (diag_aabb_range) {
                    for (int t = 0; t < tile_n; t++)
                        proj[ip*H*W + (iv0+t)*W + iu] = (float)(s_end[t] - s_start[t]);
                }
            }
        }
    }
#undef FP_TILE
    free(R_all); free(T_all);
}

/* ── iterative reconstruction loop ────────────────────────────────────── */
void reconstruct_cpu(const float *proj_measured, float *volume,
                     const CBpara *p, int epochs, const char *conv_log)
{
    int Nxz = p->Volumen_num_xz;
    int Ny   = p->Volumen_num_y;
    int W    = p->detector_width;
    int H    = p->detector_height;
    int np   = p->num_projs;

    size_t vol_size  = (size_t)Nxz * Nxz * Ny;
    size_t proj_size = (size_t)np * H * W;   /* [np][H][W] before prep */
    size_t proj_size_t = (size_t)np * W * H; /* [np][W][H] after  prep */

    float *b        = (float *)malloc(proj_size_t * sizeof(float));
    float *ratio    = (float *)malloc(proj_size_t * sizeof(float));
    float *ratio_bp = (float *)malloc(proj_size_t * sizeof(float)); /* prep'd */
    float *bp_ratio = (float *)malloc(vol_size    * sizeof(float));
    float *bp_ones  = (float *)malloc(vol_size    * sizeof(float));

    /* v_prev scratch for --log-convergence's rel_change; unused if conv_log
     * is NULL, so no extra allocation cost in the default (off) path. */
    float *v_prev = conv_log ? (float *)malloc(vol_size * sizeof(float)) : NULL;

    float *ones_raw = (float *)malloc(proj_size * sizeof(float));
    float *ones_p   = (float *)malloc(proj_size_t * sizeof(float));
    for (size_t i = 0; i < proj_size; i++) ones_raw[i] = 1.f;
    cone_weight_cpu(ones_raw, p);
    prep_proj_for_bp(ones_raw, ones_p, np, W, H, (float)p->voxelSize);
    bp_cpu(ones_p, bp_ones, p);
    free(ones_raw); free(ones_p);

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t_ep = get_time_sec();

        if (conv_log) memcpy(v_prev, volume, vol_size * sizeof(float));

        double t0 = get_time_sec();
        fp_cpu(volume, b, p);
        double t1 = get_time_sec();

        for (size_t i = 0; i < proj_size; i++)
            ratio[i] = (b[i] > 1e-3f) ? proj_measured[i] / b[i] : 0.f;

        cone_weight_cpu(ratio, p);
        prep_proj_for_bp(ratio, ratio_bp, np, W, H, (float)p->voxelSize);
        double t2 = get_time_sec();
        bp_cpu(ratio_bp, bp_ratio, p);
        double t3 = get_time_sec();

        for (size_t i = 0; i < vol_size; i++) {
            float denom = bp_ones[i];
            if (denom > 1e-10f)
                volume[i] *= bp_ratio[i] / denom;
        }
        double t_ep_total = get_time_sec() - t_ep;
        printf("  epoch %3d/%d  total=%.2fs  fp=%.2fs  bp=%.2fs\n",
               epoch+1, epochs, t_ep_total, t1-t0, t3-t2);

        if (conv_log)
            log_convergence(conv_log, epoch, t_ep_total,
                             proj_measured, b, proj_size,
                             volume, (epoch == 0) ? NULL : v_prev, vol_size);
    }

    free(b); free(ratio); free(ratio_bp); free(bp_ratio); free(bp_ones);
    if (v_prev) free(v_prev);
}
