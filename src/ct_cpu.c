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

                float uf = (ai - a_min) / da;
                if (uf < 0.f || uf >= (float)(W-1)) continue;
                int u0 = (int)uf; float du = uf - u0;

                const float *slice = proj + ip * W * H;

                for (int iz = 0; iz < Ny; iz++) {
                    float zpr = ((float)iz - radius_z) * vs;
                    float bi  = zpr * zf;
                    float vf  = (bi - b_min) / db;
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

    /* collapse(2): ip × iu gives np×W independent work items */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ip = 0; ip < np; ip++) {
        for (int iu = 0; iu < W; iu++) {
            float uu = ((float)iu + 0.5f - W * 0.5f) * pxsz;
            float d0 = uu / SDD, d2 = 1.f;
            /* precompute rd components for this column (iv only changes vv) */
            float (*R)[3] = R_all[ip];
            float *T      = T_all[ip];

            for (int iv = 0; iv < H; iv++) {
                float vv = ((float)iv + 0.5f - H * 0.5f) * pxsz;
                float d1 = vv / SDD;

                float rdx = R[0][0]*d0 + R[0][1]*d1 + R[0][2]*d2;
                float rdy = R[1][0]*d0 + R[1][1]*d1 + R[1][2]*d2;
                float rdz = R[2][0]*d0 + R[2][1]*d1 + R[2][2]*d2;

                float rd_norm  = sqrtf(rdx*rdx + rdy*rdy + rdz*rdz);
                float step_val = dt * rd_norm;   /* accumulated outside sample loop */

                /* ray start in world space at near_t */
                float ox = T[0] + rdx * near_t;
                float oy = T[1] + rdy * near_t;
                float oz = T[2] + rdz * near_t;
                /* incremental step — avoids multiply per sample */
                float dox = rdx * dt, doy = rdy * dt, doz = rdz * dt;

                float val = 0.f;
                float wx = ox, wy = oy, wz = oz;

                for (int s = 0; s < n_samples; s++) {
                    float xi = wx * inv_sv_xz + shift_xz;
                    float yi = wy * inv_sv_y  + shift_y;
                    float zi = wz * inv_sv_xz + shift_xz;

                    int x0 = (int)xi, y0 = (int)yi, z0 = (int)zi;
                    /* single bounds check — no branch inside interpolation */
                    if ((unsigned)x0 < (unsigned)(Nxz-1) &&
                        (unsigned)y0 < (unsigned)(Ny -1) &&
                        (unsigned)z0 < (unsigned)(Nxz-1)) {
                        float dx = xi-x0, dy = yi-y0, dz = zi-z0;
                        float nx = 1.f-dx, ny_ = 1.f-dy, nz_ = 1.f-dz;
                        int base = x0*(Nxz*Ny) + y0*Ny + z0;
                        int sNy  = Nxz*Ny;
                        val += volume[base]         * nx * ny_ * nz_
                             + volume[base+sNy]     * dx * ny_ * nz_
                             + volume[base+Ny]      * nx * dy  * nz_
                             + volume[base+sNy+Ny]  * dx * dy  * nz_
                             + volume[base+1]       * nx * ny_ * dz
                             + volume[base+sNy+1]   * dx * ny_ * dz
                             + volume[base+Ny+1]    * nx * dy  * dz
                             + volume[base+sNy+Ny+1]* dx * dy  * dz;
                    }
                    wx += dox; wy += doy; wz += doz;
                }
                proj[ip*H*W + iv*W + iu] = val * step_val;
            }
        }
    }
    free(R_all); free(T_all);
}

/* ── iterative reconstruction loop ────────────────────────────────────── */
void reconstruct_cpu(const float *proj_measured, float *volume,
                     const CBpara *p, int epochs)
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

    float *ones_raw = (float *)malloc(proj_size * sizeof(float));
    float *ones_p   = (float *)malloc(proj_size_t * sizeof(float));
    for (size_t i = 0; i < proj_size; i++) ones_raw[i] = 1.f;
    cone_weight_cpu(ones_raw, p);
    prep_proj_for_bp(ones_raw, ones_p, np, W, H, (float)p->voxelSize);
    bp_cpu(ones_p, bp_ones, p);
    free(ones_raw); free(ones_p);

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t_ep = get_time_sec();

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
        printf("  epoch %3d/%d  total=%.2fs  fp=%.2fs  bp=%.2fs\n",
               epoch+1, epochs, get_time_sec()-t_ep, t1-t0, t3-t2);
    }

    free(b); free(ratio); free(ratio_bp); free(bp_ratio); free(bp_ones);
}
