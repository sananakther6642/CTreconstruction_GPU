#include "ct_cpu.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── index helpers ─────────────────────────────────────────────────────── */
#define PROJ_IDX(p, h, w, H, W)  ((p)*(H)*(W) + (h)*(W) + (w))
#define VOL_IDX(x, y, z, Nxz, Ny) ((x)*(Nxz)*(Ny) + (y)*(Ny) + (z))

/* ── bilinear interpolation on a W×H image stored as [W][H] (col-major)
   matching Python's indexing='ij' meshgrid ─────────────────────────────── */
static float bilinear(const float *img, int W, int H, float u, float v)
{
    /* img stored [W rows][H cols] → img[iw*H + ih] */
    float uf = u + (W - 1) * 0.5f;
    float vf = v + (H - 1) * 0.5f;

    int u0 = (int)floorf(uf);
    int v0 = (int)floorf(vf);
    float du = uf - u0;
    float dv = vf - v0;

    int u1 = u0 + 1, v1 = v0 + 1;

    float c00 = (u0 >= 0 && u0 < W && v0 >= 0 && v0 < H) ? img[u0*H + v0] : 0.f;
    float c10 = (u1 >= 0 && u1 < W && v0 >= 0 && v0 < H) ? img[u1*H + v0] : 0.f;
    float c01 = (u0 >= 0 && u0 < W && v1 >= 0 && v1 < H) ? img[u0*H + v1] : 0.f;
    float c11 = (u1 >= 0 && u1 < W && v1 >= 0 && v1 < H) ? img[u1*H + v1] : 0.f;

    return c00*(1-du)*(1-dv) + c10*du*(1-dv) + c01*(1-du)*dv + c11*du*dv;
}

/* trilinear interpolation in volume [Nxz][Nxz][Ny] */
static float trilinear(const float *vol, int Nxz, int Ny,
                        float xi, float yi, float zi)
{
    int x0 = (int)floorf(xi), x1 = x0+1;
    int y0 = (int)floorf(yi), y1 = y0+1;
    int z0 = (int)floorf(zi), z1 = z0+1;
    float dx = xi-x0, dy = yi-y0, dz = zi-z0;

#define VGET(x,y,z) ( ((x)>=0&&(x)<Nxz&&(y)>=0&&(y)<Nxz&&(z)>=0&&(z)<Ny) \
                      ? vol[VOL_IDX((x),(y),(z),Nxz,Ny)] : 0.f )
    float c000=VGET(x0,y0,z0), c100=VGET(x1,y0,z0);
    float c010=VGET(x0,y1,z0), c110=VGET(x1,y1,z0);
    float c001=VGET(x0,y0,z1), c101=VGET(x1,y0,z1);
    float c011=VGET(x0,y1,z1), c111=VGET(x1,y1,z1);
#undef VGET
    return c000*(1-dx)*(1-dy)*(1-dz) + c100*dx*(1-dy)*(1-dz)
         + c010*(1-dx)*dy*(1-dz)     + c110*dx*dy*(1-dz)
         + c001*(1-dx)*(1-dy)*dz     + c101*dx*(1-dy)*dz
         + c011*(1-dx)*dy*dz         + c111*dx*dy*dz;
}

/* ── cone weight ───────────────────────────────────────────────────────── */
void cone_weight_cpu(float *proj, const CBpara *p)
{
    int W = p->detector_width, H = p->detector_height;
    float SDD = (float)p->SDD;
    float px  = (float)p->pixelSize;

    for (int ip = 0; ip < p->num_projs; ip++) {
        float *slice = proj + ip * H * W;
        for (int iw = 0; iw < W; iw++) {
            float u = (-(float)(iw - (W-1)/2.0)) * px;   /* matches Python a */
            for (int ih = 0; ih < H; ih++) {
                float v = ((float)(ih - (H-1)/2.0)) * px; /* matches Python b */
                float w = SDD / sqrtf(SDD*SDD + u*u + v*v);
                slice[iw*H + ih] *= w;
            }
        }
    }
}

/* ── backprojection ────────────────────────────────────────────────────── */
void bp_cpu(const float *proj, float *volume, const CBpara *p)
{
    int Nxz = p->Volumen_num_xz;
    int Ny   = p->Volumen_num_y;
    int W    = p->detector_width;
    int H    = p->detector_height;
    float SOD = (float)p->SOD;
    float SDD = (float)p->SDD;
    float vs  = (float)p->voxelSize;
    float px  = (float)p->pixelSize;
    int   np  = p->num_projs;

    float radius    = Nxz / 2.0f - 0.5f;
    float radius_z  = Ny  / 2.0f - 0.5f;

    memset(volume, 0, (size_t)Nxz * Nxz * Ny * sizeof(float));

    for (int ix = 0; ix < Nxz; ix++) {
        float xpr = ((float)ix - radius) * vs;
        for (int iy = 0; iy < Nxz; iy++) {
            float ypr = ((float)iy - radius) * vs;
            for (int iz = 0; iz < Ny; iz++) {
                float zpr = ((float)iz - radius_z) * vs;
                float sum = 0.f;

                for (int ip = 0; ip < np; ip++) {
                    float angle = (float)p->angles[ip];
                    float ca = cosf(angle), sa = sinf(angle);
                    float U  = SOD + ypr*sa + xpr*ca;
                    float t  = ypr*ca - xpr*sa;
                    float ai = SDD * t / U;
                    float bi = zpr * SDD / U;

                    const float *slice = proj + ip * H * W;
                    float val = bilinear(slice, W, H, ai / px, bi / px);
                    sum += val * (SOD*SOD) / (U*U);
                }

                /* Final scale and flip to match Python [::-1,::-1,:] */
                volume[VOL_IDX(Nxz-1-ix, Nxz-1-iy, iz, Nxz, Ny)] =
                    sum * (float)M_PI / (float)np;
            }
        }
        if (ix % 32 == 0) {
            printf("\r  bp_cpu: %d/%d slices done", ix+1, Nxz);
            fflush(stdout);
        }
    }
    printf("\n");
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
    float px  = (float)p->pixelSize;
    int   np  = p->num_projs;

    float sVoxel_xz = Nxz * vs;
    float sVoxel_y  = Ny  * vs;
    float dist_max  = sVoxel_xz * 1.42f;
    float near_t    = fmaxf(0.f, SOD - dist_max);
    float far_t     = fminf(SOD * 2.f, SOD + dist_max);
    int   n_samples = (int)ceilf(Nxz * 2.f);  /* sample_ratio=2 */

    memset(proj, 0, (size_t)np * H * W * sizeof(float));

    for (int ip = 0; ip < np; ip++) {
        float angle = (float)p->angles[ip];
        float R[3][3], T[3];
        angle2pose(SOD, angle, R, T);

        float *slice = proj + ip * H * W;

        for (int iu = 0; iu < W; iu++) {
            for (int iv = 0; iv < H; iv++) {
                float uu = ((float)iu + 0.5f - W*0.5f) * px;
                float vv = ((float)iv + 0.5f - H*0.5f) * px;

                /* ray direction in world space */
                float dirs[3] = {uu/SDD, vv/SDD, 1.f};
                float rd[3];
                for (int k=0;k<3;k++) {
                    rd[k] = R[k][0]*dirs[0] + R[k][1]*dirs[1] + R[k][2]*dirs[2];
                }
                float rd_norm = sqrtf(rd[0]*rd[0]+rd[1]*rd[1]+rd[2]*rd[2]);

                float val = 0.f;
                float dt  = (far_t - near_t) / (float)(n_samples - 1);

                for (int s = 0; s < n_samples; s++) {
                    float t = near_t + s * dt;
                    float pt[3];
                    for (int k=0;k<3;k++) pt[k] = T[k] + rd[k] * t;

                    /* map to voxel indices (same as Python) */
                    float xi = (pt[0] + sVoxel_xz*0.5f) / sVoxel_xz * Nxz - 0.5f;
                    float yi = (pt[1] + sVoxel_y *0.5f) / sVoxel_y  * Ny  - 0.5f;
                    float zi = (pt[2] + sVoxel_xz*0.5f) / sVoxel_xz * Nxz - 0.5f;

                    float density = trilinear(volume, Nxz, Ny, xi, yi, zi);
                    val += density * dt * rd_norm;
                }
                /* proj stored as [H][W] to match Python proj[i][j] */
                slice[iv*W + iu] = val;
            }
        }
        if (ip % 5 == 0) {
            printf("\r  fp_cpu: %d/%d projections done", ip+1, np);
            fflush(stdout);
        }
    }
    printf("\n");
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
    size_t proj_size = (size_t)np * H * W;

    float *b        = (float *)malloc(proj_size * sizeof(float));
    float *ratio    = (float *)malloc(proj_size * sizeof(float));
    float *bp_ratio = (float *)malloc(vol_size  * sizeof(float));
    float *bp_ones  = (float *)malloc(vol_size  * sizeof(float));
    float *ones_p   = (float *)malloc(proj_size * sizeof(float));

    /* Precompute bp(ones) — the sensitivity / normalization volume */
    for (size_t i = 0; i < proj_size; i++) ones_p[i] = 1.f;
    bp_cpu(ones_p, bp_ones, p);

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("[epoch %d/%d]\n", epoch+1, epochs);

        fp_cpu(volume, b, p);

        for (size_t i = 0; i < proj_size; i++)
            ratio[i] = (b[i] != 0.f) ? proj_measured[i] / b[i] : 0.f;

        bp_cpu(ratio, bp_ratio, p);

        for (size_t i = 0; i < vol_size; i++) {
            float denom = bp_ones[i];
            if (denom > 1e-10f)
                volume[i] *= bp_ratio[i] / denom;
        }
    }

    free(b); free(ratio); free(bp_ratio); free(bp_ones); free(ones_p);
}
