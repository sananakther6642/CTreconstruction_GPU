/*
 * fp_buffer.cl — Forward projection kernel using OpenCL buffers.
 *
 * Work-item: one detector pixel (iu, iv) for one projection angle (ip).
 * Ray marching through the volume with trilinear interpolation.
 *
 * Volume layout:  [Nxz][Nxz][Ny]  (x outer, z inner)
 * Proj layout:    [num_projs][H][W]  → slice[iv*W + iu]
 */

/* Build rotation matrix R and translation T from angle (same as Python angle2pose) */
static void angle2pose_cl(float SOD, float angle,
                           float R[3][3], float T[3])
{
    float phi1 = -M_PI_2_F;
    float phi2 =  M_PI_2_F;

    float c1=cos(phi1), s1=sin(phi1);
    float c2=cos(phi2), s2=sin(phi2);
    float ca=cos(angle), sa=sin(angle);

    /* R1 (x-axis rotation phi1) */
    float R1[3][3]={{1,0,0},{0,c1,-s1},{0,s1,c1}};
    /* R2 (z-axis rotation phi2) */
    float R2[3][3]={{c2,-s2,0},{s2,c2,0},{0,0,1}};
    /* R3 (z-axis rotation angle) */
    float R3[3][3]={{ca,-sa,0},{sa,ca,0},{0,0,1}};

    /* tmp = R3 @ R2 */
    float tmp[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        tmp[i][j]=0.f;
        for (int k=0;k<3;k++) tmp[i][j]+=R3[i][k]*R2[k][j];
    }
    /* R = tmp @ R1 */
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        R[i][j]=0.f;
        for (int k=0;k<3;k++) R[i][j]+=tmp[i][k]*R1[k][j];
    }
    T[0]=SOD*ca; T[1]=SOD*sa; T[2]=0.f;
}

static float trilinear_buf(__global const float *vol,
                            int Nxz, int Ny,
                            float xi, float yi, float zi)
{
    int x0=(int)floor(xi), x1=x0+1;
    int y0=(int)floor(yi), y1=y0+1;
    int z0=(int)floor(zi), z1=z0+1;
    float dx=xi-x0, dy=yi-y0, dz=zi-z0;

#define VGET(x,y,z) (((x)>=0&&(x)<Nxz&&(y)>=0&&(y)<Nxz&&(z)>=0&&(z)<Ny) \
                     ? vol[(x)*Nxz*Ny+(y)*Ny+(z)] : 0.f)
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

__kernel void fp_buffer(
    __global const float *volume,   /* [Nxz * Nxz * Ny] */
    __global const float *angles,   /* [num_projs] */
    __global       float *proj,     /* [num_projs * H * W] output */
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
    int iu = get_global_id(0);  /* detector width  [0, W) */
    int iv = get_global_id(1);  /* detector height [0, H) */
    int ip = get_global_id(2);  /* projection index */

    if (iu >= W || iv >= H || ip >= num_projs) return;

    float sVoxel_xz = Nxz * voxelSize;
    float sVoxel_y  = Ny  * voxelSize;
    float dist_max  = sVoxel_xz * 1.42f;
    float near_t    = fmax(0.f, SOD - dist_max);
    float far_t     = fmin(SOD * 2.f, SOD + dist_max);
    float dt        = (far_t - near_t) / (float)(n_samples - 1);

    float angle = angles[ip];
    float R[3][3], T[3];
    angle2pose_cl(SOD, angle, R, T);

    float uu = ((float)iu + 0.5f - W * 0.5f) * pixelSize;
    float vv = ((float)iv + 0.5f - H * 0.5f) * pixelSize;

    float dirs[3] = {uu/SDD, vv/SDD, 1.f};
    float rd[3];
    for (int k=0;k<3;k++)
        rd[k] = R[k][0]*dirs[0] + R[k][1]*dirs[1] + R[k][2]*dirs[2];

    float rd_norm = sqrt(rd[0]*rd[0] + rd[1]*rd[1] + rd[2]*rd[2]);

    float val = 0.f;
    for (int s = 0; s < n_samples; s++) {
        float t = near_t + s * dt;
        float pt0 = T[0] + rd[0]*t;
        float pt1 = T[1] + rd[1]*t;
        float pt2 = T[2] + rd[2]*t;

        float xi = (pt0 + sVoxel_xz*0.5f) / sVoxel_xz * Nxz - 0.5f;
        float yi = (pt1 + sVoxel_y *0.5f) / sVoxel_y  * Ny  - 0.5f;
        float zi = (pt2 + sVoxel_xz*0.5f) / sVoxel_xz * Nxz - 0.5f;

        /* Match CPU: only sample when all 8 trilinear neighbors are in-volume. */
        if (xi >= 0.f && xi < (float)(Nxz - 1) &&
            yi >= 0.f && yi < (float)(Ny  - 1) &&
            zi >= 0.f && zi < (float)(Nxz - 1)) {
            val += trilinear_buf(volume, Nxz, Ny, xi, yi, zi) * dt * rd_norm;
        }
    }

    /* proj stored [ip][iv][iu] matching Python proj[i][j] */
    proj[ip * H * W + iv * W + iu] = val;
}
