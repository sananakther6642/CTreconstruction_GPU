/*
 * fp_image.cl — Forward projection using OpenCL image3D for the volume.
 *
 * Volume stored as CL_MEM_OBJECT_IMAGE3D (read-only).
 * Hardware trilinear sampler replaces manual trilinear_buf.
 *
 * Image3D layout matches volume [Nxz][Nxz][Ny]:
 *   width  = Ny   (z dimension)
 *   height = Nxz  (y dimension)
 *   depth  = Nxz  (x dimension)
 * Texel coord = (zi+0.5, yi+0.5, xi+0.5) in texel space.
 */

__constant sampler_t vol_samp =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE   |
    CLK_FILTER_LINEAR;

static void angle2pose_img(float SOD, float angle,
                            float R[3][3], float T[3])
{
    float phi1 = -M_PI_2_F;
    float phi2 =  M_PI_2_F;
    float c1=cos(phi1), s1=sin(phi1);
    float c2=cos(phi2), s2=sin(phi2);
    float ca=cos(angle), sa=sin(angle);

    float R1[3][3]={{1,0,0},{0,c1,-s1},{0,s1,c1}};
    float R2[3][3]={{c2,-s2,0},{s2,c2,0},{0,0,1}};
    float R3[3][3]={{ca,-sa,0},{sa,ca,0},{0,0,1}};

    float tmp[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        tmp[i][j]=0.f;
        for (int k=0;k<3;k++) tmp[i][j]+=R3[i][k]*R2[k][j];
    }
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
        R[i][j]=0.f;
        for (int k=0;k<3;k++) R[i][j]+=tmp[i][k]*R1[k][j];
    }
    T[0]=SOD*ca; T[1]=SOD*sa; T[2]=0.f;
}

__kernel void fp_image(
    __read_only  image3d_t     volume_img,  /* [Nxz][Nxz][Ny] as 3D image */
    __global const float      *angles,
    __global       float      *proj,
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

    float angle = angles[ip];
    float R[3][3], T[3];
    angle2pose_img(SOD, angle, R, T);

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

        /* voxel-space fractional indices */
        float xi = (pt0 + sVoxel_xz*0.5f) / sVoxel_xz * Nxz - 0.5f;
        float yi = (pt1 + sVoxel_y *0.5f) / sVoxel_y  * Ny  - 0.5f;
        float zi = (pt2 + sVoxel_xz*0.5f) / sVoxel_xz * Nxz - 0.5f;

        /*
         * image3d texel space: width=Ny(z), height=Nxz(y), depth=Nxz(x)
         * coord = (zi+0.5, yi+0.5, xi+0.5)
         */
        if (xi > -0.5f && xi < (float)Nxz - 0.5f &&
            yi > -0.5f && yi < (float)Ny   - 0.5f &&
            zi > -0.5f && zi < (float)Nxz - 0.5f) {
            float4 coord = (float4)(zi + 0.5f, yi + 0.5f, xi + 0.5f, 0.f);
            float density = read_imagef(volume_img, vol_samp, coord).x;
            val += density * dt * rd_norm;
        }
    }

    proj[ip * H * W + iv * W + iu] = val;
}
