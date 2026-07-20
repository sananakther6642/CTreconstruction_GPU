/*
 * fp_image.cl — Forward projection using OpenCL image3D for the volume.
 * Also contains float_to_half utility kernel for half-precision vol_img upload.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/* Convert float volume buffer to half buffer for bandwidth-efficient vol_img upload.
 * vol_img stored as CL_HALF_FLOAT → halves texture bandwidth in fp_image.
 * read_imagef auto-converts half→float on read (no kernel change needed). */
__kernel void float_to_half(__global const float *src, __global half *dst, int n)
{
    int i = get_global_id(0);
    if (i < n) dst[i] = (half)src[i];
}

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
 *
 * R_mats: [num_projs * 9] row-major rotation matrices precomputed on host.
 * T_vecs: [num_projs * 3] translation vectors precomputed on host.
 * Eliminates angle2pose_img() per work-item (cos/sin + 2x matmul per pixel).
 */

__constant sampler_t vol_samp =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE   |
    CLK_FILTER_LINEAR;

__kernel void fp_image(
    __read_only  image3d_t       volume_img,   /* [Nxz][Nxz][Ny] as 3D image */
    __constant   float          *R_mats,       /* [num_projs * 9] row-major R per angle */
    __constant   float          *T_vecs,       /* [num_projs * 3] T per angle */
    __global     float          *proj,
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

    __constant float *R = R_mats + ip * 9;
    __constant float *T = T_vecs + ip * 3;

    float uu = ((float)iu + 0.5f - W * 0.5f) * pixelSize;
    float vv = ((float)iv + 0.5f - H * 0.5f) * pixelSize;

    float dirs[3] = {uu/SDD, vv/SDD, 1.f};
    float rd[3];
    for (int k=0;k<3;k++)
        rd[k] = R[k*3+0]*dirs[0] + R[k*3+1]*dirs[1] + R[k*3+2]*dirs[2];

    float rd_norm  = native_sqrt(rd[0]*rd[0] + rd[1]*rd[1] + rd[2]*rd[2]);
    float step_val = dt * rd_norm;

    float inv_sv_xz = (float)Nxz / sVoxel_xz;
    float inv_sv_y  = (float)Ny  / sVoxel_y;
    float shift_xz  = 0.5f * Nxz - 0.5f;
    float shift_y   = 0.5f * Ny  - 0.5f;

    /* Incremental stepping: start + step each sample, eliminates multiply/sample */
    float ox = T[0] + rd[0] * near_t;
    float oy = T[1] + rd[1] * near_t;
    float oz = T[2] + rd[2] * near_t;
    float dox = rd[0] * dt, doy = rd[1] * dt, doz = rd[2] * dt;
    float wx = ox, wy = oy, wz = oz;

    float val = 0.f;
    for (int s = 0; s < n_samples; s++) {
        float xi = wx * inv_sv_xz + shift_xz;
        float yi = wy * inv_sv_y  + shift_y;
        float zi = wz * inv_sv_xz + shift_xz;

        if (xi >= 0.f && xi < (float)(Nxz - 1) &&
            yi >= 0.f && yi < (float)(Ny  - 1) &&
            zi >= 0.f && zi < (float)(Nxz - 1)) {
            float4 coord = (float4)(zi + 0.5f, yi + 0.5f, xi + 0.5f, 0.f);
            val += read_imagef(volume_img, vol_samp, coord).x;
        }
        wx += dox; wy += doy; wz += doz;
    }
    val *= step_val;

    proj[ip * H * W + iv * W + iu] = val;
}
