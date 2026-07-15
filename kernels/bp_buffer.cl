/*
 * bp_buffer.cl — Backprojection kernel using OpenCL buffers.
 *
 * Each work-item processes one voxel (ix, iy, iz).
 * Inner loop iterates over all projection angles.
 * Bilinear interpolation done manually (no texture hardware).
 *
 * Volume layout:  [Nxz][Nxz][Ny]  → flat index ix*Nxz*Ny + iy*Ny + iz
 * Proj layout:    [num_projs][W][H] → flat index ip*W*H + iw*H + ih
 *   (W = detector_width, H = detector_height, col-major within slice)
 */

static float bilinear_buf(__global const float *img,
                          int W, int H,
                          float uf, float vf)
{
    /* uf, vf: pixel-space coords centered at 0 */
    float u = uf + (W - 1) * 0.5f;
    float v = vf + (H - 1) * 0.5f;

    int u0 = (int)floor(u), u1 = u0 + 1;
    int v0 = (int)floor(v), v1 = v0 + 1;
    float du = u - u0, dv = v - v0;

    float c00 = (u0>=0&&u0<W&&v0>=0&&v0<H) ? img[u0*H+v0] : 0.f;
    float c10 = (u1>=0&&u1<W&&v0>=0&&v0<H) ? img[u1*H+v0] : 0.f;
    float c01 = (u0>=0&&u0<W&&v1>=0&&v1<H) ? img[u0*H+v1] : 0.f;
    float c11 = (u1>=0&&u1<W&&v1>=0&&v1<H) ? img[u1*H+v1] : 0.f;

    return c00*(1-du)*(1-dv) + c10*du*(1-dv) + c01*(1-du)*dv + c11*du*dv;
}

__kernel void bp_buffer(
    __global const float *proj,       /* [num_projs * W * H] cone-weighted */
    __global const float *angles,     /* [num_projs] radians */
    __global       float *volume,     /* [Nxz * Nxz * Ny] output */
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

    if (ix >= Nxz || iy >= Nxz || iz >= Ny) return;

    float radius   = Nxz * 0.5f - 0.5f;
    float radius_z = Ny  * 0.5f - 0.5f;

    float xpr = ((float)ix - radius)   * voxelSize;
    float ypr = ((float)iy - radius)   * voxelSize;
    float zpr = ((float)iz - radius_z) * voxelSize;

    float sum = 0.f;

    for (int ip = 0; ip < num_projs; ip++) {
        float angle = angles[ip];
        float ca = cos(angle), sa = sin(angle);

        float U  = SOD + ypr*sa + xpr*ca;
        float t  = ypr*ca - xpr*sa;
        float ai = SDD * t / U;
        float bi = zpr * SDD / U;

        /* ai/bi are in mm; match CPU/Python detector-u sign convention */
        float uf = -ai / pixelSize;
        float vf = bi / pixelSize;

        __global const float *slice = proj + ip * W * H;
        float val = bilinear_buf(slice, W, H, uf, vf);
        sum += val * (SOD*SOD) / (U*U);
    }

    sum *= M_PI_F / (float)num_projs;

    /* Python output has [::-1, ::-1, :] flip */
    int out_ix = Nxz - 1 - ix;
    int out_iy = Nxz - 1 - iy;
    volume[out_ix * Nxz * Ny + out_iy * Ny + iz] = sum;
}

/*
 * preprocess_proj — mirrors Python's bp_func first line:
 *   proj[:, ::-1, :].transpose(0,2,1)  then  /= voxelSize
 *
 * src: [num_projs * H * W]  (row-major: proj[ip][ih][iw])
 * dst: [num_projs * W * H]  (col-major: proj[ip][iw][ih])
 *
 * Work-item: one element (ip, iw, ih) in dst space.
 */
__kernel void preprocess_proj(
    __global const float *src,
    __global       float *dst,
    int   W,
    int   H,
    float voxelSize
)
{
    int iw = get_global_id(0);
    int ih = get_global_id(1);
    int ip = get_global_id(2);
    if (iw >= W || ih >= H) return;

    /* flip H-axis: src row = H-1-ih; src is [H][W] per slice */
    float val = src[ip * H * W + (H - 1 - ih) * W + iw] / voxelSize;
    /* transpose: dst is [W][H] per slice */
    dst[ip * W * H + iw * H + ih] = val;
}

/* ── Cone-weight kernel (apply once before iteration loop) ──────────────
 * proj: [num_projs * W * H], modified in-place
 * Work-item: one pixel (ip, iw, ih)
 */
__kernel void cone_weight(
    __global float *proj,
    int   W,
    int   H,
    float SDD,
    float pixelSize
)
{
    int iw = get_global_id(0);  /* detector width index */
    int ih = get_global_id(1);  /* detector height index */
    int ip = get_global_id(2);  /* projection index */

    if (iw >= W || ih >= H) return;

    float u = (-(float)(iw - (W-1)*0.5f)) * pixelSize;
    float v = (  (float)(ih - (H-1)*0.5f)) * pixelSize;
    float w = SDD / sqrt(SDD*SDD + u*u + v*v);

    proj[ip * W * H + iw * H + ih] *= w;
}

/*
 * cone_weight_hw — cone weight for raw [np][H][W] layout (row-major).
 * Same formula as cone_weight but index order matches HDF5-loaded data.
 */
__kernel void cone_weight_hw(
    __global float *proj,
    int   W,
    int   H,
    float SDD,
    float pixelSize
)
{
    int iw = get_global_id(0);
    int ih = get_global_id(1);
    int ip = get_global_id(2);
    if (iw >= W || ih >= H) return;
    float u = (-(float)(iw - (W-1)*0.5f)) * pixelSize;
    float v = (  (float)(ih - (H-1)*0.5f)) * pixelSize;
    float w = SDD / sqrt(SDD*SDD + u*u + v*v);
    proj[ip * H * W + ih * W + iw] *= w;
}

/* ── Divide projections element-wise: ratio = p0 / b ──────────────────── */
__kernel void proj_divide(
    __global const float *p0,
    __global const float *b,
    __global       float *ratio,
    int n
)
{
    int i = get_global_id(0);
    if (i >= n) return;
    ratio[i] = (b[i] > 1e-3f) ? p0[i] / b[i] : 0.f;
}

/* ── Update volume: v0 *= bp_ratio / bp_ones ──────────────────────────── */
__kernel void vol_update(
    __global       float *volume,
    __global const float *bp_ratio,
    __global const float *bp_ones,
    int n
)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float denom = bp_ones[i];
    if (denom > 1e-10f)
        volume[i] *= bp_ratio[i] / denom;
}
