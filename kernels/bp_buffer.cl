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

    /* Whole-cell zero rule matching the Python reference
     * (scipy RegularGridInterpolator, fill_value=0): if any corner of the
     * interpolation cell is outside the grid, the WHOLE sample is 0 — not
     * a per-tap zero-pad blend. */
    if (u0 < 0 || u1 >= W || v0 < 0 || v1 >= H) return 0.f;

    float c00 = img[u0*H+v0];
    float c10 = img[u1*H+v0];
    float c01 = img[u0*H+v1];
    float c11 = img[u1*H+v1];

    return c00*(1-du)*(1-dv) + c10*du*(1-dv) + c01*(1-du)*dv + c11*du*dv;
}

__kernel void bp_buffer(
    __global const float *proj,       /* [num_projs * W * H] cone-weighted */
    __constant float2    *angle_cs,   /* [num_projs] (.x=cos, .y=sin) */
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
        float2 cs = angle_cs[ip];
        float ca = cs.x, sa = cs.y;

        float U  = SOD + ypr*sa + xpr*ca;
        float inv_U = native_recip(U);
        float t  = ypr*ca - xpr*sa;
        float ai = SDD * t * inv_U;
        float bi = zpr * SDD * inv_U;

        /* ai/bi are in mm; match CPU/Python detector-u sign convention */
        float uf = -ai / pixelSize;
        float vf = bi / pixelSize;

        __global const float *slice = proj + ip * W * H;
        float val = bilinear_buf(slice, W, H, uf, vf);
        sum += val * (SOD*SOD) * inv_U * inv_U;
    }

    sum *= M_PI_F / (float)num_projs;

    /* Python output has [::-1, ::-1, :] flip */
    int out_ix = Nxz - 1 - ix;
    int out_iy = Nxz - 1 - iy;
    volume[out_ix * Nxz * Ny + out_iy * Ny + iz] = sum;
}

/*
 * preprocess_proj — cone_weight fused with flip+transpose+/voxelSize.
 *
 * src: [np][H][W] row-major   dst: [np][W][H] col-major
 * Fuses what was two kernel passes (cone_weight_hw then preprocess_proj).
 */
__kernel void preprocess_proj(
    __global const float *src,
    __global       float *dst,
    int   W,
    int   H,
    float voxelSize,
    float SDD,
    float pixelSize
)
{
    int iw = get_global_id(0);
    int ih = get_global_id(1);
    int ip = get_global_id(2);
    if (iw >= W || ih >= H) return;

    /* cone weight at pixel (iw, ih) */
    float u = (-(float)(iw - (W-1)*0.5f)) * pixelSize;
    float v = (  (float)(ih - (H-1)*0.5f)) * pixelSize;
    float cw = SDD / sqrt(SDD*SDD + u*u + v*v);

    /* flip H-axis, transpose, scale */
    float val = src[ip * H * W + (H - 1 - ih) * W + iw] * cw / voxelSize;
    dst[ip * W * H + iw * H + ih] = val;
}

/*
 * cone_weight_hw — kept for ones_raw path where we can't fuse
 * (ones are filled with 1.0, no src to read through preprocess in one shot
 *  without an extra buffer — kept as in-place pass for that case).
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

/* ── Divide projections element-wise: ratio = p0 / b (float4 vectorized) ─ */
__kernel void proj_divide(
    __global const float *p0,
    __global const float *b,
    __global       float *ratio,
    int n
)
{
    int i4 = get_global_id(0);
    int base = i4 * 4;
    if (base + 3 < n) {
        float4 p = vload4(i4, p0);
        float4 d = vload4(i4, b);
        float4 r;
        r.x = (d.x > 1e-3f) ? p.x / d.x : 0.f;
        r.y = (d.y > 1e-3f) ? p.y / d.y : 0.f;
        r.z = (d.z > 1e-3f) ? p.z / d.z : 0.f;
        r.w = (d.w > 1e-3f) ? p.w / d.w : 0.f;
        vstore4(r, i4, ratio);
    } else {
        /* tail: handle remaining 0-3 elements */
        for (int j = base; j < n; j++)
            ratio[j] = (b[j] > 1e-3f) ? p0[j] / b[j] : 0.f;
    }
}

/* ── Update volume: v0 *= bp_ratio / bp_ones (float4 vectorized) ─────── */
__kernel void vol_update(
    __global       float *volume,
    __global const float *bp_ratio,
    __global const float *bp_ones,
    int n
)
{
    int i4 = get_global_id(0);
    int base = i4 * 4;
    if (base + 3 < n) {
        float4 v  = vload4(i4, volume);
        float4 br = vload4(i4, bp_ratio);
        float4 bo = vload4(i4, bp_ones);
        float4 out;
        out.x = (bo.x > 1e-10f) ? v.x * br.x / bo.x : v.x;
        out.y = (bo.y > 1e-10f) ? v.y * br.y / bo.y : v.y;
        out.z = (bo.z > 1e-10f) ? v.z * br.z / bo.z : v.z;
        out.w = (bo.w > 1e-10f) ? v.w * br.w / bo.w : v.w;
        vstore4(out, i4, volume);
    } else {
        for (int j = base; j < n; j++) {
            float denom = bp_ones[j];
            if (denom > 1e-10f)
                volume[j] *= bp_ratio[j] / denom;
        }
    }
}
