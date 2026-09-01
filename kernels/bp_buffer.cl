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

/*
 * OSEM: ip_start/ip_count select a contiguous angle
 * subrange instead of always summing all num_projs angles. Callers use
 * ip_start=0, ip_count=num_projs for plain MLEM (--subsets 1, the
 * default) -- identical to the pre-OSEM behavior. M_PI_F/num_projs is
 * deliberately left as num_projs (not ip_count): it appears in both
 * bp(ratio) and bp(ones) and cancels exactly in the v*=bp(ratio)/bp(ones)
 * update regardless of which angle range is summed, so leaving it alone
 * avoids a redundant rescale and keeps this kernel's only change the
 * loop bounds. Requires the angle stack to have been permuted at load
 * time (see utils.c compute_osem_permutation/permute_projections_inplace)
 * so that subset k IS the contiguous range [ip_start, ip_start+ip_count).
 */
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
    float pixelSize,
    int   ip_start,
    int   ip_count
)
{
    /* iz on dim 0 (fastest-varying local id), ix on dim 2 (slowest) --
     * see the host-side comment at run_bp_buffer's lws for why: iz drives
     * vf, the stride-1 direction in `proj`'s [W][H] layout (bilinear_buf's
     * v0/v1 taps), while ix/iy drive uf, the stride-H (2KB) direction. A
     * warp needs iz to be its fastest-varying id so consecutive lanes read
     * near-consecutive vf, not near-consecutive uf. */
    int iz = get_global_id(0);
    int iy = get_global_id(1);
    int ix = get_global_id(2);

    if (ix >= Nxz || iy >= Nxz || iz >= Ny) return;

    float radius   = Nxz * 0.5f - 0.5f;
    float radius_z = Ny  * 0.5f - 0.5f;

    float xpr = ((float)ix - radius)   * voxelSize;
    float ypr = ((float)iy - radius)   * voxelSize;
    float zpr = ((float)iz - radius_z) * voxelSize;

    float sum = 0.f;

    for (int ip = ip_start; ip < ip_start + ip_count; ip++) {
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
 * divide_preprocess_img — fuses three former steps into one kernel:
 *   1. proj_divide:      ratio = p0/b (with the same b>1e-3 guard)
 *   2. preprocess_proj:  cone_weight + flip(H-axis) + transpose + /voxelSize
 *   3. clEnqueueCopyBufferToImage: buffer -> ratio_img
 * into a single pass that reads p0/b and writes straight into an
 * image2d_array_t (width=H, height=W, depth=np).
 *
 * Kernel fusion: the intermediate d_ratio and d_ratio_prep buffers
 * were each write-once/read-once and never used anywhere else -- classic
 * fusable intermediates, plus the buffer->image copy was pure data
 * movement (~0.80GB/epoch at 512^3) the GPU already had the values for.
 * Projection-side traffic per epoch drops from the original three steps'
 * combined ~2.8x proj_bytes to this kernel's 2x (read p0, read b) -- the
 * image write isn't counted as host-visible buffer traffic. Same
 * divide-by-zero guard and same cone-weight/flip/transpose math as the
 * kernels it replaces; only the fusion point moved, no arithmetic changed
 * (see ct_gpu.c's run_divide_preprocess_img for the index-equivalence
 * derivation against the original two-kernel version).
 *
 * Not float4-vectorized like proj_divide was: this kernel is 3D-indexed
 * (iw,ih,ip) to match the flip+transpose, and on GCN a 64-wide wavefront
 * issuing 64 consecutive scalar 4-byte loads along iw coalesces into the
 * same memory transactions vload4 would -- no loss from dropping the
 * explicit vectorization.
 */
/* ip_start/ip_count: OSEM support, mirrors fp_image's pattern -- the host
 * launches with offset[2]=ip_start, gws[2]=ip_count (rounded up to lws),
 * so get_global_id(2) never goes below ip_start; only the upper bound
 * needs guarding here for the lws round-up. num_projs itself is not
 * needed as a separate arg since ip_start+ip_count <= num_projs always
 * (subset ranges partition [0,num_projs) by construction). */
__kernel void divide_preprocess_img(
    __global const float *p0,
    __global const float *b,
    __write_only image2d_array_t dst,
    int   W,
    int   H,
    float voxelSize,
    float SDD,
    float pixelSize,
    int   ip_start,
    int   ip_count
)
{
    int iw = get_global_id(0);
    int ih = get_global_id(1);
    int ip = get_global_id(2);
    if (iw >= W || ih >= H || ip >= ip_start + ip_count) return;

    int src_idx = ip * H * W + (H - 1 - ih) * W + iw;
    float p0v = p0[src_idx];
    float bv  = b[src_idx];
    float ratio = (bv > 1e-3f) ? p0v / bv : 0.f;

    float u = (-(float)(iw - (W-1)*0.5f)) * pixelSize;
    float v = (  (float)(ih - (H-1)*0.5f)) * pixelSize;
    float cw = SDD / sqrt(SDD*SDD + u*u + v*v);

    float val = ratio * cw / voxelSize;
    write_imagef(dst, (int4)(ih, iw, ip, 0), (float4)(val, 0.f, 0.f, 0.f));
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
/* ones_thresh: bp_ones cutoff for the FOV mask (see src/utils.h). Voxels
 * below it are outside the reliably-sampled region and are zeroed, NOT
 * frozen -- the volume starts at 1.0 (main.c), so freezing would leave a
 * bright rim. Host passes 1e-10f when masking is off, reproducing the
 * previous guard exactly. */
__kernel void vol_update(
    __global       float *volume,
    __global const float *bp_ratio,
    __global const float *bp_ones,
    int n,
    float ones_thresh,
    int   mask_on      /* 1 = zero below threshold, 0 = legacy freeze */
)
{
    int i4 = get_global_id(0);
    int base = i4 * 4;
    if (base + 3 < n) {
        float4 v  = vload4(i4, volume);
        float4 br = vload4(i4, bp_ratio);
        float4 bo = vload4(i4, bp_ones);
        float4 out;
        float4 fb = mask_on ? (float4)(0.f,0.f,0.f,0.f) : v;
        out.x = (bo.x > ones_thresh) ? v.x * br.x / bo.x : fb.x;
        out.y = (bo.y > ones_thresh) ? v.y * br.y / bo.y : fb.y;
        out.z = (bo.z > ones_thresh) ? v.z * br.z / bo.z : fb.z;
        out.w = (bo.w > ones_thresh) ? v.w * br.w / bo.w : fb.w;
        vstore4(out, i4, volume);
    } else {
        for (int j = base; j < n; j++) {
            float denom = bp_ones[j];
            volume[j] = (denom > ones_thresh) ? volume[j] * bp_ratio[j] / denom
                                              : (mask_on ? 0.f : volume[j]);
        }
    }
}

/*
 * vol_update_img — same update as vol_update, but also writes the result
 * directly into vol_img (a 3D image), eliminating the separate
 * clEnqueueCopyBufferToImage(d_vol -> vol_img) the epoch loop used to do
 * right before fp_image's next call (~1.07GB/epoch at 512^3: 537MB read +
 * 537MB write). Requires cl_khr_3d_image_writes (OpenCL 1.2 has no 3D
 * read-write images without it) -- confirmed supported on this Hawaii
 * device via a runtime capability check in ct_gpu.c. float32 mode only; --half still uses the
 * separate float_to_half + copy path since half-precision needs an
 * actual format conversion this kernel doesn't do.
 *
 * the vol_img fusion path. Flat buffer index j decomposes as
 * j = x*(Nxz*Ny) + y*Ny + z (matching the existing buffer layout), and
 * the image is (width=Ny/z-axis, height=Nxz/y-axis, depth=Nxz/x-axis) --
 * verified against fp_image.cl's own read_imagef(volume_img, samp,
 * (zi,yi,xi)) coordinate convention. Since Ny (256 or 512) is always a
 * multiple of 4, four consecutive flat indices in one vec4 always share
 * the same (x,y) and differ only in z -- safe to decompose into four
 * scalar write_imagef calls at (z,z+1,z+2,z+3) with the same (y,x).
 */
__kernel void vol_update_img(
    __global       float *volume,
    __global const float *bp_ratio,
    __global const float *bp_ones,
    __write_only image3d_t vol_img,
    int Nxz,
    int Ny,
    int n,
    float ones_thresh,
    int   mask_on      /* 1 = zero below threshold, 0 = legacy freeze */
)
{
    int i4 = get_global_id(0);
    int base = i4 * 4;
    if (base + 3 < n) {
        float4 v  = vload4(i4, volume);
        float4 br = vload4(i4, bp_ratio);
        float4 bo = vload4(i4, bp_ones);
        float4 out;
        float4 fb = mask_on ? (float4)(0.f,0.f,0.f,0.f) : v;
        out.x = (bo.x > ones_thresh) ? v.x * br.x / bo.x : fb.x;
        out.y = (bo.y > ones_thresh) ? v.y * br.y / bo.y : fb.y;
        out.z = (bo.z > ones_thresh) ? v.z * br.z / bo.z : fb.z;
        out.w = (bo.w > ones_thresh) ? v.w * br.w / bo.w : fb.w;
        vstore4(out, i4, volume);

        int x = base / (Nxz * Ny);
        int rem = base % (Nxz * Ny);
        int y = rem / Ny;
        int z = rem % Ny;
        write_imagef(vol_img, (int4)(z,   y, x, 0), (float4)(out.x, 0.f, 0.f, 0.f));
        write_imagef(vol_img, (int4)(z+1, y, x, 0), (float4)(out.y, 0.f, 0.f, 0.f));
        write_imagef(vol_img, (int4)(z+2, y, x, 0), (float4)(out.z, 0.f, 0.f, 0.f));
        write_imagef(vol_img, (int4)(z+3, y, x, 0), (float4)(out.w, 0.f, 0.f, 0.f));
    } else {
        for (int j = base; j < n; j++) {
            float denom = bp_ones[j];
            float val = (denom > ones_thresh) ? volume[j] * bp_ratio[j] / denom
                                              : (mask_on ? 0.f : volume[j]);
            volume[j] = val;
            int x = j / (Nxz * Ny);
            int rem = j % (Nxz * Ny);
            int y = rem / Ny;
            int z = rem % Ny;
            write_imagef(vol_img, (int4)(z, y, x, 0), (float4)(val, 0.f, 0.f, 0.f));
        }
    }
}
