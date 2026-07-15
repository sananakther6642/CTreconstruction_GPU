"""
Run Python reference reconstruction and save output for MSE validation.
Uses fixed random seed for reproducibility. 10 epochs only (Python fp is slow ~minutes/epoch).
"""
import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
import h5py, sys, time

np.random.seed(42)

# ── copy of reference functions (unmodified) ──────────────────────────────────

def _get_pixelposition(width):
    return np.arange(width) - (width - 1) / 2

def bp_func(proj, cb_para):
    proj = proj[:, ::-1, :].transpose(0, 2, 1)
    angles = cb_para['angles']
    Volumen_num_xz = cb_para['Volumen_num_xz']
    Volumen_num_y  = cb_para['Volumen_num_y']
    voxelSize  = cb_para['voxelSize']
    SDD        = cb_para['SDD']
    SOD        = cb_para['SOD']
    pixelSize  = cb_para['pixelSize']
    num_projs  = angles.shape[0]
    proj /= voxelSize
    a = -_get_pixelposition(cb_para['detector_width'])  * pixelSize
    b =  _get_pixelposition(cb_para['detector_height']) * pixelSize
    xu, yu = np.meshgrid(a, b, indexing='ij')
    cone_weight = SDD / np.sqrt(SDD**2 + xu**2 + yu**2)
    for i in range(num_projs):
        proj[i] *= cone_weight
    reconstructed = np.zeros((Volumen_num_xz, Volumen_num_xz, Volumen_num_y))
    radius   = Volumen_num_xz // 2 - 0.5
    radius_z = Volumen_num_y  // 2 - 0.5
    xpr, ypr, zpr = np.meshgrid(
        (np.arange(Volumen_num_xz) - radius)   * voxelSize,
        (np.arange(Volumen_num_xz) - radius)   * voxelSize,
        (np.arange(Volumen_num_y)  - radius_z) * voxelSize,
        indexing='ij')
    for one_proj, angle in zip(proj, angles):
        t  = ypr * np.cos(angle) - xpr * np.sin(angle)
        U  = SOD + ypr * np.sin(angle) + np.cos(angle) * xpr
        ai = SDD * t / U
        bi = zpr * SDD / U
        interp = RegularGridInterpolator((a, b), one_proj, bounds_error=False, fill_value=0)
        reconstructed += interp((ai, bi)) * (SOD**2) / (U**2)
    reconstructed = reconstructed * np.pi / num_projs
    return np.array(reconstructed, dtype=np.float32)[::-1, ::-1, :]

def angle2pose(SOD, angle):
    phi1 = -np.pi / 2
    R1 = np.array([[1,0,0],[0,np.cos(phi1),-np.sin(phi1)],[0,np.sin(phi1),np.cos(phi1)]])
    phi2 = np.pi / 2
    R2 = np.array([[np.cos(phi2),-np.sin(phi2),0],[np.sin(phi2),np.cos(phi2),0],[0,0,1]])
    R3 = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([SOD * np.cos(angle), SOD * np.sin(angle), 0])
    T = np.eye(4); T[:-1,:-1] = rot; T[:-1,-1] = trans
    return T

def fp_func(cb_para, volume, sample_ratio=2):
    """Vectorized fp_func — same math as reference but no Python inner loops.
    Processes all pixels for one angle at once via batched map_coordinates.
    Memory: ~2 GB peak for 256^3 volume, 512x512 detector, 512 samples."""
    angles    = cb_para['angles']
    H, W      = cb_para['detector_height'], cb_para['detector_width']
    SDD, SOD  = cb_para['SDD'], cb_para['SOD']
    pixelSize = cb_para['pixelSize']
    voxelSize = cb_para['voxelSize']
    nVoxel_xz = cb_para['Volumen_num_xz']
    nVoxel_y  = cb_para['Volumen_num_y']
    sVoxel_xz = nVoxel_xz * voxelSize
    sVoxel_y  = nVoxel_y  * voxelSize
    nVoxel = np.array([nVoxel_xz, nVoxel_y, nVoxel_xz], dtype=np.float32)
    sVoxel = np.array([sVoxel_xz, sVoxel_y, sVoxel_xz], dtype=np.float32)

    dist_max  = sVoxel_xz * 1.42
    near      = max(0.0, SOD - dist_max)
    far       = min(SOD * 2.0, SOD + dist_max)
    n_samples = math.ceil(nVoxel_xz * sample_ratio)

    # detector pixel grid [W, H]
    ii, jj = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32), indexing='ij')
    uu = (ii + 0.5 - W / 2) * pixelSize   # [W, H]
    vv = (jj + 0.5 - H / 2) * pixelSize

    projs = []
    for angle in angles:
        pose  = np.array(angle2pose(SOD, angle), dtype=np.float32)
        R     = pose[:3, :3]   # [3,3]
        trans = pose[:3, -1]   # [3]

        # ray directions: [W, H, 3]
        dirs   = np.stack([uu / SDD, vv / SDD, np.ones_like(uu)], axis=-1)
        rays_d = (R @ dirs.reshape(-1, 3).T).T.reshape(W, H, 3).astype(np.float32)
        rays_o = trans  # broadcast [3]

        # sample t values with jitter [n_samples]
        t_vals = np.linspace(0., 1., n_samples, dtype=np.float32)
        z_vals = near * (1. - t_vals) + far * t_vals
        mids   = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper  = np.concatenate([mids, z_vals[-1:]])
        lower  = np.concatenate([z_vals[:1], mids])
        z_vals = lower + (upper - lower) * np.random.rand(n_samples).astype(np.float32)

        # dists [n_samples]: same for all pixels (rays_d norm varies but applied below)
        dists_t = np.concatenate([z_vals[1:] - z_vals[:-1],
                                   np.array([1e-10], dtype=np.float32)])  # [n_samples]

        rd_norm = np.linalg.norm(rays_d, axis=-1)  # [W, H]
        proj    = np.zeros((W, H), dtype=np.float32)

        # process in strips of 32 columns to cap memory at ~200 MB/strip
        strip = 32
        for w0 in range(0, W, strip):
            w1 = min(w0 + strip, W)
            rd_s = rays_d[w0:w1]                      # [strip, H, 3]
            rn_s = rd_norm[w0:w1]                     # [strip, H]
            uu_s = uu[w0:w1]                          # [strip, H]

            pts = rays_o + rd_s[:, :, None, :] * z_vals[None, None, :, None]
            pts_idx = (pts + sVoxel / 2) / sVoxel * nVoxel - 0.5
            flat = pts_idx.reshape(-1, 3).T

            raw = map_coordinates(volume,
                                  [flat[0], flat[1], flat[2]],
                                  order=1, mode='constant', cval=0.0
                                  ).reshape(w1-w0, H, n_samples).astype(np.float32)

            proj[w0:w1] = np.sum(raw * dists_t[None, None, :] * rn_s[:, :, None], axis=-1)
            del pts, pts_idx, flat, raw

        projs.append(proj.T)  # [H, W]

    return np.stack(projs, axis=0)  # [num_projs, H, W]

# ── main ──────────────────────────────────────────────────────────────────────

path_data = '/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5'
out_path  = 'output_python.hdf5'
EPOCHS    = 1    # 1 epoch sufficient for MSE validation vs C/GPU

print(f"Loading {path_data} ...")
with h5py.File(path_data, 'r') as f:
    voxelSize      = f['voxelSize'][()]
    Volumen_num_xz = int(f['Volumen_num_xz'][()])
    Volumen_num_y  = int(f['Volumen_num_y'][()])
    SDD            = f['SDD'][()]
    SOD            = f['SOD'][()]
    pixelSize      = f['pixelSize'][()]
    num_projs      = int(f['num_projs'][()])
    detector_width = int(f['detector_width'][()])
    detector_height= int(f['detector_height'][()])
    angles         = f['Angle'][()]
    projection_0   = f['Projection'][:,:,:]

cb_para = {
    'angles': angles, 'pixelSize': pixelSize, 'voxelSize': voxelSize,
    'Volumen_num_xz': Volumen_num_xz, 'Volumen_num_y': Volumen_num_y,
    'SDD': SDD, 'SOD': SOD,
    'detector_width': detector_width, 'detector_height': detector_height,
}

v0 = np.ones((Volumen_num_xz, Volumen_num_xz, Volumen_num_xz), dtype=np.float32)

print("Running bp_func only (1 call on all-ones) to validate bp_func vs C bp_cpu ...")
t0 = time.time()
bp_ones = bp_func(np.ones_like(projection_0), cb_para)
print(f"  bp(ones) done in {time.time()-t0:.1f}s  min={bp_ones.min():.4f} max={bp_ones.max():.4f}")

print("Running bp_func on measured projections (1 call) ...")
t0 = time.time()
bp_meas = bp_func(projection_0.copy(), cb_para)
print(f"  bp(p0)  done in {time.time()-t0:.1f}s  min={bp_meas.min():.4f} max={bp_meas.max():.4f}")

# one MLEM step without fp: v0 *= bp(p0) / bp(ones)  — tests bp correctness
v0 *= bp_meas / np.where(bp_ones > 1e-10, bp_ones, 1.0)
print(f"  v0 after 1 bp-only update: min={v0.min():.4f} max={v0.max():.4f}")

print(f"Saving to {out_path} ...")
with h5py.File(out_path, 'w') as f:
    f.create_dataset('voxelSize', data=voxelSize)
    f.create_dataset('Volume',    data=v0)

print("Done.")
