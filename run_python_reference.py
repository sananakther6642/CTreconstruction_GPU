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
    angles    = cb_para['angles']
    H, W      = cb_para['detector_height'], cb_para['detector_width']
    SDD, SOD  = cb_para['SDD'], cb_para['SOD']
    pixelSize = cb_para['pixelSize']
    voxelSize = cb_para['voxelSize']
    nVoxel_xz = cb_para['Volumen_num_xz']
    nVoxel_y  = cb_para['Volumen_num_y']
    sVoxel_xz = nVoxel_xz * voxelSize
    sVoxel_y  = nVoxel_y  * voxelSize
    nVoxel = np.array([nVoxel_xz, nVoxel_y, nVoxel_xz])
    sVoxel = np.array([sVoxel_xz, sVoxel_y, sVoxel_xz])
    rays = []
    for angle in angles:
        pose = np.array(angle2pose(SOD, angle), dtype=np.float32)
        i, j = np.meshgrid(np.linspace(0,W-1,W), np.linspace(0,H-1,H), indexing='ij')
        uu = (i.T + 0.5 - W/2) * pixelSize
        vv = (j.T + 0.5 - H/2) * pixelSize
        dirs   = np.stack([uu/SDD, vv/SDD, np.ones_like(uu)], axis=-1)
        rays_d = np.sum(np.matmul(pose[:3,:3], dirs[...,None]), axis=-1)
        rays_o = np.broadcast_to(pose[:3,-1], rays_d.shape)
        rays.append(np.concatenate([rays_o, rays_d], axis=-1))
    rays     = np.stack(rays, axis=0)
    dist_max = sVoxel_xz * 1.42
    near     = max(0, SOD - dist_max)
    far      = min(SOD * 2, SOD + dist_max)
    n_samples = math.ceil(nVoxel_xz * sample_ratio)
    projs = []
    for index_proj in range(rays.shape[0]):
        rays_per_proj = rays[index_proj]
        rays_o, rays_d = rays_per_proj[...,:3], rays_per_proj[...,3:6]
        t_vals = np.linspace(0., 1., n_samples)
        z_vals = near * (1. - t_vals) + far * t_vals
        mids   = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper  = np.concatenate([mids, z_vals[...,-1:]], axis=-1)
        lower  = np.concatenate([z_vals[...,:1], mids],  axis=-1)
        t_rand = np.random.rand(*z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
        pts    = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        pts    = (pts + sVoxel / 2) / sVoxel * nVoxel - 0.5
        proj   = np.zeros([H, W])
        for i in range(pts.shape[0]):
            for j in range(pts.shape[1]):
                pts_per_ray = pts[i][j]
                ray_d       = rays_d[i][j]
                raw = map_coordinates(volume,
                    [pts_per_ray[...,0].ravel(),
                     pts_per_ray[...,1].ravel(),
                     pts_per_ray[...,2].ravel()],
                    order=1, mode='constant', cval=0.0)
                dists = z_vals[1:] - z_vals[:-1]
                dists = np.concatenate([dists, np.ones([1])*1e-10], axis=-1)
                dists = dists * np.linalg.norm(ray_d, axis=-1)
                proj[i][j] = np.sum(raw * dists, axis=-1)
        projs.append(proj)
    return np.stack(projs, axis=0)

# ── main ──────────────────────────────────────────────────────────────────────

path_data = '/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5'
out_path  = 'output_python.hdf5'
EPOCHS    = 10   # Python fp_func is very slow; 10 epochs takes ~30-60 min

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

print(f"Running {EPOCHS} epochs (Python reference — slow, ~minutes/epoch) ...")
bp_ones = bp_func(np.ones_like(projection_0), cb_para)

for i in range(EPOCHS):
    t0 = time.time()
    b      = fp_func(cb_para, v0, sample_ratio=2)
    result = np.divide(projection_0, b, out=np.zeros_like(projection_0), where=(b != 0))
    v0    *= bp_func(result, cb_para) / bp_ones
    print(f"  epoch {i+1}/{EPOCHS}  {time.time()-t0:.1f}s  min={v0.min():.4f} max={v0.max():.4f}")

print(f"Saving to {out_path} ...")
with h5py.File(out_path, 'w') as f:
    f.create_dataset('voxelSize', data=voxelSize)
    f.create_dataset('Volume',    data=v0)

print("Done.")
