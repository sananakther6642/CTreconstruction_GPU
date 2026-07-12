import torch
from torch.utils.cpp_extension import load
import numpy as np
import h5py
import time
import math
import os

# =====================================================================
# LOAD EXTERNAL NATIVE C++ MODULE FILE FROM DISK
# =====================================================================
print(">> Locating and compiling external 'backend.cpp' file...")
backend_cpu = load(
    name="backend_cpu", 
    sources=['backend.cpp'], 
    extra_cflags=['-O3', '-fopenmp']
)
print(">> Native C++ Module Compilation Successful!")

# =====================================================================
# HOST GEOMETRY UTILITIES
# =====================================================================
def angle2pose(SOD, angle):
    phi1, phi2 = -np.pi / 2, np.pi / 2
    R1 = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi1), -np.sin(phi1)], [0.0, np.sin(phi1), np.cos(phi1)]])
    R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0], [np.sin(phi2), np.cos(phi2), 0.0], [0.0, 0.0, 1.0]])
    R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:-1, :-1] = np.dot(np.dot(R3, R2), R1)
    T[:-1, -1] = np.array([SOD * np.cos(angle), SOD * np.sin(angle), 0.0])
    return T

def _get_pixelposition(width):
    return np.arange(width) - (width - 1) / 2

# =====================================================================
# PYTHON INTERFACE WRAPPERS FOR INTERACTIVE RECONSTRUCTION
# =====================================================================
def fp_func(cb_para, volume_tensor, sample_ratio=2):
    angles = cb_para['angles']
    H, W = cb_para['detector_height'], cb_para['detector_width']
    SDD, SOD = cb_para['SDD'], cb_para['SOD']
    pixelSize, voxelSize = cb_para['pixelSize'], cb_para['voxelSize']
    nVoxel_xz, nVoxel_y = cb_para['Volumen_num_xz'], cb_para['Volumen_num_y']
    n_samples = math.ceil(nVoxel_xz * sample_ratio)

    sVoxel = torch.tensor([nVoxel_xz * voxelSize, nVoxel_y * voxelSize, nVoxel_xz * voxelSize], dtype=torch.float32)
    nVoxel = torch.tensor([nVoxel_xz, nVoxel_y, nVoxel_xz], dtype=torch.float32)

    rays = []
    i, j = np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H), indexing="ij")
    uu = ((i.T + 0.5 - W / 2) * pixelSize).astype(np.float32)
    vv = ((j.T + 0.5 - H / 2) * pixelSize).astype(np.float32)
    dirs = np.stack([uu / SDD, vv / SDD, np.ones_like(uu)], axis=-1)

    for angle in angles:
        pose = angle2pose(SOD, angle).astype(np.float32)
        rays_d = np.sum(np.matmul(pose[:3, :3], dirs[..., None]), axis=-1)
        rays_o = np.broadcast_to(pose[:3, -1], rays_d.shape)
        rays.append(np.concatenate([rays_o, rays_d], axis=-1))

    rays = torch.from_numpy(np.stack(rays, axis=0))
    dist_max = (nVoxel_xz * voxelSize) * 1.42
    near, far = max(0.0, SOD - dist_max), min(SOD * 2, SOD + dist_max)

    t_vals = torch.linspace(0., 1., n_samples)
    z_vals_base = near * (1. - t_vals) + far * t_vals
    mids = .5 * (z_vals_base[1:] + z_vals_base[:-1])
    upper = torch.cat([mids, z_vals_base[-1:]])
    lower = torch.cat([z_vals_base[:1], mids])

    return backend_cpu.forward_cpp(volume_tensor, rays[..., :3], rays[..., 3:6], lower, upper, sVoxel, nVoxel)

def bp_func(proj_tensor, cb_para):
    angles = torch.from_numpy(cb_para['angles'].astype(np.float32))
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    Volumen_num_xz = cb_para['Volumen_num_xz']
    Volumen_num_y = cb_para['Volumen_num_y']
    voxelSize = cb_para['voxelSize']   
    SDD = cb_para['SDD']    
    SOD = cb_para['SOD'] 

    proj_working = proj_tensor.clone()
    proj_working = torch.flip(proj_working, dims=[1]).permute(0, 2, 1)
    proj_working /= voxelSize 

    a = (-_get_pixelposition(cb_para['detector_width']) * cb_para['pixelSize']).astype(np.float32)
    b = (_get_pixelposition(cb_para['detector_height']) * cb_para['pixelSize']).astype(np.float32)
    
    xu, yu = np.meshgrid(a, b, indexing='ij')
    cone_weight = torch.from_numpy((SDD / np.sqrt(SDD**2 + xu**2 + yu**2)).astype(np.float32))
    
    for i in range(angles.size(0)):   
        proj_working[i] *= cone_weight
       
    reconstructed = backend_cpu.backproject_cpp(
        proj_working, cos_a, sin_a, a[0], b[0], a[1]-a[0], b[1]-b[0],
        Volumen_num_xz, Volumen_num_y, voxelSize, SDD, SOD
    )
  
    reconstructed = torch.flip(reconstructed, dims=[0, 1])
    return reconstructed * (math.pi / angles.size(0))

# =====================================================================
# MAIN RECONSTRUCTION MANAGEMENT LOOP
# =====================================================================
if __name__ == '__main__':
    path_data = '/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5'
    output_path = 'reconstruction_result_cpp.hdf5'

    with h5py.File(path_data, 'r') as f:
        cb_para = {
            'angles': f['Angle'][()], 'pixelSize': f['pixelSize'][()], 'voxelSize': f['voxelSize'][()],  
            'Volumen_num_xz': int(f['Volumen_num_xz'][()]), 'Volumen_num_y': int(f['Volumen_num_y'][()]), 
            'SDD': f['SDD'][()], 'SOD': f['SOD'][()],   
            'detector_width': int(f['detector_width'][()]), 'detector_height': int(f['detector_height'][()])
        }
        projection_0 = torch.from_numpy(f['Projection'][:, :, :]).float()
        v0 = torch.ones((cb_para['Volumen_num_xz'], cb_para['Volumen_num_xz'], cb_para['Volumen_num_xz']), dtype=torch.float32)

        print(">> Initializing reference denominator map computation...")
        t_start = time.time()
        bp_ones = bp_func(torch.ones_like(projection_0), cb_para)
        
        epochs = 20
        print(f">> Starting iterative reconstruction solver loop ({epochs} epochs)...")
        for epoch in range(epochs):
            epoch_start = time.time()
            
            b = fp_func(cb_para, v0, sample_ratio=2) 
            result = torch.where(b != 0, projection_0 / b, torch.zeros_like(projection_0))
            v0 *= (bp_func(result, cb_para) / bp_ones)
            
            print(f"Epoch {epoch+1}/{epochs} completed in {time.time() - epoch_start:.2f}s")
          
        print(f"\n>> Total Processing Loop Duration: {time.time() - t_start:.2f}s")

        # =====================================================================
        # EXPORT DATASET TO DISK
        # =====================================================================
        print(">> Extracting memory layers for structured file write...")
        volume_numpy = v0.cpu().numpy()
        
        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('Volume', data=volume_numpy, dtype='float32')
            f_out.create_dataset('voxelSize', data=cb_para['voxelSize'])
            
        print(f">> Dataset successfully saved to target location:")
        print(f"   {os.path.abspath(output_path)}")
