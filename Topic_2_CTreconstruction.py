import numpy as np
import math
from scipy import interpolate
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import h5py

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates



def _get_pixelposition(width):
    return np.arange(width)-(width-1)/2

def bp_func(proj,cb_para):
    
    proj=proj[:, ::-1, :] .transpose(0,2,1)

    angles=cb_para['angles']
    Volumen_num_xz=cb_para['Volumen_num_xz']
    Volumen_num_y=cb_para['Volumen_num_y']
    detector_width=cb_para['detector_width']
    detector_height=cb_para['detector_height']
    voxelSize = cb_para['voxelSize']   
    SDD = cb_para['SDD']    
    SOD= cb_para['SOD'] 
    pixelSize = cb_para['pixelSize'] 

    num_projs=angles.shape[0]

    proj/=voxelSize 
    a=-_get_pixelposition(detector_width)* pixelSize   
    b=_get_pixelposition(detector_height)* pixelSize   
    
    xu,yu = np.meshgrid(a,b, indexing='ij')
    cone_weight=SDD/np.sqrt(SDD**2+xu**2+yu**2)   
    
    for i in range(num_projs):   
        proj[i]*=cone_weight
       
    reconstructed = np.zeros((Volumen_num_xz, Volumen_num_xz,Volumen_num_y))
    radius = Volumen_num_xz // 2-0.5
    radius_z = Volumen_num_y // 2-0.5
    xpr,ypr,zpr = np.meshgrid((np.arange(Volumen_num_xz)-radius)*voxelSize,(np.arange(Volumen_num_xz)-radius)*voxelSize,(np.arange(Volumen_num_y)-radius_z)*voxelSize , indexing='ij')
    
 
    for one_proj, angle in zip(proj, angles):   
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        U=SOD+ypr * np.sin(angle)  + np.cos(angle) *xpr
        ai=SDD*t/U
        bi=zpr*SDD/U
        interpolant= RegularGridInterpolator((a, b), one_proj, bounds_error=False, fill_value=0)
     
        reconstructed += interpolant((ai,bi))*(SOD**2)/(U**2)
  
    reconstructed=reconstructed*np.pi/num_projs

    return np.array(reconstructed,dtype=np.float32)[::-1, ::-1, :]

def angle2pose(SOD, angle):
    phi1 = -np.pi / 2
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi1), -np.sin(phi1)],
            [0.0, np.sin(phi1), np.cos(phi1)],
        ]
    )
    phi2 = np.pi / 2
    R2 = np.array(
        [
            [np.cos(phi2), -np.sin(phi2), 0.0],
            [np.sin(phi2), np.cos(phi2), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R3 = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([SOD * np.cos(angle), SOD * np.sin(angle), 0])
    T = np.eye(4)
    T[:-1, :-1] = rot
    T[:-1, -1] = trans
    return T

def fp_func(cb_para, volume,sample_ratio=2):
    angles = cb_para['angles']

    H = cb_para['detector_height']
    W = cb_para['detector_width']
    SDD = cb_para['SDD']
    SOD = cb_para['SOD']

    pixelSize = cb_para['pixelSize']
    voxelSize = cb_para['voxelSize']

    nVoxel_xz = cb_para['Volumen_num_xz']
    nVoxel_y = cb_para['Volumen_num_y']

    sVoxel_xz = nVoxel_xz * voxelSize
    sVoxel_y = nVoxel_y * voxelSize

    nVoxel = np.array([nVoxel_xz, nVoxel_y, nVoxel_xz])
    sVoxel = np.array([sVoxel_xz, sVoxel_y, sVoxel_xz])

    # get rays for every projection
    rays = []

    for angle in angles:
        pose = np.array(angle2pose(SOD, angle), dtype=np.float32)
        rays_o, rays_d = None, None
        i, j = np.meshgrid(
            np.linspace(0, W - 1, W),
            np.linspace(0, H - 1, H),
            indexing="ij",
        )
        uu = (i.T + 0.5 - W / 2) * pixelSize
        vv = (j.T + 0.5 - H / 2) * pixelSize
        dirs = np.stack([uu / SDD, vv / SDD, np.ones_like(uu)],axis=-1)
        rays_d = np.sum(np.matmul(pose[:3, :3], dirs[..., None]), axis=-1)
        rays_o = np.broadcast_to(pose[:3, -1], rays_d.shape)
        rays.append(np.concatenate([rays_o, rays_d], axis=-1))

    rays = np.stack(rays, axis=0)

    # get the range to sample points from a ray
    dist_max = sVoxel_xz * 1.42
    near = np.max([0, SOD - dist_max])
    far = np.min([SOD * 2, SOD + dist_max])

    # get projections
    n_samples = math.ceil(nVoxel_xz * sample_ratio)
    projs = []

    for index_proj in range(rays.shape[0]):
        rays_per_proj = rays[index_proj]
        rays_o, rays_d= rays_per_proj[...,:3], rays_per_proj[...,3:6]

        t_vals = np.linspace(0., 1., n_samples)
        z_vals = near * (1. - t_vals) + far * (t_vals)

        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = np.concatenate([mids, z_vals[..., -1:]], axis=-1)
        lower = np.concatenate([z_vals[..., :1], mids], axis=-1)
        t_rand = np.random.rand(*z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = (pts + sVoxel / 2) / sVoxel * nVoxel - 0.5

        proj = np.zeros([H, W])
        for i in range(pts.shape[0]):
            for j in range(pts.shape[1]):
                pts_per_ray = pts[i][j]
                ray_d = rays_d[i][j]

                x_idx = pts_per_ray[..., 0]
                y_idx = pts_per_ray[..., 1]
                z_idx = pts_per_ray[..., 2]

                raw = map_coordinates(
                    volume,
                    [x_idx.ravel(), y_idx.ravel(), z_idx.ravel()],
                    order=1,
                    mode="constant",
                    cval=0.0
                )

                dists = z_vals[1:] - z_vals[:-1]

                last_dist = np.ones([1]) * 1e-10
                dists = np.concatenate([dists, last_dist], axis=-1)

                dists = dists * np.linalg.norm(ray_d, axis=-1)

                proj[i][j] = np.sum(raw * dists, axis=-1)

        projs.append(proj)

    return np.stack(projs, axis=0)


if __name__ == '__main__':
    
    ## use this two projection datasets
    path_data='/lgrp/edu-2026-1-gpulab/proj_256_75.hdf5'
    # path_data='/lgrp/edu-2026-1-gpulab/proj_512_75.hdf5'

    with h5py.File(path_data,'r') as f:

        voxelSize=f['voxelSize'][()] 
        Volumen_num_xz=int(f['Volumen_num_xz'][()] )
        Volumen_num_y=int(f['Volumen_num_y'][()] )
        SDD = f['SDD'][()] 
        SOD =f['SOD'][()]  
        magnification=SDD/SOD
        pixelSize =f['pixelSize'][()] 
  
        num_projs=int(f['num_projs'][()] )
        detector_width=int(f['detector_width'][()] )
        detector_height=int(f['detector_height'][()] )
        angles=f['Angle'][()] 
    
        
        cb_para={
            'angles' : angles,
            'pixelSize' :pixelSize ,
            'voxelSize' : voxelSize,  
            'Volumen_num_xz':Volumen_num_xz, 
            'Volumen_num_y':Volumen_num_y, 
            'SDD':  SDD,   
            'SOD' :SOD,   
            'detector_width':detector_width,
            'detector_height':detector_height,
            }

        projection_0=f['Projection'][:,:,:]
        # was (Volumen_num_xz,Volumen_num_xz,Volumen_num_xz) -- z-axis must
        # use Volumen_num_y, matching bp_func's own reconstructed array shape
        # (line 42) and every other axis-size use in this file. Latent on
        # the cubic datasets actually used here, fixed for correctness.
        v0=np.ones((Volumen_num_xz,Volumen_num_xz,Volumen_num_y)).astype(np.float32)


        proj_f= lambda vol: fp_func(cb_para, vol,sample_ratio=2)
        bp_f = lambda projection:  bp_func(projection,cb_para)

        ## input parameters, (which can be load later)
        Epochs=100   # can choose any number until get good reconstruction results

        # bp(ones) is the MLEM normalizer -- it does not depend on v0 and is
        # identical every epoch. Was recomputed inside the loop 100 times;
        # hoisted out to match how the C/GPU implementations in this project
        # compute it once before the iterative loop.
        bp_ones = bp_f(np.ones_like(projection_0))

        import time
        for i in range(Epochs):
            t0 = time.time()
            b=proj_f(v0)
            result = np.divide( projection_0,b, out=np.zeros_like( projection_0), where=(b != 0))
            ratio=bp_f( result  )/bp_ones
            v0*=ratio
            print(f"epoch {i+1}/{Epochs}  {time.time()-t0:.1f}s", flush=True)

        # save v0  to a hdf5 file
        out_path = 'output_python_reconstruction.hdf5'  # was '...' (Ellipsis) -- crashed on save
        with h5py.File(out_path,'w') as file:
            file.create_dataset('voxelSize', data=voxelSize)
            file.create_dataset('Volume', data=v0)








    
