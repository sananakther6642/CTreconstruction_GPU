#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <vector>

// Explicitly forced inline for trilinear interpolation
inline float trilinear(const torch::TensorAccessor<float,3>& vol, float x, float y, float z, int nx, int ny, int nz) {
    if (x < 0 || x >= nx - 1 || y < 0 || y >= ny - 1 || z < 0 || z >= nz - 1) return 0.0f;
    int x0 = (int)x, y0 = (int)y, z0 = (int)z;
    float xd = x - x0, yd = y - y0, zd = z - z0;
    
    float c00 = vol[x0][y0][z0] * (1.0f - xd) + vol[x0+1][y0][z0] * xd;
    float c01 = vol[x0][y0][z0+1] * (1.0f - xd) + vol[x0+1][y0][z0+1] * xd;
    float c10 = vol[x0][y0+1][z0] * (1.0f - xd) + vol[x0+1][y0+1][z0] * xd;
    float c11 = vol[x0][y0+1][z0+1] * (1.0f - xd) + vol[x0+1][y0+1][z0+1] * xd;
    
    return (c00 * (1.0f - yd) + c10 * yd) * (1.0f - zd) + (c01 * (1.0f - yd) + c11 * yd) * zd;
}

torch::Tensor forward_cpp(torch::Tensor volume, torch::Tensor r_o, torch::Tensor r_d, 
                          torch::Tensor lower, torch::Tensor upper, torch::Tensor sVoxel, torch::Tensor nVoxel) {
    int num_projs = r_o.size(0), H = r_o.size(1), W = r_o.size(2), n_samples = lower.size(0);
    auto projs = torch::zeros({num_projs, H, W}, torch::kFloat32);
    
    auto vol_a = volume.accessor<float,3>();
    auto ro_a = r_o.accessor<float,4>(); auto rd_a = r_d.accessor<float,4>();
    auto lw_a = lower.accessor<float,1>();
    auto pr_a = projs.accessor<float,3>();
    
    int nx = volume.size(0), ny = volume.size(1), nz = volume.size(2);
    float svx = sVoxel[0].item<float>(), svy = sVoxel[1].item<float>(), svz = sVoxel[2].item<float>();
    float nvx = nVoxel[0].item<float>(), nvy = nVoxel[1].item<float>(), nvz = nVoxel[2].item<float>();

    float scale_x = nvx / svx;
    float scale_y = nvy / svy;
    float scale_z = nvz / svz;
    float shift_x = (svx / 2.0f) * scale_x - 0.5f;
    float shift_y = (svy / 2.0f) * scale_y - 0.5f;
    float shift_z = (svz / 2.0f) * scale_z - 0.5f;

    std::vector<float> lw_vec(n_samples);
    for(int k = 0; k < n_samples; ++k) lw_vec[k] = lw_a[k];

    #pragma omp parallel for collapse(3)
    for (int p = 0; p < num_projs; ++p) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                float ox = ro_a[p][i][j][0], oy = ro_a[p][i][j][1], oz = ro_a[p][i][j][2];
                float dx = rd_a[p][i][j][0], dy = rd_a[p][i][j][1], dz = rd_a[p][i][j][2];
                
                float norm_rd = std::sqrt(dx * dx + dy * dy + dz * dz);
                float ray_sum = 0.0f;
                
                for (int k = 0; k < n_samples; ++k) {
                    float last_zv = lw_vec[k];
                    float dist = (k < n_samples - 1) ? (lw_vec[k+1] - last_zv) * norm_rd : 1e-10f * norm_rd;
                    
                    float px = ox + dx * last_zv;
                    float py = oy + dy * last_zv;
                    float pz = oz + dz * last_zv;
                    
                    float vx = px * scale_x + shift_x;
                    float vy = py * scale_y + shift_y;
                    float vz = pz * scale_z + shift_z;
                    
                    ray_sum += trilinear(vol_a, vx, vy, vz, nx, ny, nz) * dist;
                }
                pr_a[p][i][j] = ray_sum;
            }
        }
    }
    return projs;
}

torch::Tensor backproject_cpp(torch::Tensor projs, torch::Tensor cos_a, torch::Tensor sin_a, 
                              float a_min, float b_min, float da, float db, 
                              int Volumen_num_xz, int Volumen_num_y, float voxelSize, float SDD, float SOD) {
    int num_projs = projs.size(0);
    int img_nx = projs.size(1), img_ny = projs.size(2);
    
    // Multi-threaded accumulation target arrays (one per thread to avoid race conditions)
    int max_threads = omp_get_max_threads();
    std::vector<torch::Tensor> thread_buffers;
    for (int t = 0; t < max_threads; ++t) {
        thread_buffers.push_back(torch::zeros({Volumen_num_xz, Volumen_num_xz, Volumen_num_y}, torch::kFloat32));
    }
    
    auto cos_a_accessor = cos_a.accessor<float,1>();
    auto sin_a_accessor = sin_a.accessor<float,1>();
    auto projs_accessor = projs.accessor<float,3>();
    
    float radius = (float)Volumen_num_xz / 2.0f - 0.5f;
    float radius_z = (float)Volumen_num_y / 2.0f - 0.5f;
    float sod_sq = SOD * SOD;

    // CRITICAL FIX: Loop order completely inverted. 
    // Projections are parsed on the outside to secure supreme L1/L2 cache spatial locality.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto local_recon_a = thread_buffers[tid].accessor<float,3>();

        #pragma omp for
        for (int p = 0; p < num_projs; ++p) {
            float c_a = cos_a_accessor[p];
            float s_a = sin_a_accessor[p];
            
            for (int i = 0; i < Volumen_num_xz; ++i) {
                float x_val = ((float)i - radius) * voxelSize;
                float x_sin = x_val * s_a;
                float x_cos = x_val * c_a;
                
                for (int j = 0; j < Volumen_num_xz; ++j) {
                    float y_val = ((float)j - radius) * voxelSize;
                    
                    float t = y_val * c_a - x_sin;
                    float U = SOD + y_val * s_a + x_cos;
                    float u_inv = 1.0f / U; // Multiplication is drastically faster than hardware division
                    
                    float ai = SDD * t * u_inv;
                    float img_idx_x = (ai - a_min) / da;
                    
                    if (img_idx_x >= 0 && img_idx_x < img_nx - 1) {
                        float weight = sod_sq * u_inv * u_inv;
                        float z_factor = SDD * u_inv;
                        
                        int x0 = (int)img_idx_x;
                        float xd = img_idx_x - x0;
                        int x0_p1 = x0 + 1;

                        for (int k = 0; k < Volumen_num_y; ++k) {
                            float z_val = ((float)k - radius_z) * voxelSize;
                            float bi = z_val * z_factor;
                            float img_idx_y = (bi - b_min) / db;
                            
                            if (img_idx_y >= 0 && img_idx_y < img_ny - 1) {
                                int y0 = (int)img_idx_y;
                                float yd = img_idx_y - y0;
                                int y0_p1 = y0 + 1;
                                
                                float b_val = (projs_accessor[p][x0][y0] * (1.0f - xd) + projs_accessor[p][x0_p1][y0] * xd) * (1.0f - yd) + 
                                              (projs_accessor[p][x0][y0+1] * (1.0f - xd) + projs_accessor[p][x0_p1][y0+1] * xd) * yd;
                                              
                                local_recon_a[i][j][k] += b_val * weight;
                            }
                        }
                    }
                }
            }
        }
    }

    // Direct reduction combining thread buffers back into master record
    auto reconstructed = torch::zeros({Volumen_num_xz, Volumen_num_xz, Volumen_num_y}, torch::kFloat32);
    for (int t = 0; t < max_threads; ++t) {
        reconstructed += thread_buffers[t];
    }
    
    return reconstructed;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpp", &forward_cpp, "Multi-threaded CPU Forward Projector");
    m.def("backproject_cpp", &backproject_cpp, "Multi-threaded CPU Backprojector");
}
