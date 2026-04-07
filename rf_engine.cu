#include "rf_engine.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <iostream>

#define PI 3.14159265358979323846f
#define C_LIGHT 299792458.0f 

// --- Device Math Helpers ---
__device__ inline Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ inline Vec3 mul(Vec3 a, float b) { return {a.x * b, a.y * b, a.z * b}; }
__device__ inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float length(Vec3 a) { return sqrtf(dot(a, a)); }
__device__ inline Vec3 normalize(Vec3 a) {
    float l = length(a);
    return l > 0 ? mul(a, 1.0f / l) : a;
}
__device__ inline Vec3 cross_prod(Vec3 a, Vec3 b) { 
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; 
}

// Atomic helpers for float arrays (Delay Spread tracking)
__device__ inline void atomicMinFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}
__device__ inline void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// SOTA: ITU-R P.2040 Material Properties mapped to LAS LiDAR Classes
__device__ void get_material_props(int cls, float freq_hz, float &eps_r, float &sigma) {
    float f_GHz = freq_hz / 1e9f;
    eps_r = 15.0f; sigma = 0.05f; // Default Soil

    if (cls == 6) { // Building / Concrete
        eps_r = 5.31f; 
        sigma = 0.0326f * powf(f_GHz, 0.8095f);
    } else if (cls >= 3 && cls <= 5) { // Vegetation
        eps_r = 1.5f; 
        sigma = 0.001f;
    } else if (cls == 9) { // Water
        eps_r = 81.0f; 
        sigma = 5.0f;
    } else if (cls == 7) { // Metal / Utilities
        eps_r = 1.0f; 
        sigma = 1.0e6f; 
    }
}

// SOTA: Weissberger's Modified Exponential Decay (MED) for Foliage Loss
__device__ float calc_vegetation_loss_power_mult(float veg_dist, float freq_hz) {
    if (veg_dist < 0.1f) return 1.0f;
    float f_GHz = freq_hz / 1e9f;
    float loss_db = 0.0f;
    if (veg_dist <= 14.0f) {
        loss_db = 0.45f * powf(f_GHz, 0.284f) * veg_dist;
    } else {
        loss_db = 1.33f * powf(f_GHz, 0.284f) * powf(veg_dist, 0.588f);
    }
    return powf(10.0f, -loss_db / 10.0f); 
}

// SOTA: Knife-Edge Loss conversion (Positive dB)
__device__ float knife_edge_loss_db(float nu) {
    if (nu <= -1.0f) return 0.0f;
    if (nu <= 0.0f) return -20.0f * log10f(0.5f - 0.62f * nu);
    if (nu <= 1.0f) return -20.0f * log10f(0.5f * expf(-0.95f * nu));
    if (nu <= 2.4f) return -20.0f * log10f(0.4f - sqrtf(0.1184f - powf(0.38f - 0.1f * nu, 2.0f)));
    return -20.0f * log10f(0.225f / nu);
}

// SOTA: Complex Vector Polarization Tracking & TE/TM Decomposition
__device__ float compute_reflection_and_update_pol(Vec3 ray_dir, Vec3 hit_normal, Vec3 &pol, float eps_r, float sigma, float freq_hz) {
    float cos_theta_i = fabs(dot(ray_dir, hit_normal));
    float sin_theta_i_sq = fmaxf(0.0f, 1.0f - cos_theta_i * cos_theta_i);
    float omega = 2.0f * PI * freq_hz;
    float eps_complex_i = sigma / (omega * 8.854e-12f);
    
    float re_eps = eps_r - sin_theta_i_sq;
    float im_eps = eps_complex_i;
    float mag_eps = sqrtf(re_eps * re_eps + im_eps * im_eps);
    float phase_eps = atan2f(im_eps, re_eps);
    
    float sq_re = sqrtf(mag_eps) * cosf(phase_eps * 0.5f);
    float sq_im = sqrtf(mag_eps) * sinf(phase_eps * 0.5f);
    
    // Perpendicular (TE)
    float num_te_re = cos_theta_i - sq_re, num_te_im = -sq_im;
    float den_te_re = cos_theta_i + sq_re, den_te_im = sq_im;
    float R_TE_mag = sqrtf((num_te_re*num_te_re + num_te_im*num_te_im) / (den_te_re*den_te_re + den_te_im*den_te_im));
    
    // Parallel (TM)
    float eps_cos_re = eps_r * cos_theta_i, eps_cos_im = -eps_complex_i * cos_theta_i;
    float num_tm_re = eps_cos_re - sq_re, num_tm_im = eps_cos_im - sq_im;
    float den_tm_re = eps_cos_re + sq_re, den_tm_im = eps_cos_im + sq_im;
    float R_TM_mag = sqrtf((num_tm_re*num_tm_re + num_tm_im*num_tm_im) / (den_tm_re*den_tm_re + den_tm_im*den_tm_im));
    
    Vec3 v_te = cross_prod(ray_dir, hit_normal);
    float v_te_len = length(v_te);
    if (v_te_len < 1e-5f) return R_TM_mag; 
    v_te = mul(v_te, 1.0f / v_te_len);
    Vec3 v_tm = cross_prod(v_te, ray_dir);
    
    float pol_te = dot(pol, v_te) * R_TE_mag;
    float pol_tm = dot(pol, v_tm) * R_TM_mag;
    float new_mag = sqrtf(pol_te*pol_te + pol_tm*pol_tm);
    
    if (new_mag > 1e-6f) {
        Vec3 new_pol = add(mul(v_te, pol_te), mul(v_tm, pol_tm));
        pol = normalize(new_pol);
    }
    return new_mag;
}

__device__ int find_highest_voxel(VoxelGrid grid, int gx, int gz) {
    if (gx < 0 || gx >= grid.dim_x || gz < 0 || gz >= grid.dim_z) return 0;
    for (int y = grid.dim_y - 1; y >= 0; y--) {
        int idx = y * (grid.dim_x * grid.dim_z) + gz * grid.dim_x + gx;
        if (grid.data[idx] > 0) return y;
    }
    return 0;
}

__device__ float calc_antenna_gain(Vec3 ray_dir, SimParams params) {
    float az_rad = params.tx_azimuth_deg * PI / 180.0f;
    float el_rad = params.tx_elevation_deg * PI / 180.0f;
    float ray_az = atan2f(ray_dir.x, ray_dir.z);
    float delta_az = ray_az - az_rad;
    while (delta_az > PI) delta_az -= 2.0f * PI;
    while (delta_az < -PI) delta_az += 2.0f * PI;
    float ray_el = asinf(fmaxf(-1.0f, fminf(1.0f, ray_dir.y)));
    float delta_el = ray_el - el_rad;
    
    // Dynamic vertical beamwidth from parameters
    float A_h = 12.0f * powf(delta_az / params.beamwidth_rad, 2.0f);
    float A_v = 12.0f * powf(delta_el / params.vertical_beamwidth_rad, 2.0f);
    float attenuation = fminf(A_h + A_v, 38.0f); 
    
    return params.tx_gain_dbi - attenuation;
}

// Raymarcher returning class code for material lookup
__device__ bool trace_voxel(Vec3 orig, Vec3 dir, float max_dist, VoxelGrid grid, float& out_t, Vec3& out_normal, float& out_veg_dist, int& out_cls) {
    float gx = (orig.x - grid.min_x) / grid.cell_size;
    float gy = (orig.y - grid.min_y) / grid.cell_size;
    float gz = (orig.z - grid.min_z) / grid.cell_size;
    int step_x = (dir.x > 0) ? 1 : ((dir.x < 0) ? -1 : 0);
    int step_y = (dir.y > 0) ? 1 : ((dir.y < 0) ? -1 : 0);
    int step_z = (dir.z > 0) ? 1 : ((dir.z < 0) ? -1 : 0);
    float tDeltaX = (step_x != 0) ? fminf(fabs(1.0f / dir.x), 1e6) : 1e6;
    float tDeltaY = (step_y != 0) ? fminf(fabs(1.0f / dir.y), 1e6) : 1e6;
    float tDeltaZ = (step_z != 0) ? fminf(fabs(1.0f / dir.z), 1e6) : 1e6;
    float tMaxX = (step_x > 0) ? (ceilf(gx) - gx) * tDeltaX : (gx - floorf(gx)) * tDeltaX;
    float tMaxY = (step_y > 0) ? (ceilf(gy) - gy) * tDeltaY : (gy - floorf(gy)) * tDeltaY;
    float tMaxZ = (step_z > 0) ? (ceilf(gz) - gz) * tDeltaZ : (gz - floorf(gz)) * tDeltaZ;
    if (tMaxX == 0.0f) tMaxX += tDeltaX;
    if (tMaxY == 0.0f) tMaxY += tDeltaY;
    if (tMaxZ == 0.0f) tMaxZ += tDeltaZ;
    int cur_x = floorf(gx), cur_y = floorf(gy), cur_z = floorf(gz);
    float t = 0.0f;
    float max_grid_dist = max_dist / grid.cell_size;
    out_veg_dist = 0.0f;

    while (t <= max_grid_dist) {
        if (cur_x >= 0 && cur_x < grid.dim_x && cur_y >= 0 && cur_y < grid.dim_y && cur_z >= 0 && cur_z < grid.dim_z) {
            int idx = cur_y * (grid.dim_x * grid.dim_z) + cur_z * grid.dim_x + cur_x;
            int cls = grid.data[idx];
            if (cls >= 3 && cls <= 5) { 
                out_veg_dist += grid.cell_size;
            } else if (cls > 0) { 
                out_t = t * grid.cell_size;
                out_cls = cls;
                return true;
            }
        }
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) { cur_x += step_x; t = tMaxX; tMaxX += tDeltaX; out_normal = {(float)-step_x, 0, 0}; } 
            else { cur_z += step_z; t = tMaxZ; tMaxZ += tDeltaZ; out_normal = {0, 0, (float)-step_z}; }
        } else {
            if (tMaxY < tMaxZ) { cur_y += step_y; t = tMaxY; tMaxY += tDeltaY; out_normal = {0, (float)-step_y, 0}; } 
            else { cur_z += step_z; t = tMaxZ; tMaxZ += tDeltaZ; out_normal = {0, 0, (float)-step_z}; }
        }
    }
    return false;
}

// SOTA: Deygout Multiple Knife-Edge Diffraction
__global__ void los_diffraction_voxel_kernel(VoxelGrid grid, SimParams params, float* rx_grid_watts, float* min_dist_grid, float* max_dist_grid) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gz = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= params.grid_width || gz >= params.grid_height) return;

    int top_y = find_highest_voxel(grid, gx, gz);
    float rx_y = grid.min_y + top_y * grid.cell_size + 1.5f; 
    float rx_x = grid.min_x + gx * grid.cell_size + grid.cell_size * 0.5f;
    float rx_z = grid.min_z + gz * grid.cell_size + grid.cell_size * 0.5f;

    Vec3 tx_pos = {params.tx_x, params.tx_y, params.tx_z};
    Vec3 rx_pos = {rx_x, rx_y, rx_z};
    Vec3 dir = sub(rx_pos, tx_pos);
    float dist = length(dir);
    if (dist < 0.1f) return;
    dir = normalize(dir);

    float lambda = C_LIGHT / params.freq_hz;
    float gain_dbi = calc_antenna_gain(dir, params);
    float eirp_dbm = params.tx_power_dbm + gain_dbi;
    float p_tx_watts = powf(10.0f, (eirp_dbm - 30.0f) / 10.0f);
    
    // Incorporate the Receiver Antenna Gain
    float rx_gain_lin = powf(10.0f, params.rx_gain_dbi / 10.0f);
    float power_density = p_tx_watts / (4.0f * PI * dist * dist);
    float a_eff = rx_gain_lin * (lambda * lambda) / (4.0f * PI);
    float p_rx_watts = power_density * a_eff;

    int num_steps = (int)ceilf(dist / (grid.cell_size * 0.5f));
    float veg_dist = 0.0f;
    
    // DEYGOUT ALGORITHM: Pass 1 (Main Peak)
    float max_nu_1 = -1e9f;
    int peak_1_idx = -1;
    float p1_x, p1_y, p1_z, p1_d1;

    for (int i = 1; i < num_steps; i++) {
        float f = (float)i / num_steps;
        float cx = tx_pos.x + f * (rx_pos.x - tx_pos.x);
        float cz = tx_pos.z + f * (rx_pos.z - tx_pos.z);
        float cy = tx_pos.y + f * (rx_pos.y - tx_pos.y);
        
        int cgx = (cx - grid.min_x) / grid.cell_size;
        int cgz = (cz - grid.min_z) / grid.cell_size;
        int cgy = find_highest_voxel(grid, cgx, cgz);
        
        if (cgy > 0) {
            int cls = grid.data[cgy * (grid.dim_x * grid.dim_z) + cgz * grid.dim_x + cgx];
            if (cls >= 3 && cls <= 5) veg_dist += grid.cell_size * 0.5f;
            
            float obs_y = grid.min_y + cgy * grid.cell_size;
            float h = obs_y - cy;
            float d1 = f * dist;
            float d2 = dist - d1;
            float nu = h * sqrtf(2.0f * (d1 + d2) / (lambda * d1 * d2));
            
            if (nu > max_nu_1) {
                max_nu_1 = nu; peak_1_idx = i; 
                p1_x = cx; p1_y = obs_y; p1_z = cz; p1_d1 = d1;
            }
        }
    }

    float diff_loss_db = 0.0f;
    if (max_nu_1 > -1.0f && peak_1_idx != -1) {
        diff_loss_db += knife_edge_loss_db(max_nu_1);
        
        // DEYGOUT: Pass 2 (Left Sub-Peak)
        if (peak_1_idx > 1) {
            float max_nu_2 = -1e9f;
            for (int i = 1; i < peak_1_idx; i++) {
                float f = (float)i / peak_1_idx;
                float cx = tx_pos.x + f * (p1_x - tx_pos.x);
                float cz = tx_pos.z + f * (p1_z - tx_pos.z);
                int cgx = (cx - grid.min_x) / grid.cell_size;
                int cgz = (cz - grid.min_z) / grid.cell_size;
                float obs_y = grid.min_y + find_highest_voxel(grid, cgx, cgz) * grid.cell_size;
                float cy = tx_pos.y + f * (p1_y - tx_pos.y);
                float h = obs_y - cy;
                float d1 = f * p1_d1, d2 = p1_d1 - d1;
                float nu = h * sqrtf(2.0f * (d1 + d2) / (lambda * d1 * d2));
                if (nu > max_nu_2) max_nu_2 = nu;
            }
            if (max_nu_2 > -1.0f) diff_loss_db += knife_edge_loss_db(max_nu_2);
        }
        
        // DEYGOUT: Pass 3 (Right Sub-Peak)
        if (peak_1_idx < num_steps - 1) {
            float max_nu_3 = -1e9f;
            int right_steps = num_steps - peak_1_idx;
            float dist_p1_rx = dist - p1_d1;
            for (int i = 1; i < right_steps; i++) {
                float f = (float)i / right_steps;
                float cx = p1_x + f * (rx_pos.x - p1_x);
                float cz = p1_z + f * (rx_pos.z - p1_z);
                int cgx = (cx - grid.min_x) / grid.cell_size;
                int cgz = (cz - grid.min_z) / grid.cell_size;
                float obs_y = grid.min_y + find_highest_voxel(grid, cgx, cgz) * grid.cell_size;
                float cy = p1_y + f * (rx_pos.y - p1_y);
                float h = obs_y - cy;
                float d1 = f * dist_p1_rx, d2 = dist_p1_rx - d1;
                float nu = h * sqrtf(2.0f * (d1 + d2) / (lambda * d1 * d2));
                if (nu > max_nu_3) max_nu_3 = nu;
            }
            if (max_nu_3 > -1.0f) diff_loss_db += knife_edge_loss_db(max_nu_3);
        }
    }

    p_rx_watts *= powf(10.0f, -diff_loss_db / 10.0f);
    p_rx_watts *= calc_vegetation_loss_power_mult(veg_dist, params.freq_hz);

    int grid_idx = gz * params.grid_width + gx;
    atomicMax((int*)&rx_grid_watts[grid_idx], __float_as_int(p_rx_watts));
    atomicMinFloat(&min_dist_grid[grid_idx], dist);
    atomicMaxFloat(&max_dist_grid[grid_idx], dist);
}

// SOTA + NEW FIX: SBR with Materials AND correctly normalized Gaussian Splatting
__global__ void sbr_voxel_kernel(VoxelGrid grid, SimParams params, float* rx_grid_watts, float* min_dist_grid, float* max_dist_grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.ray_count) return;

    float phi = acosf(1.0f - 2.0f * (idx + 0.5f) / params.ray_count);
    double theta_d = 3.141592653589793238 * (1.0 + sqrt(5.0)) * (double)idx;
    float theta = (float)fmod(theta_d, 2.0 * 3.141592653589793238);
    Vec3 ray_dir = { sinf(phi) * sinf(theta), cosf(phi), sinf(phi) * cosf(theta) };
    Vec3 ray_orig = {params.tx_x, params.tx_y, params.tx_z};
    
    // Initial Polarization Vector 
    Vec3 up_vec = {0, 1, 0};
    Vec3 right = cross_prod(ray_dir, up_vec);
    if (length(right) < 0.001f) right = {1, 0, 0};
    Vec3 pol = normalize(cross_prod(right, ray_dir));

    float lambda = C_LIGHT / params.freq_hz;
    float gain_dbi = calc_antenna_gain(ray_dir, params);
    float eirp_dbm = params.tx_power_dbm + gain_dbi;
    float p_watts = powf(10.0f, (eirp_dbm - 30.0f) / 10.0f);
    
    // Incorporate the Receiver Antenna Gain
    float rx_gain_lin = powf(10.0f, params.rx_gain_dbi / 10.0f);
    float d_omega = 4.0f * PI / params.ray_count;
    float p_ray_watts_init = p_watts * (d_omega / (4.0f * PI)); 
    
    float accumulated_dist = 0.0f;
    float ray_power_multiplier = 1.0f;

    for (int bounce = 0; bounce < params.max_bounces; bounce++) {
        float hit_t = 0;
        float veg_dist = 0;
        int hit_cls = 0;
        Vec3 hit_normal = {0,0,0};
        
        if (!trace_voxel(ray_orig, ray_dir, 5000.0f, grid, hit_t, hit_normal, veg_dist, hit_cls)) break; 

        ray_power_multiplier *= calc_vegetation_loss_power_mult(veg_dist, params.freq_hz);

        Vec3 hit_point = add(ray_orig, mul(ray_dir, hit_t));
        accumulated_dist += hit_t;
        
        if (bounce > 0) {
            int rx_x = (int)((hit_point.x - params.bounds_min_x) / params.cell_size);
            int rx_z = (int)((hit_point.z - params.bounds_min_z) / params.cell_size);

            float A_t = d_omega * accumulated_dist * accumulated_dist; 
            
            Vec3 flat_ground = {0.0f, 1.0f, 0.0f};
            float cos_ground = fmaxf(0.05f, fabs(dot(ray_dir, flat_ground)));
            float A_footprint = A_t / cos_ground;
            float A_voxel = params.cell_size * params.cell_size;
            
            float spread_area = fmaxf(A_footprint, A_voxel); 
            float power_density = (p_ray_watts_init * ray_power_multiplier) / spread_area;
            float a_eff = rx_gain_lin * (lambda * lambda) / (4.0f * PI);
            float p_rx_watts_center = power_density * a_eff;
            
            float r_footprint = sqrtf(spread_area / PI);
            int radius_cells = (int)ceilf(r_footprint / params.cell_size);
            radius_cells = min(radius_cells, 6);

            if (radius_cells <= 0) {
                if (rx_x >= 0 && rx_x < params.grid_width && rx_z >= 0 && rx_z < params.grid_height) {
                    int grid_idx = rx_z * params.grid_width + rx_x;
                    atomicMax((int*)&rx_grid_watts[grid_idx], __float_as_int(p_rx_watts_center));
                    atomicMinFloat(&min_dist_grid[grid_idx], accumulated_dist);
                    atomicMaxFloat(&max_dist_grid[grid_idx], accumulated_dist);
                }
            } else {
                float r_cells_sq = (float)(radius_cells * radius_cells);
                float sigma_sq = r_cells_sq / 4.0f;
                for (int dx = -radius_cells; dx <= radius_cells; dx++) {
                    for (int dz = -radius_cells; dz <= radius_cells; dz++) {
                        float dist_sq = (float)(dx*dx + dz*dz);
                        if (dist_sq <= r_cells_sq) {
                            int px = rx_x + dx;
                            int pz = rx_z + dz;
                            if (px >= 0 && px < params.grid_width && pz >= 0 && pz < params.grid_height) {
                                float weight = expf(-dist_sq / (2.0f * sigma_sq));
                                float splat_watts = p_rx_watts_center * weight;
                                if (splat_watts > 1e-18f) {
                                    int grid_idx = pz * params.grid_width + px;
                                    atomicMax((int*)&rx_grid_watts[grid_idx], __float_as_int(splat_watts));
                                    atomicMinFloat(&min_dist_grid[grid_idx], accumulated_dist);
                                    atomicMaxFloat(&max_dist_grid[grid_idx], accumulated_dist);
                                }
                            }
                        }
                    }
                }
            }
        }

        // SOTA: ITU-R P.2040 Material Reflection & Polarization Shift
        float eps_r, sigma;
        get_material_props(hit_cls, params.freq_hz, eps_r, sigma);
        float reflection_mag = compute_reflection_and_update_pol(ray_dir, hit_normal, pol, eps_r, sigma, params.freq_hz);
        
        float cos_theta_i = fabs(dot(ray_dir, hit_normal));
        float roughness_term = (4.0f * PI * 0.1f * cos_theta_i) / lambda; // Realistic 10cm dirt roughness
        float specular_rho = expf(-0.5f * roughness_term * roughness_term);
        
        ray_power_multiplier *= (reflection_mag * reflection_mag) * specular_rho;
        if (ray_power_multiplier < 1e-9f) break;

        ray_orig = add(hit_point, mul(hit_normal, grid.cell_size * 0.1f)); 
        ray_dir = sub(ray_dir, mul(hit_normal, 2.0f * dot(ray_dir, hit_normal)));
        ray_dir = normalize(ray_dir);
    }
}

extern "C" void run_rf_simulation(
    const VoxelGrid& grid, const SimParams& params, 
    std::vector<float>& out_rx_power_dbm, std::vector<float>& out_delay_spread_ns) 
{
    int grid_size = params.grid_width * params.grid_height;
    out_rx_power_dbm.resize(grid_size, -120.0f);
    out_delay_spread_ns.resize(grid_size, 0.0f);

    uint8_t* d_grid_data;
    float *d_rx_watts, *d_min_dist, *d_max_dist;

    int total_voxels = grid.dim_x * grid.dim_y * grid.dim_z;
    cudaMalloc(&d_grid_data, total_voxels * sizeof(uint8_t));
    cudaMemcpy(d_grid_data, grid.data, total_voxels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_rx_watts, grid_size * sizeof(float));
    cudaMemset(d_rx_watts, 0, grid_size * sizeof(float));

    cudaMalloc(&d_min_dist, grid_size * sizeof(float));
    cudaMalloc(&d_max_dist, grid_size * sizeof(float));
    
    std::vector<float> init_min(grid_size, 1e9f);
    std::vector<float> init_max(grid_size, 0.0f);
    cudaMemcpy(d_min_dist, init_min.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_dist, init_max.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice);

    VoxelGrid d_grid = grid;
    d_grid.data = d_grid_data; 

    dim3 blockSize(16, 16);
    dim3 gridSize((params.grid_width + blockSize.x - 1) / blockSize.x, 
                  (params.grid_height + blockSize.y - 1) / blockSize.y);
    los_diffraction_voxel_kernel<<<gridSize, blockSize>>>(d_grid, params, d_rx_watts, d_min_dist, d_max_dist);
    cudaDeviceSynchronize();

    int threads_per_block = 256;
    int blocks = (params.ray_count + threads_per_block - 1) / threads_per_block;
    sbr_voxel_kernel<<<blocks, threads_per_block>>>(d_grid, params, d_rx_watts, d_min_dist, d_max_dist);
    cudaDeviceSynchronize();

    std::vector<float> h_rx_watts(grid_size);
    std::vector<float> h_min(grid_size);
    std::vector<float> h_max(grid_size);
    cudaMemcpy(h_rx_watts.data(), d_rx_watts, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min.data(), d_min_dist, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max.data(), d_max_dist, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < grid_size; i++) {
        float p_watts = h_rx_watts[i];
        if (p_watts > 1e-15f) {
            out_rx_power_dbm[i] = 10.0f * log10f(p_watts) + 30.0f; 
        } else {
            out_rx_power_dbm[i] = -120.0f;
        }

        if (h_max[i] > h_min[i] && h_min[i] < 1e8f) {
            float dist_diff = h_max[i] - h_min[i];
            out_delay_spread_ns[i] = (dist_diff / C_LIGHT) * 1e9f; 
        }
    }

    cudaFree(d_grid_data);
    cudaFree(d_rx_watts);
    cudaFree(d_min_dist);
    cudaFree(d_max_dist);
}