#include "rf_engine.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <iostream>

#define PI 3.14159265358979323846f
#define C_LIGHT 299792458.0f 
#define EPSILON_R 15.0f      
#define SIGMA 0.005f         
#define ROUGHNESS_RMS 1.5f   

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

// UTD (Uniform Theory of Diffraction) Single Knife-Edge approx
__device__ float knife_edge_diffraction(float d1, float d2, float h, float lambda) {
    float nu = h * sqrtf(2.0f * (d1 + d2) / (lambda * d1 * d2));
    if (nu <= -1.0f) return 1.0f; // LOS clear
    if (nu > 2.4f) return 0.225f / nu; // Deep shadow
    
    float loss_db = 0;
    if (nu >= -1.0f && nu <= 0.0f) loss_db = 20.0f * log10f(0.5f - 0.62f * nu);
    else if (nu > 0.0f && nu <= 1.0f) loss_db = 20.0f * log10f(0.5f * expf(-0.95f * nu));
    else if (nu > 1.0f && nu <= 2.4f) loss_db = 20.0f * log10f(0.4f - sqrtf(0.1184f - (0.38f - 0.1f * nu) * (0.38f - 0.1f * nu)));
    
    return powf(10.0f, -loss_db / 20.0f); 
}

// Helper to find the max height of the voxel column at (gx, gz)
__device__ int find_highest_voxel(VoxelGrid grid, int gx, int gz) {
    if (gx < 0 || gx >= grid.dim_x || gz < 0 || gz >= grid.dim_z) return 0;
    for (int y = grid.dim_y - 1; y >= 0; y--) {
        int idx = y * (grid.dim_x * grid.dim_z) + gz * grid.dim_x + gx;
        if (grid.data[idx] > 0) return y;
    }
    return 0;
}

// Fresnel Reflection Coefficient
__device__ float fresnel_reflection(Vec3 incident, Vec3 normal, float freq_hz) {
    float cos_theta_i = -dot(incident, normal);
    if (cos_theta_i < 0) cos_theta_i = -cos_theta_i; 
    float omega = 2.0f * PI * freq_hz;
    float eps_complex = EPSILON_R - (SIGMA / (omega * 8.854e-12f)); 
    float sin_theta_i_sq = 1.0f - cos_theta_i * cos_theta_i;
    float sqrt_term = sqrtf(fmaxf(0.0f, eps_complex - sin_theta_i_sq));
    return (cos_theta_i - sqrt_term) / (cos_theta_i + sqrt_term + 1e-6f);
}

// Fast 3D Voxel Raymarching (Digital Differential Analyzer)
__device__ bool trace_voxel(Vec3 orig, Vec3 dir, float max_dist, VoxelGrid grid, float& out_t, Vec3& out_normal) {
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

    int cur_x = floorf(gx);
    int cur_y = floorf(gy);
    int cur_z = floorf(gz);

    float t = 0.0f;
    float max_grid_dist = max_dist / grid.cell_size;

    while (t <= max_grid_dist) {
        if (cur_x >= 0 && cur_x < grid.dim_x && 
            cur_y >= 0 && cur_y < grid.dim_y && 
            cur_z >= 0 && cur_z < grid.dim_z) {
            
            int idx = cur_y * (grid.dim_x * grid.dim_z) + cur_z * grid.dim_x + cur_x;
            if (grid.data[idx] > 0) { 
                out_t = t * grid.cell_size;
                return true;
            }
        }

        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                cur_x += step_x; t = tMaxX; tMaxX += tDeltaX;
                out_normal = {(float)-step_x, 0, 0};
            } else {
                cur_z += step_z; t = tMaxZ; tMaxZ += tDeltaZ;
                out_normal = {0, 0, (float)-step_z};
            }
        } else {
            if (tMaxY < tMaxZ) {
                cur_y += step_y; t = tMaxY; tMaxY += tDeltaY;
                out_normal = {0, (float)-step_y, 0};
            } else {
                cur_z += step_z; t = tMaxZ; tMaxZ += tDeltaZ;
                out_normal = {0, 0, (float)-step_z};
            }
        }
    }
    return false;
}

// 3GPP 3D Directional Antenna Pattern (Horizontal & Vertical)
__device__ float calc_antenna_gain(Vec3 ray_dir, SimParams params) {
    float az_rad = params.tx_azimuth_deg * PI / 180.0f;
    float el_rad = params.tx_elevation_deg * PI / 180.0f;

    // Calculate horizontal angle difference
    float ray_az = atan2f(ray_dir.x, ray_dir.z);
    float delta_az = ray_az - az_rad;
    while (delta_az > PI) delta_az -= 2.0f * PI;
    while (delta_az < -PI) delta_az += 2.0f * PI;

    // Calculate vertical angle difference (clamp to prevent NaN on rounding)
    float ray_el = asinf(fmaxf(-1.0f, fminf(1.0f, ray_dir.y)));
    float delta_el = ray_el - el_rad;

    // Real cell antennas are tall and skinny panel arrays. 
    // They have wide horizontal beamwidths (e.g. 65 deg) but tight vertical ones (e.g. 10 deg)
    float vertical_beamwidth_rad = 10.0f * (PI / 180.0f);

    // 3GPP Attenuation formulas
    float A_h = 12.0f * powf(delta_az / params.beamwidth_rad, 2.0f);
    float A_v = 12.0f * powf(delta_el / vertical_beamwidth_rad, 2.0f);

    // Combine and clamp to a stricter Front-to-Back ratio (38 dB)
    float attenuation = fminf(A_h + A_v, 38.0f); 
    
    return params.tx_gain_dbi - attenuation;
}

// Pass 1: Direct LOS & Diffraction Voxel Grid pass
__global__ void los_diffraction_voxel_kernel(VoxelGrid grid, SimParams params, float* rx_grid_watts) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gz = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= params.grid_width || gz >= params.grid_height) return;

    int top_y = find_highest_voxel(grid, gx, gz);
    float rx_y = grid.min_y + top_y * grid.cell_size + 1.5f; // 1.5m Mobile Antenna Height

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
    float e_init = sqrtf(30.0f * p_tx_watts);
    float e_mag = e_init * (lambda / (4.0f * PI * dist)); 
    
    // Find highest obstacle along the 2D path
    int num_steps = (int)ceilf(dist / (grid.cell_size * 0.5f));
    float max_h = -1e9f;
    float d1 = dist * 0.5f;
    float d2 = dist * 0.5f;

    for (int i = 1; i < num_steps; i++) {
        float f = (float)i / num_steps;
        float cx = tx_pos.x + f * (rx_pos.x - tx_pos.x);
        float cz = tx_pos.z + f * (rx_pos.z - tx_pos.z);
        
        int cgx = (cx - grid.min_x) / grid.cell_size;
        int cgz = (cz - grid.min_z) / grid.cell_size;
        
        int c_top_y = find_highest_voxel(grid, cgx, cgz);
        float obs_y = grid.min_y + c_top_y * grid.cell_size;
        float los_y = tx_pos.y + f * (rx_pos.y - tx_pos.y);
        
        float cur_h = obs_y - los_y;
        if (cur_h > max_h) {
            max_h = cur_h;
            d1 = f * dist;
            d2 = (1.0f - f) * dist;
        }
    }

    if (max_h > 0.01f) {
        e_mag *= knife_edge_diffraction(d1, d2, max_h, lambda);
    }

    float e_mag_sq = e_mag * e_mag;
    float a_eff = (lambda * lambda) / (4.0f * PI);
    float p_rx_watts = (e_mag_sq / (120.0f * PI)) * a_eff;

    int grid_idx = gz * params.grid_width + gx;
    atomicMax((int*)&rx_grid_watts[grid_idx], __float_as_int(p_rx_watts));
}

// CUDA Kernel: Shooting and Bouncing Rays (SBR) against Voxel Grid
__global__ void sbr_voxel_kernel(
    VoxelGrid grid, SimParams params, float* rx_grid_watts) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.ray_count) return;

    // 1. Generate Spherical Fibonacci Ray
    float phi = acosf(1.0f - 2.0f * (idx + 0.5f) / params.ray_count);
    double theta_d = 3.141592653589793238 * (1.0 + sqrt(5.0)) * (double)idx;
    float theta = (float)fmod(theta_d, 2.0 * 3.141592653589793238);
    
    Vec3 ray_dir = { sinf(phi) * sinf(theta), cosf(phi), sinf(phi) * cosf(theta) };
    Vec3 ray_orig = {params.tx_x, params.tx_y, params.tx_z};
    
    float lambda = C_LIGHT / params.freq_hz;

    // Apply Antenna Gain
    float gain_dbi = calc_antenna_gain(ray_dir, params);
    float eirp_dbm = params.tx_power_dbm + gain_dbi;
    float p_watts = powf(10.0f, (eirp_dbm - 30.0f) / 10.0f);
    float e_init = sqrtf(30.0f * p_watts); 
    
    float current_e_mag = e_init;
    float accumulated_dist = 0.0f;

    // Raycast Bouncing Loop
    for (int bounce = 0; bounce < params.max_bounces; bounce++) {
        float hit_t = 0;
        Vec3 hit_normal = {0,0,0};
        
        if (!trace_voxel(ray_orig, ray_dir, 5000.0f, grid, hit_t, hit_normal)) break; 

        Vec3 hit_point = add(ray_orig, mul(ray_dir, hit_t));
        accumulated_dist += hit_t;
        
        int rx_x = (int)((hit_point.x - params.bounds_min_x) / params.cell_size);
        int rx_z = (int)((hit_point.z - params.bounds_min_z) / params.cell_size);

        if (rx_x >= 0 && rx_x < params.grid_width && rx_z >= 0 && rx_z < params.grid_height) {
            
            // Calculate absolute power of the ray hitting the voxel
            float fspl_mag = current_e_mag * (lambda / (4.0f * PI * accumulated_dist));
            float e_mag_sq = fspl_mag * fspl_mag;
            float a_eff = (lambda * lambda) / (4.0f * PI);
            float p_rx_watts = (e_mag_sq / (120.0f * PI)) * a_eff;
            
            int grid_idx = rx_z * params.grid_width + rx_x;
            
            // Incoherent Peak Power Mapping: 
            // We use a bit-casting trick to perform an atomicMax on positive floats.
            // This logs the single strongest ray that hits this grid cell, killing the interference pattern!
            atomicMax((int*)&rx_grid_watts[grid_idx], __float_as_int(p_rx_watts));
        }

        // Apply Reflection & Scattering Loss for next bounce
        float cos_theta_i = fabs(dot(ray_dir, hit_normal));
        float R = fresnel_reflection(ray_dir, hit_normal, params.freq_hz);
        float roughness_term = (4.0f * PI * ROUGHNESS_RMS * cos_theta_i) / lambda;
        float specular_rho = expf(-0.5f * roughness_term * roughness_term);
        
        current_e_mag *= (R * specular_rho); 

        ray_orig = add(hit_point, mul(hit_normal, grid.cell_size * 0.1f)); 
        ray_dir = sub(ray_dir, mul(hit_normal, 2.0f * dot(ray_dir, hit_normal)));
        ray_dir = normalize(ray_dir);
    }
}

extern "C" void run_rf_simulation(
    const VoxelGrid& grid, const SimParams& params, std::vector<float>& out_rx_power_dbm) 
{
    int grid_size = params.grid_width * params.grid_height;
    out_rx_power_dbm.resize(grid_size, -120.0f);

    uint8_t* d_grid_data;
    float *d_rx_watts;

    int total_voxels = grid.dim_x * grid.dim_y * grid.dim_z;
    cudaMalloc(&d_grid_data, total_voxels * sizeof(uint8_t));
    cudaMemcpy(d_grid_data, grid.data, total_voxels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_rx_watts, grid_size * sizeof(float));
    cudaMemset(d_rx_watts, 0, grid_size * sizeof(float));

    VoxelGrid d_grid = grid;
    d_grid.data = d_grid_data; 

    // Launch Pass 1: Direct LOS & Diffraction Grid Kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((params.grid_width + blockSize.x - 1) / blockSize.x, 
                  (params.grid_height + blockSize.y - 1) / blockSize.y);
    los_diffraction_voxel_kernel<<<gridSize, blockSize>>>(d_grid, params, d_rx_watts);
    cudaDeviceSynchronize();

    // Launch Pass 2: Ray Bouncing (SBR)
    int threads_per_block = 256;
    int blocks = (params.ray_count + threads_per_block - 1) / threads_per_block;

    sbr_voxel_kernel<<<blocks, threads_per_block>>>(d_grid, params, d_rx_watts);
    cudaDeviceSynchronize();

    std::vector<float> h_rx_watts(grid_size);
    cudaMemcpy(h_rx_watts.data(), d_rx_watts, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert Watts directly to dBm
    for (int i = 0; i < grid_size; i++) {
        float p_watts = h_rx_watts[i];
        if (p_watts > 1e-15f) {
            out_rx_power_dbm[i] = 10.0f * log10f(p_watts) + 30.0f; 
        } else {
            out_rx_power_dbm[i] = -120.0f;
        }
    }

    cudaFree(d_grid_data);
    cudaFree(d_rx_watts);
}