#include "rf_engine.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <iostream>

#define PI 3.14159265358979323846f
#define C_LIGHT 299792458.0f // Speed of light in m/s
#define EPSILON_R 15.0f      // Relative permittivity of average ground
#define SIGMA 0.005f         // Conductivity of average ground in S/m
#define ROUGHNESS_RMS 1.5f   // 1.5m root-mean-square terrain roughness

// --- Device Math Helpers ---
__device__ inline Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ inline Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ inline Vec3 mul(Vec3 a, float b) { return {a.x * b, a.y * b, a.z * b}; }
__device__ inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ inline float length(Vec3 a) { return sqrtf(dot(a, a)); }
__device__ inline Vec3 normalize(Vec3 a) {
    float l = length(a);
    return l > 0 ? mul(a, 1.0f / l) : a;
}

// Möller-Trumbore Ray-Triangle Intersection
__device__ bool intersect_triangle(Vec3 orig, Vec3 dir, Triangle tri, float& t, Vec3& normal) {
    const float EPSILON = 1e-6f;
    Vec3 edge1 = sub(tri.v1, tri.v0);
    Vec3 edge2 = sub(tri.v2, tri.v0);
    Vec3 h = cross(dir, edge2);
    float a = dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) return false; // Parallel
    float f = 1.0f / a;
    Vec3 s = sub(orig, tri.v0);
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    Vec3 q = cross(s, edge1);
    float v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * dot(edge2, q);
    if (t > EPSILON) {
        normal = normalize(cross(edge1, edge2));
        return true;
    }
    return false;
}

// Fresnel Reflection Coefficient (TE Polarization)
__device__ float fresnel_reflection(Vec3 incident, Vec3 normal, float freq_hz) {
    float cos_theta_i = -dot(incident, normal);
    if (cos_theta_i < 0) cos_theta_i = -cos_theta_i; // Ensure positive incidence angle
    
    // Complex permittivity
    float omega = 2.0f * PI * freq_hz;
    float eps_complex = EPSILON_R - (SIGMA / (omega * 8.854e-12f)); 
    
    float sin_theta_i_sq = 1.0f - cos_theta_i * cos_theta_i;
    float sqrt_term = sqrtf(fmaxf(0.0f, eps_complex - sin_theta_i_sq));
    
    // R_TE formulation
    float r_te = (cos_theta_i - sqrt_term) / (cos_theta_i + sqrt_term + 1e-6f);
    return r_te;
}

// UTD (Uniform Theory of Diffraction) Single Knife-Edge approx via Fresnel Integral formulation
__device__ float knife_edge_diffraction(float d1, float d2, float h, float lambda) {
    // Fresnel-Kirchhoff diffraction parameter
    float nu = h * sqrtf(2.0f * (d1 + d2) / (lambda * d1 * d2));
    if (nu <= -1.0f) return 1.0f; // LOS clear
    if (nu > 2.4f) return 0.225f / nu; // Deep shadow approximation
    
    // Lee's piecewise approximation for Diffraction Loss in linear scale
    float loss_db = 0;
    if (nu >= -1.0f && nu <= 0.0f) loss_db = 20.0f * log10f(0.5f - 0.62f * nu);
    else if (nu > 0.0f && nu <= 1.0f) loss_db = 20.0f * log10f(0.5f * expf(-0.95f * nu));
    else if (nu > 1.0f && nu <= 2.4f) loss_db = 20.0f * log10f(0.4f - sqrtf(0.1184f - (0.38f - 0.1f * nu) * (0.38f - 0.1f * nu)));
    
    return powf(10.0f, -loss_db / 20.0f); // Convert positive dB loss back to linear E-field multiplier
}

// 3GPP Parabolic Antenna Pattern Approximation
__device__ float calc_antenna_gain(Vec3 ray_dir, SimParams params) {
    // Convert Azimuth/Elevation to pointing vector
    float az_rad = params.tx_azimuth_deg * PI / 180.0f;
    float el_rad = params.tx_elevation_deg * PI / 180.0f;
    
    // Assuming Z is North, X is East, Y is Up
    Vec3 main_lobe = {
        sinf(az_rad) * cosf(el_rad),
        sinf(el_rad),
        cosf(az_rad) * cosf(el_rad)
    };
    
    float angle = acosf(fmaxf(-1.0f, fminf(1.0f, dot(ray_dir, main_lobe))));
    
    // Gain decay: G = G_max - 12 * (angle / beamwidth)^2
    float attenuation = 12.0f * powf(angle / params.beamwidth_rad, 2.0f);
    float gain_db = params.tx_gain_dbi - attenuation;
    
    // Front-to-back ratio floor (assume 25 dB)
    return fmaxf(gain_db, params.tx_gain_dbi - 25.0f);
}

// Pass 1: Direct LOS & Diffraction Grid pass
__global__ void los_diffraction_kernel(
    Triangle* mesh, int num_triangles,
    SimParams params,
    float* rx_grid_real, float* rx_grid_imag)
{
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_z = blockIdx.y * blockDim.y + threadIdx.y;

    if (grid_x >= params.grid_width || grid_z >= params.grid_height) return;

    float rx_x = params.bounds_min_x + grid_x * params.cell_size + (params.cell_size * 0.5f);
    float rx_z = params.bounds_min_z + grid_z * params.cell_size + (params.cell_size * 0.5f);

    // Fast vertical raycast to find ground height at this Rx cell
    float rx_y = 0.0f; 
    float min_y_t = 1e9f;
    Vec3 ray_o = {rx_x, 5000.0f, rx_z};
    Vec3 ray_d = {0.0f, -1.0f, 0.0f};
    for(int i=0; i<num_triangles; ++i) {
        float t; Vec3 n;
        if(intersect_triangle(ray_o, ray_d, mesh[i], t, n)) {
            if(t < min_y_t) { min_y_t = t; rx_y = 5000.0f - t; }
        }
    }
    rx_y += 1.5f; // Mobile user height (1.5m AGL)

    Vec3 tx_pos = {params.tx_lat, params.tx_alt, params.tx_lon};
    Vec3 rx_pos = {rx_x, rx_y, rx_z};
    
    Vec3 dir = sub(rx_pos, tx_pos);
    float dist = length(dir);
    dir = normalize(dir);

    float lambda = C_LIGHT / params.freq_hz;
    float k = 2.0f * PI / lambda;
    
    // Calculate Directional Antenna Gain
    float gain_dbi = calc_antenna_gain(dir, params);
    float eirp_dbm = params.tx_power_dbm + gain_dbi;
    
    float p_watts = powf(10.0f, (eirp_dbm - 30.0f) / 10.0f);
    float e_init = sqrtf(30.0f * p_watts);
    float e_mag = e_init * (lambda / (4.0f * PI * dist)); // FSPL
    
    // Knife edge obstruction check
    float max_h = 0.0f;
    float block_d1 = dist * 0.5f; 
    float block_d2 = dist * 0.5f;

    for(int i=0; i<num_triangles; ++i) {
        float t; Vec3 n;
        // Check if terrain intersects the direct LOS line
        if(intersect_triangle(tx_pos, dir, mesh[i], t, n)) {
            if(t > 0.1f && t < dist) {
                // Estimate obstacle height (h) for UTD
                Vec3 v0_to_tx = sub(mesh[i].v0, tx_pos);
                float proj = dot(v0_to_tx, dir);
                Vec3 proj_pt = add(tx_pos, mul(dir, proj));
                float h = length(sub(mesh[i].v0, proj_pt));
                
                if (h > max_h) {
                    max_h = h;
                    block_d1 = proj;
                    block_d2 = dist - proj;
                }
            }
        }
    }

    if (max_h > 0.01f) {
        e_mag *= knife_edge_diffraction(block_d1, block_d2, max_h, lambda);
    }

    float phase = -k * dist;
    int idx = grid_z * params.grid_width + grid_x;
    rx_grid_real[idx] = e_mag * cosf(phase);
    rx_grid_imag[idx] = e_mag * sinf(phase);
}

// CUDA Kernel: Shooting and Bouncing Rays (SBR)
__global__ void sbr_kernel(
    Triangle* mesh, int num_triangles,
    SimParams params,
    float* rx_grid_real, float* rx_grid_imag) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.ray_count) return;

    // 1. Generate Spherical Fibonacci Ray (Using double precision to fix ray clumping bug)
    float phi = acosf(1.0f - 2.0f * (idx + 0.5f) / params.ray_count);
    double theta_d = 3.141592653589793238 * (1.0 + sqrt(5.0)) * (double)idx;
    float theta = (float)fmod(theta_d, 2.0 * 3.141592653589793238);
    
    Vec3 ray_dir = {
        sinf(phi) * sinf(theta), // X is East
        cosf(phi),               // Y is Up
        sinf(phi) * cosf(theta)  // Z is North
    };
    
    Vec3 ray_orig = {params.tx_lat, params.tx_alt, params.tx_lon};
    float lambda = C_LIGHT / params.freq_hz;
    float k = 2.0f * PI / lambda;

    // Apply Directional Antenna Gain for this specific ray
    float gain_dbi = calc_antenna_gain(ray_dir, params);
    float eirp_dbm = params.tx_power_dbm + gain_dbi;
    float p_watts = powf(10.0f, (eirp_dbm - 30.0f) / 10.0f);
    float e_init = sqrtf(30.0f * p_watts); 
    
    float current_e_mag = e_init;
    float accumulated_dist = 0.0f;
    float phase_shift = 0.0f;

    // Raycast Bouncing Loop (Start at bounce 1, since bounce 0 is handled by los_diffraction_kernel)
    for (int bounce = 1; bounce <= params.max_bounces; bounce++) {
        float min_t = 1e9f;
        Vec3 hit_normal = {0,0,0};
        bool hit = false;
        
        // Brute force mesh intersection (For production: replace with OptiX or BVH)
        for (int i = 0; i < num_triangles; i++) {
            float t;
            Vec3 n;
            if (intersect_triangle(ray_orig, ray_dir, mesh[i], t, n)) {
                if (t < min_t) {
                    min_t = t;
                    hit_normal = n;
                    hit = true;
                }
            }
        }

        if (!hit) break; // Ray escaped to sky

        Vec3 hit_point = add(ray_orig, mul(ray_dir, min_t));
        accumulated_dist += min_t;
        
        // Map hit point to Rx Grid coordinates to accumulate power
        int grid_x = (int)((hit_point.x - params.bounds_min_x) / params.cell_size);
        int grid_z = (int)((hit_point.z - params.bounds_min_z) / params.cell_size);

        if (grid_x >= 0 && grid_x < params.grid_width && grid_z >= 0 && grid_z < params.grid_height) {
            
            // FSPL: Free Space Path Loss E-field magnitude degradation
            // E = E0 * (lambda / (4 * pi * d))
            float fspl_mag = current_e_mag * (lambda / (4.0f * PI * accumulated_dist));
            
            // Phase accumulation
            phase_shift = -k * accumulated_dist;

            // Map to Complex E-Field
            float e_real = fspl_mag * cosf(phase_shift);
            float e_imag = fspl_mag * sinf(phase_shift);
            
            int grid_idx = grid_z * params.grid_width + grid_x;
            
            // Atomically add complex field components (Multipath Superposition)
            atomicAdd(&rx_grid_real[grid_idx], e_real);
            atomicAdd(&rx_grid_imag[grid_idx], e_imag);
        }

        // Apply Fresnel Reflection + Rayleigh Diffuse Scattering Loss
        float cos_theta_i = fabs(dot(ray_dir, hit_normal));
        float R = fresnel_reflection(ray_dir, hit_normal, params.freq_hz);
        
        // Rayleigh Roughness specular reduction factor
        float roughness_term = (4.0f * PI * ROUGHNESS_RMS * cos_theta_i) / lambda;
        float specular_rho = expf(-0.5f * roughness_term * roughness_term);
        
        current_e_mag *= (R * specular_rho); 

        // Update ray for next bounce
        ray_orig = add(hit_point, mul(hit_normal, 1e-4f)); // Offset to avoid self-intersection
        // r = d - 2(d.n)n
        ray_dir = sub(ray_dir, mul(hit_normal, 2.0f * dot(ray_dir, hit_normal)));
        ray_dir = normalize(ray_dir);
    }
}

// C++ Bridge to manage CUDA Memory and Launch
extern "C" void run_rf_simulation(
    const std::vector<Triangle>& mesh,
    const SimParams& params,
    std::vector<float>& out_rx_power_dbm) 
{
    int grid_size = params.grid_width * params.grid_height;
    out_rx_power_dbm.resize(grid_size, -120.0f); // Default to noise floor

    Triangle* d_mesh;
    float *d_rx_real, *d_rx_imag;

    cudaMalloc(&d_mesh, mesh.size() * sizeof(Triangle));
    cudaMemcpy(d_mesh, mesh.data(), mesh.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&d_rx_real, grid_size * sizeof(float));
    cudaMalloc(&d_rx_imag, grid_size * sizeof(float));
    cudaMemset(d_rx_real, 0, grid_size * sizeof(float));
    cudaMemset(d_rx_imag, 0, grid_size * sizeof(float));

    // 1. Launch Direct LOS & Diffraction Grid Kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((params.grid_width + blockSize.x - 1) / blockSize.x, 
                  (params.grid_height + blockSize.y - 1) / blockSize.y);
    los_diffraction_kernel<<<gridSize, blockSize>>>(d_mesh, mesh.size(), params, d_rx_real, d_rx_imag);
    cudaDeviceSynchronize();

    // 2. Launch Ray Launching Kernel (Multipath Reflections)
    int threads_per_block = 256;
    int blocks = (params.ray_count + threads_per_block - 1) / threads_per_block;
    sbr_kernel<<<blocks, threads_per_block>>>(d_mesh, mesh.size(), params, d_rx_real, d_rx_imag);
    cudaDeviceSynchronize();

    std::vector<float> h_rx_real(grid_size);
    std::vector<float> h_rx_imag(grid_size);
    cudaMemcpy(h_rx_real.data(), d_rx_real, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rx_imag.data(), d_rx_imag, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Post-process complex E-field to Received Power (dBm)
    for (int i = 0; i < grid_size; i++) {
        float real = h_rx_real[i];
        float imag = h_rx_imag[i];
        
        float e_mag_sq = real * real + imag * imag;
        
        if (e_mag_sq > 1e-15f) {
            // Convert E-Field^2 back to Power (Watts), then dBm
            // P_rx = E^2 / (120*pi) * A_eff
            float lambda = C_LIGHT / params.freq_hz;
            float a_eff = (lambda * lambda) / (4.0f * PI); // Antenna aperture
            float p_watts = (e_mag_sq / (120.0f * PI)) * a_eff;
            out_rx_power_dbm[i] = 10.0f * log10f(p_watts) + 30.0f; // Convert W to dBm
        } else {
            out_rx_power_dbm[i] = -120.0f; // Noise floor
        }
    }

    cudaFree(d_mesh);
    cudaFree(d_rx_real);
    cudaFree(d_rx_imag);
}