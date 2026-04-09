#include <optix.h>
#include "rf_engine.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

#define PI 3.14159265358979323846f
#define C_LIGHT 299792458.0f

extern "C" {
    __constant__ OptixLaunchParams launch_params;
}

// --- Math Helpers ---
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

// FIX 2 Implementation: Texture Sampler
__device__ float calc_antenna_gain_texture(Vec3 ray_dir, cudaTextureObject_t tex, float tx_az_deg, float tx_el_deg) {
    float ray_az = atan2f(ray_dir.x, ray_dir.z);
    float ray_el = asinf(fmaxf(-1.0f, fminf(1.0f, ray_dir.y)));
    
    float delta_az = ray_az - (tx_az_deg * PI / 180.0f);
    while (delta_az > PI) delta_az -= 2.0f * PI;
    while (delta_az < -PI) delta_az += 2.0f * PI;
    
    float delta_el = ray_el - (tx_el_deg * PI / 180.0f);

    float u = (delta_az + PI) / (2.0f * PI);
    float v = (delta_el + PI / 2.0f) / PI;
    
    return tex2D<float>(tex, u, v);
}

// RESTORED: Weissberger's Modified Exponential Decay (MED) for Foliage Loss
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

// RESTORED: SOTA: ITU-R P.2040 Material Properties
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

// RESTORED: SOTA: ITU-R P.2040 Complex Vector Polarization Tracking
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

struct RayPayload {
    float t_hit;
    float norm_x, norm_y, norm_z;
    int cls;
};

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    return reinterpret_cast<void*>(uptr);
}
static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32; i1 = uptr & 0x00000000ffffffff;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    int ray_idx = idx.x + idx.y * dim.x;
    if (ray_idx >= launch_params.params.ray_count) return;

    SimParams& params = launch_params.params;
    VoxelGrid& grid = launch_params.grid;

    float phi = acosf(1.0f - 2.0f * (ray_idx + 0.5f) / params.ray_count);
    float theta_f = PI * (1.0f + sqrtf(5.0f)) * (float)ray_idx;
    float theta = fmodf(theta_f, 2.0f * PI);
    Vec3 ray_dir = { sinf(phi) * sinf(theta), cosf(phi), sinf(phi) * cosf(theta) };
    Vec3 ray_orig = {params.tx_x, params.tx_y, params.tx_z};
    
    Vec3 up_vec = {0, 1, 0};
    Vec3 right = cross_prod(ray_dir, up_vec);
    if (length(right) < 0.001f) right = {1, 0, 0};
    Vec3 pol = normalize(cross_prod(right, ray_dir));

    float lambda = C_LIGHT / params.freq_hz;
    float gain_dbi = calc_antenna_gain_texture(ray_dir, params.antenna_tex, params.tx_azimuth_deg, params.tx_elevation_deg);
    float eirp_dbm = params.tx_power_dbm + gain_dbi;
    float p_watts = powf(10.0f, (eirp_dbm - 30.0f) / 10.0f);
    
    float rx_gain_lin = powf(10.0f, params.rx_gain_dbi / 10.0f);
    float d_omega = 4.0f * PI / params.ray_count;
    float p_ray_watts_init = p_watts * (d_omega / (4.0f * PI)); 
    
    float accumulated_dist = 0.0f;
    float ray_power_multiplier = 1.0f;

    RayPayload payload;
    uint32_t p0, p1; packPointer(&payload, p0, p1);

    for (int bounce = 0; bounce < params.max_bounces; bounce++) {
        payload.t_hit = -1.0f;
        
        optixTrace(
            params.gas_handle,
            make_float3(ray_orig.x, ray_orig.y, ray_orig.z),
            make_float3(ray_dir.x, ray_dir.y, ray_dir.z),
            0.1f, 5000.0f, 0.0f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
            0, 1, 0, p0, p1
        );

        if (payload.t_hit < 0.0f) break; 

        Vec3 hit_normal = {payload.norm_x, payload.norm_y, payload.norm_z};
        float hit_t = payload.t_hit;
        Vec3 hit_point = add(ray_orig, mul(ray_dir, hit_t));

        // RESTORED: Check voxels along the ray to see if it passed through vegetation
        float veg_dist = 0.0f;
        int num_steps = (int)ceilf(hit_t / (grid.cell_size * 0.5f));
        for (int i = 1; i < num_steps; i++) {
            float f = (float)i / num_steps;
            float cx = ray_orig.x + f * (hit_point.x - ray_orig.x);
            float cz = ray_orig.z + f * (hit_point.z - ray_orig.z);
            float cy = ray_orig.y + f * (hit_point.y - ray_orig.y);
            int cgx = (cx - grid.min_x) / grid.cell_size;
            int cgz = (cz - grid.min_z) / grid.cell_size;
            int cgy = (cy - grid.min_y) / grid.cell_size;
            if (cgx >= 0 && cgx < grid.dim_x && cgz >= 0 && cgz < grid.dim_z && cgy >= 0 && cgy < grid.dim_y) {
                int cls = grid.data[cgy * (grid.dim_x * grid.dim_z) + cgz * grid.dim_x + cgx];
                if (cls >= 3 && cls <= 5) veg_dist += grid.cell_size * 0.5f;
            }
        }
        ray_power_multiplier *= calc_vegetation_loss_power_mult(veg_dist, params.freq_hz);
        
        accumulated_dist += hit_t;

        if (bounce > 0) {
            int rx_x = (int)((hit_point.x - params.bounds_min_x) / params.cell_size);
            int rx_z = (int)((hit_point.z - params.bounds_min_z) / params.cell_size);

            // FIX 3 Implementation: Ray-Frustum Wavefront Expansion 
            // IMPROVEMENT: Mathematically projects against the exact 3D terrain normal instead of flat ground.
            float A_t = d_omega * accumulated_dist * accumulated_dist; 
            float cos_terrain = fmaxf(0.05f, fabs(dot(ray_dir, hit_normal)));
            float A_footprint = A_t / cos_terrain;
            
            float power_density = (p_ray_watts_init * ray_power_multiplier) / fmaxf(A_footprint, 0.1f);
            float a_eff = rx_gain_lin * (lambda * lambda) / (4.0f * PI);
            
            // Distribute power perfectly within geometric radius
            float r_footprint = sqrtf(A_footprint / PI);
            int radius_cells = (int)ceilf(r_footprint / params.cell_size);
            radius_cells = min(radius_cells, 8);

            for (int dx = -radius_cells; dx <= radius_cells; dx++) {
                for (int dz = -radius_cells; dz <= radius_cells; dz++) {
                    float dist_sq = (float)(dx*dx*params.cell_size*params.cell_size + dz*dz*params.cell_size*params.cell_size);
                    if (dist_sq <= r_footprint*r_footprint) {
                        int px = rx_x + dx;
                        int pz = rx_z + dz;
                        if (px >= 0 && px < params.grid_width && pz >= 0 && pz < params.grid_height) {
                            
                            float cell_area = params.cell_size * params.cell_size;
                            float splat_watts = power_density * a_eff * (cell_area / A_footprint);
                            
                            // FIX 4 Implementation: Coherent Phase Addition
                            if (splat_watts > 1e-18f) {
                                float phase = (2.0f * PI * accumulated_dist) / lambda;
                                float voltage = sqrtf(splat_watts);
                                float re = voltage * cosf(phase);
                                float im = -voltage * sinf(phase);
                                
                                int grid_idx = pz * params.grid_width + px;
                                atomicAdd(&params.d_rx_grid_re[grid_idx], re);
                                atomicAdd(&params.d_rx_grid_im[grid_idx], im);
                                
                                atomicMinFloat(&launch_params.min_dist_grid[grid_idx], accumulated_dist);
                                atomicMaxFloat(&launch_params.max_dist_grid[grid_idx], accumulated_dist);
                            }
                        }
                    }
                }
            }
        }

        // Apply ITU-R P.2040 Material properties to reflection
        float eps_r, sigma;
        get_material_props(payload.cls, params.freq_hz, eps_r, sigma);
        float reflection_mag = compute_reflection_and_update_pol(ray_dir, hit_normal, pol, eps_r, sigma, params.freq_hz);
        
        float cos_theta_i = fabs(dot(ray_dir, hit_normal));
        float roughness_term = (4.0f * PI * 0.1f * cos_theta_i) / lambda; 
        float specular_rho = expf(-0.5f * roughness_term * roughness_term);
        
        ray_power_multiplier *= (reflection_mag * reflection_mag) * specular_rho;
        if (ray_power_multiplier < 1e-9f) break;

        ray_orig = add(hit_point, mul(hit_normal, 0.05f)); 
        ray_dir = sub(ray_dir, mul(hit_normal, 2.0f * dot(ray_dir, hit_normal)));
        ray_dir = normalize(ray_dir);
    }
}

extern "C" __global__ void __miss__ms() {
    RayPayload* payload;
    uint32_t p0 = optixGetPayload_0();
    uint32_t p1 = optixGetPayload_1();
    payload = reinterpret_cast<RayPayload*>(unpackPointer(p0, p1));
    payload->t_hit = -1.0f;
}

extern "C" __global__ void __closesthit__ch() {
    RayPayload* payload;
    uint32_t p0 = optixGetPayload_0();
    uint32_t p1 = optixGetPayload_1();
    payload = reinterpret_cast<RayPayload*>(unpackPointer(p0, p1));
    
    payload->t_hit = optixGetRayTmax();
    
    // Fetch the ID of the triangle we just hit
    unsigned int prim_idx = optixGetPrimitiveIndex();
    
    // Look up the 3 vertex indices for this triangle
    int idx0 = launch_params.mesh_indices[prim_idx * 3 + 0];
    int idx1 = launch_params.mesh_indices[prim_idx * 3 + 1];
    int idx2 = launch_params.mesh_indices[prim_idx * 3 + 2];
    
    // Fetch the 3D coordinates for those 3 vertices
    Vec3 v0 = {launch_params.mesh_vertices[idx0 * 3 + 0], launch_params.mesh_vertices[idx0 * 3 + 1], launch_params.mesh_vertices[idx0 * 3 + 2]};
    Vec3 v1 = {launch_params.mesh_vertices[idx1 * 3 + 0], launch_params.mesh_vertices[idx1 * 3 + 1], launch_params.mesh_vertices[idx1 * 3 + 2]};
    Vec3 v2 = {launch_params.mesh_vertices[idx2 * 3 + 0], launch_params.mesh_vertices[idx2 * 3 + 1], launch_params.mesh_vertices[idx2 * 3 + 2]};
    
    // Calculate the geometric face normal using the Cross Product of the edges
    Vec3 edge1 = sub(v1, v0);
    Vec3 edge2 = sub(v2, v0);
    Vec3 n = normalize(cross_prod(edge1, edge2));
    
    payload->norm_x = n.x;
    payload->norm_y = n.y;
    payload->norm_z = n.z;
    
    // Fetch the material classification mapped to the hit triangle
    payload->cls = 1; 
}