#ifndef RF_ENGINE_CUH
#define RF_ENGINE_CUH

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

// 3D Vector Math Structure
struct Vec3 {
    float x, y, z;
};

// Represents the 3D Voxel Grid in memory
struct VoxelGrid {
    uint8_t* data;
    int dim_x, dim_y, dim_z;
    float min_x, min_y, min_z;
    float cell_size;
};

// Continuous Triangle Mesh for OptiX
struct TriangleMesh {
    std::vector<float> vertices; 
    std::vector<int> indices;    
    std::vector<int> materials;  
};

// Simulation Configuration Parameters
struct SimParams {
    float tx_x, tx_y, tx_z;       
    float freq_hz;                
    float tx_power_dbm;           
    float tx_azimuth_deg;         
    float tx_elevation_deg;       
    float tx_gain_dbi;            
    float rx_gain_dbi;            
    float beamwidth_rad;          
    float vertical_beamwidth_rad; 
    
    float bounds_min_x, bounds_min_z; 
    float cell_size;              
    int grid_width, grid_height;  
    int max_bounces;              
    int ray_count;                

    float* d_rx_grid_re;
    float* d_rx_grid_im;
    
    unsigned long long gas_handle;
    cudaTextureObject_t antenna_tex;
};

// OptiX Launch Parameters Payload
struct OptixLaunchParams {
    SimParams params;
    VoxelGrid grid; // Needed by OptiX to calculate Weissberger foliage loss
    float* min_dist_grid;
    float* max_dist_grid;
    float* mesh_vertices;
    int* mesh_indices;
};

extern "C" void run_rf_simulation(
    const VoxelGrid& grid,
    const TriangleMesh& mesh,
    const std::vector<float>& antenna_pattern,
    const SimParams& host_params,
    std::vector<float>& out_rx_power_dbm,
    std::vector<float>& out_delay_spread_ns
);

#endif // RF_ENGINE_CUH