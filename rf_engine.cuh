#ifndef RF_ENGINE_CUH
#define RF_ENGINE_CUH

#include <vector>
#include <cstdint>

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

// Simulation Configuration Parameters
struct SimParams {
    float tx_x, tx_y, tx_z;       // Transmitter location (Easting, Altitude, Northing)
    float freq_hz;                // Frequency in Hz
    float tx_power_dbm;           // Tx Power in dBm
    float tx_azimuth_deg;         // Antenna Azimuth (0=N, 90=E)
    float tx_elevation_deg;       // Antenna Downtilt/Uptilt
    float tx_gain_dbi;            // Antenna Gain in dBi
    float beamwidth_rad;          // Antenna 3dB Beamwidth
    
    float bounds_min_x, bounds_min_z; // Grid physical bounds
    float cell_size;              // Distance between Rx grid points
    int grid_width, grid_height;  // Rx matrix dimensions
    int max_bounces;              // Max ray bounces
    int ray_count;                // Total rays to launch
};

// C++ Bridge Function to launch CUDA simulation
extern "C" void run_rf_simulation(
    const VoxelGrid& grid,
    const SimParams& params,
    std::vector<float>& out_rx_power_dbm,
    std::vector<float>& out_delay_spread_ns
);

#endif // RF_ENGINE_CUH