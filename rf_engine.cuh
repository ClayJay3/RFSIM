#ifndef RF_ENGINE_CUH
#define RF_ENGINE_CUH

#include <vector>

// 3D Vector Math Structure
struct Vec3 {
    float x, y, z;
};

// Represents a 3D Triangle from the DEM Mesh
struct Triangle {
    Vec3 v0, v1, v2;
};

// Complex Number Structure for Electric Field Superposition
struct Complex {
    float real;
    float imag;
};

// Simulation Configuration Parameters
struct SimParams {
    float tx_lat, tx_lon, tx_alt; // Transmitter location
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
    const std::vector<Triangle>& mesh,
    const SimParams& params,
    std::vector<float>& out_rx_power_dbm
);

#endif // RF_ENGINE_CUH