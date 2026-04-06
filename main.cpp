#include "crow.h"
#include "rf_engine.cuh"
#include <sqlite3.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

struct Point3D { float e, n, a; int cls; };

// Function to pull LiDAR from your custom SQLite DB and build a 3D Voxel Grid
bool build_voxel_grid(
    float center_e, float center_n, float radius, float cell_size,
    std::vector<uint8_t>& voxel_data, VoxelGrid& grid_info, std::vector<float>& surface_heightmap) 
{
    sqlite3* db;
    // Check current directory first, EXPLICITLY requesting Read-Only access to avoid Docker permission errors
    int rc = sqlite3_open_v2("terrain.db", &db, SQLITE_OPEN_READONLY, nullptr);
    if (rc != SQLITE_OK) {
        // Fallback: Check parent directory if running from the build/ folder
        rc = sqlite3_open_v2("../terrain.db", &db, SQLITE_OPEN_READONLY, nullptr);
        if (rc != SQLITE_OK) {
            std::cerr << "[ERROR] Failed to open terrain.db. SQLite error code: " << rc << std::endl;
            if (db) {
                std::cerr << "[ERROR] SQLite extended error: " << sqlite3_errmsg(db) << std::endl;
                sqlite3_close(db);
            }
            std::cerr << "[ERROR] Please ensure terrain.db is located in the project root directory!" << std::endl;
            return false;
        } else {
            std::cout << "[INFO] Successfully opened ../terrain.db (root directory) in Read-Only mode." << std::endl;
        }
    } else {
        std::cout << "[INFO] Successfully opened terrain.db in Read-Only mode." << std::endl;
    }

    // Utilizing the R-Tree virtual index for blazing-fast spatial bounding-box lookups
    std::string q = "SELECT p.easting, p.northing, p.altitude, p.class_code "
                    "FROM ProcessedLiDARPoints_idx AS idx "
                    "JOIN ProcessedLiDARPoints AS p ON p.id = idx.id "
                    "WHERE idx.min_x BETWEEN ? AND ? AND idx.min_y BETWEEN ? AND ?";

    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db, q.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_double(stmt, 1, center_e - radius);
    sqlite3_bind_double(stmt, 2, center_e + radius);
    sqlite3_bind_double(stmt, 3, center_n - radius);
    sqlite3_bind_double(stmt, 4, center_n + radius);

    std::vector<Point3D> points;
    float min_alt = 1e9, max_alt = -1e9;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        float e = sqlite3_column_double(stmt, 0);
        float n = sqlite3_column_double(stmt, 1);
        float a = sqlite3_column_double(stmt, 2);
        int cls = sqlite3_column_int(stmt, 3);
        
        points.push_back({e, n, a, cls});
        if (a < min_alt) min_alt = a;
        if (a > max_alt) max_alt = a;
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    std::cout << "[INFO] Successfully closed terrain.db. Fetched " << points.size() << " LiDAR points for this bounding box." << std::endl;

    if (points.empty()) {
        std::cerr << "[WARNING] 0 LiDAR points found in the specified radius! Check your coordinates." << std::endl;
        return false;
    }

    // Define Grid Dimensions
    grid_info.min_x = center_e - radius;
    grid_info.min_z = center_n - radius;
    grid_info.min_y = min_alt;
    grid_info.cell_size = cell_size;
    
    grid_info.dim_x = ceil((radius * 2.0f) / cell_size);
    grid_info.dim_z = ceil((radius * 2.0f) / cell_size);
    grid_info.dim_y = ceil((max_alt - min_alt) / cell_size) + 1;

    voxel_data.assign(grid_info.dim_x * grid_info.dim_y * grid_info.dim_z, 0);
    
    // Fill Voxels
    for (const auto& p : points) {
        int gx = (p.e - grid_info.min_x) / cell_size;
        int gz = (p.n - grid_info.min_z) / cell_size;
        int gy = (p.a - grid_info.min_y) / cell_size;
        
        if (gx >= 0 && gx < grid_info.dim_x && gz >= 0 && gz < grid_info.dim_z && gy >= 0 && gy < grid_info.dim_y) {
            int idx = gy * (grid_info.dim_x * grid_info.dim_z) + gz * grid_info.dim_x + gx;
            // Differentiate Ground (2) vs Building (6) vs Trees (3-5)
            // Storing classification directly into the voxel array for advanced material logic later
            voxel_data[idx] = (p.cls > 0) ? p.cls : 1; 
        }
    }

    grid_info.data = voxel_data.data();

    // Generate a lightweight 2.5D top-surface heightmap to send to the web browser
    // This allows the frontend to easily render a mesh that wraps the LiDAR shape (buildings included)
    surface_heightmap.assign(grid_info.dim_x * grid_info.dim_z, min_alt);
    for (int x = 0; x < grid_info.dim_x; x++) {
        for (int z = 0; z < grid_info.dim_z; z++) {
            for (int y = grid_info.dim_y - 1; y >= 0; y--) {
                int idx = y * (grid_info.dim_x * grid_info.dim_z) + z * grid_info.dim_x + x;
                if (voxel_data[idx] > 0) {
                    surface_heightmap[z * grid_info.dim_x + x] = min_alt + (y * cell_size);
                    break;
                }
            }
        }
    }

    return true;
}

int main() {
    crow::SimpleApp app;

    CROW_ROUTE(app, "/")([]() {
        std::ifstream file("index.html");
        if (!file.is_open()) file.open("../index.html");
        if (!file.is_open()) return crow::response(404, "index.html not found!");
        std::stringstream buffer;
        buffer << file.rdbuf();
        crow::response res(buffer.str());
        res.set_header("Content-Type", "text/html");
        return res;
    });

    CROW_ROUTE(app, "/api/simulate").methods(crow::HTTPMethod::POST)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");

        try {
            // New Input: Easting / Northing to match your LiDAR DB
            float easting = body.has("easting") ? body["easting"].d() : 500000.0;
            float northing = body.has("northing") ? body["northing"].d() : 4000000.0;
            float radius = body.has("radius") ? body["radius"].d() : 100.0;
            float voxel_res = body.has("resolution") ? body["resolution"].d() : 2.0; // 2m voxels
            
            float freq = body.has("freq_mhz") ? body["freq_mhz"].d() * 1e6 : 900e6;
            float tx_h = body.has("tx_height") ? body["tx_height"].d() : 10.0;
            float tx_p = body.has("tx_power") ? body["tx_power"].d() : 43.0;
            float tx_az = body.has("tx_azimuth") ? body["tx_azimuth"].d() : 180.0;
            float tx_el = body.has("tx_elevation") ? body["tx_elevation"].d() : -2.0;
            float tx_gain = body.has("tx_gain") ? body["tx_gain"].d() : 18.0;
            float beamwidth = body.has("beamwidth") ? body["beamwidth"].d() : 65.0;

            std::vector<uint8_t> voxel_data;
            VoxelGrid grid_info;
            std::vector<float> surface_mesh;

            if (!build_voxel_grid(easting, northing, radius, voxel_res, voxel_data, grid_info, surface_mesh)) {
                return crow::response(400, "No LiDAR points found in this area. Check your DB and coordinates.");
            }

            // Estimate ground height at center
            float center_ground = surface_mesh[(grid_info.dim_z/2) * grid_info.dim_x + (grid_info.dim_x/2)];

            SimParams params;
            params.tx_x = easting;
            params.tx_y = center_ground + tx_h;
            params.tx_z = northing;
            params.freq_hz = freq;
            params.tx_power_dbm = tx_p;
            params.tx_azimuth_deg = tx_az;
            params.tx_elevation_deg = tx_el;
            params.tx_gain_dbi = tx_gain;
            params.beamwidth_rad = beamwidth * (M_PI / 180.0);
            
            params.bounds_min_x = grid_info.min_x;
            params.bounds_min_z = grid_info.min_z;
            params.grid_width = grid_info.dim_x;
            params.grid_height = grid_info.dim_z;
            params.cell_size = grid_info.cell_size;
            params.max_bounces = 3; 
            params.ray_count = 500000; 

            std::vector<float> rx_power_dbm;
            run_rf_simulation(grid_info, params, rx_power_dbm);

            crow::json::wvalue res;
            res["grid_w"] = grid_info.dim_x;
            res["grid_h"] = grid_info.dim_z;
            res["cell_size"] = grid_info.cell_size;
            res["min_alt"] = grid_info.min_y;
            
            res["surface_heights"] = surface_mesh;
            res["heatmap_dbm"] = std::vector<float>(rx_power_dbm.begin(), rx_power_dbm.end());

            return crow::response(res);
        } catch (const std::exception& e) {
            std::cerr << "Simulation Error: " << e.what() << std::endl;
            return crow::response(500, "Internal Server Error");
        }
    });

    app.port(8080).multithreaded().run();
    return 0;
}