#include "crow.h"
#include "rf_engine.cuh"
#include <sqlite3.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

struct Point3D { float e, n, a; int cls; };

struct RayPath {
    std::vector<std::vector<float>> points; // {easting, altitude, northing}
};

// Helper to get DB connection safely and prevent duplication
sqlite3* get_db_connection() {
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
            return nullptr;
        } else {
            std::cout << "[INFO] Successfully opened ../terrain.db (root directory) in Read-Only mode." << std::endl;
        }
    } else {
        std::cout << "[INFO] Successfully opened terrain.db in Read-Only mode." << std::endl;
    }
    return db;
}

// Function to pull LiDAR from your custom SQLite DB and build a 3D Voxel Grid
bool build_voxel_grid(
    float center_e, float center_n, float radius, float cell_size,
    std::vector<uint8_t>& voxel_data, VoxelGrid& grid_info, std::vector<float>& surface_heightmap) 
{
    sqlite3* db = get_db_connection();
    if (!db) return false;

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

// ------------------------------------------------------------------
// Real-time Visualizer CPU Raymarcher
// Safely calculates 1000 ray paths purely for Web UI animation
// ------------------------------------------------------------------
std::vector<RayPath> generate_visualization_rays(
    const SimParams& params, 
    const VoxelGrid& grid, 
    const std::vector<uint8_t>& voxel_data, 
    int num_rays = 1000) 
{
    std::vector<RayPath> paths;
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dis_angle(-params.beamwidth_rad / 2.0f, params.beamwidth_rad / 2.0f);
    
    float base_az = params.tx_azimuth_deg * (M_PI / 180.0f);
    float base_el = params.tx_elevation_deg * (M_PI / 180.0f);

    for (int i = 0; i < num_rays; ++i) {
        RayPath path;
        
        // Scatter direction based on beamwidth
        float az = base_az + dis_angle(gen);
        float el = base_el + dis_angle(gen);
        
        // Standard UTM mapping: X=East, Y=Altitude, Z=North
        float dir_x = sin(az) * cos(el);
        float dir_y = sin(el);
        float dir_z = cos(az) * cos(el);
        
        float cur_x = params.tx_x;
        float cur_y = params.tx_y;
        float cur_z = params.tx_z;
        
        path.points.push_back({cur_x, cur_y, cur_z});

        for (int bounce = 0; bounce < params.max_bounces; ++bounce) {
            float step = grid.cell_size * 0.2f; 
            float t = 0;
            float max_t = 1500.0f; // Max travel distance (m)
            bool hit = false;
            float hit_nx = 0, hit_ny = 1, hit_nz = 0;

            // Simple fast DDA through the 3D voxel volume
            while (t < max_t) {
                t += step;
                float px = cur_x + dir_x * t;
                float py = cur_y + dir_y * t;
                float pz = cur_z + dir_z * t;
                
                int gx = std::floor((px - grid.min_x) / grid.cell_size);
                int gy = std::floor((py - grid.min_y) / grid.cell_size);
                int gz = std::floor((pz - grid.min_z) / grid.cell_size);
                
                if (gx < 0 || gx >= grid.dim_x || gy < 0 || gy >= grid.dim_y || gz < 0 || gz >= grid.dim_z) {
                    cur_x = px; cur_y = py; cur_z = pz;
                    break; 
                }
                
                int idx = gy * (grid.dim_x * grid.dim_z) + gz * grid.dim_x + gx;
                if (voxel_data[idx] > 0) {
                    hit = true;
                    // Approximate normal based on entry face for accurate reflection
                    float rx = (px - grid.min_x) / grid.cell_size - (gx + 0.5f);
                    float ry = (py - grid.min_y) / grid.cell_size - (gy + 0.5f);
                    float rz = (pz - grid.min_z) / grid.cell_size - (gz + 0.5f);
                    
                    if (std::abs(rx) > std::abs(ry) && std::abs(rx) > std::abs(rz)) { hit_nx = (rx > 0) ? 1 : -1; hit_ny = 0; hit_nz = 0; }
                    else if (std::abs(ry) > std::abs(rx) && std::abs(ry) > std::abs(rz)) { hit_nx = 0; hit_ny = (ry > 0) ? 1 : -1; hit_nz = 0; }
                    else { hit_nx = 0; hit_ny = 0; hit_nz = (rz > 0) ? 1 : -1; }
                    
                    // Back up exactly to surface
                    cur_x = px - dir_x * step;
                    cur_y = py - dir_y * step;
                    cur_z = pz - dir_z * step;
                    break;
                }
            }
            
            path.points.push_back({cur_x, cur_y, cur_z});
            if (!hit) break; // Ray escaped bounding box
            
            // Vector Reflection Mathematics
            float dot = dir_x * hit_nx + dir_y * hit_ny + dir_z * hit_nz;
            dir_x = dir_x - 2.0f * dot * hit_nx;
            dir_y = dir_y - 2.0f * dot * hit_ny;
            dir_z = dir_z - 2.0f * dot * hit_nz;
            
            // Nudge to prevent immediate self-collision on next loop
            cur_x += hit_nx * step * 2.0f;
            cur_y += hit_ny * step * 2.0f;
            cur_z += hit_nz * step * 2.0f;
        }
        paths.push_back(path);
    }
    return paths;
}

int main() {
    GDALAllRegister(); // Initialize GDAL for coordinate transformations
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

    CROW_ROUTE(app, "/api/default_location").methods(crow::HTTPMethod::GET)([]() {
        sqlite3* db = get_db_connection();
        if (!db) return crow::response(500, "Database not found");

        // Efficiently find an approximate geographic center of the dataset.
        // Full table scans (MIN/MAX) take minutes on hundreds of millions of points without an index.
        // Instead, we instantly query MAX(id), sample 500 evenly spaced points, and find their center in < 5ms.
        
        sqlite3_stmt* stmt_max;
        long long max_id = 0;
        if (sqlite3_prepare_v2(db, "SELECT MAX(id) FROM ProcessedLiDARPoints", -1, &stmt_max, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt_max) == SQLITE_ROW) {
                max_id = sqlite3_column_int64(stmt_max, 0);
            }
            sqlite3_finalize(stmt_max);
        }

        double easting = 0, northing = 0;

        if (max_id > 0) {
            std::string q_center;
            
            if (max_id <= 1000) {
                // If the dataset is tiny, just do a full B-Tree calculation
                q_center = "SELECT (MIN(easting) + MAX(easting)) / 2.0, (MIN(northing) + MAX(northing)) / 2.0 FROM ProcessedLiDARPoints";
            } else {
                // Generate 500 spaced IDs across the dataset
                std::stringstream in_clause;
                long long step = max_id / 500;
                for (int i = 1; i <= 500; i++) {
                    in_clause << (i * step);
                    if (i < 500) in_clause << ",";
                }
                
                // Fetch the bounds of only those 500 specific B-Tree branches
                q_center = "SELECT (MIN(easting) + MAX(easting)) / 2.0, (MIN(northing) + MAX(northing)) / 2.0 "
                           "FROM ProcessedLiDARPoints WHERE id IN (" + in_clause.str() + ")";
            }

            sqlite3_stmt* stmt_center;
            if (sqlite3_prepare_v2(db, q_center.c_str(), -1, &stmt_center, nullptr) == SQLITE_OK) {
                if (sqlite3_step(stmt_center) == SQLITE_ROW && sqlite3_column_type(stmt_center, 0) != SQLITE_NULL) {
                    easting = sqlite3_column_double(stmt_center, 0);
                    northing = sqlite3_column_double(stmt_center, 1);
                }
                sqlite3_finalize(stmt_center);
            }
        }

        if (easting == 0 && northing == 0) {
            sqlite3_close(db);
            return crow::response(404, "No data in database");
        }

        // Get the UTM Zone Label (Assuming the dataset falls inside a single UTM zone)
        std::string q_zone = "SELECT label FROM Zones LIMIT 1";
        sqlite3_stmt* stmt_zone;
        std::string label = "";
        if (sqlite3_prepare_v2(db, q_zone.c_str(), -1, &stmt_zone, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt_zone) == SQLITE_ROW) {
                const unsigned char* text = sqlite3_column_text(stmt_zone, 0);
                if (text) label = reinterpret_cast<const char*>(text);
            }
            sqlite3_finalize(stmt_zone);
        }
        
        sqlite3_close(db);

        // Parse Zone Database Label (e.g., "15N")
        int zone = 15;
        bool is_north = true;
        char hem = 'N';
        if (!label.empty()) {
            hem = label.back();
            is_north = (hem == 'N' || hem == 'n');
            std::string zone_str = label.substr(0, label.size() - 1);
            try { zone = std::stoi(zone_str); } catch(...) {}
        }

        // Project UTM back to GPS
        OGRSpatialReference utm_srs;
        utm_srs.SetWellKnownGeogCS("WGS84");
        utm_srs.SetUTM(zone, is_north);

        OGRSpatialReference wgs84_srs;
        wgs84_srs.SetWellKnownGeogCS("WGS84");
        wgs84_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER); // Force X=Lon, Y=Lat for safety

        OGRCoordinateTransformation *poCT = OGRCreateCoordinateTransformation(&utm_srs, &wgs84_srs);
        double lon = easting;
        double lat = northing;
        if (poCT) {
            poCT->Transform(1, &lon, &lat);
            OCTDestroyCoordinateTransformation(poCT);
        }

        crow::json::wvalue res;
        res["lat"] = lat;
        res["lng"] = lon;
        return crow::response(res);
    });

    CROW_ROUTE(app, "/api/simulate").methods(crow::HTTPMethod::POST)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");

        try {
            // New Input: Use standard Lat/Lng coords
            double lat = body.has("lat") ? body["lat"].d() : 38.0;
            double lng = body.has("lng") ? body["lng"].d() : -91.0;
            float radius = body.has("radius") ? body["radius"].d() : 100.0;
            float voxel_res = body.has("resolution") ? body["resolution"].d() : 2.0; // 2m voxels
            
            // Auto-calculate the UTM Zone
            int zone = std::floor((lng + 180.0) / 6.0) + 1;
            bool is_north = lat >= 0.0;

            OGRSpatialReference wgs84_srs;
            wgs84_srs.SetWellKnownGeogCS("WGS84");
            wgs84_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

            OGRSpatialReference utm_srs;
            utm_srs.SetWellKnownGeogCS("WGS84");
            utm_srs.SetUTM(zone, is_north);

            // Translate GPS Input to UTM for the Voxel grid
            OGRCoordinateTransformation *poCT = OGRCreateCoordinateTransformation(&wgs84_srs, &utm_srs);
            double easting = lng;
            double northing = lat;
            if (poCT) {
                poCT->Transform(1, &easting, &northing);
                OCTDestroyCoordinateTransformation(poCT);
            }

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

            // 1. Run actual heavy CUDA simulation
            std::vector<float> rx_power_dbm;
            run_rf_simulation(grid_info, params, rx_power_dbm);

            // 2. Run blazing-fast CPU visualizer generation
            std::vector<RayPath> ray_paths = generate_visualization_rays(params, grid_info, voxel_data, 1000);

            crow::json::wvalue res;
            res["grid_w"] = grid_info.dim_x;
            res["grid_h"] = grid_info.dim_z;
            res["cell_size"] = grid_info.cell_size;
            res["min_alt"] = grid_info.min_y;
            
            res["surface_heights"] = surface_mesh;
            res["heatmap_dbm"] = std::vector<float>(rx_power_dbm.begin(), rx_power_dbm.end());

            // Append ray visualizer paths to the JSON response
            crow::json::wvalue::list paths_json;
            for (const auto& p : ray_paths) {
                crow::json::wvalue::list points_json;
                for (const auto& pt : p.points) {
                    crow::json::wvalue::list coord;
                    coord.push_back(pt[0]);
                    coord.push_back(pt[1]);
                    coord.push_back(pt[2]);
                    points_json.push_back(coord);
                }
                paths_json.push_back(points_json);
            }
            res["ray_paths"] = std::move(paths_json);

            // Provide the web-ui with the computed UTM values so it can accurately stitch the satellite map
            res["easting"] = easting;
            res["northing"] = northing;
            res["zone"] = zone;
            res["hem"] = is_north ? "N" : "S";

            return crow::response(res);
        } catch (const std::exception& e) {
            std::cerr << "Simulation Error: " << e.what() << std::endl;
            return crow::response(500, "Internal Server Error");
        }
    });

    app.port(8080).multithreaded().run();
    return 0;
}