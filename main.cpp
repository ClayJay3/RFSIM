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
    std::vector<std::vector<float>> points; 
};

inline Vec3 cross_prod(Vec3 a, Vec3 b) { 
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; 
}
inline float length(Vec3 a) { return sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }
inline Vec3 normalize(Vec3 a) {
    float l = length(a);
    return l > 0 ? Vec3{a.x/l, a.y/l, a.z/l} : a;
}

sqlite3* get_db_connection() {
    sqlite3* db;
    int rc = sqlite3_open_v2("terrain.db", &db, SQLITE_OPEN_READONLY, nullptr);
    if (rc != SQLITE_OK) {
        rc = sqlite3_open_v2("../terrain.db", &db, SQLITE_OPEN_READONLY, nullptr);
        if (rc != SQLITE_OK) {
            std::cerr << "[ERROR] Failed to open terrain.db. SQLite error code: " << rc << std::endl;
            return nullptr;
        } 
    } 
    return db;
}

// FIX 1: Lightweight Block Meshing Algorithm
// Converts the Voxel Grid into a continuous 3D Triangle Mesh for Hardware Raytracing
void generate_optix_mesh(const VoxelGrid& grid, TriangleMesh& mesh) {
    float s = grid.cell_size;
    for (int y = 0; y < grid.dim_y; y++) {
        for (int z = 0; z < grid.dim_z; z++) {
            for (int x = 0; x < grid.dim_x; x++) {
                int idx = y * (grid.dim_x * grid.dim_z) + z * grid.dim_x + x;
                if (grid.data[idx] > 0) {
                    float px = grid.min_x + x * s;
                    float py = grid.min_y + y * s;
                    float pz = grid.min_z + z * s;
                    
                    // Top Face
                    if (y == grid.dim_y - 1 || grid.data[idx + grid.dim_x*grid.dim_z] == 0) {
                        int v_idx = mesh.vertices.size() / 3;
                        mesh.vertices.insert(mesh.vertices.end(), {
                            px, py+s, pz,  px+s, py+s, pz,  px+s, py+s, pz+s,  px, py+s, pz+s
                        });
                        mesh.indices.insert(mesh.indices.end(), {v_idx, v_idx+2, v_idx+1, v_idx, v_idx+3, v_idx+2});
                        mesh.materials.insert(mesh.materials.end(), {grid.data[idx], grid.data[idx]});
                    }
                    // Right Face
                    if (x == grid.dim_x - 1 || grid.data[idx + 1] == 0) {
                        int v_idx = mesh.vertices.size() / 3;
                        mesh.vertices.insert(mesh.vertices.end(), {
                            px+s, py, pz,  px+s, py+s, pz,  px+s, py+s, pz+s,  px+s, py, pz+s
                        });
                        mesh.indices.insert(mesh.indices.end(), {v_idx, v_idx+1, v_idx+2, v_idx, v_idx+2, v_idx+3});
                        mesh.materials.insert(mesh.materials.end(), {grid.data[idx], grid.data[idx]});
                    }
                    // Front Face
                    if (z == grid.dim_z - 1 || grid.data[idx + grid.dim_x] == 0) {
                        int v_idx = mesh.vertices.size() / 3;
                        mesh.vertices.insert(mesh.vertices.end(), {
                            px, py, pz+s,  px+s, py, pz+s,  px+s, py+s, pz+s,  px, py+s, pz+s
                        });
                        mesh.indices.insert(mesh.indices.end(), {v_idx, v_idx+1, v_idx+2, v_idx, v_idx+2, v_idx+3});
                        mesh.materials.insert(mesh.materials.end(), {grid.data[idx], grid.data[idx]});
                    }
                    // Left Face
                    if (x == 0 || grid.data[idx - 1] == 0) {
                        int v_idx = mesh.vertices.size() / 3;
                        mesh.vertices.insert(mesh.vertices.end(), {
                            px, py, pz,  px, py, pz+s,  px, py+s, pz+s,  px, py+s, pz
                        });
                        mesh.indices.insert(mesh.indices.end(), {v_idx, v_idx+1, v_idx+2, v_idx, v_idx+2, v_idx+3});
                        mesh.materials.insert(mesh.materials.end(), {grid.data[idx], grid.data[idx]});
                    }
                    // Back Face
                    if (z == 0 || grid.data[idx - grid.dim_x] == 0) {
                        int v_idx = mesh.vertices.size() / 3;
                        mesh.vertices.insert(mesh.vertices.end(), {
                            px, py, pz,  px, py+s, pz,  px+s, py+s, pz,  px+s, py, pz
                        });
                        mesh.indices.insert(mesh.indices.end(), {v_idx, v_idx+1, v_idx+2, v_idx, v_idx+2, v_idx+3});
                        mesh.materials.insert(mesh.materials.end(), {grid.data[idx], grid.data[idx]});
                    }
                }
            }
        }
    }
}

// FIX 2: Synthetic 3D Antenna Pattern Generator
// Maps Directivity, side-lobes, and back-lobes into a 2D Texture Array
std::vector<float> generate_antenna_texture(float gain_dbi, float horiz_beam, float vert_beam) {
    std::vector<float> tex(360 * 180, -30.0f); // Default deep null
    for (int el = 0; el < 180; el++) {
        for (int az = 0; az < 360; az++) {
            float az_off = std::fmod(az + 180.0f, 360.0f) - 180.0f; // -180 to 180
            float el_off = el - 90.0f; // -90 to 90
            
            // Main Lobe
            float h_loss = 12.0f * std::pow(az_off / horiz_beam, 2.0f);
            float v_loss = 12.0f * std::pow(el_off / vert_beam, 2.0f);
            float main_lobe = gain_dbi - std::min(h_loss + v_loss, 38.0f);
            
            // Sidelobes (Bessel-like ripple)
            float side_lobe = gain_dbi - 18.0f - 5.0f * std::sin(std::abs(az_off) * M_PI / 20.0f);
            
            // Backlobe
            float back_lobe = (std::abs(az_off) > 120.0f) ? (gain_dbi - 25.0f) : -100.0f;
            
            tex[el * 360 + az] = std::max({main_lobe, side_lobe, back_lobe});
        }
    }
    return tex;
}

bool build_voxel_grid(
    float center_e, float center_n, float radius, float cell_size,
    std::vector<uint8_t>& voxel_data, VoxelGrid& grid_info, std::vector<float>& surface_heightmap) 
{
    sqlite3* db = get_db_connection();
    if (!db) return false;

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

    if (points.empty()) return false;

    grid_info.min_x = center_e - radius;
    grid_info.min_z = center_n - radius;
    grid_info.min_y = min_alt;
    grid_info.cell_size = cell_size;
    grid_info.dim_x = ceil((radius * 2.0f) / cell_size);
    grid_info.dim_z = ceil((radius * 2.0f) / cell_size);
    grid_info.dim_y = ceil((max_alt - min_alt) / cell_size) + 1;

    voxel_data.assign(grid_info.dim_x * grid_info.dim_y * grid_info.dim_z, 0);
    
    for (const auto& p : points) {
        int gx = (p.e - grid_info.min_x) / cell_size;
        int gz = (p.n - grid_info.min_z) / cell_size;
        int gy = (p.a - grid_info.min_y) / cell_size;
        
        if (gx >= 0 && gx < grid_info.dim_x && gz >= 0 && gz < grid_info.dim_z && gy >= 0 && gy < grid_info.dim_y) {
            int idx = gy * (grid_info.dim_x * grid_info.dim_z) + gz * grid_info.dim_x + gx;
            voxel_data[idx] = (p.cls > 0) ? p.cls : 1; 
        }
    }

    grid_info.data = voxel_data.data();

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

// Note: Kept strictly unchanged as requested. Visualization ray paths still use voxels.
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
        float az = base_az + dis_angle(gen);
        float el = base_el + dis_angle(gen);
        Vec3 ray_dir = { sinf(az) * cosf(el), sinf(el), cosf(az) * cosf(el) };
        
        Vec3 up = {0, 1, 0};
        Vec3 right = cross_prod(ray_dir, up);
        if (length(right) < 0.001f) right = {1, 0, 0};
        Vec3 pol = normalize(cross_prod(right, ray_dir));
        
        float cur_x = params.tx_x;
        float cur_y = params.tx_y;
        float cur_z = params.tx_z;
        
        path.points.push_back({cur_x, cur_y, cur_z, pol.x, pol.y, pol.z});

        for (int bounce = 0; bounce < params.max_bounces; ++bounce) {
            float step = grid.cell_size * 0.2f; 
            float t = 0;
            float max_t = 1500.0f; 
            bool hit = false;
            Vec3 hit_normal = {0, 1, 0};

            while (t < max_t) {
                t += step;
                float px = cur_x + ray_dir.x * t;
                float py = cur_y + ray_dir.y * t;
                float pz = cur_z + ray_dir.z * t;
                
                int gx = std::floor((px - grid.min_x) / grid.cell_size);
                int gy = std::floor((py - grid.min_y) / grid.cell_size);
                int gz = std::floor((pz - grid.min_z) / grid.cell_size);
                
                if (gx < 0 || gx >= grid.dim_x || gy < 0 || gy >= grid.dim_y || gz < 0 || gz >= grid.dim_z) {
                    cur_x = px; cur_y = py; cur_z = pz;
                    break; 
                }
                
                int idx = gy * (grid.dim_x * grid.dim_z) + gz * grid.dim_x + gx;
                int cls = voxel_data[idx];
                
                if (cls > 0 && !(cls >= 3 && cls <= 5)) { 
                    hit = true;
                    float rx = (px - grid.min_x) / grid.cell_size - (gx + 0.5f);
                    float ry = (py - grid.min_y) / grid.cell_size - (gy + 0.5f);
                    float rz = (pz - grid.min_z) / grid.cell_size - (gz + 0.5f);
                    
                    if (std::abs(rx) > std::abs(ry) && std::abs(rx) > std::abs(rz)) { hit_normal = {(rx > 0) ? 1.0f : -1.0f, 0, 0}; }
                    else if (std::abs(ry) > std::abs(rx) && std::abs(ry) > std::abs(rz)) { hit_normal = {0, (ry > 0) ? 1.0f : -1.0f, 0}; }
                    else { hit_normal = {0, 0, (rz > 0) ? 1.0f : -1.0f}; }
                    
                    cur_x = px - ray_dir.x * step;
                    cur_y = py - ray_dir.y * step;
                    cur_z = pz - ray_dir.z * step;
                    break;
                }
            }
            
            path.points.push_back({cur_x, cur_y, cur_z, pol.x, pol.y, pol.z});
            if (!hit) break; 
            
            float dot_dir = ray_dir.x * hit_normal.x + ray_dir.y * hit_normal.y + ray_dir.z * hit_normal.z;
            ray_dir = {ray_dir.x - 2.0f * dot_dir * hit_normal.x, ray_dir.y - 2.0f * dot_dir * hit_normal.y, ray_dir.z - 2.0f * dot_dir * hit_normal.z};
            
            float dot_pol = pol.x * hit_normal.x + pol.y * hit_normal.y + pol.z * hit_normal.z;
            pol = {pol.x - 2.0f * dot_pol * hit_normal.x, pol.y - 2.0f * dot_pol * hit_normal.y, pol.z - 2.0f * dot_pol * hit_normal.z};
            pol = normalize(pol);

            cur_x += hit_normal.x * step * 2.0f;
            cur_y += hit_normal.y * step * 2.0f;
            cur_z += hit_normal.z * step * 2.0f;
        }
        paths.push_back(path);
    }
    return paths;
}

int main() {
    GDALAllRegister(); 
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

        sqlite3_stmt* stmt_max;
        long long max_id = 0;
        if (sqlite3_prepare_v2(db, "SELECT MAX(id) FROM ProcessedLiDARPoints", -1, &stmt_max, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt_max) == SQLITE_ROW) max_id = sqlite3_column_int64(stmt_max, 0);
            sqlite3_finalize(stmt_max);
        }

        double easting = 0, northing = 0;
        if (max_id > 0) {
            std::string q_center;
            if (max_id <= 1000) {
                q_center = "SELECT (MIN(easting) + MAX(easting)) / 2.0, (MIN(northing) + MAX(northing)) / 2.0 FROM ProcessedLiDARPoints";
            } else {
                std::stringstream in_clause;
                long long step = max_id / 500;
                for (int i = 1; i <= 500; i++) {
                    in_clause << (i * step);
                    if (i < 500) in_clause << ",";
                }
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

        int zone = 15;
        bool is_north = true;
        char hem = 'N';
        if (!label.empty()) {
            hem = label.back();
            is_north = (hem == 'N' || hem == 'n');
            std::string zone_str = label.substr(0, label.size() - 1);
            try { zone = std::stoi(zone_str); } catch(...) {}
        }

        OGRSpatialReference utm_srs;
        utm_srs.SetWellKnownGeogCS("WGS84");
        utm_srs.SetUTM(zone, is_north);
        OGRSpatialReference wgs84_srs;
        wgs84_srs.SetWellKnownGeogCS("WGS84");
        wgs84_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER); 

        OGRCoordinateTransformation *poCT = OGRCreateCoordinateTransformation(&utm_srs, &wgs84_srs);
        double lon = easting, lat = northing;
        if (poCT) {
            poCT->Transform(1, &lon, &lat);
            OCTDestroyCoordinateTransformation(poCT);
        }

        crow::json::wvalue res;
        res["lat"] = lat; res["lng"] = lon;
        return crow::response(res);
    });

    CROW_ROUTE(app, "/api/simulate").methods(crow::HTTPMethod::POST)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");

        try {
            double lat = body.has("lat") ? body["lat"].d() : 38.0;
            double lng = body.has("lng") ? body["lng"].d() : -91.0;
            float radius = body.has("radius") ? body["radius"].d() : 100.0;
            float voxel_res = body.has("resolution") ? body["resolution"].d() : 2.0;
            int max_bounces = body.has("max_bounces") ? body["max_bounces"].i() : 3;
            
            int zone = std::floor((lng + 180.0) / 6.0) + 1;
            bool is_north = lat >= 0.0;

            OGRSpatialReference wgs84_srs;
            wgs84_srs.SetWellKnownGeogCS("WGS84");
            wgs84_srs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            OGRSpatialReference utm_srs;
            utm_srs.SetWellKnownGeogCS("WGS84");
            utm_srs.SetUTM(zone, is_north);

            OGRCoordinateTransformation *poCT = OGRCreateCoordinateTransformation(&wgs84_srs, &utm_srs);
            double easting = lng, northing = lat;
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
            float rx_gain = body.has("rx_gain") ? body["rx_gain"].d() : 0.0; 
            float beamwidth = body.has("beamwidth") ? body["beamwidth"].d() : 65.0;
            float v_beamwidth = body.has("v_beamwidth") ? body["v_beamwidth"].d() : 10.0; 

            std::vector<uint8_t> voxel_data;
            VoxelGrid grid_info;
            std::vector<float> surface_mesh;

            if (!build_voxel_grid(easting, northing, radius, voxel_res, voxel_data, grid_info, surface_mesh)) {
                return crow::response(400, "No LiDAR points found in this area.");
            }
            
            // FIX 1 Execution: Generate OptiX Triangle Mesh from Voxels
            TriangleMesh optix_mesh;
            generate_optix_mesh(grid_info, optix_mesh);

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
            params.rx_gain_dbi = rx_gain; 
            params.beamwidth_rad = beamwidth * (M_PI / 180.0);
            params.vertical_beamwidth_rad = v_beamwidth * (M_PI / 180.0); 
            
            params.bounds_min_x = grid_info.min_x;
            params.bounds_min_z = grid_info.min_z;
            params.grid_width = grid_info.dim_x;
            params.grid_height = grid_info.dim_z;
            params.cell_size = grid_info.cell_size;
            params.max_bounces = max_bounces; 
            params.ray_count = 500000; 
            
            // FIX 2 Execution: Generate 3D Hardware Antenna Texture Pattern
            std::vector<float> antenna_pattern = generate_antenna_texture(tx_gain, beamwidth, v_beamwidth);

            std::vector<float> rx_power_dbm;
            std::vector<float> delay_spread_ns;
            
            // Pass Mesh and Antenna Pattern into Engine
            run_rf_simulation(grid_info, optix_mesh, antenna_pattern, params, rx_power_dbm, delay_spread_ns);

            std::vector<RayPath> ray_paths = generate_visualization_rays(params, grid_info, voxel_data, 1000);

            crow::json::wvalue res;
            res["grid_w"] = grid_info.dim_x;
            res["grid_h"] = grid_info.dim_z;
            res["cell_size"] = grid_info.cell_size;
            res["min_alt"] = grid_info.min_y;
            res["surface_heights"] = surface_mesh;
            res["heatmap_dbm"] = std::vector<float>(rx_power_dbm.begin(), rx_power_dbm.end());
            res["delay_spread_ns"] = std::vector<float>(delay_spread_ns.begin(), delay_spread_ns.end());

            crow::json::wvalue::list paths_json;
            for (const auto& p : ray_paths) {
                crow::json::wvalue::list points_json;
                for (const auto& pt : p.points) {
                    crow::json::wvalue::list coord;
                    coord.push_back(pt[0]); coord.push_back(pt[1]); coord.push_back(pt[2]); 
                    coord.push_back(pt[3]); coord.push_back(pt[4]); coord.push_back(pt[5]); 
                    points_json.push_back(coord);
                }
                paths_json.push_back(points_json);
            }
            res["ray_paths"] = std::move(paths_json);
            res["easting"] = easting; res["northing"] = northing;
            res["zone"] = zone; res["hem"] = is_north ? "N" : "S";

            return crow::response(res);
        } catch (const std::exception& e) {
            std::cerr << "Simulation Error: " << e.what() << std::endl;
            return crow::response(500, "Internal Server Error");
        }
    });

    app.port(8080).multithreaded().run();
    return 0;
}