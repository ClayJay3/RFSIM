#include "crow.h"
#include "rf_engine.cuh"
#include <gdal_priv.h>
#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

// Helper for libcurl to write data to memory
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Fallback Procedural Terrain Generator if API fails
std::vector<Triangle> generate_procedural_terrain(float lat, float lon, float radius, int resolution, float& bounds_x, float& bounds_z, float& cell_size) {
    std::vector<Triangle> mesh;
    bounds_x = -radius;
    bounds_z = -radius;
    cell_size = (2.0f * radius) / resolution;

    std::vector<std::vector<Vec3>> grid(resolution, std::vector<Vec3>(resolution));
    
    for (int i = 0; i < resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            float x = bounds_x + j * cell_size;
            float z = bounds_z + i * cell_size;
            // Generate some hilly terrain via combined sine waves
            float y = 50.0f * std::sin(x / 200.0f) * std::cos(z / 200.0f) + 
                      20.0f * std::sin(x / 50.0f) + 10.0f; 
            grid[i][j] = {x, y, z};
        }
    }

    // Convert grid to triangles
    for (int i = 0; i < resolution - 1; ++i) {
        for (int j = 0; j < resolution - 1; ++j) {
            Vec3 v0 = grid[i][j];
            Vec3 v1 = grid[i][j+1];
            Vec3 v2 = grid[i+1][j];
            Vec3 v3 = grid[i+1][j+1];
            mesh.push_back({v0, v2, v1}); // Triangle 1
            mesh.push_back({v1, v2, v3}); // Triangle 2
        }
    }
    return mesh;
}

// Fetch Real DEM data using GDAL and Curl (OpenTopography API)
std::vector<Triangle> fetch_real_terrain(float lat, float lon, float radius, float& bounds_x, float& bounds_z, float& cell_size) {
    // Note: In production, convert radius (meters) to lat/lon degrees approximation
    float degree_offset = radius / 111320.0f; 
    float south = lat - degree_offset;
    float north = lat + degree_offset;
    float west = lon - degree_offset;
    float east = lon + degree_offset;

    // Upgraded to SRTMGL1 (30m resolution) for much higher fidelity
    std::string url = "https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=" + 
                      std::to_string(south) + "&north=" + std::to_string(north) + 
                      "&west=" + std::to_string(west) + "&east=" + std::to_string(east) + "&outputFormat=GTiff";

    CURL* curl = curl_easy_init();
    std::string readBuffer;
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        
        if (res != CURLE_OK || readBuffer.empty()) {
            std::cerr << "DEM fetch failed. Using procedural fallback." << std::endl;
            return generate_procedural_terrain(lat, lon, radius, 100, bounds_x, bounds_z, cell_size);
        }
    }

    // Save temporary TIFF to be read by GDAL
    std::string tmp_file = "/tmp/dem_tile.tif";
    std::ofstream out(tmp_file, std::ios::binary);
    out.write(readBuffer.c_str(), readBuffer.size());
    out.close();

    // Read with GDAL
    GDALDataset* poDataset = (GDALDataset*) GDALOpen(tmp_file.c_str(), GA_ReadOnly);
    if (!poDataset) {
        return generate_procedural_terrain(lat, lon, radius, 100, bounds_x, bounds_z, cell_size);
    }

    GDALRasterBand* poBand = poDataset->GetRasterBand(1);
    int width = poBand->GetXSize();
    int height = poBand->GetYSize();

    std::vector<float> elevation(width * height);
    CPLErr err = poBand->RasterIO(GF_Read, 0, 0, width, height, elevation.data(), width, height, GDT_Float32, 0, 0);
    if (err != CE_None) {
        GDALClose(poDataset);
        std::cerr << "GDAL RasterIO failed. Using procedural fallback." << std::endl;
        return generate_procedural_terrain(lat, lon, radius, 100, bounds_x, bounds_z, cell_size);
    }
    GDALClose(poDataset);

    // Map GDAL Matrix to Mesh Triangles
    std::vector<Triangle> mesh;
    bounds_x = -radius;
    bounds_z = -radius;
    cell_size = (2.0f * radius) / width;

    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            float vx = bounds_x + x * cell_size;
            float vz = bounds_z + y * cell_size;

            // Apply 4/3 Earth Curvature for Atmospheric Refraction
            // h_drop = d^2 / (2 * Re), where Re = 4/3 * 6371000 meters
            auto apply_curvature = [](float vx, float vz, float elev) {
                float dist_sq = vx*vx + vz*vz;
                float drop = dist_sq / (2.0f * 8494666.0f);
                return elev - drop;
            };

            Vec3 v0 = {vx, apply_curvature(vx, vz, elevation[y * width + x]), vz};
            Vec3 v1 = {vx + cell_size, apply_curvature(vx + cell_size, vz, elevation[y * width + (x + 1)]), vz};
            Vec3 v2 = {vx, apply_curvature(vx, vz + cell_size, elevation[(y + 1) * width + x]), vz + cell_size};
            Vec3 v3 = {vx + cell_size, apply_curvature(vx + cell_size, vz + cell_size, elevation[(y + 1) * width + (x + 1)]), vz + cell_size};

            mesh.push_back({v0, v2, v1});
            mesh.push_back({v1, v2, v3});
        }
    }
    return mesh;
}

int main() {
    GDALAllRegister();
    crow::SimpleApp app;

    // Serve Frontend
    CROW_ROUTE(app, "/")([]() {
        std::ifstream file("index.html");
        
        // Fallback: If not in the current folder, check the parent folder
        if (!file.is_open()) {
            file.open("../index.html");
            if (!file.is_open()) {
                return crow::response(404, "<h1>Error 404</h1><p>index.html not found! Make sure the file exists in the same directory you are running the server from.</p>");
            }
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        
        // Explicitly tell the browser this is an HTML document
        crow::response res(buffer.str());
        res.set_header("Content-Type", "text/html");
        return res;
    });

    // API Endpoint for RF Simulation
    CROW_ROUTE(app, "/api/simulate").methods(crow::HTTPMethod::POST)([](const crow::request& req) {
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(400, "Invalid JSON");

        try {
            // Safely parse JSON with fallbacks in case the browser sends an older cached payload
            float lat = body.has("lat") ? body["lat"].d() : 37.7749;
            float lon = body.has("lon") ? body["lon"].d() : -122.4194;
            float radius = body.has("radius") ? body["radius"].d() : 1000.0;
            float freq = body.has("freq_mhz") ? body["freq_mhz"].d() * 1e6 : 900e6;
            float tx_h = body.has("tx_height") ? body["tx_height"].d() : 30.0;
            float tx_p = body.has("tx_power") ? body["tx_power"].d() : 43.0;
            
            // New antenna parameters (with safe defaults)
            float tx_az = body.has("tx_azimuth") ? body["tx_azimuth"].d() : 180.0;
            float tx_el = body.has("tx_elevation") ? body["tx_elevation"].d() : -2.0;
            float tx_gain = body.has("tx_gain") ? body["tx_gain"].d() : 18.0;
            float beamwidth = body.has("beamwidth") ? body["beamwidth"].d() : 65.0;

            // 1. Fetch & Build Terrain Mesh
            float bounds_x, bounds_z, cell_size;
            std::vector<Triangle> mesh = fetch_real_terrain(lat, lon, radius, bounds_x, bounds_z, cell_size);

            // Calculate Transmit Center (Local Cartesian)
            float tx_local_x = 0; // Centered
            float tx_local_z = 0;
            // Interpolate ground height at center for Tx Alt
            float ground_height = 10.0f; // Default fallback
            if (!mesh.empty()) ground_height = mesh[mesh.size()/2].v0.y;

            // 2. Setup CUDA Parameters
            SimParams params;
            params.tx_lat = tx_local_x;
            params.tx_lon = tx_local_z;
            params.tx_alt = ground_height + tx_h;
            params.freq_hz = freq;
            params.tx_power_dbm = tx_p;
            params.tx_azimuth_deg = tx_az;
            params.tx_elevation_deg = tx_el;
            params.tx_gain_dbi = tx_gain;
            params.beamwidth_rad = beamwidth * (M_PI / 180.0);
            params.bounds_min_x = bounds_x;
            params.bounds_min_z = bounds_z;
            
            // Define Rx Grid Resolution (e.g. 100x100 resolution heatmap)
            int grid_res = 100;
            params.grid_width = grid_res;
            params.grid_height = grid_res;
            params.cell_size = (2.0f * radius) / grid_res;
            params.max_bounces = 3; 
            params.ray_count = 500000; // Shoot 500k rays

            // 3. Run GPU Physics Simulation
            std::vector<float> rx_power_dbm;
            run_rf_simulation(mesh, params, rx_power_dbm);

            // 4. Send Results Back
            crow::json::wvalue res;
            res["grid_w"] = grid_res;
            res["grid_h"] = grid_res;
            res["cell_size"] = params.cell_size;
            
            // Serialize mesh for frontend rendering (subsample for performance)
            std::vector<crow::json::wvalue> mesh_json;
            for(size_t i=0; i < mesh.size(); i += 2) { // sending 1/2 of triangles
                crow::json::wvalue tri;
                tri["v0"] = std::vector<float>{mesh[i].v0.x, mesh[i].v0.y, mesh[i].v0.z};
                tri["v1"] = std::vector<float>{mesh[i].v1.x, mesh[i].v1.y, mesh[i].v1.z};
                tri["v2"] = std::vector<float>{mesh[i].v2.x, mesh[i].v2.y, mesh[i].v2.z};
                mesh_json.push_back(std::move(tri));
            }
            res["mesh"] = std::move(mesh_json);

            // Serialize Heatmap Array
            std::vector<float> heatmap_array(rx_power_dbm.begin(), rx_power_dbm.end());
            res["heatmap_dbm"] = heatmap_array;

            return crow::response(res);

        } catch (const std::exception& e) {
            std::cerr << "Simulation Error: " << e.what() << std::endl;
            return crow::response(500, "Internal Server Error");
        }
    });

    app.port(8080).multithreaded().run();
    return 0;
}