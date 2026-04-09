#include "rf_engine.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define PI 3.14159265358979323846f
#define C_LIGHT 299792458.0f 

// --- ROBUST ERROR CHECKING MACROS ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0)

#define OPTIX_CHECK(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX Error (Code " << res << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0)


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

// RESTORED: Vegetation Model for LOS
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

// RESTORED: Knife-Edge Loss Model
__device__ float knife_edge_loss_db(float nu) {
    if (nu <= -1.0f) return 0.0f;
    if (nu <= 0.0f) return -20.0f * log10f(0.5f - 0.62f * nu);
    if (nu <= 1.0f) return -20.0f * log10f(0.5f * expf(-0.95f * nu));
    if (nu <= 2.4f) return -20.0f * log10f(0.4f - sqrtf(0.1184f - powf(0.38f - 0.1f * nu, 2.0f)));
    return -20.0f * log10f(0.225f / nu);
}

__device__ int find_highest_voxel(VoxelGrid grid, int gx, int gz) {
    if (gx < 0 || gx >= grid.dim_x || gz < 0 || gz >= grid.dim_z) return 0;
    for (int y = grid.dim_y - 1; y >= 0; y--) {
        int idx = y * (grid.dim_x * grid.dim_z) + gz * grid.dim_x + gx;
        if (grid.data[idx] > 0) return y;
    }
    return 0;
}

// RESTORED FULL DEYGOUT + COHERENT PHASE INTEGRATION
__global__ void los_diffraction_voxel_kernel(VoxelGrid grid, SimParams params, float* min_dist_grid, float* max_dist_grid) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gz = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= params.grid_width || gz >= params.grid_height) return;

    int top_y = find_highest_voxel(grid, gx, gz);
    float rx_y = grid.min_y + top_y * grid.cell_size + 1.5f; 
    float rx_x = grid.min_x + gx * grid.cell_size + grid.cell_size * 0.5f;
    float rx_z = grid.min_z + gz * grid.cell_size + grid.cell_size * 0.5f;

    Vec3 tx_pos = {params.tx_x, params.tx_y, params.tx_z};
    Vec3 rx_pos = {rx_x, rx_y, rx_z};
    Vec3 dir = sub(rx_pos, tx_pos);
    float dist = length(dir);
    if (dist < 0.1f) return;
    dir = normalize(dir);

    float lambda = C_LIGHT / params.freq_hz;
    
    // Sample texture for primary LOS path gain
    float ray_az = atan2f(dir.x, dir.z);
    float ray_el = asinf(fmaxf(-1.0f, fminf(1.0f, dir.y)));
    float delta_az = ray_az - (params.tx_azimuth_deg * PI / 180.0f);
    while (delta_az > PI) delta_az -= 2.0f * PI;
    while (delta_az < -PI) delta_az += 2.0f * PI;
    float delta_el = ray_el - (params.tx_elevation_deg * PI / 180.0f);
    float u = (delta_az + PI) / (2.0f * PI);
    float v = (delta_el + PI / 2.0f) / PI;
    float gain_dbi = tex2D<float>(params.antenna_tex, u, v);

    float eirp_dbm = params.tx_power_dbm + gain_dbi;
    float p_tx_watts = powf(10.0f, (eirp_dbm - 30.0f) / 10.0f);
    
    float rx_gain_lin = powf(10.0f, params.rx_gain_dbi / 10.0f);
    float power_density = p_tx_watts / (4.0f * PI * dist * dist);
    float a_eff = rx_gain_lin * (lambda * lambda) / (4.0f * PI);
    float p_rx_watts = power_density * a_eff;

    int num_steps = (int)ceilf(dist / (grid.cell_size * 0.5f));
    float veg_dist = 0.0f;
    
    // RESTORED DEYGOUT ALGORITHM: Pass 1 (Main Peak)
    float max_nu_1 = -1e9f;
    int peak_1_idx = -1;
    float p1_x, p1_y, p1_z, p1_d1;

    for (int i = 1; i < num_steps; i++) {
        float f = (float)i / num_steps;
        float cx = tx_pos.x + f * (rx_pos.x - tx_pos.x);
        float cz = tx_pos.z + f * (rx_pos.z - tx_pos.z);
        float cy = tx_pos.y + f * (rx_pos.y - tx_pos.y);
        
        int cgx = (cx - grid.min_x) / grid.cell_size;
        int cgz = (cz - grid.min_z) / grid.cell_size;
        int cgy = find_highest_voxel(grid, cgx, cgz);
        
        if (cgy > 0) {
            int cls = grid.data[cgy * (grid.dim_x * grid.dim_z) + cgz * grid.dim_x + cgx];
            if (cls >= 3 && cls <= 5) veg_dist += grid.cell_size * 0.5f;
            
            float obs_y = grid.min_y + cgy * grid.cell_size;
            float h = obs_y - cy;
            float d1 = f * dist;
            float d2 = dist - d1;
            float nu = h * sqrtf(2.0f * (d1 + d2) / (lambda * d1 * d2));
            
            if (nu > max_nu_1) {
                max_nu_1 = nu; peak_1_idx = i; 
                p1_x = cx; p1_y = obs_y; p1_z = cz; p1_d1 = d1;
            }
        }
    }

    float diff_loss_db = 0.0f;
    if (max_nu_1 > -1.0f && peak_1_idx != -1) {
        diff_loss_db += knife_edge_loss_db(max_nu_1);
        
        // RESTORED DEYGOUT: Pass 2 (Left Sub-Peak)
        if (peak_1_idx > 1) {
            float max_nu_2 = -1e9f;
            for (int i = 1; i < peak_1_idx; i++) {
                float f = (float)i / peak_1_idx;
                float cx = tx_pos.x + f * (p1_x - tx_pos.x);
                float cz = tx_pos.z + f * (p1_z - tx_pos.z);
                int cgx = (cx - grid.min_x) / grid.cell_size;
                int cgz = (cz - grid.min_z) / grid.cell_size;
                float obs_y = grid.min_y + find_highest_voxel(grid, cgx, cgz) * grid.cell_size;
                float cy = tx_pos.y + f * (p1_y - tx_pos.y);
                float h = obs_y - cy;
                float d1 = f * p1_d1, d2 = p1_d1 - d1;
                float nu = h * sqrtf(2.0f * (d1 + d2) / (lambda * d1 * d2));
                if (nu > max_nu_2) max_nu_2 = nu;
            }
            if (max_nu_2 > -1.0f) diff_loss_db += knife_edge_loss_db(max_nu_2);
        }
        
        // RESTORED DEYGOUT: Pass 3 (Right Sub-Peak)
        if (peak_1_idx < num_steps - 1) {
            float max_nu_3 = -1e9f;
            int right_steps = num_steps - peak_1_idx;
            float dist_p1_rx = dist - p1_d1;
            for (int i = 1; i < right_steps; i++) {
                float f = (float)i / right_steps;
                float cx = p1_x + f * (rx_pos.x - p1_x);
                float cz = p1_z + f * (rx_pos.z - p1_z);
                int cgx = (cx - grid.min_x) / grid.cell_size;
                int cgz = (cz - grid.min_z) / grid.cell_size;
                float obs_y = grid.min_y + find_highest_voxel(grid, cgx, cgz) * grid.cell_size;
                float cy = p1_y + f * (rx_pos.y - p1_y);
                float h = obs_y - cy;
                float d1 = f * dist_p1_rx, d2 = dist_p1_rx - d1;
                float nu = h * sqrtf(2.0f * (d1 + d2) / (lambda * d1 * d2));
                if (nu > max_nu_3) max_nu_3 = nu;
            }
            if (max_nu_3 > -1.0f) diff_loss_db += knife_edge_loss_db(max_nu_3);
        }
    }

    p_rx_watts *= powf(10.0f, -diff_loss_db / 10.0f);
    p_rx_watts *= calc_vegetation_loss_power_mult(veg_dist, params.freq_hz);

    int grid_idx = gz * params.grid_width + gx;
    
    // Integrate primary LOS ray into Coherent Phase Grid instead of raw max Watts
    float phase = (2.0f * PI * dist) / lambda;
    float voltage = sqrtf(p_rx_watts);
    float re = voltage * cosf(phase);
    float im = -voltage * sinf(phase);
    
    atomicAdd(&params.d_rx_grid_re[grid_idx], re);
    atomicAdd(&params.d_rx_grid_im[grid_idx], im);
    atomicAdd(&params.d_rx_grid_incoherent_watts[grid_idx], p_rx_watts);
    
    atomicMinFloat(&min_dist_grid[grid_idx], dist);
    atomicMaxFloat(&max_dist_grid[grid_idx], dist);
}

// Catch OptiX internal debug messages and explicitly flush them to the console
static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
    std::cerr << "[OptiX " << tag << "] " << message << std::endl;
    std::cerr.flush();
}

extern "C" void run_rf_simulation(
    const VoxelGrid& grid, const TriangleMesh& mesh, const std::vector<float>& antenna_pattern,
    const SimParams& host_params, 
    std::vector<float>& out_coherent_dbm, 
    std::vector<float>& out_incoherent_dbm, 
    std::vector<float>& out_phase_rad, 
    std::vector<float>& out_tof_ns, 
    std::vector<float>& out_delay_spread_ns) 
{
    int grid_size = host_params.grid_width * host_params.grid_height;
    if (grid_size <= 0) throw std::runtime_error("Invalid grid dimensions calculated.");
    
    uint32_t num_vertices = mesh.vertices.size() / 3;
    uint32_t num_triangles = mesh.indices.size() / 3;
    if (num_vertices == 0 || num_triangles == 0) {
        throw std::runtime_error("Triangle mesh has 0 vertices/triangles! Cannot build OptiX BVH from flat terrain.");
    }

    out_coherent_dbm.resize(grid_size, -120.0f);
    out_incoherent_dbm.resize(grid_size, -120.0f);
    out_phase_rad.resize(grid_size, 0.0f);
    out_tof_ns.resize(grid_size, 0.0f);
    out_delay_spread_ns.resize(grid_size, 0.0f);

    SimParams d_params = host_params;

    // --- Create 3D Antenna Texture for Hardware Fetching ---
    cudaArray_t cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, 360, 180));
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, antenna_pattern.data(), 360 * sizeof(float), 360 * sizeof(float), 180, cudaMemcpyHostToDevice));
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;   
    texDesc.addressMode[1] = cudaAddressModeClamp;  
    texDesc.filterMode = cudaFilterModeLinear;      
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;                   
    
    CUDA_CHECK(cudaCreateTextureObject(&d_params.antenna_tex, &resDesc, &texDesc, nullptr));

    // --- Allocate Complex Coherent Grids ---
    CUDA_CHECK(cudaMalloc(&d_params.d_rx_grid_re, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_params.d_rx_grid_im, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_params.d_rx_grid_incoherent_watts, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_params.d_rx_grid_re, 0, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_params.d_rx_grid_im, 0, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_params.d_rx_grid_incoherent_watts, 0, grid_size * sizeof(float)));

    float *d_min_dist, *d_max_dist;
    CUDA_CHECK(cudaMalloc(&d_min_dist, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_dist, grid_size * sizeof(float)));
    std::vector<float> init_min(grid_size, 1e9f); 
    std::vector<float> init_max(grid_size, 0.0f);
    CUDA_CHECK(cudaMemcpy(d_min_dist, init_min.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max_dist, init_max.data(), grid_size * sizeof(float), cudaMemcpyHostToDevice));

    uint8_t* d_grid_data;
    CUDA_CHECK(cudaMalloc(&d_grid_data, grid.dim_x * grid.dim_y * grid.dim_z * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_grid_data, grid.data, grid.dim_x * grid.dim_y * grid.dim_z * sizeof(uint8_t), cudaMemcpyHostToDevice));
    VoxelGrid d_grid = grid; d_grid.data = d_grid_data; 

    // --- OPTIX INITIALIZATION ---
    OptixResult initRes = optixInit();
    if (initRes == 7801) { // OPTIX_ERROR_UNSUPPORTED_ABI_VERSION
        throw std::runtime_error("Hardware Error: OptiX ABI mismatch at optixInit(). Your Docker container is failing to mount the host's OptiX driver. Rebuild the container with the updated Dockerfile containing 'ENV NVIDIA_DRIVER_CAPABILITIES=all'. If the error persists, you must downgrade to NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh.");
    }
    OPTIX_CHECK(initRes);
    
    OptixDeviceContext context;
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    
    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &context));

    // Build GAS (Geometry Acceleration Structure) from Marching Cubes Mesh
    float* d_vertices; int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_vertices, mesh.vertices.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_vertices, mesh.vertices.data(), mesh.vertices.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_indices, mesh.indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_indices, mesh.indices.data(), mesh.indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    CUdeviceptr v_ptr = reinterpret_cast<CUdeviceptr>(d_vertices);
    CUdeviceptr i_ptr = reinterpret_cast<CUdeviceptr>(d_indices);
    
    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    // CRITICAL FIX: Set strides to 0 so OptiX safely assumes arrays are densely packed
    triangleInput.triangleArray.vertexStrideInBytes = 0; 
    triangleInput.triangleArray.numVertices = num_vertices;
    triangleInput.triangleArray.vertexBuffers = &v_ptr;
    
    triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    // CRITICAL FIX: Set strides to 0 so OptiX safely assumes arrays are densely packed
    triangleInput.triangleArray.indexStrideInBytes = 0; 
    triangleInput.triangleArray.numIndexTriplets = num_triangles;
    triangleInput.triangleArray.indexBuffer = i_ptr;
    
    uint32_t triangleFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    triangleInput.triangleArray.flags = triangleFlags;
    triangleInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &triangleInput, 1, &gasBufferSizes));
    
    CUdeviceptr d_temp_buffer, d_gas_output;
    CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gas_output, gasBufferSizes.outputSizeInBytes));
    
    OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &triangleInput, 1, d_temp_buffer, gasBufferSizes.tempSizeInBytes,
                    d_gas_output, gasBufferSizes.outputSizeInBytes, &d_params.gas_handle, nullptr, 0));
    CUDA_CHECK(cudaFree((void*)d_temp_buffer));

    // Load PTX and Create OptiX Pipeline
    std::string ptx_path = std::string(PTX_FILE_DIR) + "/optix_kernels.ptx";
    std::ifstream ptx_file(ptx_path);
    if (!ptx_file.good()) {
        throw std::runtime_error("CRITICAL: Could not find compiled OptiX PTX file at " + ptx_path);
    }
    std::string ptx((std::istreambuf_iterator<char>(ptx_file)), std::istreambuf_iterator<char>());

    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "launch_params";

    OptixModule module;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OptixResult moduleRes = optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions, ptx.c_str(), ptx.size(), log, &sizeof_log, &module);
    if (moduleRes != OPTIX_SUCCESS) {
        throw std::runtime_error("OptiX Module Compilation Failed: " + std::string(log));
    }

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroup raygenPG, missPG, hitgroupPG;
    OptixProgramGroupDesc rgDesc = {}; rgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; rgDesc.raygen.module = module; rgDesc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(context, &rgDesc, 1, &pgOptions, nullptr, nullptr, &raygenPG));
    
    OptixProgramGroupDesc msDesc = {}; msDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS; msDesc.miss.module = module; msDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(context, &msDesc, 1, &pgOptions, nullptr, nullptr, &missPG));

    OptixProgramGroupDesc hgDesc = {}; hgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; hgDesc.hitgroup.moduleCH = module; hgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK(optixProgramGroupCreate(context, &hgDesc, 1, &pgOptions, nullptr, nullptr, &hitgroupPG));

    OptixPipeline pipeline;
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    OptixProgramGroup pgs[] = {raygenPG, missPG, hitgroupPG};
    
    sizeof_log = sizeof(log);
    OptixResult pipeRes = optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions, pgs, 3, log, &sizeof_log, &pipeline);
    if (pipeRes != OPTIX_SUCCESS) {
        throw std::runtime_error("OptiX Pipeline Link Failed: " + std::string(log));
    }

    // SBT Setup
    struct SbtRecord { __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    SbtRecord rgSbt, msSbt, hgSbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rgSbt));
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &msSbt));
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, &hgSbt));
    
    CUdeviceptr d_rg, d_ms, d_hg;
    CUDA_CHECK(cudaMalloc((void**)&d_rg, sizeof(SbtRecord))); CUDA_CHECK(cudaMemcpy((void*)d_rg, &rgSbt, sizeof(SbtRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_ms, sizeof(SbtRecord))); CUDA_CHECK(cudaMemcpy((void*)d_ms, &msSbt, sizeof(SbtRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_hg, sizeof(SbtRecord))); CUDA_CHECK(cudaMemcpy((void*)d_hg, &hgSbt, sizeof(SbtRecord), cudaMemcpyHostToDevice));

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = d_rg;
    sbt.missRecordBase = d_ms; sbt.missRecordStrideInBytes = sizeof(SbtRecord); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = d_hg; sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord); sbt.hitgroupRecordCount = 1;

    OptixLaunchParams h_launch_params;
    h_launch_params.params = d_params;
    h_launch_params.grid = d_grid; 
    h_launch_params.min_dist_grid = d_min_dist;
    h_launch_params.max_dist_grid = d_max_dist;
    h_launch_params.mesh_vertices = d_vertices;
    h_launch_params.mesh_indices = d_indices;

    CUdeviceptr d_launch_params;
    CUDA_CHECK(cudaMalloc((void**)&d_launch_params, sizeof(OptixLaunchParams)));
    CUDA_CHECK(cudaMemcpy((void*)d_launch_params, &h_launch_params, sizeof(OptixLaunchParams), cudaMemcpyHostToDevice));

    // --- KERNEL EXECUTION ---
    dim3 blockSize(16, 16);
    dim3 gridSize((host_params.grid_width + blockSize.x - 1) / blockSize.x, 
                  (host_params.grid_height + blockSize.y - 1) / blockSize.y);
    los_diffraction_voxel_kernel<<<gridSize, blockSize>>>(d_grid, d_params, d_min_dist, d_max_dist);
    CUDA_CHECK(cudaDeviceSynchronize());

    OPTIX_CHECK(optixLaunch(pipeline, 0, d_launch_params, sizeof(OptixLaunchParams), &sbt, host_params.ray_count, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- COHERENT WAVE RECONSTRUCTION ---
    std::vector<float> h_re(grid_size), h_im(grid_size), h_incoh(grid_size), h_min(grid_size), h_max(grid_size);
    CUDA_CHECK(cudaMemcpy(h_re.data(), d_params.d_rx_grid_re, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_im.data(), d_params.d_rx_grid_im, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_incoh.data(), d_params.d_rx_grid_incoherent_watts, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min.data(), d_min_dist, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max.data(), d_max_dist, grid_size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < grid_size; i++) {
        // Fast Fading emerges organically from summing the interference of complex waves
        float re = h_re[i];
        float im = h_im[i];
        float p_coh = (re * re) + (im * im);
        float p_incoh = h_incoh[i];

        if (p_coh > 1e-15f) out_coherent_dbm[i] = 10.0f * log10f(p_coh) + 30.0f; 
        else out_coherent_dbm[i] = -120.0f;
        
        if (p_incoh > 1e-15f) out_incoherent_dbm[i] = 10.0f * log10f(p_incoh) + 30.0f;
        else out_incoherent_dbm[i] = -120.0f;
        
        out_phase_rad[i] = atan2f(im, re);

        if (h_max[i] > h_min[i] && h_min[i] < 1e8f) {
            float dist_diff = h_max[i] - h_min[i];
            out_tof_ns[i] = (h_min[i] / C_LIGHT) * 1e9f;
            out_delay_spread_ns[i] = (dist_diff / C_LIGHT) * 1e9f; 
        } else if (h_min[i] < 1e8f) {
            out_tof_ns[i] = (h_min[i] / C_LIGHT) * 1e9f;
            out_delay_spread_ns[i] = 0.0f;
        }
    }

    // Cleanup
    cudaFree(d_grid_data);
    cudaFree(d_params.d_rx_grid_re); cudaFree(d_params.d_rx_grid_im); cudaFree(d_params.d_rx_grid_incoherent_watts);
    cudaFree(d_min_dist); cudaFree(d_max_dist);
    cudaFree(d_vertices); cudaFree(d_indices);
    optixPipelineDestroy(pipeline); optixProgramGroupDestroy(raygenPG);
    optixModuleDestroy(module); optixDeviceContextDestroy(context);
    cudaDestroyTextureObject(d_params.antenna_tex); cudaFreeArray(cuArray);
}