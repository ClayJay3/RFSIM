cmake_minimum_required(VERSION 3.18)
project(RF_GPU_Sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find GDAL and CURL
find_package(GDAL REQUIRED)
find_package(CURL REQUIRED)

# Find SQLite3 for the local LiDAR Database
find_package(SQLite3 REQUIRED)

# --- Crow Web Framework Configuration ---
include(FetchContent)
FetchContent_Declare(
  crow_fetch
  GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
  GIT_TAG v1.2.0 
)

FetchContent_GetProperties(crow_fetch)
if(NOT crow_fetch_POPULATED)
  FetchContent_Populate(crow_fetch)
  add_library(Crow INTERFACE)
  target_include_directories(Crow INTERFACE ${crow_fetch_SOURCE_DIR}/include)
  target_compile_definitions(Crow INTERFACE ASIO_STANDALONE)
  add_library(Crow::Crow ALIAS Crow)
endif()

add_executable(rf_sim_server 
    main.cpp 
    rf_engine.cu
)

target_include_directories(rf_sim_server PRIVATE 
    ${GDAL_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link SQLite3 alongside other libraries
target_link_libraries(rf_sim_server PRIVATE 
    CUDA::cudart
    ${GDAL_LIBRARIES}
    ${CURL_LIBRARIES}
    SQLite::SQLite3
    Crow::Crow
    pthread
)

set_target_properties(rf_sim_server PROPERTIES CUDA_ARCHITECTURES "75;80;86;89")

file(COPY index.html DESTINATION ${CMAKE_CURRENT_BINARY_DIR})