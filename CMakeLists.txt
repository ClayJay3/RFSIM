cmake_minimum_required(VERSION 3.18)
project(RF_GPU_Sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find GDAL
find_package(GDAL REQUIRED)

# Find CURL
find_package(CURL REQUIRED)

# --- Crow Web Framework Configuration ---
# Crow is header-only. Instead of running its CMake script (which looks for Boost),
# we just download the repo and link directly to its include directory.
include(FetchContent)
FetchContent_Declare(
  crow_fetch
  GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
  GIT_TAG v1.2.0 # Upgraded to a more modern, stable tag
)

FetchContent_GetProperties(crow_fetch)
if(NOT crow_fetch_POPULATED)
  FetchContent_Populate(crow_fetch)
  
  # Manually create the Crow::Crow target
  add_library(Crow INTERFACE)
  target_include_directories(Crow INTERFACE ${crow_fetch_SOURCE_DIR}/include)
  
  # CRITICAL: Tell Crow to use libasio-dev instead of Boost.Asio
  target_compile_definitions(Crow INTERFACE ASIO_STANDALONE)
  
  add_library(Crow::Crow ALIAS Crow)
endif()

# Define the executable
add_executable(rf_sim_server 
    main.cpp 
    rf_engine.cu
)

# Target Include Directories
target_include_directories(rf_sim_server PRIVATE 
    ${GDAL_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Libraries
target_link_libraries(rf_sim_server PRIVATE 
    CUDA::cudart
    ${GDAL_LIBRARIES}
    ${CURL_LIBRARIES}
    Crow::Crow
    pthread
)

# CUDA Architecture targeting (Turing, Ampere, Ada)
set_target_properties(rf_sim_server PROPERTIES CUDA_ARCHITECTURES "75;80;86;89")

file(COPY index.html DESTINATION ${CMAKE_CURRENT_BINARY_DIR})