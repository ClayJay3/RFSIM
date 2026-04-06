FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install core build dependencies, GDAL, CURL, ASIO, and Boost (required by Crow's CMake)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    gdb \
    git \
    libgdal-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev \
    libssl-dev \
    zlib1g-dev \
    libasio-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace