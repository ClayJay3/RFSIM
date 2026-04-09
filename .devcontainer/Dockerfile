FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# CRITICAL FIX: OptiX requires graphical libraries to be mounted into the container.
# Without this, optixInit() hits a stub library and throws an ABI mismatch (Code 7801).
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install core build dependencies, GDAL, CURL, ASIO, and Boost
# Added libtinfo5 and libxml2 which are often required by NVIDIA installers
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    gdb \
    git \
    wget \
    libgdal-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev \
    libssl-dev \
    zlib1g-dev \
    libasio-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libsqlite3-dev \
    libtinfo5 \
    libxml2 \
    && rm -rf /var/lib/apt/lists/*

# Install official NVIDIA OptiX SDK from the locally provided installer
# Note: This assumes the installer is in the project root and context is set to '..'
COPY NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64.sh /tmp/optix-installer.sh
RUN chmod +x /tmp/optix-installer.sh && \
    mkdir -p /tmp/optix_sdk /usr/local/include/optix && \
    /tmp/optix-installer.sh --skip-license --prefix=/tmp/optix_sdk --exclude-subdir && \
    cp -r /tmp/optix_sdk/include/* /usr/local/include/optix/ && \
    rm -rf /tmp/optix_sdk /tmp/optix-installer.sh

WORKDIR /workspace