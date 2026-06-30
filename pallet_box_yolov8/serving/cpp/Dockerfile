FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake wget unzip \
    libopencv-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Download ONNX Runtime GPU
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz \
    && tar -xzf onnxruntime-linux-x64-gpu-1.17.1.tgz \
    && cp onnxruntime-linux-x64-gpu-1.17.1/lib/* /usr/local/lib/ \
    && cp -r onnxruntime-linux-x64-gpu-1.17.1/include/* /usr/local/include/ \
    && ldconfig \
    && rm -rf onnxruntime-linux-x64-gpu-1.17.1*

# Download header-only libraries
RUN mkdir -p /build/third_party \
    && wget -q -O /build/third_party/httplib.h https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.15.3/httplib.h \
    && wget -q -O /build/third_party/nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp \
    || mkdir -p /build/third_party/nlohmann \
    && wget -q -O /build/third_party/nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp

# Copy source and build
COPY CMakeLists.txt main.cpp /build/
RUN mkdir -p /build/third_party/nlohmann \
    && (test -f /build/third_party/nlohmann/json.hpp || wget -q -O /build/third_party/nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp) \
    && cmake -B build_dir -DCMAKE_BUILD_TYPE=Release \
       -DONNXRUNTIME_INCLUDE_DIR=/usr/local/include \
       -DONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so \
    && cmake --build build_dir --config Release -j$(nproc) \
    && cp build_dir/yolo_server /usr/local/bin/

WORKDIR /workspace
EXPOSE 8001

CMD ["yolo_server"]
