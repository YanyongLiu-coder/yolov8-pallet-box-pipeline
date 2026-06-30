FROM nvcr.io/nvidia/tensorrt:23.08-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/yolo_trt_cpp_lab

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    ultralytics==8.2.103 \
    onnx==1.16.2 \
    onnxsim==0.4.36 \
    onnxruntime==1.23.2 \
    onnxslim==0.1.34 \
    opencv-python-headless==4.10.0.84 \
    numpy==1.26.4

COPY . /workspace/yolo_trt_cpp_lab

# Build both CLI and server
RUN cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j"$(nproc)"

EXPOSE 8002

CMD ["/bin/bash"]
