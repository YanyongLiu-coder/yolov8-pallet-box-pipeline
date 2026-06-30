#!/usr/bin/env bash
set -euo pipefail

ENGINE_PATH="${1:-models/best.fp16.engine}"
PORT="${2:-8002}"

if [[ ! -x build/yolo_trt_server ]]; then
  cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(nproc)"
fi

echo "Starting TensorRT inference server on port ${PORT}..."
build/yolo_trt_server --engine "${ENGINE_PATH}" --port "${PORT}"
