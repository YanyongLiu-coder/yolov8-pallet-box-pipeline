#!/usr/bin/env bash
set -euo pipefail

ENGINE_PATH="${1:-models/best.fp16.engine}"
IMAGE_PATH="${2:?Usage: bash scripts/run_cpp_infer.sh ENGINE IMAGE [OUTPUT]}"
OUTPUT_PATH="${3:-outputs/result.jpg}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

if [[ ! -x build/yolo_trt_infer ]]; then
  cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(nproc)"
fi

build/yolo_trt_infer \
  --engine "${ENGINE_PATH}" \
  --image "${IMAGE_PATH}" \
  --labels assets/pallet_box.names \
  --output "${OUTPUT_PATH}" \
  --conf 0.25 \
  --iou 0.7

echo "Result image saved to: ${OUTPUT_PATH}"

