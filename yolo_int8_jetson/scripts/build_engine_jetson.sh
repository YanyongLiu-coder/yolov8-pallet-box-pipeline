#!/usr/bin/env bash
# Build TensorRT INT8 engine on Jetson Orin.
#
# IMPORTANT: The engine file is hardware-specific. An engine built on Jetson
# Orin cannot run on x86 GPU and vice versa. You MUST build on the target device.
#
# Usage:
#   # FP16 (no calibration needed):
#   bash scripts/build_engine_jetson.sh --fp16
#
#   # INT8 (requires calibration cache):
#   bash scripts/build_engine_jetson.sh --int8
#
# Prerequisites for INT8:
#   python3 scripts/calibrate_int8.py \
#     --onnx models/best.onnx \
#     --images-dir models/calib_images \
#     --cache models/calibration.cache

set -euo pipefail

MODE="${1:---fp16}"
ONNX_PATH="${2:-models/best.onnx}"
CALIB_CACHE="models/calibration.cache"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ERROR: ONNX model not found: ${ONNX_PATH}"
  exit 1
fi

# Detect Jetson platform
if [[ -f /etc/nv_tegra_release ]]; then
  echo "Detected Jetson platform:"
  cat /etc/nv_tegra_release
else
  echo "WARNING: Not running on Jetson. Engine will only work on THIS GPU."
fi

mkdir -p models

case "${MODE}" in
  --fp16)
    ENGINE_PATH="models/best.fp16.engine"
    echo "Building FP16 engine for Jetson..."
    trtexec \
      --onnx="${ONNX_PATH}" \
      --saveEngine="${ENGINE_PATH}" \
      --fp16 \
      --workspace=2048 \
      --memPoolSize=workspace:2048MiB
    ;;
  --int8)
    ENGINE_PATH="models/best.int8.engine"
    if [[ ! -f "${CALIB_CACHE}" ]]; then
      echo "ERROR: Calibration cache not found: ${CALIB_CACHE}"
      echo "Run calibrate_int8.py on this device first."
      exit 1
    fi
    echo "Building INT8 engine for Jetson..."
    trtexec \
      --onnx="${ONNX_PATH}" \
      --saveEngine="${ENGINE_PATH}" \
      --int8 \
      --fp16 \
      --calib="${CALIB_CACHE}" \
      --workspace=2048 \
      --memPoolSize=workspace:2048MiB
    ;;
  *)
    echo "Usage: $0 [--fp16|--int8] [ONNX_PATH]"
    exit 1
    ;;
esac

echo ""
echo "Engine saved to: ${ENGINE_PATH}"
echo "Engine size: $(du -h ${ENGINE_PATH} | cut -f1)"
echo ""
echo "Test inference:"
echo "  ./build/yolo_trt_infer --engine ${ENGINE_PATH} --image test.jpg"
echo ""
echo "Start server:"
echo "  ./build/yolo_trt_server --engine ${ENGINE_PATH} --port 8002"
