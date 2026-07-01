#!/usr/bin/env bash
# Build INT8 TensorRT engine using trtexec with pre-generated calibration cache.
#
# Usage:
#   bash scripts/build_engine_int8.sh [ONNX_PATH] [ENGINE_PATH] [CALIB_CACHE]
#
# Prerequisites:
#   Run calibrate_int8.py first to generate the calibration cache.

set -euo pipefail

ONNX_PATH="${1:-models/best.onnx}"
ENGINE_PATH="${2:-models/best.int8.engine}"
CALIB_CACHE="${3:-models/calibration.cache}"
VERBOSE="${VERBOSE:-0}"

if [[ ! -f "${CALIB_CACHE}" ]]; then
  echo "ERROR: Calibration cache not found: ${CALIB_CACHE}"
  echo "Run calibrate_int8.py first to generate it."
  exit 1
fi

mkdir -p "$(dirname "${ENGINE_PATH}")"

ARGS=(
  --onnx="${ONNX_PATH}"
  --saveEngine="${ENGINE_PATH}"
  --int8
  --fp16
  --calib="${CALIB_CACHE}"
  --workspace=4096
)

if [[ "${VERBOSE}" == "1" ]]; then
  ARGS+=(--verbose)
fi

echo "Building INT8 engine..."
echo "  ONNX:  ${ONNX_PATH}"
echo "  Cache: ${CALIB_CACHE}"
echo "  Output: ${ENGINE_PATH}"

trtexec "${ARGS[@]}"

echo ""
echo "INT8 TensorRT engine saved to: ${ENGINE_PATH}"
echo "Use with:"
echo "  build/yolo_trt_infer --engine ${ENGINE_PATH} --image test.jpg"
echo "  build/yolo_trt_server --engine ${ENGINE_PATH} --port 8002"
