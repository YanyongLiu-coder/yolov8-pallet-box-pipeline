#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-models/best.onnx}"
ENGINE_PATH="${2:-models/best.fp16.engine}"
VERBOSE="${VERBOSE:-0}"

mkdir -p "$(dirname "${ENGINE_PATH}")"

ARGS=(
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --workspace=4096
)

if [[ "${VERBOSE}" == "1" ]]; then
  ARGS+=(--verbose)
fi

trtexec "${ARGS[@]}"

echo "TensorRT engine saved to: ${ENGINE_PATH}"
