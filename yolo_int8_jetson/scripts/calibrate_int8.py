"""
INT8 Calibration Script for TensorRT.

Generates a calibration cache file by running representative images through
the ONNX model. The cache is used by trtexec to build an INT8 engine.

Usage:
    python3 scripts/calibrate_int8.py \
        --onnx models/best.onnx \
        --images-dir /path/to/calibration_images \
        --cache models/calibration.cache \
        --batch-size 8 \
        --num-images 200
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError:
    raise SystemExit(
        "This script requires tensorrt and pycuda.\n"
        "Install with: pip install pycuda\n"
        "TensorRT Python bindings come with the TensorRT container or JetPack SDK."
    )

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_SHAPE = (3, 640, 640)  # CHW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate INT8 calibration cache for TensorRT.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model.")
    parser.add_argument("--images-dir", required=True, help="Directory with calibration images.")
    parser.add_argument("--cache", default="models/calibration.cache", help="Output calibration cache path.")
    parser.add_argument("--batch-size", type=int, default=8, help="Calibration batch size.")
    parser.add_argument("--num-images", type=int, default=200, help="Max number of calibration images to use.")
    return parser.parse_args()


def preprocess_image(image_path: str, input_h: int = 640, input_w: int = 640) -> np.ndarray:
    """Preprocess a single image: letterbox resize, normalize, HWC->CHW, BGR->RGB."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = min(input_w / w, input_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (input_w - new_w) // 2
    pad_y = (input_h - new_h) // 2

    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # BGR -> RGB, normalize to [0, 1], HWC -> CHW
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = canvas.astype(np.float32) / 255.0
    canvas = canvas.transpose(2, 0, 1)  # CHW
    return canvas


class YoloCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Entropy Calibrator for YOLOv8 models."""

    def __init__(self, image_dir: str, cache_file: str, batch_size: int = 8, num_images: int = 200):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size

        # Collect image paths
        image_dir = Path(image_dir)
        all_images = sorted([
            str(p) for p in image_dir.rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        ])
        self.image_paths = all_images[:num_images]
        self.num_images = len(self.image_paths)
        self.current_index = 0

        # Allocate device memory for one batch
        self.input_size = int(np.prod(INPUT_SHAPE)) * batch_size
        self.device_input = cuda.mem_alloc(self.input_size * 4)  # float32 = 4 bytes

        print(f"Calibrator initialized: {self.num_images} images, batch_size={batch_size}")

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names) -> list:
        """Feed one batch of images to the calibrator."""
        if self.current_index >= self.num_images:
            return None

        batch_end = min(self.current_index + self.batch_size, self.num_images)
        batch_images = []

        for i in range(self.current_index, batch_end):
            img = preprocess_image(self.image_paths[i])
            if img is not None:
                batch_images.append(img)

        if not batch_images:
            return None

        # Pad batch if needed
        while len(batch_images) < self.batch_size:
            batch_images.append(batch_images[-1])

        batch_data = np.ascontiguousarray(np.stack(batch_images[:self.batch_size]))
        cuda.memcpy_htod(self.device_input, batch_data.tobytes())
        self.current_index = batch_end

        progress = min(100, int(self.current_index / self.num_images * 100))
        print(f"  Calibrating... {progress}% ({self.current_index}/{self.num_images})")

        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        """Read existing calibration cache if available."""
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache to disk."""
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"Calibration cache written to: {self.cache_file}")


def build_int8_engine(onnx_path: str, calibrator: YoloCalibrator, engine_path: str = None):
    """Build a TensorRT INT8 engine using the calibrator."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # Allow FP16 fallback for sensitive layers
    config.int8_calibrator = calibrator

    # Build engine
    print("Building INT8 TensorRT engine (this may take 1-3 minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build INT8 engine")

    if engine_path:
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        size_mb = len(serialized_engine) / (1024 * 1024)
        print(f"INT8 engine saved to: {engine_path} ({size_mb:.1f} MB)")

    return serialized_engine


def main():
    args = parse_args()

    # Create calibrator
    calibrator = YoloCalibrator(
        image_dir=args.images_dir,
        cache_file=args.cache,
        batch_size=args.batch_size,
        num_images=args.num_images,
    )

    # Build INT8 engine
    engine_path = args.onnx.replace(".onnx", ".int8.engine")
    build_int8_engine(args.onnx, calibrator, engine_path)

    print("\nDone! You can now use the INT8 engine for inference:")
    print(f"  CLI:    build/yolo_trt_infer --engine {engine_path} --image test.jpg")
    print(f"  Server: build/yolo_trt_server --engine {engine_path} --port 8002")


if __name__ == "__main__":
    main()
