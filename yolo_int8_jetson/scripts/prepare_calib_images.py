"""
Prepare calibration images for INT8 quantization.

Copies a representative subset of training images to a calibration directory.
Aims to cover diverse scenarios (different camera positions, stack sizes, etc.)

Usage:
    python3 scripts/prepare_calib_images.py \
        --src /path/to/train_images \
        --dst models/calib_images \
        --num 200
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare calibration images for INT8 quantization.")
    parser.add_argument("--src", required=True, help="Source image directory (e.g., training images).")
    parser.add_argument("--dst", default="models/calib_images", help="Output calibration image directory.")
    parser.add_argument("--num", type=int, default=200, help="Number of images to select.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    all_images = sorted([p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    print(f"Found {len(all_images)} images in {src}")

    if len(all_images) == 0:
        raise SystemExit(f"No images found in {src}")

    rng = random.Random(args.seed)
    selected = all_images if len(all_images) <= args.num else rng.sample(all_images, args.num)
    selected.sort()

    dst.mkdir(parents=True, exist_ok=True)
    for img_path in selected:
        shutil.copy2(img_path, dst / img_path.name)

    print(f"Copied {len(selected)} calibration images to: {dst}")
    print(f"\nNext step: run calibration")
    print(f"  python3 scripts/calibrate_int8.py \\")
    print(f"    --onnx models/best.onnx \\")
    print(f"    --images-dir {dst} \\")
    print(f"    --cache models/calibration.cache")


if __name__ == "__main__":
    main()
