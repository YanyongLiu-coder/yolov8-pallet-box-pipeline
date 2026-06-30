from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 weights to ONNX for TensorRT.")
    parser.add_argument("--weights", default="models/best.pt", help="YOLOv8 .pt path or model name.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size.")
    parser.add_argument("--output", default="models/best.onnx", help="Destination ONNX path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        batch=1,
    )

    exported_path = Path(exported)
    if exported_path.resolve() != output.resolve():
        shutil.copy2(exported_path, output)

    print(f"ONNX exported to: {output}")


if __name__ == "__main__":
    main()

