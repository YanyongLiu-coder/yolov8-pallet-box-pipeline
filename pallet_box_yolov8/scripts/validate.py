from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLOv8 detection model.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="datasets/pallet_box/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.val(data=args.data, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()

