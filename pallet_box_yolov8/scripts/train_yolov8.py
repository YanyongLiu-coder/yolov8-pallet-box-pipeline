from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for pallet box stack detection.")
    parser.add_argument("--data", default="datasets/pallet_box/dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="pallet_box_yolov8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=30,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        degrees=5.0,
        translate=0.05,
        scale=0.4,
        fliplr=0.5,
    )


if __name__ == "__main__":
    main()

