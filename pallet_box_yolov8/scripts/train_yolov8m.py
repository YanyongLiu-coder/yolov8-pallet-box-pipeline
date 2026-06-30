"""Train YOLOv8m for pallet box stack detection with AMP disabled."""
from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8m.pt")
    model.train(
        data="datasets/pallet_box/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project="runs/detect",
        name="pallet_box_yolov8m",
        patience=30,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        amp=False,
        degrees=5.0,
        translate=0.05,
        scale=0.4,
        fliplr=0.5,
    )


if __name__ == "__main__":
    main()
