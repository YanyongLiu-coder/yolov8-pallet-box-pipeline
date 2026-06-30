"""Export YOLOv8m best.pt to ONNX format for C++ inference."""
from ultralytics import YOLO


def main() -> None:
    model = YOLO("/workspace/pallet_box_yolov8/runs/detect/runs/detect/pallet_box_yolov8m-4/weights/best.pt")
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        opset=17,
        dynamic=False,
    )
    print("ONNX export complete.")


if __name__ == "__main__":
    main()
