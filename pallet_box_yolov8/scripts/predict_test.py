"""Run prediction on test images and save visualizations."""
from ultralytics import YOLO


def main() -> None:
    model = YOLO("/workspace/pallet_box_yolov8/runs/detect/runs/detect/pallet_box_yolov8m-4/weights/best.pt")
    results = model.predict(
        source="/workspace/pallet_box_yolov8/datasets/pallet_box/images/test",
        imgsz=640,
        conf=0.25,
        device=0,
        project="/workspace/pallet_box_yolov8/runs/predict",
        name="test_visualize",
        save=True,
        save_txt=True,
        save_conf=True,
        exist_ok=True,
    )
    print(f"Predicted {len(results)} images")
    for r in results:
        boxes = r.boxes
        print(f"  {r.path}: {len(boxes)} boxes detected")
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            xyxy = b.xyxy[0].tolist()
            print(f"    class={cls_id} conf={conf:.3f} box=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]")


if __name__ == "__main__":
    main()
