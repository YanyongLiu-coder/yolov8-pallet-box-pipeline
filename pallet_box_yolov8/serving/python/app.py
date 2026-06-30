"""YOLOv8m inference REST API service using FastAPI + ultralytics."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

MODEL_PATH = "/workspace/models/best.pt"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
DEVICE = "0"

app = FastAPI(title="YOLOv8m Pallet Box Detection API (Python)")
model: YOLO | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    print(f"Loading model from {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=640, device=DEVICE, verbose=False)
    print("Model loaded and warmed up.")


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    start_time = time.time()

    # Read image bytes
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)

    import cv2
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Run inference
    infer_start = time.time()
    results = model.predict(
        image,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )
    infer_time = time.time() - infer_start

    # Parse results
    result = results[0]
    boxes_data = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        boxes_data.append({
            "class_id": cls_id,
            "class_name": result.names[cls_id],
            "confidence": round(conf, 4),
            "bbox": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            },
        })

    total_time = time.time() - start_time

    return JSONResponse(content={
        "image_name": file.filename,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "num_boxes": len(boxes_data),
        "boxes": boxes_data,
        "timing_ms": {
            "inference": round(infer_time * 1000, 2),
            "total": round(total_time * 1000, 2),
        },
    })


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
