# YOLO Visual Detection Projects

End-to-end object detection engineering practice based on the YOLO model family, covering the full pipeline from data annotation, model training, to inference deployment.

## Projects

### pallet_box_yolov8/

**Goods Stack Detection (goods_stack)** — Complete YOLOv8 training + inference service project.

- Extract data from LabelMe annotations and convert to YOLO format
- Fine-tune YOLOv8m model (mAP50=97.3%)
- Two RESTful inference services: Python (FastAPI) and C++ (ONNX Runtime)
- Performance benchmarking comparison

See [pallet_box_yolov8/README.md](pallet_box_yolov8/README.md)

### yolo_trt_cpp_lab/

**YOLO TensorRT C++ Inference** — High-performance YOLO inference using TensorRT in C++.

See [yolo_trt_cpp_lab/README.md](yolo_trt_cpp_lab/README.md)
