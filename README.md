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

**YOLO TensorRT C++ Inference** — High-performance YOLO inference using TensorRT FP16 in C++.

See [yolo_trt_cpp_lab/README.md](yolo_trt_cpp_lab/README.md)

### yolo_int8_jetson/

**INT8 Quantization + Jetson Orin Deployment** — TensorRT INT8 quantization with calibration, optimized for edge deployment on NVIDIA Jetson Orin.

- INT8 calibration pipeline (prepare images → calibrate → build engine)
- Jetson Orin native build and Docker support
- FP16 vs INT8 accuracy validation
- ~2x speedup over FP16, ~50% model size reduction

See [yolo_int8_jetson/README.md](yolo_int8_jetson/README.md)
