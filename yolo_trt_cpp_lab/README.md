# YOLOv8m Pallet Box → TensorRT C++ Inference

Export the pallet_box_yolov8m model (goods_stack detection) to ONNX, compile it into a TensorRT FP16 engine, and run high-performance inference in C++.

Two usage modes are provided:
- **CLI mode**: Single-image inference from the command line, outputs a result image with bounding boxes
- **HTTP server mode**: REST API endpoint, accepts image uploads and returns JSON detection results

## Directory Structure

```
yolo_trt_cpp_lab/
├── Dockerfile
├── cpp/
│   ├── CMakeLists.txt
│   └── src/
│       ├── main.cpp          # CLI inference (single image detection + visualization)
│       └── server.cpp        # HTTP inference service (TensorRT backend)
├── scripts/
│   ├── export_yolov8_onnx.py # Export to ONNX
│   ├── build_engine.sh       # Build FP16 engine with trtexec
│   ├── run_cpp_infer.sh      # Launch CLI inference
│   └── run_server.sh         # Launch HTTP server
├── assets/
│   └── pallet_box.names      # Class names file (goods_stack)
└── models/                   # Store .pt / .onnx / .engine files
```

## Full Workflow

### 1. Build Docker Image

```bash
cd /home/localadmin/build/yolo_trt_cpp_lab
docker build -t yolo-trt-cpp:0.2 .
```

Based on `nvcr.io/nvidia/tensorrt:23.08-py3` (TensorRT 8.6 + CUDA 12 + cuDNN 8).

### 2. Enter the Container

```bash
docker run --rm -it --gpus all \
  -v /home/localadmin/build/yolo_trt_cpp_lab:/workspace/yolo_trt_cpp_lab \
  -v /home/localadmin/build/pallet_box_yolov8/runs/detect/runs/detect/pallet_box_yolov8m-4/weights:/workspace/yolo_trt_cpp_lab/models \
  -w /workspace/yolo_trt_cpp_lab \
  yolo-trt-cpp:0.2
```

This mounts the trained `best.pt` and `best.onnx` into the container's `models/` directory.

### 3. Export ONNX (if not already done)

```bash
python3 scripts/export_yolov8_onnx.py \
  --weights models/best.pt \
  --imgsz 640 \
  --output models/best.onnx
```

### 4. Build TensorRT FP16 Engine

```bash
bash scripts/build_engine.sh models/best.onnx models/best.fp16.engine
```

Takes approximately 30-60 seconds on an RTX A6000. The engine file is ~50MB and runs 3-5x faster than ONNX Runtime.

### 5a. CLI Inference (Single Image)

```bash
bash scripts/run_cpp_infer.sh \
  models/best.fp16.engine \
  /path/to/test_image.jpg \
  outputs/result.jpg
```

Detection results are printed to the terminal; `outputs/result.jpg` contains the annotated image with bounding boxes.

### 5b. HTTP Inference Server

```bash
bash scripts/run_server.sh models/best.fp16.engine 8002
```

Test request:
```bash
curl -X POST http://localhost:8002/detect \
  -F "file=@test_image.jpg" | python3 -m json.tool
```

Response format:
```json
{
  "image_name": "test_image.jpg",
  "image_size": {"width": 1080, "height": 1920},
  "num_boxes": 2,
  "boxes": [
    {
      "class_id": 0,
      "class_name": "goods_stack",
      "confidence": 0.8923,
      "bbox": {"x1": 355.0, "y1": 1015.0, "x2": 1080.0, "y2": 1774.0}
    }
  ],
  "timing_ms": {"inference": 3.5, "total": 8.2}
}
```

## One-Command Docker Deployment (HTTP Server)

```bash
# Start the server after building the engine
docker run -d --name yolo-trt-server \
  --gpus device=0 -p 8002:8002 \
  -v /home/localadmin/build/yolo_trt_cpp_lab:/workspace/yolo_trt_cpp_lab \
  -v /home/localadmin/build/pallet_box_yolov8/runs/detect/runs/detect/pallet_box_yolov8m-4/weights:/workspace/yolo_trt_cpp_lab/models \
  -w /workspace/yolo_trt_cpp_lab \
  yolo-trt-cpp:0.2 \
  bash -c './build/yolo_trt_server --engine models/best.fp16.engine --port 8002'
```

## Expected Performance

TensorRT FP16 inference performance on RTX A6000:

| Approach | Inference Latency | Expected Throughput |
|----------|-------------------|---------------------|
| Python (PyTorch CUDA) | ~10 ms | ~34 req/s |
| C++ (ONNX Runtime CUDA) | ~20 ms | ~15 req/s |
| **C++ (TensorRT FP16)** | **~3-5 ms** | **~100+ req/s** |

Why TensorRT outperforms PyTorch:
- Compile-time operator fusion (conv+bn+relu merged into a single kernel)
- Automatic selection of the optimal CUDA kernel for the current GPU (tactic selection)
- FP16 precision doubles throughput on A6000
- No Python GIL, no framework dispatch overhead

## Relationship with pallet_box_yolov8

```
pallet_box_yolov8/              <- Training + data preparation
  └── runs/detect/.../weights/
        ├── best.pt             <- Original PyTorch weights
        └── best.onnx           <- Exported ONNX model

yolo_trt_cpp_lab/               <- Deployment + high-performance inference
  └── models/
        ├── best.onnx           <- Mounted from above
        └── best.fp16.engine    <- TensorRT compiled artifact
```
