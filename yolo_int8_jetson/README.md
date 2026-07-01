# YOLOv8m INT8 Quantization + Jetson Orin Deployment

INT8 quantization of the pallet_box_yolov8m model (goods_stack detection) using TensorRT, optimized for deployment on NVIDIA Jetson Orin.

## Why INT8 Quantization

| Precision | Inference Latency (Jetson Orin) | Model Size | GPU Memory |
|-----------|--------------------------------|------------|------------|
| FP32 | ~45 ms | ~100 MB | ~200 MB |
| FP16 | ~12-18 ms | ~50 MB | ~100 MB |
| **INT8** | **~6-10 ms** | **~25 MB** | **~60 MB** |

INT8 achieves ~2x speedup over FP16 by:
- Using 8-bit integer Tensor Core pipelines (2x throughput vs FP16)
- Halving memory bandwidth requirements (1 byte vs 2 bytes per value)
- Fitting 2x more data in GPU cache, reducing memory latency

## Directory Structure

```
yolo_int8_jetson/
├── Dockerfile.jetson              # Docker image for Jetson Orin (L4T r36.x)
├── cpp/
│   ├── CMakeLists.txt             # Build config (produces CLI + server binaries)
│   └── src/
│       ├── main.cpp               # CLI inference (works with any .engine file)
│       └── server.cpp             # HTTP server (works with any .engine file)
├── scripts/
│   ├── prepare_calib_images.py    # Step 1: Select calibration images
│   ├── calibrate_int8.py          # Step 2: Generate calibration cache
│   ├── build_engine_int8.sh       # Step 3: Build INT8 engine (x86 server)
│   ├── build_engine_jetson.sh     # Step 3: Build INT8 engine (Jetson)
│   └── validate_int8.py           # Step 4: Compare FP16 vs INT8 accuracy
├── assets/
│   └── pallet_box.names           # Class names (goods_stack)
├── models/                        # Stores .onnx, .cache, .engine files
└── README.md
```

## Complete Workflow

### Step 1: Prepare Calibration Images

Select ~200 representative images from the training set that cover diverse scenarios:

```bash
python3 scripts/prepare_calib_images.py \
  --src /path/to/pallet_box_yolov8/datasets/pallet_box/images/train \
  --dst models/calib_images \
  --num 200
```

### Step 2: Generate Calibration Cache

Run the ONNX model on calibration images to profile activation value ranges per layer:

```bash
python3 scripts/calibrate_int8.py \
  --onnx models/best.onnx \
  --images-dir models/calib_images \
  --cache models/calibration.cache \
  --batch-size 8 \
  --num-images 200
```

This produces `models/calibration.cache` — a portable file that records the dynamic range of each tensor in the network.

### Step 3: Build INT8 Engine

**On x86 server (RTX A6000):**
```bash
bash scripts/build_engine_int8.sh \
  models/best.onnx \
  models/best.int8.engine \
  models/calibration.cache
```

**On Jetson Orin:**
```bash
bash scripts/build_engine_jetson.sh --int8
```

> **Important**: Engine files are hardware-specific. You must build on the target device.

### Step 4: Run Inference

The same C++ binaries work with both FP16 and INT8 engines:

```bash
# CLI mode
./build/yolo_trt_infer --engine models/best.int8.engine --image test.jpg --output result.jpg

# HTTP server mode
./build/yolo_trt_server --engine models/best.int8.engine --port 8002
```

Test the server:
```bash
curl -X POST http://localhost:8002/detect \
  -F "file=@test.jpg" | python3 -m json.tool
```

### Step 5: Validate Accuracy

Compare FP16 vs INT8 detection accuracy (start both servers on different ports):

```bash
python3 scripts/validate_int8.py \
  --fp16-url http://localhost:8002/detect \
  --int8-url http://localhost:8003/detect \
  --images-dir /path/to/val_images
```

Expected output: >95% box match rate for the goods_stack detection task.

## Jetson Orin Deployment

### Option A: Build Natively

```bash
# On Jetson Orin with JetPack 6.x installed

# 1. Build C++ binaries
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. Copy model files from server
scp server:/path/to/best.onnx models/
scp server:/path/to/calibration.cache models/

# 3. Build engine on Jetson (hardware-specific)
bash scripts/build_engine_jetson.sh --int8

# 4. Run
./build/yolo_trt_server --engine models/best.int8.engine --port 8002
```

### Option B: Docker

```bash
# Build image on Jetson
docker build -f Dockerfile.jetson -t yolo-int8-jetson .

# Run (mount models directory with engine file)
docker run --runtime nvidia -p 8002:8002 \
  -v /path/to/models:/workspace/yolo_int8_jetson/models \
  yolo-int8-jetson
```

### Cross-Platform Workflow

```
x86 Server (training)                    Jetson Orin (deployment)
┌─────────────────────────┐              ┌─────────────────────────┐
│ 1. Train model (best.pt)│              │                         │
│ 2. Export ONNX          │── copy ─────▶│ best.onnx               │
│ 3. Run calibration      │              │ calibration.cache       │
│    (calibration.cache)  │── copy ─────▶│                         │
│                         │              │ 4. Build engine ON Orin │
│                         │              │    (best.int8.engine)   │
│                         │              │ 5. Run inference        │
│                         │              │    ~6-10ms / image      │
└─────────────────────────┘              └─────────────────────────┘

Portable files:    best.onnx, calibration.cache, source code
Non-portable:      *.engine (must build on target hardware)
```

## How INT8 Calibration Works

```
┌──────────────────────────────────────────────────────────────┐
│  Calibration Process (one-time)                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  For each layer in the network:                              │
│    1. Run calibration images through the FP32 model          │
│    2. Collect activation value statistics (min, max, hist)   │
│    3. Find optimal threshold T that minimizes KL-divergence  │
│       between FP32 distribution and quantized distribution   │
│    4. Compute scale = T / 127                                │
│    5. Store scale per tensor in calibration.cache            │
│                                                              │
│  At inference time:                                          │
│    FP32_value = INT8_value * scale                           │
│    INT8_value = round(FP32_value / scale)                    │
│                                                              │
│  Sensitive layers (e.g., first conv, detection head) may     │
│  fall back to FP16 automatically if INT8 causes too much     │
│  error — this is handled by TensorRT's mixed precision.      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Performance Summary

| Platform | FP16 | INT8 | INT8 Speedup |
|----------|------|------|--------------|
| RTX A6000 (server) | 3-5 ms | 2-3 ms | ~1.5x |
| Jetson Orin (edge) | 12-18 ms | 6-10 ms | ~2x |
| Jetson Orin NX 16GB | 20-30 ms | 10-15 ms | ~2x |

INT8 is especially valuable on Jetson where:
- GPU compute is limited → smaller ops execute faster
- Memory bandwidth is the bottleneck → halved data movement
- Power budget is constrained → less computation = less wattage
- VRAM is scarce (8-16 GB) → smaller model leaves room for other tasks
