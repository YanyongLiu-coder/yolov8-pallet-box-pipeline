# Pallet Box YOLOv8 Training Project

Detect goods stacks on pallets (`goods_stack` class) using YOLOv8 detection framework with transfer learning from LabelMe annotations.

## Core Principle: Transfer Learning from Pretrained Models

### Why Few Annotations Enable Custom Scene Recognition

The core approach of this project is **Transfer Learning**:

1. **YOLOv8 pretrained model** (e.g., `yolov8m.pt`) has already been trained on the COCO dataset (330K images, 80 general classes). The backbone network has learned universal visual feature extraction — edges, textures, shapes, color distributions, etc.

2. **Fine-tuning**: We only need a small amount of annotated data (256 images in this project) to teach the model to recognize our specific target (`goods_stack`) on top of the existing general features. This requires far less data than training from scratch.

3. **Key premise**: We simply provide correct annotations — telling the model "this region in this image is a goods stack." The model automatically learns the visual feature patterns of that region and can detect the same class of objects in new images.

### Overall Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Data Preparation                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Raw Images + LabelMe Annotations                                          │
│       │                                                                    │
│       ▼                                                                    │
│  ┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐   │
│  │ inspect_labelme │ ──▶ │ labelme_to_yolo  │ ──▶ │  YOLO Dataset     │   │
│  │ Check quality   │     │ Convert + split   │     │ images/ + labels/ │   │
│  └─────────────────┘     └──────────────────┘     └───────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                         Model Training                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  YOLOv8m Pretrained Weights (COCO)                                         │
│       │                                                                    │
│       ▼                                                                    │
│  ┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐   │
│  │ train_yolov8m   │ ──▶ │  Training (GPU)  │ ──▶ │  best.pt weights  │   │
│  │ Load + fine-tune│     │  100 epochs       │     │  Custom model     │   │
│  └─────────────────┘     └──────────────────┘     └───────────────────┘   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                         Validation & Inference                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐     ┌──────────────────┐                             │
│  │ validate.py     │     │ predict_test.py  │                             │
│  │ Evaluate mAP    │     │ Visualize results │                             │
│  └─────────────────┘     └──────────────────┘                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### LabelMe Annotation → YOLO Format Conversion

**LabelMe JSON format** (one `.json` file per image):
```json
{
  "imagePath": "image_001.jpg",
  "imageWidth": 1920,
  "imageHeight": 1080,
  "shapes": [
    {
      "label": "goods_stack",
      "shape_type": "rectangle",
      "points": [[100, 200], [500, 800]]   // top-left and bottom-right pixel coords
    }
  ]
}
```

**YOLO TXT format** (one `.txt` file per image):
```
0 0.312500 0.462963 0.208333 0.555556
```

Meaning: `class_id  x_center  y_center  width  height`, all coordinates normalized to [0, 1].

**Coordinate conversion formula**:
```
x_center = (x_min + x_max) / 2 / image_width
y_center = (y_min + y_max) / 2 / image_height
width    = (x_max - x_min) / image_width
height   = (y_max - y_min) / image_height
```

Normalization ensures the model training is independent of original image resolution — the same annotation remains valid at any scale.

---

## Script Execution Flow

### Step 1: analyze_data.py — Dataset Overview Analysis

**Purpose**: Quickly understand the overall data landscape to inform subsequent decisions.

**Execution flow**:
```
Input: Raw data directory (images + LabelMe json files mixed together)
  │
  ├─ 1. Scan directory, separate image files and json files by extension
  │     image_files = [.jpg, .jpeg, .png, .bmp]
  │     json_files  = [.json]
  │
  ├─ 2. Match check: does each image have a corresponding .json?
  │     Has json → labeled_images
  │     No json  → unlabeled_images
  │
  ├─ 3. Iterate all json files, collect annotation statistics
  │     - Whether each json's shapes array is empty
  │     - Aggregate all label names and their counts
  │     - Aggregate shape_type (rectangle/polygon)
  │
  ├─ 4. Group by filename prefix, count camera position coverage
  │     "03_005" → how many images at this position, how many labeled
  │
  └─ Output: Print statistics report to console
```

**Example output**:
```
Total images: 673
Total json files: 256
Images WITH annotation: 256
Images WITHOUT annotation: 417

Json with shapes (labeled): 256
Json without shapes (empty): 0

Label distribution:
  goods_stack: 427

Shape types:
  rectangle: 427
```

---

### Step 2: inspect_labelme.py — Annotation Quality Check

**Purpose**: Verify that LabelMe annotation class names and types match expectations, ensuring data is usable.

**Execution flow**:
```
Input: --src data directory
  │
  ├─ 1. Scan images and json files in the directory
  │
  ├─ 2. Iterate each json file:
  │     ├─ Read JSON content
  │     ├─ Extract imagePath field → record annotated images
  │     ├─ Iterate shapes array:
  │     │     ├─ Count label names (e.g., "goods_stack")
  │     │     └─ Count shape_type (e.g., "rectangle")
  │     └─ Accumulate to global Counter
  │
  └─ 3. Output statistics:
        - Total images / json count / labeled / unlabeled
        - Each label name and occurrence count
        - Each shape_type and occurrence count
```

**Why this step is needed**: Confirms that label names in annotations match `configs/classes.txt` (e.g., if annotations say "Goods_Stack" but classes.txt says "goods_stack", conversion will fail).

---

### Step 3: labelme_to_yolo.py — Core Conversion and Dataset Splitting

**Purpose**: Convert LabelMe annotations to YOLO training format and automatically split into train/val/test.

**Execution flow**:

```
Input:
  --src     Raw data directory (images + json mixed together)
  --out     Output YOLO dataset directory
  --classes Class names file (configs/classes.txt)
  │
  ├─ 1. Read class file, build mapping
  │     classes.txt: "goods_stack"
  │     class_to_id: {"goods_stack": 0}
  │
  ├─ 2. Iterate all .json files, convert each one
  │     For each json file, execute convert_json():
  │     │
  │     ├─ 2a. Read JSON, extract imageWidth/imageHeight
  │     │
  │     ├─ 2b. Find corresponding image file
  │     │       Prefer imagePath field from json
  │     │       Otherwise try same-name .jpg/.jpeg/.png/.bmp
  │     │
  │     ├─ 2c. Iterate shapes array, for each annotation box:
  │     │     ├─ Validate label name exists in class_to_id
  │     │     ├─ Extract points (rectangle corners or polygon vertices)
  │     │     ├─ Call points_to_yolo_box() for coordinate conversion:
  │     │     │     ├─ Compute x_min, y_min, x_max, y_max from all vertices
  │     │     │     ├─ Clip to image boundaries (prevent overflow)
  │     │     │     ├─ Compute center point and dimensions
  │     │     │     └─ Divide by image size to normalize to [0, 1]
  │     │     └─ Generate one YOLO line: "0 0.389895 0.678329 0.779791 0.322690"
  │     │
  │     └─ 2d. Return (image_path, [YOLO label lines])
  │
  ├─ 3. Collect all valid samples (images with annotation boxes)
  │     items = [(img_path, label_lines), ...]
  │
  ├─ 4. Randomly split dataset (seed=42 for reproducibility)
  │     random.shuffle(items)
  │     train: first 80%
  │     val:   middle 15%
  │     test:  last 5%
  │
  ├─ 5. Write to filesystem
  │     For each split (train/val/test):
  │     ├─ Create images/{split}/ directory, copy images
  │     └─ Create labels/{split}/ directory, write .txt labels
  │
  └─ 6. Generate dataset.yaml
        path: /absolute/path/to/dataset
        train: images/train
        val: images/val
        test: images/test
        names:
          0: goods_stack
```

**Output directory structure**:
```
datasets/pallet_box/
├── dataset.yaml           # YOLOv8 dataset configuration
├── images/
│   ├── train/             # 204 training images
│   ├── val/               # 38 validation images
│   └── test/              # 14 test images
└── labels/
    ├── train/             # 204 training labels (one .txt per image)
    ├── val/               # 38 validation labels
    └── test/              # 14 test labels
```

**Label file example** (`03_009_1_1_20260626070603_rotated.txt`):
```
0 0.407005 0.689651 0.814010 0.460824
```
Meaning: class 0 (goods_stack), center at 40.7% width and 69.0% height, box width 81.4% of image, box height 46.1%.

---

### Step 4: train_yolov8m.py — Model Training

**Purpose**: Load pretrained YOLOv8m weights and fine-tune on our dataset.

**Execution flow**:

```
  ├─ 1. Load pretrained model
  │     model = YOLO("yolov8m.pt")
  │     - Automatically downloads yolov8m.pt (25.9M parameters)
  │     - Pretrained on COCO 80 classes, already has general visual features
  │
  ├─ 2. Transfer weights
  │     - Backbone convolution layer weights are directly reused
  │     - Detection head is re-initialized (80→1 classes)
  │     - Log shows "Transferred 469/475 items from pretrained weights"
  │       i.e., 469 of 475 layers transfer directly, 6 layers (detection head) retrained
  │
  ├─ 3. Training loop (100 epochs)
  │     Each epoch:
  │     ├─ Load batch from train images (16 images)
  │     ├─ Data augmentation:
  │     │   - Mosaic: 4 images stitched into 1 (first 90 epochs)
  │     │   - Random rotation ±5°, translation 5%, scale 0.6~1.4x
  │     │   - Horizontal flip 50%, HSV color jitter
  │     ├─ Forward pass: image → model → predicted boxes
  │     ├─ Compute loss (3 components):
  │     │   - box_loss: bounding box position and size error
  │     │   - cls_loss: classification confidence error
  │     │   - dfl_loss: distribution focal loss (box boundary precision)
  │     ├─ Backpropagation: compute gradients
  │     ├─ Optimizer updates weights (SGD + cosine learning rate decay)
  │     └─ Validate on val set, record mAP
  │
  ├─ 4. Early stopping
  │     patience=30: stop if no mAP improvement for 30 epochs
  │     Triggered at epoch 94; best model at epoch 64
  │
  └─ 5. Save outputs
        runs/detect/pallet_box_yolov8m/
        ├── weights/best.pt    # Best mAP weights
        ├── weights/last.pt    # Last epoch weights
        ├── results.csv        # Per-epoch metrics
        ├── results.png        # Training curves
        └── confusion_matrix.png
```

**Key training parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| model | yolov8m.pt | Medium model, balancing accuracy and speed |
| epochs | 100 | Maximum training epochs |
| patience | 30 | Early stopping patience |
| batch | 16 | Batch size |
| imgsz | 640 | Input images resized to 640px |
| cos_lr | True | Cosine annealing learning rate |
| close_mosaic | 10 | Disable mosaic for last 10 epochs |
| amp | False | Disable mixed precision (container compatibility) |

---

### Step 5: validate.py — Model Validation

**Purpose**: Quantitatively evaluate model performance on the validation set.

**Execution flow**:
```
  ├─ 1. Load trained weights best.pt
  │
  ├─ 2. Run inference on each validation image
  │     - Forward pass to get predicted boxes
  │     - Compare with ground truth labels, compute IoU
  │
  └─ 3. Compute evaluation metrics
        - Precision: ratio of correct predictions among all predictions
        - Recall: ratio of detected targets among all ground truth targets
        - mAP50: average precision at IoU threshold 0.5
        - mAP50-95: average precision across IoU 0.5 to 0.95 (stricter)
```

**Training results**:
| Metric | Value |
|--------|-------|
| Precision | 0.953 |
| Recall | 0.966 |
| mAP50 | 0.973 |
| mAP50-95 | 0.812 |

---

### Step 6: predict_test.py — Inference Visualization

**Purpose**: Run detection on test images and draw bounding boxes for visual inspection.

**Execution flow**:
```
  ├─ 1. Load best.pt weights
  │
  ├─ 2. For each image in the test directory:
  │     ├─ Preprocess: resize to 640px, pad to square
  │     ├─ Forward inference: get candidate boxes + confidence + class
  │     ├─ NMS: remove overlapping redundant boxes (keep highest score at IoU > 0.7)
  │     └─ Filter: discard boxes with confidence < 0.25
  │
  ├─ 3. Draw results
  │     Draw detection boxes, class names, confidence scores on original image
  │
  └─ 4. Save to runs/predict/test_visualize/
        - 14 annotated images with bounding boxes
        - labels/ directory with detection result text files
```

---

## Project Overview

| Item | Info |
|------|------|
| Detection target | `goods_stack` (goods stacks on pallets) |
| Annotation tool | LabelMe (rectangle boxes) |
| Model | YOLOv8m (25.9M parameters, 79.1 GFLOPs) |
| Training environment | CUDA GPU (e.g., NVIDIA RTX series) |
| Data source | Private business dataset (raw images not included in public repo) |

## Dataset Statistics

Raw data directory (local example): `/path/to/labelme_dataset`

| Metric | Count |
|--------|-------|
| Total images | 673 |
| Images with LabelMe annotation | 256 |
| Unannotated images | 417 |
| Total annotation boxes | 427 goods_stack rectangles |
| Camera sources | 03/04/05 across 195 positions |

Dataset split (train/val/test = 0.8/0.15/0.05):

| Split | Images | Boxes |
|-------|--------|-------|
| train | 204 | 348 |
| val | 38 | 63 |
| test | 14 | 16 |

## Project Structure

```
pallet_box_yolov8/
├── configs/
│   └── classes.txt              # Class definition (goods_stack)
├── datasets/
│   └── pallet_box/
│       ├── dataset.yaml         # YOLOv8 dataset configuration
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── scripts/
│   ├── analyze_data.py          # [Step 1] Dataset overview analysis
│   ├── inspect_labelme.py       # [Step 2] Annotation quality check
│   ├── labelme_to_yolo.py       # [Step 3] Format conversion + dataset split
│   ├── train_yolov8.py          # [Step 4] General training script (with args)
│   ├── train_yolov8m.py         # [Step 4] YOLOv8m training (AMP disabled)
│   ├── validate.py              # [Step 5] Model validation
│   ├── predict.py               # [Step 6] General inference
│   ├── predict_test.py          # [Step 6] Test set inference visualization
│   └── export_onnx.py           # Export ONNX model
├── serving/
│   ├── python/
│   │   ├── app.py               # FastAPI inference service
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── cpp/
│   │   ├── main.cpp             # C++ ONNX Runtime inference service
│   │   ├── CMakeLists.txt
│   │   └── Dockerfile
│   └── benchmark.py             # Performance comparison benchmark script
├── runs/
│   ├── detect/train_logs/       # Training logs (args.yaml, results.csv, train_log.txt)
│   └── predict/test_visualize/  # Test set detection visualization (14 annotated images)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `ultralytics>=8.2.0`, `opencv-python>=4.8.0`

## Quick Reproduction Commands

```bash
# 1. Check annotations
python scripts/inspect_labelme.py --src "/path/to/labelme_dataset"

# 2. Convert dataset
python scripts/labelme_to_yolo.py \
  --src "/path/to/labelme_dataset" \
  --out datasets/pallet_box \
  --classes configs/classes.txt

# 3. Train (requires GPU)
python scripts/train_yolov8m.py

# 4. Validate
python scripts/validate.py --weights runs/detect/pallet_box_yolov8m/weights/best.pt

# 5. Inference
python scripts/predict.py \
  --weights runs/detect/pallet_box_yolov8m/weights/best.pt \
  --source "/path/to/labelme_dataset" \
  --conf 0.25
```

## Docker Deployment Example

### Docker Training

```bash
docker run -d --name yolov8m-pallet-train \
  --gpus device=0 \
  --shm-size=8g \
  -v /path/to/pallet_box_yolov8:/workspace/pallet_box_yolov8 \
  -w /workspace/pallet_box_yolov8 \
  yolov8-obb-train \
  bash -c 'pip install numpy==1.26.4 -q; python3 scripts/train_yolov8m.py'
```

### Notes

- `--shm-size=8g`: DataLoader with multiple workers requires sufficient shared memory
- `numpy==1.26.4`: PyTorch in the container is incompatible with numpy 2.x
- `amp=False`: AMP check fails in this container environment
- `path` in dataset.yaml must match the container mount path

## Applying to New Scenarios

To detect other targets (e.g., forklifts, empty pallets, damaged boxes):

1. Annotate new targets with LabelMe rectangle boxes
2. Edit `configs/classes.txt` to add new classes
3. Re-run `labelme_to_yolo.py` to convert the dataset
4. Run the training script

The model automatically adapts to the new class definitions.

---

## RESTful Inference Services

After training, the model is packaged into two HTTP inference services for comparison:

### API Endpoint

Both services provide the same interface:

**POST /detect** — Upload an image, return detection results

Request: `multipart/form-data`, field `file` contains the image file

Response example:
```json
{
  "image_name": "03_008_1_1_20260626070605_rotated.jpg",
  "image_size": {"width": 1080, "height": 1920},
  "num_boxes": 3,
  "boxes": [
    {
      "class_id": 0,
      "class_name": "goods_stack",
      "confidence": 0.8237,
      "bbox": {"x1": 0.0, "y1": 936.3, "x2": 235.3, "y2": 1542.4}
    }
  ],
  "timing_ms": {"inference": 10.39, "total": 22.46}
}
```

**GET /health** — Health check

### Python Service (serving/python/)

| Item | Description |
|------|-------------|
| Framework | FastAPI + Uvicorn |
| Inference engine | ultralytics (PyTorch CUDA) |
| Model format | best.pt (native PyTorch) |
| Port | 8000 |
| GPU | NVIDIA RTX A6000 (device=0) |

```bash
# Deploy
docker run -d --name yolo-python-server \
  --gpus device=0 --shm-size=2g -p 8000:8000 \
  -v /path/to/weights:/workspace/models \
  -v /path/to/serving/python:/app -w /app \
  yolov8-obb-train \
  bash -c 'pip install numpy==1.26.4 fastapi uvicorn python-multipart -q; python3 app.py'
```

### C++ Service (serving/cpp/)

| Item | Description |
|------|-------------|
| HTTP framework | cpp-httplib (header-only) |
| Inference engine | ONNX Runtime 1.17.1 (CUDA EP) |
| Model format | best.onnx (98.8MB) |
| Image processing | OpenCV 4.5 |
| JSON | nlohmann/json |
| Port | 8001 |
| GPU | NVIDIA RTX A6000 (device=1) |

```bash
# Build image
docker build -t yolo-cpp-server serving/cpp/

# Deploy
docker run -d --name yolo-cpp-server \
  --gpus device=1 -p 8001:8001 \
  -v /path/to/weights:/workspace/models \
  yolo-cpp-server
```

### Performance Benchmark

Test conditions: 14 test images, 5 rounds, sequential requests (concurrency=1), local loopback network

```bash
python serving/benchmark.py \
  --images-dir datasets/pallet_box/images/test \
  --rounds 5 --max-images 14
```

**Results:**

| Metric | Python (PyTorch CUDA) | C++ (ONNX Runtime CUDA) |
|--------|------|------|
| Inference latency (mean) | **10.39 ms** | 20.51 ms |
| Inference latency (p95) | **11.24 ms** | 22.75 ms |
| Server total latency (mean) | **22.46 ms** | 55.10 ms |
| Client total latency (mean) | **29.69 ms** | 66.29 ms |
| Throughput (sequential) | **33.7 req/s** | 15.1 req/s |

### Performance Analysis

The Python service is actually faster than C++ in GPU inference scenarios. Reasons:

1. **PyTorch CUDA kernel optimization**: ultralytics' PyTorch inference path uses highly optimized CUDA kernels (cuDNN auto-tuning, memory pooling), battle-tested by a large community.

2. **ONNX Runtime CUDA EP overhead**: ORT's CUDA Execution Provider performs runtime operator mapping and memory management, which introduces more dispatch overhead than PyTorch's native CUDA path for dense convolution models like YOLOv8.

3. **Preprocessing differences**: The Python service uses PyTorch tensor operations for preprocessing (directly on GPU), while the C++ service does letterbox + normalize on CPU with OpenCV before transferring to GPU.

4. **Model format**: .pt is PyTorch's native format requiring no format conversion at inference time; .onnx requires ORT to parse the execution graph and map it to CUDA kernels.

**Conclusion**: For GPU inference scenarios, PyTorch's native approach is already highly efficient. C++ + ONNX Runtime is better suited for CPU deployment or embedded scenarios. To surpass PyTorch on GPU, use TensorRT (expected 2-4x speedup).

## Future Improvements

- Add more annotations: 417 unannotated images can further improve generalization
- Cover more scenarios: different lighting, occlusion, stack heights, empty pallets
- Model export: TensorRT for production deployment (expected faster than ONNX Runtime)
- Multi-class extension: distinguish full/partial/empty pallet subtypes
- C++ service optimization: replace with TensorRT EP or direct TensorRT C++ API to surpass Python performance
