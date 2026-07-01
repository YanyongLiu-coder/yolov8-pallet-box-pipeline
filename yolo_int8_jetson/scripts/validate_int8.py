"""
Validate INT8 quantized model against FP16 baseline.

Compares detection results between FP16 and INT8 engines on the validation set
to measure accuracy degradation from quantization.

Usage:
    python3 scripts/validate_int8.py \
        --fp16-url http://localhost:8002/detect \
        --int8-url http://localhost:8003/detect \
        --images-dir /path/to/val_images
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FP16 vs INT8 detection accuracy.")
    parser.add_argument("--fp16-url", default="http://localhost:8002/detect")
    parser.add_argument("--int8-url", default="http://localhost:8003/detect")
    parser.add_argument("--images-dir", required=True, help="Validation images directory.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=50)
    return parser.parse_args()


def detect(url: str, image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"file": (image_path.name, f, "image/jpeg")})
    if resp.status_code != 200:
        return {"num_boxes": 0, "boxes": []}
    return resp.json()


def iou(box_a: dict, box_b: dict) -> float:
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def compare_detections(fp16_result: dict, int8_result: dict, iou_thresh: float = 0.5) -> dict:
    fp16_boxes = fp16_result.get("boxes", [])
    int8_boxes = int8_result.get("boxes", [])

    matched = 0
    for fb in fp16_boxes:
        for ib in int8_boxes:
            if iou(fb["bbox"], ib["bbox"]) > iou_thresh:
                matched += 1
                break

    return {
        "fp16_count": len(fp16_boxes),
        "int8_count": len(int8_boxes),
        "matched": matched,
        "fp16_infer_ms": fp16_result.get("timing_ms", {}).get("inference", 0),
        "int8_infer_ms": int8_result.get("timing_ms", {}).get("inference", 0),
    }


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    images = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    images = images[:args.max_images]

    print(f"Comparing FP16 vs INT8 on {len(images)} images")
    print(f"  FP16: {args.fp16_url}")
    print(f"  INT8: {args.int8_url}")
    print()

    results = []
    for i, img_path in enumerate(images):
        fp16_res = detect(args.fp16_url, img_path)
        int8_res = detect(args.int8_url, img_path)
        comp = compare_detections(fp16_res, int8_res)
        results.append(comp)

        if (i + 1) % 10 == 0 or i == len(images) - 1:
            print(f"  Processed {i+1}/{len(images)} images")

    total_fp16 = sum(r["fp16_count"] for r in results)
    total_int8 = sum(r["int8_count"] for r in results)
    total_matched = sum(r["matched"] for r in results)
    avg_fp16_ms = sum(r["fp16_infer_ms"] for r in results) / len(results) if results else 0
    avg_int8_ms = sum(r["int8_infer_ms"] for r in results) / len(results) if results else 0

    match_rate = total_matched / total_fp16 * 100 if total_fp16 > 0 else 0

    print(f"\n{'='*50}")
    print(f"QUANTIZATION ACCURACY COMPARISON")
    print(f"{'='*50}")
    print(f"  Images evaluated:    {len(images)}")
    print(f"  FP16 total boxes:    {total_fp16}")
    print(f"  INT8 total boxes:    {total_int8}")
    print(f"  Matched (IoU>0.5):   {total_matched}")
    print(f"  Match rate:          {match_rate:.1f}%")
    print(f"")
    print(f"  FP16 avg inference:  {avg_fp16_ms:.2f} ms")
    print(f"  INT8 avg inference:  {avg_int8_ms:.2f} ms")
    if avg_int8_ms > 0:
        print(f"  Speedup:             {avg_fp16_ms/avg_int8_ms:.2f}x")
    print()

    if match_rate >= 95:
        print("  RESULT: INT8 quantization quality EXCELLENT (>95% match)")
    elif match_rate >= 90:
        print("  RESULT: INT8 quantization quality GOOD (>90% match)")
    elif match_rate >= 80:
        print("  RESULT: INT8 quantization quality ACCEPTABLE (>80% match)")
    else:
        print("  RESULT: INT8 quantization quality POOR (<80% match, consider more calibration data)")


if __name__ == "__main__":
    main()
