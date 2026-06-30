from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect LabelMe annotations.")
    parser.add_argument("--src", required=True, help="Directory containing images and LabelMe json files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Source directory does not exist: {src}")

    image_files = [p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    json_files = sorted(src.glob("*.json"))

    labels: Counter[str] = Counter()
    shape_types: Counter[str] = Counter()
    annotated_images = set()

    for json_path in json_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = data.get("imagePath") or json_path.with_suffix(".jpg").name
        annotated_images.add(image_path)
        for shape in data.get("shapes", []):
            labels[str(shape.get("label", ""))] += 1
            shape_types[str(shape.get("shape_type", ""))] += 1

    print(f"Images: {len(image_files)}")
    print(f"Json files: {len(json_files)}")
    print(f"Images with json: {len(annotated_images)}")
    print(f"Images without json: {len(image_files) - len(annotated_images)}")
    print("\nLabels:")
    for label, count in labels.most_common():
        print(f"  {label}: {count}")
    print("\nShape types:")
    for shape_type, count in shape_types.most_common():
        print(f"  {shape_type}: {count}")


if __name__ == "__main__":
    main()

