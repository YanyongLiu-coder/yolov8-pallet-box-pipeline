from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LabelMe rectangles/polygons to YOLO detection dataset.")
    parser.add_argument("--src", required=True, help="Directory containing images and LabelMe json files.")
    parser.add_argument("--out", required=True, help="Output YOLO dataset directory.")
    parser.add_argument("--classes", required=True, help="Class names txt, one class per line.")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-unlabeled-as-negative",
        action="store_true",
        help="Include images without json as negative samples with empty label files. Use only if they truly have no target.",
    )
    return parser.parse_args()


def read_classes(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No classes found in {path}")
    return names


def find_image(src: Path, image_name: str, json_path: Path) -> Path | None:
    candidates = [src / image_name]
    candidates.extend(json_path.with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
    return None


def points_to_yolo_box(points: list[list[float]], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(image_width), max(xs))
    y_max = min(float(image_height), max(ys))

    box_width = max(0.0, x_max - x_min)
    box_height = max(0.0, y_max - y_min)
    x_center = x_min + box_width / 2.0
    y_center = y_min + box_height / 2.0

    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def convert_json(json_path: Path, src: Path, class_to_id: dict[str, int]) -> tuple[Path, list[str]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_width = int(data.get("imageWidth") or 0)
    image_height = int(data.get("imageHeight") or 0)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size in {json_path}")

    image_name = data.get("imagePath") or json_path.with_suffix(".jpg").name
    image_path = find_image(src, image_name, json_path)
    if image_path is None:
        raise FileNotFoundError(f"Image not found for {json_path}")

    lines: list[str] = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        if label not in class_to_id:
            raise ValueError(f"Unknown label '{label}' in {json_path}. Add it to classes.txt if needed.")

        points = shape.get("points") or []
        if len(points) < 2:
            continue

        x_center, y_center, width, height = points_to_yolo_box(points, image_width, image_height)
        if width <= 0 or height <= 0:
            continue

        class_id = class_to_id[label]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return image_path, lines


def split_items(items: list[tuple[Path, list[str]]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[tuple[Path, list[str]]]]:
    random.Random(seed).shuffle(items)
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def write_split(split: str, items: list[tuple[Path, list[str]]], out: Path) -> None:
    image_dir = out / "images" / split
    label_dir = out / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_lines in items:
        target_image = image_dir / image_path.name
        target_label = label_dir / f"{image_path.stem}.txt"
        shutil.copy2(image_path, target_image)
        target_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def write_dataset_yaml(out: Path, names: list[str]) -> None:
    dataset_path = str(out.resolve()).replace("\\", "/")
    lines = [
        f"path: {dataset_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    (out / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    class_names = read_classes(Path(args.classes))
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    json_files = sorted(src.glob("*.json"))
    items: list[tuple[Path, list[str]]] = []
    used_images: set[Path] = set()

    for index, json_path in enumerate(json_files, start=1):
        if index == 1 or index == len(json_files) or index % 20 == 0:
            print(f"Converting json {index}/{len(json_files)}")
        image_path, label_lines = convert_json(json_path, src, class_to_id)
        if label_lines:
            items.append((image_path, label_lines))
            used_images.add(image_path.resolve())

    if args.include_unlabeled_as_negative:
        for image_path in src.iterdir():
            if image_path.suffix.lower() in IMAGE_SUFFIXES and image_path.resolve() not in used_images:
                items.append((image_path, []))

    if not items:
        raise SystemExit("No usable labeled samples found.")

    splits = split_items(items, args.train_ratio, args.val_ratio, args.seed)
    for split, split_items_value in splits.items():
        write_split(split, split_items_value, out)

    write_dataset_yaml(out, class_names)

    print(f"Dataset written to: {out.resolve()}")
    for split, split_items_value in splits.items():
        boxes = sum(len(labels) for _, labels in split_items_value)
        print(f"{split}: {len(split_items_value)} images, {boxes} boxes")
    print(f"Dataset yaml: {(out / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()
