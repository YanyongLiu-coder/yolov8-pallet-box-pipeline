from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a YOLO test image, with a synthetic fallback.")
    parser.add_argument("--output", default="images/bus.jpg")
    parser.add_argument("--url", default="https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg")
    return parser.parse_args()


def write_fallback(path: Path) -> None:
    image = np.full((640, 960, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (120, 180), (820, 520), (40, 90, 180), -1)
    cv2.rectangle(image, (190, 250), (310, 520), (30, 30, 30), -1)
    cv2.rectangle(image, (620, 250), (740, 520), (30, 30, 30), -1)
    cv2.circle(image, (260, 540), 55, (20, 20, 20), -1)
    cv2.circle(image, (680, 540), 55, (20, 20, 20), -1)
    cv2.putText(image, "fallback test image", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(str(path), image)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(args.url, output)
        print(f"Downloaded test image to: {output}")
    except Exception as exc:
        print(f"Download failed: {exc}. Writing synthetic fallback image.")
        write_fallback(output)
        print(f"Fallback test image saved to: {output}")


if __name__ == "__main__":
    main()
