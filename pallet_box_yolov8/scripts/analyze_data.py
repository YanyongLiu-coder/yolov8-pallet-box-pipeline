"""Analyze the raw dataset: count images, jsons, labeled/unlabeled images."""
import argparse
import json
from pathlib import Path
from collections import Counter

parser = argparse.ArgumentParser(description="Analyze a LabelMe-style image dataset.")
parser.add_argument("--data-dir", required=True, help="Directory containing images and LabelMe JSON files.")
args = parser.parse_args()

data_dir = Path(args.data_dir)
if not data_dir.exists():
    raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

images = sorted([f for f in data_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
jsons = sorted([f for f in data_dir.iterdir() if f.suffix.lower() == '.json'])

labeled_images = []
unlabeled_images = []
for img in images:
    json_file = img.with_suffix('.json')
    if json_file.exists():
        labeled_images.append(img)
    else:
        unlabeled_images.append(img)

print(f'Total images: {len(images)}')
print(f'Total json files: {len(jsons)}')
print(f'Images WITH annotation: {len(labeled_images)}')
print(f'Images WITHOUT annotation: {len(unlabeled_images)}')
print()

labels = Counter()
shape_types = Counter()
empty_jsons = 0
non_empty_jsons = 0
for jf in jsons:
    data = json.loads(jf.read_text())
    shapes = data.get('shapes', [])
    if not shapes:
        empty_jsons += 1
    else:
        non_empty_jsons += 1
    for s in shapes:
        labels[s.get('label', '')] += 1
        shape_types[s.get('shape_type', '')] += 1

print(f'Json with shapes (labeled): {non_empty_jsons}')
print(f'Json without shapes (empty): {empty_jsons}')
print(f'\nLabel distribution:')
for label, count in labels.most_common():
    print(f'  {label}: {count}')
print(f'\nShape types:')
for st, count in shape_types.most_common():
    print(f'  {st}: {count}')

print(f'\n--- Image prefix distribution (camera_position) ---')
prefixes = Counter()
for img in images:
    parts = img.stem.split('_')
    prefix = parts[0] + '_' + parts[1]
    prefixes[prefix] += 1

labeled_prefixes = Counter()
for img in labeled_images:
    parts = img.stem.split('_')
    prefix = parts[0] + '_' + parts[1]
    labeled_prefixes[prefix] += 1

print(f'Total unique positions: {len(prefixes)}')
print(f'\nPositions (total / labeled):')
for p, c in sorted(prefixes.items()):
    lc = labeled_prefixes.get(p, 0)
    print(f'  {p}: {c} total, {lc} labeled')

print(f'\n--- Unlabeled images sample (first 15) ---')
for img in unlabeled_images[:15]:
    print(f'  {img.name}')
