"""Benchmark script to compare Python and C++ inference services."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def send_request(url: str, image_path: Path) -> dict:
    """Send a single detection request and return timing info."""
    start = time.time()
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"file": (image_path.name, f, "image/jpeg")})
    total_client = (time.time() - start) * 1000  # ms

    if resp.status_code != 200:
        return {"error": resp.text, "client_total_ms": total_client}

    data = resp.json()
    data["client_total_ms"] = round(total_client, 2)
    return data


def benchmark_service(
    url: str,
    images: list[Path],
    rounds: int = 3,
    concurrency: int = 1,
) -> dict:
    """Run benchmark against a service endpoint."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {url}")
    print(f"Images: {len(images)}, Rounds: {rounds}, Concurrency: {concurrency}")
    print(f"{'='*60}")

    # Warmup (3 requests)
    print("Warming up...")
    for img in images[:3]:
        send_request(url, img)

    # Benchmark
    all_infer_times = []
    all_total_times = []
    all_client_times = []

    for round_idx in range(rounds):
        round_start = time.time()

        if concurrency == 1:
            # Sequential
            for img in images:
                result = send_request(url, img)
                if "error" not in result:
                    all_infer_times.append(result["timing_ms"]["inference"])
                    all_total_times.append(result["timing_ms"]["total"])
                    all_client_times.append(result["client_total_ms"])
        else:
            # Concurrent
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(send_request, url, img) for img in images]
                for future in as_completed(futures):
                    result = future.result()
                    if "error" not in result:
                        all_infer_times.append(result["timing_ms"]["inference"])
                        all_total_times.append(result["timing_ms"]["total"])
                        all_client_times.append(result["client_total_ms"])

        round_time = (time.time() - round_start) * 1000
        print(f"  Round {round_idx + 1}/{rounds}: {round_time:.0f}ms total")

    # Statistics
    stats = {}
    if all_infer_times:
        stats = {
            "num_requests": len(all_infer_times),
            "inference_ms": {
                "mean": round(statistics.mean(all_infer_times), 2),
                "median": round(statistics.median(all_infer_times), 2),
                "p95": round(sorted(all_infer_times)[int(len(all_infer_times) * 0.95)], 2),
                "min": round(min(all_infer_times), 2),
                "max": round(max(all_infer_times), 2),
            },
            "server_total_ms": {
                "mean": round(statistics.mean(all_total_times), 2),
                "median": round(statistics.median(all_total_times), 2),
                "p95": round(sorted(all_total_times)[int(len(all_total_times) * 0.95)], 2),
            },
            "client_total_ms": {
                "mean": round(statistics.mean(all_client_times), 2),
                "median": round(statistics.median(all_client_times), 2),
                "p95": round(sorted(all_client_times)[int(len(all_client_times) * 0.95)], 2),
            },
            "throughput_rps": round(len(all_infer_times) / (sum(all_client_times) / 1000), 2) if concurrency == 1 else None,
        }

    print(f"\nResults:")
    print(f"  Inference  - mean: {stats['inference_ms']['mean']:.2f}ms, "
          f"median: {stats['inference_ms']['median']:.2f}ms, "
          f"p95: {stats['inference_ms']['p95']:.2f}ms")
    print(f"  Server E2E - mean: {stats['server_total_ms']['mean']:.2f}ms, "
          f"median: {stats['server_total_ms']['median']:.2f}ms")
    print(f"  Client E2E - mean: {stats['client_total_ms']['mean']:.2f}ms, "
          f"median: {stats['client_total_ms']['median']:.2f}ms")
    if stats.get("throughput_rps"):
        print(f"  Throughput - {stats['throughput_rps']:.1f} req/s (sequential)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 inference services.")
    parser.add_argument("--python-url", default="http://localhost:8000/detect")
    parser.add_argument("--cpp-url", default="http://localhost:8001/detect")
    parser.add_argument("--images-dir", required=True, help="Directory with test images")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_images:
        images = images[:args.max_images]

    print(f"Test images: {len(images)} from {images_dir}")

    results = {}

    # Test Python service
    try:
        resp = requests.get(args.python_url.replace("/detect", "/health"), timeout=5)
        if resp.status_code == 200:
            results["python"] = benchmark_service(args.python_url, images, args.rounds, args.concurrency)
    except Exception as e:
        print(f"\nPython service not available: {e}")

    # Test C++ service
    try:
        resp = requests.get(args.cpp_url.replace("/detect", "/health"), timeout=5)
        if resp.status_code == 200:
            results["cpp"] = benchmark_service(args.cpp_url, images, args.rounds, args.concurrency)
    except Exception as e:
        print(f"\nC++ service not available: {e}")

    # Comparison
    if "python" in results and "cpp" in results:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        py = results["python"]
        cpp = results["cpp"]
        speedup_infer = py["inference_ms"]["mean"] / cpp["inference_ms"]["mean"]
        speedup_total = py["server_total_ms"]["mean"] / cpp["server_total_ms"]["mean"]
        print(f"  Inference speedup (C++/Python): {speedup_infer:.2f}x")
        print(f"  Server E2E speedup (C++/Python): {speedup_total:.2f}x")
        print(f"\n  {'Metric':<20} {'Python':>12} {'C++':>12} {'Speedup':>10}")
        print(f"  {'-'*54}")
        print(f"  {'Infer mean (ms)':<20} {py['inference_ms']['mean']:>12.2f} {cpp['inference_ms']['mean']:>12.2f} {speedup_infer:>9.2f}x")
        print(f"  {'Infer p95 (ms)':<20} {py['inference_ms']['p95']:>12.2f} {cpp['inference_ms']['p95']:>12.2f}")
        print(f"  {'Server E2E (ms)':<20} {py['server_total_ms']['mean']:>12.2f} {cpp['server_total_ms']['mean']:>12.2f} {speedup_total:>9.2f}x")
        print(f"  {'Client E2E (ms)':<20} {py['client_total_ms']['mean']:>12.2f} {cpp['client_total_ms']['mean']:>12.2f}")

    # Save results
    output_file = Path("benchmark_results.json")
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
