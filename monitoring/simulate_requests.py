"""M5 post-deployment tracking: send a batch of (real or simulated) requests
with known true labels to the running inference service, then compare
predictions against ground truth to track live model performance.

Usage:
    python monitoring/simulate_requests.py --base-url http://localhost:8000 \
        --test-dir data/processed/test --n-per-class 25
"""
import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def sample_images(test_dir: Path, n_per_class: int, seed: int) -> list:
    rng = random.Random(seed)
    samples = []
    for cls in ("cats", "dogs"):
        class_dir = test_dir / cls
        paths = sorted(class_dir.glob("*.jpg"))
        chosen = rng.sample(paths, min(n_per_class, len(paths)))
        true_label = "cat" if cls == "cats" else "dog"
        samples.extend((p, true_label) for p in chosen)
    rng.shuffle(samples)
    return samples


def call_predict(base_url: str, image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        start = time.perf_counter()
        resp = requests.post(f"{base_url}/predict", files=files, timeout=10)
        latency_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    body = resp.json()
    body["client_latency_ms"] = round(latency_ms, 2)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--test-dir", default="data/processed/test")
    parser.add_argument("--n-per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    samples = sample_images(Path(args.test_dir), args.n_per_class, args.seed)
    if not samples:
        raise SystemExit(f"No images found under {args.test_dir}. Run preprocessing first.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "post_deployment_requests.csv"
    is_new_log = not log_path.exists()

    correct = 0
    latencies = []

    with open(log_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if is_new_log:
            writer.writerow(["timestamp", "image", "true_label", "predicted_label",
                              "prob_cat", "prob_dog", "server_latency_ms", "client_latency_ms", "correct"])

        for image_path, true_label in samples:
            result = call_predict(args.base_url, image_path)
            is_correct = result["label"] == true_label
            correct += int(is_correct)
            latencies.append(result["client_latency_ms"])

            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                image_path.name,
                true_label,
                result["label"],
                result["probabilities"]["cat"],
                result["probabilities"]["dog"],
                result["latency_ms"],
                result["client_latency_ms"],
                is_correct,
            ])

    accuracy = correct / len(samples)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_requests": len(samples),
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
    }

    with open(out_dir / "post_deployment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[monitoring] sent {len(samples)} requests, live accuracy={accuracy:.3f}")
    print(f"[monitoring] per-request log appended to {log_path}")
    print(f"[monitoring] summary written to {out_dir / 'post_deployment_summary.json'}")


if __name__ == "__main__":
    main()
