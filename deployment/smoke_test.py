"""Post-deploy smoke test: hits /health and makes one /predict call.

Exits non-zero (failing the pipeline) if either check fails.

Usage:
    python deployment/smoke_test.py --base-url http://localhost:8000
"""
import argparse
import io
import sys
import time

import requests
from PIL import Image


def make_sample_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(90, 140, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def wait_for_health(base_url: str, retries: int, delay: float) -> None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            resp.raise_for_status()
            body = resp.json()
            if body.get("status") == "ok" and body.get("model_loaded"):
                print(f"[smoke-test] health OK on attempt {attempt}: {body}")
                return
            last_error = f"unexpected health body: {body}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        print(f"[smoke-test] health check attempt {attempt}/{retries} failed: {last_error}")
        time.sleep(delay)
    raise SystemExit(f"[smoke-test] FAILED: health check never succeeded ({last_error})")


def run_prediction_check(base_url: str) -> None:
    files = {"file": ("smoke.jpg", make_sample_image_bytes(), "image/jpeg")}
    resp = requests.post(f"{base_url}/predict", files=files, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    assert body["label"] in {"cat", "dog"}, f"unexpected label: {body}"
    assert "cat" in body["probabilities"] and "dog" in body["probabilities"]
    print(f"[smoke-test] prediction OK: {body}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()

    wait_for_health(args.base_url, args.retries, args.delay)
    run_prediction_check(args.base_url)
    print("[smoke-test] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
