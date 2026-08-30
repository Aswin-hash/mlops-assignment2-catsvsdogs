import io

from fastapi.testclient import TestClient
from PIL import Image

import src.api.main as api_main
from src.models.model import build_model


def _fake_image_bytes() -> bytes:
    img = Image.new("RGB", (300, 200), color=(10, 200, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_and_predict_endpoints():
    # Inject a randomly-initialized model so the test never depends on a
    # real trained checkpoint being present on disk.
    api_main._model = build_model("simple_cnn").eval()
    client = TestClient(api_main.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": True, "version": "1.1.0"}

    files = {"file": ("pet.jpg", _fake_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"cat", "dog"}
    assert "cat" in body["probabilities"] and "dog" in body["probabilities"]

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert b"predictions_total" in metrics.content


def test_predict_rejects_non_image_file():
    api_main._model = build_model("simple_cnn").eval()
    client = TestClient(api_main.app)

    files = {"file": ("note.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
