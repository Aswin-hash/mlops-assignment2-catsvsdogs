import io

import torch
from PIL import Image

from src.models.model import build_model
from src.models.predict import image_bytes_to_tensor, predict_tensor


def _fake_image_bytes(size=(300, 200), color=(10, 200, 30)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_image_bytes_to_tensor_shape():
    tensor = image_bytes_to_tensor(_fake_image_bytes(), size=224)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_predict_tensor_returns_valid_label_and_probabilities():
    model = build_model("simple_cnn")
    model.eval()
    tensor = image_bytes_to_tensor(_fake_image_bytes(), size=224)

    result = predict_tensor(model, tensor)

    assert result["label"] in {"cat", "dog"}
    probs = result["probabilities"]
    assert set(probs.keys()) == {"cat", "dog"}
    assert abs(probs["cat"] + probs["dog"] - 1.0) < 1e-3
    assert 0.0 <= probs["cat"] <= 1.0
    assert 0.0 <= probs["dog"] <= 1.0


def test_predict_tensor_label_matches_higher_probability():
    model = build_model("simple_cnn")
    model.eval()
    tensor = image_bytes_to_tensor(_fake_image_bytes(), size=224)

    result = predict_tensor(model, tensor)
    probs = result["probabilities"]
    expected_label = "dog" if probs["dog"] >= probs["cat"] else "cat"

    assert result["label"] == expected_label
