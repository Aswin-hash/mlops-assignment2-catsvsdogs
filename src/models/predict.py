"""Inference utilities shared by the training/eval script and the FastAPI service."""
import io
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image

from src.data.preprocess import resize_and_normalize
from src.models.model import build_model

LABELS = {0: "cat", 1: "dog"}
IMAGE_SIZE = 224


def load_model(checkpoint_path: Union[str, Path], model_name: str = "simple_cnn", device: str = "cpu"):
    """Load a trained model checkpoint into eval mode."""
    model = build_model(model_name)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def image_bytes_to_tensor(image_bytes: bytes, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Decode raw image bytes into a normalized (1, 3, H, W) float tensor."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        array = resize_and_normalize(img, size=size)  # (H, W, 3) in [0, 1]
    chw = np.transpose(array, (2, 0, 1))  # (3, H, W)
    tensor = torch.from_numpy(chw).unsqueeze(0).float()  # (1, 3, H, W)
    return tensor


@torch.no_grad()
def predict_tensor(model, tensor: torch.Tensor) -> Dict[str, object]:
    """Run inference and return a label + class probabilities."""
    logit = model(tensor)
    prob_dog = torch.sigmoid(logit).item()
    prob_cat = 1.0 - prob_dog
    label = LABELS[1] if prob_dog >= 0.5 else LABELS[0]
    return {
        "label": label,
        "probabilities": {"cat": round(prob_cat, 4), "dog": round(prob_dog, 4)},
    }


@torch.no_grad()
def predict_image_bytes(model, image_bytes: bytes) -> Dict[str, object]:
    tensor = image_bytes_to_tensor(image_bytes)
    return predict_tensor(model, tensor)
