import numpy as np
from PIL import Image

from src.data.preprocess import resize_and_normalize, split_paths


def test_resize_and_normalize_shape_and_range():
    img = Image.new("RGB", (500, 333), color=(120, 40, 200))
    array = resize_and_normalize(img, size=224)

    assert array.shape == (224, 224, 3)
    assert array.dtype == np.float32
    assert array.min() >= 0.0
    assert array.max() <= 1.0


def test_resize_and_normalize_converts_grayscale_to_rgb():
    img = Image.new("L", (100, 100), color=128)
    array = resize_and_normalize(img, size=64)

    assert array.shape == (64, 64, 3)


def test_split_paths_ratios_and_no_overlap():
    paths = [f"img_{i}.jpg" for i in range(100)]
    train, val, test = split_paths(paths, train_split=0.8, val_split=0.1, seed=42)

    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(paths)


def test_split_paths_is_deterministic_for_same_seed():
    paths = [f"img_{i}.jpg" for i in range(50)]
    result_a = split_paths(paths, 0.8, 0.1, seed=7)
    result_b = split_paths(paths, 0.8, 0.1, seed=7)

    assert result_a == result_b
