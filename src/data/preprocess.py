"""Pre-processing utilities: resize/normalize images and build the
train/val/test split from the raw Kaggle Cats-vs-Dogs download.

The raw dataset (kaggle dataset `salader/dogsvscats`) is laid out as:
    data/raw/train/cats/*.jpg
    data/raw/train/dogs/*.jpg
    data/raw/test/cats/*.jpg
    data/raw/test/dogs/*.jpg

We pool every labelled image and re-split it ourselves so the
train/val/test ratios match the assignment spec exactly.
"""
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

CLASSES = ["cats", "dogs"]


def resize_and_normalize(image: Image.Image, size: int = 224) -> np.ndarray:
    """Convert an image to RGB, resize to (size, size) and scale pixels to [0, 1].

    Pure function (no file I/O) so it is easy to unit test.
    """
    rgb = image.convert("RGB")
    resized = rgb.resize((size, size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array


def collect_labeled_images(raw_dir: Path) -> Dict[str, List[Path]]:
    """Gather all cat/dog image paths from every split folder under raw_dir."""
    images: Dict[str, List[Path]] = {cls: [] for cls in CLASSES}
    for split_dir in raw_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for cls in CLASSES:
            class_dir = split_dir / cls
            if class_dir.is_dir():
                images[cls].extend(sorted(class_dir.glob("*.jpg")))
    return images


def split_paths(
    paths: List[Path], train_split: float, val_split: float, seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Deterministically shuffle and split a list of paths into train/val/test."""
    shuffled = paths[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def build_processed_dataset(
    raw_dir: Path,
    processed_dir: Path,
    image_size: int,
    max_per_class: int,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> Dict[str, int]:
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6

    images_by_class = collect_labeled_images(raw_dir)
    counts: Dict[str, int] = {}

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    for cls, paths in images_by_class.items():
        capped = paths[:max_per_class] if max_per_class else paths
        train, val, test = split_paths(capped, train_split, val_split, seed)
        for split_name, split_paths_list in (("train", train), ("val", val), ("test", test)):
            out_dir = processed_dir / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for src_path in split_paths_list:
                with Image.open(src_path) as img:
                    array = resize_and_normalize(img, size=image_size)
                out_img = Image.fromarray((array * 255).astype(np.uint8))
                out_img.save(out_dir / src_path.name)
            counts[f"{split_name}_{cls}"] = len(split_paths_list)

    return counts


def load_params(params_path: Path) -> dict:
    import yaml  # dev-only dependency, kept out of the lean inference image
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=str, default="params.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(Path(args.params))["data"]

    counts = build_processed_dataset(
        raw_dir=Path(params["raw_dir"]),
        processed_dir=Path(params["processed_dir"]),
        image_size=params["image_size"],
        max_per_class=params["max_per_class"],
        train_split=params["train_split"],
        val_split=params["val_split"],
        test_split=params["test_split"],
        seed=params["seed"],
    )
    print("Processed dataset counts:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
