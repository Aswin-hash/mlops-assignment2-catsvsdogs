"""Download the Cats vs Dogs dataset from Kaggle into data/raw.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (see
https://www.kaggle.com/docs/api). Usage:

    python -m src.data.download --dest data/raw
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "salader/dogsvscats"


def _kaggle_executable() -> str:
    """Locate the `kaggle` console script installed alongside this interpreter."""
    scripts_dir = Path(sys.executable).parent
    candidate = scripts_dir / ("kaggle.exe" if sys.platform == "win32" else "kaggle")
    return str(candidate) if candidate.exists() else "kaggle"


def download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        _kaggle_executable(), "datasets", "download",
        "-d", DATASET,
        "-p", str(dest),
        "--unzip",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Dataset downloaded and extracted to {dest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=str, default="data/raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(Path(args.dest))


if __name__ == "__main__":
    main()
